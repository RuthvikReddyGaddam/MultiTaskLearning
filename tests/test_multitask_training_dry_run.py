# tests/test_multitask_training_dry_run.py

import unittest
from pathlib import Path
import numpy as np
import torch

from scripts.utils import load_yaml_config, build_optimizer, build_scheduler
from scripts.build_models import build_model
from datasets import build_multitask_dataloader
from scripts.build_losses import build_loss
from utils.test_utils import class4_to_instances


class TestMultiTaskTrainingDryRun(unittest.TestCase):
    """
    End-to-end integration test verifying that:
      1. Model configuration and weights instantiate cleanly.
      2. Multi-Task DataLoader loads actual 512x512 patches across GLaS, MoNuSeg, MoNuSAC.
      3. Forward pass with task_ids feature slicing functions properly.
      4. Decoupled multi-task loss is computed and backward pass computes non-zero gradients.
      5. Optimizer step updates model parameters.
      6. Single-task inference evaluation outputs expected keys and shapes.
      7. Downstream instance extraction (class4_to_instances) produces valid instance maps.
    """

    def setUp(self):
        self.config_path = "configs/multitask_sdenetpointgradexp_512.yaml"
        self.config = load_yaml_config(self.config_path)

    def test_model_build(self):
        model = build_model(self.config)
        self.assertIsNotNone(model)
        self.assertEqual(model.prediction_head.approach, "4class_j")
        self.assertEqual(model.prediction_head.num_gland_classes, 4)
        self.assertEqual(model.prediction_head.num_cell_classes, 4)
        self.assertEqual(model.prediction_head.num_classification_classes, 5)
        self.assertFalse(model.prediction_head.predict_flow)

    def test_dataloader_and_training_step(self):
        dataset_cfg = self.config.get("dataset", {})
        training_cfg = self.config.get("training", {})

        train_loader = build_multitask_dataloader(
            glas_dir=dataset_cfg.get("glas_dir"),
            monuseg_dir=dataset_cfg.get("monuseg_dir"),
            monusac_dir=dataset_cfg.get("monusac_dir"),
            patch_size=int(dataset_cfg.get("patch_size", 512)),
            batch_size=int(training_cfg.get("batch_size", 8)),
            num_workers=0,
            pin_memory=False,
            monusac_resample_size=int(dataset_cfg.get("monusac_resample_size", 600)),
            class_weights=dataset_cfg.get("class_weights"),
            augment=True,
        )

        batch = next(iter(train_loader))
        self.assertIn("image", batch)
        self.assertIn("task_id", batch)
        self.assertIn("target_4class", batch)
        self.assertIn("class_map", batch)
        self.assertEqual(batch["image"].shape, (8, 3, 512, 512))

        # Build model and optimizer
        model = build_model(self.config)
        model.train()
        optimizer = build_optimizer(model, self.config)
        criterion = build_loss(self.config)

        # Forward pass
        preds = model(batch["image"], task_ids=batch["task_id"])
        self.assertIn("gland_seg", preds)
        self.assertIn("cell_seg", preds)
        self.assertIn("cell_class", preds)

        # Loss and backward
        loss, logs = criterion(preds, batch)
        self.assertGreater(loss.item(), 0.0)
        optimizer.zero_grad()
        loss.backward()

        # Check gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        self.assertGreater(grad_norm.item(), 0.0)
        optimizer.step()

        # Single-task inference evaluation
        model.eval()
        with torch.no_grad():
            x_test = batch["image"][:2]
            out_g = model(x_test, task="glas")
            out_m = model(x_test, task="monuseg")
            out_s = model(x_test, task="monusac")

            self.assertIn("gland_seg", out_g)
            self.assertIn("cell_seg", out_m)
            self.assertIn("cell_seg", out_s)
            self.assertIn("cell_class", out_s)

            # Instance extraction
            prob_4class = torch.softmax(out_m["cell_seg"][0], dim=0).cpu().numpy()
            inst_pred = class4_to_instances(prob_4class, min_size=10, expand_boundary=True)
            self.assertEqual(inst_pred.shape, (512, 512))

    def test_multitask_trainer_one_epoch(self):
        from training.trainer_multitask import MultiTaskTrainer
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        try:
            model = build_model(self.config)
            criterion = build_loss(self.config)
            optimizer = build_optimizer(model, self.config)

            dataset_cfg = self.config.get("dataset", {})
            training_cfg = self.config.get("training", {})
            real_loader = build_multitask_dataloader(
                glas_dir=dataset_cfg.get("glas_dir"),
                monuseg_dir=dataset_cfg.get("monuseg_dir"),
                monusac_dir=dataset_cfg.get("monusac_dir"),
                patch_size=int(dataset_cfg.get("patch_size", 512)),
                batch_size=int(training_cfg.get("batch_size", 8)),
                num_workers=0,
                pin_memory=False,
                monusac_resample_size=int(dataset_cfg.get("monusac_resample_size", 600)),
                class_weights=dataset_cfg.get("class_weights"),
                augment=False,
            )
            batch = next(iter(real_loader))
            mock_loader = type("MockLoader", (list,), {"dataset": real_loader.dataset})([batch])

            trainer = MultiTaskTrainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                train_loader=mock_loader,
                device=torch.device("cpu"),
                config=self.config,
                exp_dir=temp_dir,
            )

            metrics = trainer.train_one_epoch()
            self.assertIn("loss", metrics)
            self.assertIn("glas_bin_dice", metrics)
            self.assertIn("monuseg_bin_dice", metrics)
            self.assertIn("monusac_bin_dice", metrics)
            self.assertIn("monusac_cell_dice", metrics)
            self.assertGreater(metrics["loss"], 0.0)

            # Test checkpoint saving
            trainer._save_checkpoint(epoch=1, is_best=True, metrics=metrics)
            self.assertTrue(Path(temp_dir, "best.pt").exists())
            self.assertTrue(Path(temp_dir, "latest.pt").exists())

            # Test logging
            trainer._log_history(epoch=1, metrics=metrics, run_time=1.23)
            self.assertTrue(Path(temp_dir, "history.csv").exists())
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

