import unittest
import torch
import torch.nn as nn
from datasets.multitask_sampler import EqualTaskBatchSampler
from scripts.utils import load_yaml_config
from scripts.build_models import build_model
from scripts.build_losses import build_loss


class TestShuffledMultiTaskBatch(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.config = load_yaml_config("configs/multitask_sdenetpointgradexp_512.yaml")

    def test_sampler_exact_ratio_and_shuffled(self):
        lengths = [100, 50, 75]
        sampler = EqualTaskBatchSampler(
            dataset_lengths=lengths,
            batch_size=8,
            shuffle=True,
            shuffle_within_batch=True,
            steps_per_epoch=10,
        )
        for batch in sampler:
            self.assertEqual(len(batch), 8)
            g_count = sum(1 for idx in batch if 0 <= idx < 100)
            m_count = sum(1 for idx in batch if 100 <= idx < 150)
            s_count = sum(1 for idx in batch if 150 <= idx < 225)
            self.assertEqual(g_count, 4, "GLaS count must be 4 (B//2)")
            self.assertEqual(m_count, 2, "MoNuSeg count must be 2 (B//4)")
            self.assertEqual(s_count, 2, "MoNuSAC count must be 2 (B//4)")

    def test_model_forward_loss_backward_with_shuffled_batch(self):
        # 1. Instantiate MultiTask model using project config
        model = build_model(self.config).to(self.device)
        model.train()

        criterion = build_loss(self.config).to(self.device)

        # 2. Construct a randomly shuffled batch of 8 patches (4 GLaS=0, 2 MoNuSeg=1, 2 MoNuSAC=2)
        B, H, W = 8, 64, 64
        # Shuffled task_ids
        task_ids = torch.tensor([1, 0, 2, 0, 0, 2, 1, 0], dtype=torch.long, device=self.device)
        images = torch.randn(B, 3, H, W, device=self.device)
        target_4class = torch.randint(0, 4, (B, H, W), device=self.device)
        class_map = torch.randint(-1, 5, (B, H, W), device=self.device)
        binary_mask = (target_4class > 0).float().unsqueeze(1)

        batch = {
            "image": images,
            "target_4class": target_4class,
            "target_3class": target_4class,
            "binary_mask": binary_mask,
            "class_map": class_map,
            "task_id": task_ids,
            "is_gland": (task_ids == 0),
            "is_monuseg": (task_ids == 1),
            "is_monusac": (task_ids == 2),
        }

        # 3. Model forward pass
        preds = model(images, task_ids=task_ids)
        self.assertIn("gland_seg", preds)
        self.assertIn("gland_neg", preds)
        self.assertIn("cell_seg", preds)
        self.assertIn("monuseg_seg", preds)
        self.assertIn("monusac_seg", preds)
        self.assertIn("cell_class", preds)

        # Verify shapes: 4 gland, 4 cell (2 monuseg + 2 monusac)
        self.assertEqual(preds["gland_seg"].shape[0], 4)
        self.assertEqual(preds["gland_neg"].shape[0], 4)
        self.assertEqual(preds["cell_seg"].shape[0], 4)
        self.assertEqual(preds["monuseg_seg"].shape[0], 2)
        self.assertEqual(preds["monusac_seg"].shape[0], 2)
        self.assertEqual(preds["cell_class"].shape[0], 2)

        # 4. Multi-task loss computation
        loss, logs = criterion(preds, batch)
        self.assertFalse(torch.isnan(loss), "Loss must not be NaN")
        self.assertTrue(loss.requires_grad)

        # Confirm all required loss logs exist
        self.assertIn("loss_glas_total", logs)
        self.assertIn("loss_glas_neg", logs)
        self.assertIn("loss_monuseg_total", logs)
        self.assertIn("loss_monusac_seg_total", logs)
        self.assertIn("loss_monusac_cls_total", logs)
        self.assertIn("total_loss", logs)

        # 5. Backward pass
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        loss.backward()

        # Confirm non-zero gradients exist across heads
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        self.assertGreater(grad_norm.item(), 0.0)
        optimizer.step()

    def test_trainer_train_one_epoch_with_shuffled_batches(self):
        from training.trainer_multitask import MultiTaskTrainer

        model = build_model(self.config).to(self.device)
        criterion = build_loss(self.config).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Mock loader with 2 shuffled batches
        B, H, W = 8, 64, 64
        batches = []
        for _ in range(2):
            # Shuffled task ids
            perm = torch.randperm(8)
            base_tasks = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
            shuffled_tasks = base_tasks[perm]

            t4 = torch.randint(0, 4, (B, H, W))
            batches.append({
                "image": torch.randn(B, 3, H, W),
                "task_id": shuffled_tasks,
                "target_4class": t4,
                "binary_mask": (t4 > 0).float().unsqueeze(1),
                "class_map": torch.randint(-1, 5, (B, H, W)),
                "is_gland": (shuffled_tasks == 0),
                "is_monuseg": (shuffled_tasks == 1),
                "is_monusac": (shuffled_tasks == 2),
            })

        class MockLoader:
            def __len__(self):
                return len(batches)
            def __iter__(self):
                return iter(batches)

        mock_loader = MockLoader()

        trainer = MultiTaskTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=mock_loader,
            device=self.device,
            config=self.config,
            exp_dir="/tmp/test_trainer_shuffled",
        )

        metrics = trainer.train_one_epoch()
        self.assertIn("loss", metrics)
        self.assertIn("glas_bin_dice", metrics)
        self.assertIn("monuseg_bin_dice", metrics)
        self.assertIn("monusac_bin_dice", metrics)
        self.assertIn("monusac_cell_dice", metrics)
        self.assertIn("cell_bin_dice", metrics)
        self.assertFalse(torch.isnan(torch.tensor(metrics["loss"])))


if __name__ == "__main__":
    unittest.main()
