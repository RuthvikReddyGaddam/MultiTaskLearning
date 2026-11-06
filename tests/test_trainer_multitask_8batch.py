import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from training.trainer_multitask import MultiTaskTrainer
from models.SDENetPointGrad.MultiTask.Model import Model
from losses.multitask_loss import MultiTaskLoss


class TestTrainer8Batch(unittest.TestCase):
    def test_train_step_8batch(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Model(approach="4class_j", num_gland_classes=4, num_cell_classes=4, num_classification_classes=5).to(device)
        criterion = MultiTaskLoss(approach="4class_j", negative_supervision_weight=0.5).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # 8-patch batch (4 GLaS, 2 MoNuSeg, 2 MoNuSAC)
        batch = {
            "image": torch.randn(8, 3, 64, 64, device=device),
            "target_4class": torch.randint(0, 4, (8, 64, 64), device=device),
            "binary_mask": torch.randint(0, 2, (8, 1, 64, 64), device=device).float(),
            "class_map": torch.full((8, 64, 64), -1, dtype=torch.long, device=device),
            "task_id": torch.tensor([0, 0, 0, 0, 1, 1, 2, 2], device=device),
            "is_gland": torch.tensor([True, True, True, True, False, False, False, False], device=device),
            "is_monuseg": torch.tensor([False, False, False, False, True, True, False, False], device=device),
            "is_monusac": torch.tensor([False, False, False, False, False, False, True, True], device=device),
            "patch_names": [f"p_{i}" for i in range(8)],
        }
        batch["class_map"][6:8] = torch.randint(0, 5, (2, 64, 64), device=device)

        class DummyLoader:
            def __iter__(self):
                return iter([batch])
            def __len__(self):
                return 1

        trainer = MultiTaskTrainer(
            model=model,
            train_loader=DummyLoader(),
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            config={"training": {"epochs": 1}},
            exp_dir="checkpoints/dummy_test_8batch",
        )
        metrics = trainer.train_one_epoch()
        self.assertIn("glas_bin_dice", metrics)
        self.assertIn("monuseg_bin_dice", metrics)
        self.assertIn("monusac_bin_dice", metrics)
        self.assertIn("loss_monusac_cls_ce", metrics)
        self.assertIn("loss_monusac_cls_dice", metrics)

    def test_train_step_gradient_accumulation_4batch(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Model(approach="4class_j", num_gland_classes=4, num_cell_classes=4, num_classification_classes=5).to(device)
        criterion = MultiTaskLoss(approach="4class_j").to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Batch 1 (4 patches: 2 GLaS, 1 MoNuSeg, 1 MoNuSAC)
        b1 = {
            "image": torch.randn(4, 3, 64, 64, device=device),
            "target_4class": torch.randint(0, 4, (4, 64, 64), device=device),
            "binary_mask": torch.randint(0, 2, (4, 1, 64, 64), device=device).float(),
            "class_map": torch.full((4, 64, 64), -1, dtype=torch.long, device=device),
            "task_id": torch.tensor([0, 0, 1, 2], device=device),
            "is_gland": torch.tensor([True, True, False, False], device=device),
            "is_monuseg": torch.tensor([False, False, True, False], device=device),
            "is_monusac": torch.tensor([False, False, False, True], device=device),
            "patch_names": [f"p_{i}" for i in range(4)],
        }
        b1["class_map"][3:4] = torch.randint(0, 5, (1, 64, 64), device=device)

        # Batch 2 (4 patches: 2 GLaS, 1 MoNuSeg, 1 MoNuSAC)
        b2 = {
            "image": torch.randn(4, 3, 64, 64, device=device),
            "target_4class": torch.randint(0, 4, (4, 64, 64), device=device),
            "binary_mask": torch.randint(0, 2, (4, 1, 64, 64), device=device).float(),
            "class_map": torch.full((4, 64, 64), -1, dtype=torch.long, device=device),
            "task_id": torch.tensor([0, 0, 1, 2], device=device),
            "is_gland": torch.tensor([True, True, False, False], device=device),
            "is_monuseg": torch.tensor([False, False, True, False], device=device),
            "is_monusac": torch.tensor([False, False, False, True], device=device),
            "patch_names": [f"p_{i+4}" for i in range(4)],
        }
        b2["class_map"][3:4] = torch.randint(0, 5, (1, 64, 64), device=device)

        class DummyLoader2:
            def __iter__(self):
                return iter([b1, b2])
            def __len__(self):
                return 2

        trainer = MultiTaskTrainer(
            model=model,
            train_loader=DummyLoader2(),
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            config={"training": {"epochs": 1, "batch_size": 4, "gradient_accumulation_steps": 2}},
            exp_dir="checkpoints/dummy_test_4batch_accum",
        )
        self.assertEqual(trainer.grad_accum_steps, 2)
        metrics = trainer.train_one_epoch()
        self.assertIn("loss", metrics)
        self.assertIn("glas_bin_dice", metrics)
        self.assertIn("monuseg_bin_dice", metrics)
        self.assertIn("monusac_bin_dice", metrics)


if __name__ == "__main__":
    unittest.main()
