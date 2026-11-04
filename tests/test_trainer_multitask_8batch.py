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
        model = Model(approach="4class_j", num_gland_classes=4, num_cell_classes=4, num_classification_classes=5).cuda()
        criterion = MultiTaskLoss(approach="4class_j", negative_supervision_weight=0.5).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # 8-patch batch (4 GLaS, 2 MoNuSeg, 2 MoNuSAC)
        batch = {
            "image": torch.randn(8, 3, 64, 64).cuda(),
            "target_4class": torch.randint(0, 4, (8, 64, 64)).cuda(),
            "binary_mask": torch.randint(0, 2, (8, 1, 64, 64)).float().cuda(),
            "class_map": torch.full((8, 64, 64), -1, dtype=torch.long).cuda(),
            "task_id": torch.tensor([0, 0, 0, 0, 1, 1, 2, 2]).cuda(),
            "is_gland": torch.tensor([True, True, True, True, False, False, False, False]).cuda(),
            "is_monuseg": torch.tensor([False, False, False, False, True, True, False, False]).cuda(),
            "is_monusac": torch.tensor([False, False, False, False, False, False, True, True]).cuda(),
            "patch_names": [f"p_{i}" for i in range(8)],
        }
        batch["class_map"][6:8] = torch.randint(0, 5, (2, 64, 64)).cuda()

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
            device=torch.device("cuda:0"),
            config={"training": {"epochs": 1}},
            exp_dir="checkpoints/dummy_test_8batch",
        )
        metrics = trainer.train_one_epoch()
        self.assertIn("glas_bin_dice", metrics)
        self.assertIn("monuseg_bin_dice", metrics)
        self.assertIn("monusac_bin_dice", metrics)
        self.assertIn("loss_glas_neg", metrics)
        self.assertIn("glas_neg_bg_ratio", metrics)
        self.assertGreaterEqual(metrics["glas_neg_bg_ratio"], 0.0)
        self.assertLessEqual(metrics["glas_neg_bg_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
