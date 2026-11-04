import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from losses.multitask_loss import MultiTaskLoss


class TestMultiTaskLoss8Batch(unittest.TestCase):
    def test_loss_on_8batch_with_negative_supervision(self):
        criterion = MultiTaskLoss(approach="4class_j", negative_supervision_weight=0.5)
        preds = {
            "gland_seg": torch.randn(4, 4, 32, 32, requires_grad=True),
            "gland_neg": torch.randn(4, 4, 32, 32, requires_grad=True),
            "cell_seg": torch.randn(4, 4, 32, 32, requires_grad=True),
            "cell_class": torch.randn(2, 5, 32, 32, requires_grad=True),
        }
        batch = {
            "target_4class": torch.randint(0, 4, (8, 32, 32)),
            "class_map": torch.full((8, 32, 32), -1, dtype=torch.long),
            "task_id": torch.tensor([0, 0, 0, 0, 1, 1, 2, 2]),
        }
        batch["class_map"][6:8] = torch.randint(0, 5, (2, 32, 32))

        loss, logs = criterion(preds, batch)
        self.assertTrue(loss.requires_grad)
        self.assertIn("loss_glas_neg", logs)
        self.assertGreater(logs["loss_glas_neg"].item(), 0.0)

        # Backprop check
        loss.backward()
        self.assertIsNotNone(preds["gland_neg"].grad)
        self.assertIsNotNone(preds["gland_seg"].grad)
        self.assertIsNotNone(preds["cell_seg"].grad)
        self.assertIsNotNone(preds["cell_class"].grad)


if __name__ == "__main__":
    unittest.main()

