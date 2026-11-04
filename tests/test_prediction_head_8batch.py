import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from models.SDENetPointGrad.MultiTask.Decoder.PredictionHead import MultiTaskPredictionHead


class TestPredictionHead8Batch(unittest.TestCase):
    def test_forward_with_strict_8_batch(self):
        head = MultiTaskPredictionHead(
            in_channels=64,
            approach="4class_j",
            num_gland_classes=4,
            num_cell_classes=4,
            num_classification_classes=5,
        )
        B = 8
        x = torch.randn(B, 64, 32, 32)
        task_ids = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
        out = head(x, task_ids=task_ids)

        self.assertIn("gland_seg", out)
        self.assertIn("gland_neg", out)
        self.assertIn("cell_seg", out)
        self.assertIn("monuseg_seg", out)
        self.assertIn("monusac_seg", out)
        self.assertIn("cell_class", out)

        # Exact tensor shapes for 2:1:1 split
        self.assertEqual(out["gland_seg"].shape, (4, 4, 32, 32))
        self.assertEqual(out["gland_neg"].shape, (4, 4, 32, 32))
        self.assertEqual(out["cell_seg"].shape, (4, 4, 32, 32))
        self.assertEqual(out["monuseg_seg"].shape, (2, 4, 32, 32))
        self.assertEqual(out["monusac_seg"].shape, (2, 4, 32, 32))
        self.assertEqual(out["cell_class"].shape, (2, 5, 32, 32))

    def test_forward_with_batch_16(self):
        head = MultiTaskPredictionHead(
            in_channels=64,
            approach="4class_j",
            num_gland_classes=4,
            num_cell_classes=4,
            num_classification_classes=5,
        )
        B = 16
        x = torch.randn(B, 64, 16, 16)
        # 8 GLaS, 4 MoNuSeg, 4 MoNuSAC
        task_ids = torch.tensor([0] * 8 + [1] * 4 + [2] * 4)
        out = head(x, task_ids=task_ids)

        self.assertEqual(out["gland_seg"].shape, (8, 4, 16, 16))
        self.assertEqual(out["gland_neg"].shape, (8, 4, 16, 16))
        self.assertEqual(out["cell_seg"].shape, (8, 4, 16, 16))
        self.assertEqual(out["monuseg_seg"].shape, (4, 4, 16, 16))
        self.assertEqual(out["monusac_seg"].shape, (4, 4, 16, 16))
        self.assertEqual(out["cell_class"].shape, (4, 5, 16, 16))


if __name__ == "__main__":
    unittest.main()

