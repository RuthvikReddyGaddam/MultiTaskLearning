# tests/test_glas_aspp_head.py

import unittest
import sys
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.SDENetPointGrad.MultiTask.Decoder.PredictionHead import (
    MultiTaskPredictionHead,
    AtrousSpatialPyramidHead,
)


class TestGlasASPPHead(unittest.TestCase):
    def setUp(self):
        self.in_channels = 64
        self.num_gland_classes = 4
        self.num_cell_classes = 4
        self.num_classification_classes = 5

        self.head = MultiTaskPredictionHead(
            in_channels=self.in_channels,
            approach="4class_j",
            num_gland_classes=self.num_gland_classes,
            num_cell_classes=self.num_cell_classes,
            num_classification_classes=self.num_classification_classes,
        )

    def test_glas_head_is_aspp_and_cell_heads_are_standard(self):
        """Verify ASPP is applied strictly to GLaS gland head, while cell heads remain standard Conv2d."""
        self.assertIsInstance(
            self.head.gland_seg,
            AtrousSpatialPyramidHead,
            "GLaS head must be an instance of AtrousSpatialPyramidHead",
        )
        self.assertIs(self.head.gland_seg, self.head.glas_seg)

        # Cell heads must remain standard Conv2d
        self.assertIsInstance(
            self.head.cell_seg,
            nn.Conv2d,
            "Cell segmentation head must remain standard Conv2d",
        )
        self.assertIsInstance(
            self.head.cell_class,
            nn.Conv2d,
            "Cell phenotype classification head must remain standard Conv2d",
        )

    def test_aspp_multi_scale_forward_and_gradient_flow(self):
        """Verify multi-scale ASPP branches (1x1, dilations 6/12/18, global pool) forward and backward."""
        B, C, H, W = 4, self.in_channels, 128, 128
        x = torch.randn(B, C, H, W, requires_grad=True)

        out = self.head.gland_seg(x)
        self.assertEqual(out.shape, (B, self.num_gland_classes, H, W))
        self.assertEqual(out.dtype, torch.float32)

        # Backward pass
        loss = out.sum()
        loss.backward()

        # Check classifier weight gradient
        self.assertIsNotNone(self.head.gland_seg.weight.grad)
        self.assertGreater(self.head.gland_seg.weight.grad.norm().item(), 0.0)

        # Check all ASPP branches received gradients
        for i, branch in enumerate(self.head.gland_seg.branches):
            has_grad = any(p.grad is not None and p.grad.norm().item() > 0.0 for p in branch.parameters())
            self.assertTrue(has_grad, f"ASPP branch {i} must receive non-zero gradients")

    def test_8_batch_multi_task_routing_with_aspp_gland(self):
        """Verify 8-batch routing: gland_seg(x[:4]), gland_neg(x[4:]), cell_seg(x[4:]), cell_class(x[6:])."""
        B = 8
        x = torch.randn(B, self.in_channels, 64, 64)
        task_ids = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])

        preds = self.head(x, task_ids=task_ids)

        self.assertEqual(preds["gland_seg"].shape, (4, 4, 64, 64))
        self.assertEqual(preds["gland_neg"].shape, (4, 4, 64, 64))
        self.assertEqual(preds["cell_seg"].shape, (4, 4, 64, 64))
        self.assertEqual(preds["monuseg_seg"].shape, (2, 4, 64, 64))
        self.assertEqual(preds["monusac_seg"].shape, (2, 4, 64, 64))
        self.assertEqual(preds["cell_class"].shape, (2, 5, 64, 64))


if __name__ == "__main__":
    unittest.main()

