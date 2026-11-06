# tests/test_model_masked_decoders.py

import unittest
import torch
import torch.nn as nn

from models.SDENetPointGrad.MultiTask.ModelMaskedDecoders import ModelMaskedDecoders
from scripts.build_models import build_model
from scripts.build_losses import build_loss
from losses.multitask_loss import MultiTaskLoss


class TestModelMaskedDecoders(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.B = 8
        self.H = 64
        self.W = 64
        self.channels = (16, 32, 32, 64)
        self.skip_channels = (32, 32, 16)
        self.decoder_channels = (64, 32, 32, 16)
        self.pooled_sizes = ((8, 8), (8, 8), (8, 8), (8, 8))

        self.model = ModelMaskedDecoders(
            in_channels=3,
            encoder_channels=self.channels,
            decoder_channels=self.decoder_channels,
            skip_channels=self.skip_channels,
            embed_dim=32,
            num_heads=2,
            sde_block=("SE_only", "Full", "Full"),
            r=(4, 4, 2),
            pooled_sizes=self.pooled_sizes,
            dropout=0.0,
            image_sde_r=2,
            geometric_in_channels=3,
            geometric_encoder_channels=self.channels,
            approach="4class_j",
            num_gland_classes=4,
            num_cell_classes=4,
            num_classification_classes=5,
        )

        # Standard 2:1:1 batch (4 GLaS, 2 MoNuSeg, 2 MoNuSAC)
        self.task_ids = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2], dtype=torch.long)
        self.x = torch.randn(self.B, 3, self.H, self.W)

    def test_model_build_via_registry(self):
        cfg = {
            "model": {
                "name": "multitask_masked_decoders",
                "multitask": True,
                "approach": "4class_j",
                "encoder": {"in_channels": 3, "encoder_channels": [16, 32, 32, 64], "sde_r": 2},
                "geometric_encoder": {"in_channels": 3, "encoder_channels": [16, 32, 32, 64]},
                "decoder": {
                    "decoder_channels": [64, 32, 32, 16],
                    "skip_channels": [32, 32, 16],
                    "embed_dim": 32,
                    "num_heads": 2,
                    "sde_block": ["SE_only", "Full", "Full"],
                    "r": [4, 4, 2],
                    "pooled_sizes": [[8, 8], [8, 8], [8, 8], [8, 8]],
                    "dropout": 0.0,
                },
                "heads": {
                    "multitask": {"enabled": True},
                    "approach": "4class_j",
                    "segmentation": {"num_gland_classes": 4, "num_cell_classes": 4},
                    "classification": {"num_classes": 5},
                },
            }
        }
        built_model = build_model(cfg)
        self.assertIsInstance(built_model, ModelMaskedDecoders)
        self.assertTrue(hasattr(built_model, "gland_decoder"))
        self.assertTrue(hasattr(built_model, "cell_seg_decoder"))
        self.assertTrue(hasattr(built_model, "cell_class_decoder"))

    def test_forward_shape_and_masking(self):
        # 1. Training mode: task-sliced predictions for memory efficiency
        self.model.train()
        train_preds = self.model(self.x, task_ids=self.task_ids)

        self.assertIn("gland_seg", train_preds)
        self.assertIn("cell_seg", train_preds)
        self.assertIn("cell_class", train_preds)

        # Shapes must match task-sliced allocations:
        # 4 GLaS patches (indices 0..3)
        self.assertEqual(train_preds["gland_seg"].shape, (4, 4, self.H, self.W))
        # 4 Cell patches (2 MoNuSeg + 2 MoNuSAC)
        self.assertEqual(train_preds["cell_seg"].shape, (4, 4, self.H, self.W))
        self.assertEqual(train_preds["monuseg_seg"].shape, (2, 4, self.H, self.W))
        self.assertEqual(train_preds["monusac_seg"].shape, (2, 4, self.H, self.W))
        # 2 MoNuSAC patches
        self.assertEqual(train_preds["cell_class"].shape, (2, 5, self.H, self.W))

        # 2. Evaluation / Dense inference mode: all decoders evaluate densely across all B patches
        self.model.eval()
        with torch.no_grad():
            eval_preds = self.model(self.x)

        self.assertEqual(eval_preds["gland_seg"].shape, (self.B, 4, self.H, self.W))
        self.assertEqual(eval_preds["cell_seg"].shape, (self.B, 4, self.H, self.W))
        self.assertEqual(eval_preds["monuseg_seg"].shape, (self.B, 4, self.H, self.W))
        self.assertEqual(eval_preds["monusac_seg"].shape, (self.B, 4, self.H, self.W))
        self.assertEqual(eval_preds["cell_class"].shape, (self.B, 5, self.H, self.W))

    def test_gradient_isolation_gland_only(self):
        self.model.zero_grad()
        preds = self.model(self.x, task_ids=self.task_ids)

        # Compute dummy loss on gland_seg
        loss = preds["gland_seg"].sum()
        loss.backward()

        # gland_decoder must receive gradients
        gland_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.gland_decoder.parameters()
        )
        self.assertTrue(gland_has_grad)

        # cell_seg_decoder must NOT receive any gradient
        for name, p in self.model.cell_seg_decoder.named_parameters():
            if p.grad is not None:
                self.assertEqual(p.grad.abs().sum().item(), 0.0, f"{name} received non-zero grad")

        # cell_class_decoder must NOT receive any gradient
        for name, p in self.model.cell_class_decoder.named_parameters():
            if p.grad is not None:
                self.assertEqual(p.grad.abs().sum().item(), 0.0, f"{name} received non-zero grad")

        # Shared image_encoder must receive gradients from gland path
        encoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.image_encoder.parameters()
        )
        self.assertTrue(encoder_has_grad)

    def test_gradient_isolation_cell_class_only(self):
        self.model.zero_grad()
        preds = self.model(self.x, task_ids=self.task_ids)

        # Compute dummy loss on cell_class
        loss = preds["cell_class"].sum()
        loss.backward()

        # cell_class_decoder must receive gradients
        cls_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.cell_class_decoder.parameters()
        )
        self.assertTrue(cls_has_grad)

        # gland_decoder must NOT receive any gradient
        for name, p in self.model.gland_decoder.named_parameters():
            if p.grad is not None:
                self.assertEqual(p.grad.abs().sum().item(), 0.0, f"{name} received non-zero grad")

        # cell_seg_decoder must NOT receive any gradient
        for name, p in self.model.cell_seg_decoder.named_parameters():
            if p.grad is not None:
                self.assertEqual(p.grad.abs().sum().item(), 0.0, f"{name} received non-zero grad")

    def test_active_patch_normalized_loss_weighting(self):
        criterion = MultiTaskLoss(
            glas_weight=1.0,
            monuseg_weight=1.0,
            monusac_weight=1.0,
            normalize_active_patches=True,
            monusac_seg_weight=0.5,
            monusac_cls_weight=0.5,
        )

        preds = self.model(self.x, task_ids=self.task_ids)

        batch = {
            "target_4class": torch.randint(0, 4, (self.B, self.H, self.W), dtype=torch.long),
            "class_map": torch.randint(0, 5, (self.B, self.H, self.W), dtype=torch.long),
            "task_id": self.task_ids,
            "is_gland": self.task_ids == 0,
            "is_monuseg": self.task_ids == 1,
            "is_monusac": self.task_ids == 2,
        }

        total_loss, logs = criterion(preds, batch)

        # In 2:1:1 batch (4 GLaS, 2 MoNuSeg, 2 MoNuSAC, total 8):
        # eff_glas_w = 1.0 * (4 / 8) = 0.5
        # eff_monuseg_w = 1.0 * (2 / 8) = 0.25
        # eff_monusac_seg_w = 1.0 * (2 / 8) * 0.5 = 0.125
        # eff_monusac_cls_w = 1.0 * (2 / 8) * 0.5 = 0.125
        self.assertAlmostEqual(logs["weight_eff_glas"], 0.5)
        self.assertAlmostEqual(logs["weight_eff_monuseg"], 0.25)
        self.assertAlmostEqual(logs["weight_eff_monusac_seg"], 0.125)
        self.assertAlmostEqual(logs["weight_eff_monusac_cls"], 0.125)

        # Check backward pass stability
        self.model.zero_grad()
        total_loss.backward()

        self.assertFalse(torch.isnan(total_loss))
        self.assertTrue(total_loss.item() > 0)


if __name__ == "__main__":
    unittest.main()

