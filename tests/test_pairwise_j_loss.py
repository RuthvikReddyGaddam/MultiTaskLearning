import unittest
import math
import torch
import torch.nn as nn
from losses.j_regularization_loss import SoftYoudenJLoss, JRegularizedMultiClassLoss
from losses.multitask_loss import MultiTaskLoss


class TestPairwiseYoudenJLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_perfect_predictions_yield_zero_loss(self):
        loss_fn = SoftYoudenJLoss(num_classes=4, epsilon=1e-7, include_background=True)
        B, C, H, W = 2, 4, 16, 16
        # Balanced target containing classes 0, 1, 2, 3
        targets = torch.randint(0, C, (B, H, W))
        # Logits that heavily favor the correct class
        logits = torch.zeros((B, C, H, W))
        one_hot = torch.nn.functional.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()
        logits = one_hot * 50.0 - (1.0 - one_hot) * 50.0

        loss = loss_fn(logits, targets)
        self.assertAlmostEqual(loss.item(), 0.0, places=4, msg="Perfect predictions should yield ~0 loss")

    def test_uniform_random_predictions_yield_log2(self):
        loss_fn = SoftYoudenJLoss(num_classes=4, epsilon=1e-7, include_background=True)
        B, C, H, W = 2, 4, 16, 16
        targets = torch.randint(0, C, (B, H, W))
        # Uniform logits -> probs = 1/C everywhere
        logits = torch.zeros((B, C, H, W))

        loss = loss_fn(logits, targets)
        expected = math.log(2.0)  # ~0.693147
        self.assertAlmostEqual(loss.item(), expected, places=4, msg="Uniform logits should yield log(2)")

    def test_absent_class_handling_no_nan(self):
        loss_fn = SoftYoudenJLoss(num_classes=4, epsilon=1e-7, include_background=True)
        B, C, H, W = 2, 4, 16, 16
        # Target with only classes 0 and 1 (classes 2 and 3 are absent)
        targets = torch.randint(0, 2, (B, H, W))
        logits = torch.randn((B, C, H, W), requires_grad=True)

        loss = loss_fn(logits, targets)
        self.assertFalse(torch.isnan(loss), "Loss must not be NaN when some classes are absent")
        self.assertFalse(torch.isinf(loss), "Loss must not be Inf when some classes are absent")

        # Backward should succeed
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertFalse(torch.isnan(logits.grad).any(), "Gradients must not contain NaN")

    def test_j_regularized_multiclass_loss_composite(self):
        composite_loss = JRegularizedMultiClassLoss(
            num_classes=4,
            ce_weight=1.0,
            j_weight=1.0,
            dice_weight=0.5,
        )
        B, C, H, W = 2, 4, 16, 16
        targets = torch.randint(0, C, (B, H, W))
        logits = torch.randn((B, C, H, W), requires_grad=True)

        res = composite_loss(logits, targets)
        self.assertIn("total_loss", res)
        self.assertIn("ce_loss", res)
        self.assertIn("j_loss", res)
        self.assertIn("dice_loss", res)

        self.assertGreater(res["total_loss"].item(), 0.0)
        self.assertGreater(res["j_loss"].item(), 0.0)

        res["total_loss"].backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(logits.grad.norm().item(), 0.0)

    def test_multitask_loss_with_pairwise_j(self):
        mt_loss = MultiTaskLoss(
            glas_weight=1.0,
            monuseg_weight=1.0,
            monusac_weight=1.0,
            j_reg_weight=1.0,
            ce_weight=1.0,
            dice_weight=0.5,
            negative_supervision_weight=0.5,
        )

        B, H, W = 8, 32, 32
        task_ids = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
        images = torch.randn(B, 3, H, W)
        target_4class = torch.randint(0, 4, (B, H, W))
        class_map = torch.randint(-1, 5, (B, H, W))

        batch = {
            "image": images,
            "target_4class": target_4class,
            "target_3class": target_4class,
            "binary_mask": (target_4class > 0).float().unsqueeze(1),
            "class_map": class_map,
            "task_id": task_ids,
            "is_gland": (task_ids == 0),
            "is_monuseg": (task_ids == 1),
            "is_monusac": (task_ids == 2),
        }

        preds = {
            "gland_seg": torch.randn(4, 4, H, W, requires_grad=True),
            "monuseg_seg": torch.randn(2, 4, H, W, requires_grad=True),
            "monusac_seg": torch.randn(2, 4, H, W, requires_grad=True),
            "cell_class": torch.randn(2, 5, H, W, requires_grad=True),
        }

        total_loss, logs = mt_loss(preds, batch)
        self.assertFalse(torch.isnan(total_loss))
        self.assertIn("loss_glas_j", logs)
        self.assertIn("loss_monuseg_j", logs)
        self.assertIn("loss_monusac_seg_j", logs)
        # J-reg is intentionally removed from cell_class in favor of unweighted CE + Dice
        self.assertIn("loss_monusac_cls_ce", logs)
        self.assertIn("loss_monusac_cls_dice", logs)
        self.assertNotIn("loss_monusac_cls_j", logs)

        total_loss.backward()
        for k, p in preds.items():
            self.assertIsNotNone(p.grad, f"Gradient for {k} should not be None")
            self.assertGreater(p.grad.norm().item(), 0.0, f"Gradient for {k} should be non-zero")


if __name__ == "__main__":
    unittest.main()

