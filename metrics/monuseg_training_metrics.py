# metrics/monuseg_training_metrics.py

from typing import Dict, Any, List
import numpy as np
import torch


class MoNuSegTrainingMetrics:
    """
    Dedicated training metrics for the MoNuSeg nuclear segmentation dataset.
    Calculates binary Dice and binary IoU efficiently on GPU across training batches.
    Used strictly during training epochs.
    """

    def __init__(self, min_instance_size: int = 10):
        self.min_instance_size = min_instance_size
        self.reset()

    def reset(self) -> None:
        """Resets metric accumulators for a new training epoch."""
        self.dices: List[float] = []
        self.ious: List[float] = []

    def update(self, pred_logits: torch.Tensor, gt_masks: torch.Tensor) -> None:
        """
        Updates metrics with a training batch.

        Args:
            pred_logits: [B, 1, H, W] or [B, H, W] raw model output logits.
            gt_masks:    [B, 1, H, W] or [B, H, W] ground truth binary mask (0 or 1).
        """
        # Pixel-level binarization
        pred_prob = torch.sigmoid(pred_logits)
        pred_bin = (pred_prob >= 0.5).float()
        gt_bin = (gt_masks > 0.5).float()

        # Fast GPU pixel-level Dice and IoU
        inter = (pred_bin * gt_bin).sum(dim=[1, 2, 3])
        total = pred_bin.sum(dim=[1, 2, 3]) + gt_bin.sum(dim=[1, 2, 3])
        union = total - inter

        batch_dice = ((2.0 * inter + 1e-6) / (total + 1e-6)).detach().cpu().tolist()
        batch_iou = ((inter + 1e-6) / (union + 1e-6)).detach().cpu().tolist()

        self.dices.extend(batch_dice)
        self.ious.extend(batch_iou)

    def compute(self) -> Dict[str, float]:
        """Computes epoch-level average training metrics for MoNuSeg."""
        return {
            "dice": float(np.mean(self.dices)) if self.dices else 0.0,
            "iou": float(np.mean(self.ious)) if self.ious else 0.0,
        }


__all__ = ["MoNuSegTrainingMetrics"]
