# metrics/monusac_training_metrics.py

from typing import Dict, Any, List
import numpy as np
import torch

CELL_CLASSES = ["epithelial", "lymphocyte", "neutrophil", "macrophage"]


class MoNuSACTrainingMetrics:
    """
    Dedicated training metrics for the MoNuSAC dual-head segmentation & classification dataset.
    Calculates:
      1. Binary metrics: bin_dice, bin_iou (nuclei vs background)
      2. Per-class metrics: dice_<cls>, iou_<cls> for each of the 4 cell classes:
         (Epithelial, Lymphocyte, Neutrophil, Macrophage)
      3. Macro-averaged cell metrics: cell_dice, cell_iou
    Computed efficiently on GPU/tensors without slow CPU instance matching.
    Used strictly during training epochs.
    """

    def __init__(self, min_instance_size: int = 5):
        self.min_instance_size = min_instance_size
        self.reset()

    def reset(self) -> None:
        """Resets all metric accumulators for a new training epoch."""
        # Binary accumulators
        self.bin_dices: List[float] = []
        self.bin_ious: List[float] = []

        # Per-class accumulators: {class_name: [scores]}
        self.cls_dices: Dict[str, List[float]] = {name: [] for name in CELL_CLASSES}
        self.cls_ious: Dict[str, List[float]] = {name: [] for name in CELL_CLASSES}

    def update(
        self,
        pred_bin: torch.Tensor,
        gt_bin: torch.Tensor,
        pred_cls: torch.Tensor,
        gt_cls: torch.Tensor,
    ) -> None:
        """
        Updates metrics with a training batch.

        Args:
            pred_bin: [B, 1, H, W] binary segmentation logits.
            gt_bin:   [B, 1, H, W] ground truth binary mask.
            pred_cls: [B, 5, H, W] classification logits (ch 0: bg, ch 1-4: 4 cell types).
            gt_cls:   [B, 5, H, W] ground truth 5-channel mask.
        """
        b = pred_bin.shape[0]

        # -------------------------------------------------------------
        # 1. Binary Segmentation Metrics (all nuclei vs background)
        # -------------------------------------------------------------
        p_bin = (torch.sigmoid(pred_bin) >= 0.5).float()
        g_bin = (gt_bin > 0.5).float()

        # Fast GPU pixel-level binary Dice and IoU
        inter_bin = (p_bin * g_bin).sum(dim=[1, 2, 3])
        total_bin = p_bin.sum(dim=[1, 2, 3]) + g_bin.sum(dim=[1, 2, 3])
        union_bin = total_bin - inter_bin

        batch_bin_dice = ((2.0 * inter_bin + 1e-6) / (total_bin + 1e-6)).detach().cpu().tolist()
        batch_bin_iou = ((inter_bin + 1e-6) / (union_bin + 1e-6)).detach().cpu().tolist()
        self.bin_dices.extend(batch_bin_dice)
        self.bin_ious.extend(batch_bin_iou)

        # -------------------------------------------------------------
        # 2. Per-Class Cell Metrics (Epithelial, Lymphocyte, Neutrophil, Macrophage)
        # -------------------------------------------------------------
        # Discrete class label maps [B, H, W] in {0, 1, 2, 3, 4}
        pred_prob = torch.softmax(pred_cls, dim=1)
        pred_labels = torch.argmax(pred_prob, dim=1)
        gt_labels = torch.argmax(gt_cls, dim=1)

        # Mask out background if binary prediction is negative
        bin_active = (p_bin[:, 0] > 0.5)
        pred_labels = torch.where(bin_active, pred_labels, torch.zeros_like(pred_labels))

        for c_idx, c_name in enumerate(CELL_CLASSES, start=1):
            p_c = (pred_labels == c_idx).float()
            g_c = (gt_labels == c_idx).float()

            inter_c = (p_c * g_c).sum(dim=[1, 2])
            tot_c = p_c.sum(dim=[1, 2]) + g_c.sum(dim=[1, 2])
            uni_c = tot_c - inter_c

            for i in range(b):
                # Only record sample if class is present in GT or Prediction
                if tot_c[i] > 0:
                    d_val = float((2.0 * inter_c[i] + 1e-6) / (tot_c[i] + 1e-6))
                    j_val = float((inter_c[i] + 1e-6) / (uni_c[i] + 1e-6))
                    self.cls_dices[c_name].append(d_val)
                    self.cls_ious[c_name].append(j_val)

    def compute(self) -> Dict[str, float]:
        """
        Computes epoch-level binary, per-class, and macro-averaged training metrics for MoNuSAC.
        """
        metrics: Dict[str, float] = {
            "bin_dice": float(np.mean(self.bin_dices)) if self.bin_dices else 0.0,
            "bin_iou": float(np.mean(self.bin_ious)) if self.bin_ious else 0.0,
        }

        cell_dice_list: List[float] = []
        cell_iou_list: List[float] = []

        for name in CELL_CLASSES:
            d_val = float(np.mean(self.cls_dices[name])) if self.cls_dices[name] else 0.0
            j_val = float(np.mean(self.cls_ious[name])) if self.cls_ious[name] else 0.0

            metrics[f"dice_{name}"] = d_val
            metrics[f"iou_{name}"] = j_val

            if self.cls_dices[name]:
                cell_dice_list.append(d_val)
            if self.cls_ious[name]:
                cell_iou_list.append(j_val)

        # Macro averages across active cell classes
        metrics["cell_dice"] = float(np.mean(cell_dice_list)) if cell_dice_list else 0.0
        metrics["cell_iou"] = float(np.mean(cell_iou_list)) if cell_iou_list else 0.0

        return metrics


__all__ = [
    "MoNuSACTrainingMetrics",
    "CELL_CLASSES",
    "compute_dice_iou_binary",
    "compute_multiclass_metrics",
]


def compute_dice_iou_binary(pred_bin: torch.Tensor, gt_bin: torch.Tensor, eps: float = 1e-6) -> tuple[float, float]:
    """Computes mean binary Dice and IoU on batch tensors [B, 1, H, W] or [B, H, W]."""
    p = (pred_bin > 0.5).float()
    g = (gt_bin > 0.5).float()
    inter = (p * g).sum(dim=(-2, -1))
    total = p.sum(dim=(-2, -1)) + g.sum(dim=(-2, -1))
    union = total - inter
    dice = ((2.0 * inter + eps) / (total + eps)).mean().item()
    iou = ((inter + eps) / (union + eps)).mean().item()
    return float(dice), float(iou)


def compute_multiclass_metrics(pred_cls: torch.Tensor, gt_cls: torch.Tensor, eps: float = 1e-6) -> dict[str, float]:
    """Computes macro-average Dice and IoU for multi-class classification heads."""
    if pred_cls.ndim == 4:
        p_argmax = pred_cls.argmax(dim=1)
    else:
        p_argmax = pred_cls.long()

    if gt_cls.ndim == 4 and gt_cls.shape[1] > 1:
        g_argmax = gt_cls.argmax(dim=1)
    elif gt_cls.ndim == 4 and gt_cls.shape[1] == 1:
        g_argmax = gt_cls.squeeze(1).long()
    else:
        g_argmax = gt_cls.long()

    num_classes = pred_cls.shape[1] if pred_cls.ndim == 4 else 5
    dices, ious = [], []
    for c in range(1, num_classes):
        p_c = (p_argmax == c).float()
        g_c = (g_argmax == c).float()
        inter = (p_c * g_c).sum()
        total = p_c.sum() + g_c.sum()
        if total > 0:
            d = ((2.0 * inter + eps) / (total + eps)).item()
            u = total - inter
            j = ((inter + eps) / (u + eps)).item()
            dices.append(d)
            ious.append(j)

    return {
        "cell_dice": float(np.mean(dices)) if dices else 0.0,
        "cell_iou": float(np.mean(ious)) if ious else 0.0,
    }
