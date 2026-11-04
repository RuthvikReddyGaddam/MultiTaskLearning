# metrics/dice.py

from typing import Union, Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Computes binary Dice coefficient between prediction and target.

    Args:
        pred: [B, 1, H, W] or [B, H, W] logits or probabilities.
        target: [B, 1, H, W] or [B, H, W] ground truth binary mask (0 or 1).
        threshold: Threshold for binarizing probabilities if not already boolean.
        smooth: Smoothing epsilon for numerical stability.

    Returns:
        Scalar tensor with Dice score.
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
    if target.ndim == 3:
        target = target.unsqueeze(1)

    # Convert logits to probabilities if necessary
    if pred.is_floating_point() and (pred.min() < 0.0 or pred.max() > 1.0):
        pred_prob = torch.sigmoid(pred)
    else:
        pred_prob = pred

    pred_bin = (pred_prob >= threshold).to(device=target.device, dtype=torch.float32)
    target_bin = (target > 0.5).to(device=target.device, dtype=torch.float32)

    intersection = (pred_bin * target_bin).sum(dim=(-2, -1))
    cardinality = pred_bin.sum(dim=(-2, -1)) + target_bin.sum(dim=(-2, -1))

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return dice.mean()


def multiclass_dice_score(
    pred_cls: torch.Tensor,
    gt_cls: torch.Tensor,
    num_classes: int = 5,
    smooth: float = 1e-6,
    include_background: bool = False,
) -> torch.Tensor:
    """
    Computes multi-class Dice score across cell classes.
    Natively accepts gt_cls as integer class labels [B, H, W] (0..num_classes-1)
    or one-hot multi-channel tensor [B, C, H, W].

    Args:
        pred_cls: [B, C, H, W] logits or softmax probabilities.
        gt_cls: [B, H, W] integer labels or [B, C, H, W] one-hot/multi-channel.
        num_classes: Total number of classes (default 5: 0=bg, 1=Epi, 2=Lym, 3=Neu, 4=Mac).
        smooth: Smoothing epsilon.
        include_background: Whether to include class 0 (background) in the mean score.

    Returns:
        Scalar tensor with average multi-class Dice score.
    """
    device = pred_cls.device
    num_classes = pred_cls.shape[1] if pred_cls.ndim == 4 else num_classes

    # Predict discrete class labels via argmax [B, H, W]
    pred_idx = torch.argmax(pred_cls, dim=1)

    # Ground truth discrete class labels directly from [B, 5, H, W]
    gt_idx = torch.argmax(gt_cls, dim=1)

    start_class = 0 if include_background else 1
    scores = []

    for c in range(start_class, num_classes):
        p_c = (pred_idx == c).float()
        g_c = (gt_idx == c).float()

        intersection = (p_c * g_c).sum()
        total = p_c.sum() + g_c.sum()

        if total == 0:
            # Class not present in prediction or ground truth
            scores.append(torch.tensor(1.0, device=device))
        else:
            scores.append((2.0 * intersection + smooth) / (total + smooth))

    if len(scores) == 0:
        return torch.tensor(1.0, device=device)

    return torch.stack(scores).mean()


def binary_dice_numpy(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    """Numpy binary dice score calculation."""
    p = (pred_bin > 0).astype(np.float32)
    g = (gt_bin > 0).astype(np.float32)
    intersection = float(np.sum(p * g))
    cardinality = float(np.sum(p) + np.sum(g))
    if cardinality == 0:
        return 1.0
    return (2.0 * intersection + smooth) / (cardinality + smooth)


__all__ = [
    "dice_score",
    "multiclass_dice_score",
    "binary_dice_numpy",
]
