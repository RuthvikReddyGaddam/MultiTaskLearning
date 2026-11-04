# losses/j_regularization_loss.py
"""
Differentiable Youden's J-Regularization Loss for Imbalanced 4-Channel Segmentation.

Reference:
  Guerrero-Pena, F. A., Fernandez, P. D. M., Tarr, P. T., Ren, T. I., Meyerowitz, E. M., & Cunha, A.
  "J Regularization Improves Imbalanced Multiclass Segmentation."
  IEEE International Symposium on Biomedical Imaging (ISBI), 2020. arXiv:1910.09783.

4-Channel Semantic Schema:
  0: Background
  1: Cell/Gland Interior (Core)
  2: Cell/Gland Boundary (Contour ring)
  3: Inter-cellular Gap (Clefts/valleys between adjacent cells via Bottom-Hat Transform)

Core Concept:
  Youden's J statistic (Informedness):
      J = Sensitivity + Specificity - 1 = True Positive Rate + True Negative Rate - 1

  Unlike standard cross-entropy which stalls or biases heavily towards dominant majority background pixels
  (outnumbering gap pixels by >100:1), J-statistic intrinsically balances all 4 classes because
  sensitivity and specificity are self-normalized by total positive and negative class populations.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftYoudenJLoss(nn.Module):
    """
    Computes differentiable Soft Youden's J Loss:
        L_J = (1 / C) * sum_{c=1}^C [(1 - Sensitivity_c) + (1 - Specificity_c)]
            = (1 / C) * sum_{c=1}^C [1 - J_c]

    When predictions are perfect, Sens_c = 1, Spec_c = 1 -> J_c = 1 -> L_J = 0.
    When predictions are completely inverted, Sens_c = 0, Spec_c = 0 -> J_c = -1 -> L_J = 2.
    """
    def __init__(
        self,
        num_classes: int = 4,
        epsilon: float = 1e-6,
        include_background: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.include_background = include_background

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicted class logits [B, C, H, W]
            targets: Ground-truth integer labels [B, H, W] with values in [0, C-1]
            mask: Optional spatial validity mask [B, H, W] (1 for valid pixels, 0 to ignore)

        Returns:
            Scalar tensor representing the Soft Youden's J Loss.
        """
        b, c, h, w = logits.shape
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]

        # Convert targets to one-hot: [B, C, H, W]
        if targets.ndim == 3:
            one_hot = F.one_hot(targets.clamp(0, c - 1), num_classes=c)  # [B, H, W, C]
            one_hot = one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        elif targets.ndim == 4 and targets.shape[1] == c:
            one_hot = targets.float()
        else:
            raise ValueError(f"Unsupported target shape: {targets.shape} for logits: {logits.shape}")

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            probs = probs * mask
            one_hot = one_hot * mask

        # Flatten spatial dimensions: [B, C, N] where N = H * W
        probs_flat = probs.view(b, c, -1)
        one_hot_flat = one_hot.view(b, c, -1)

        # TP, FP, TN, FN across batch and spatial dimensions per class
        tp = torch.sum(probs_flat * one_hot_flat, dim=(0, 2))  # [C]
        p_total = torch.sum(one_hot_flat, dim=(0, 2))          # [C]
        fp = torch.sum(probs_flat * (1.0 - one_hot_flat), dim=(0, 2))  # [C]
        n_total = torch.sum(1.0 - one_hot_flat, dim=(0, 2))   # [C]
        tn = n_total - fp

        # Sensitivity (TPR) and Specificity (TNR)
        sensitivity = (tp + self.epsilon) / (p_total + self.epsilon)
        specificity = (tn + self.epsilon) / (n_total + self.epsilon)

        # Loss per class = 1 - J = (1 - Sens) + (1 - Spec)
        j_loss_per_class = (1.0 - sensitivity) + (1.0 - specificity)

        start_idx = 0 if self.include_background else 1
        loss = torch.mean(j_loss_per_class[start_idx:])
        return loss


class MultiClassDiceLoss(nn.Module):
    """Soft multi-class Dice loss."""
    def __init__(self, num_classes: int = 4, epsilon: float = 1e-6, include_background: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.include_background = include_background

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, c, h, w = logits.shape
        probs = F.softmax(logits, dim=1)

        if targets.ndim == 3:
            one_hot = F.one_hot(targets.clamp(0, c - 1), num_classes=c).permute(0, 3, 1, 2).float()
        else:
            one_hot = targets.float()

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            probs = probs * mask
            one_hot = one_hot * mask

        probs_flat = probs.view(b, c, -1)
        one_hot_flat = one_hot.view(b, c, -1)

        intersection = torch.sum(probs_flat * one_hot_flat, dim=(0, 2))
        cardinality = torch.sum(probs_flat + one_hot_flat, dim=(0, 2))

        dice = (2.0 * intersection + self.epsilon) / (cardinality + self.epsilon)
        dice_loss = 1.0 - dice

        start_idx = 0 if self.include_background else 1
        return torch.mean(dice_loss[start_idx:])


class JRegularizedMultiClassLoss(nn.Module):
    """
    Composite Multi-Class Loss combining:
      L = ce_weight * L_CE + j_weight * L_J + dice_weight * L_Dice
    """
    def __init__(
        self,
        num_classes: int = 4,
        ce_weight: float = 1.0,
        j_weight: float = 1.0,
        dice_weight: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        include_background: bool = True,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.j_weight = j_weight
        self.dice_weight = dice_weight

        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.j_loss_fn = SoftYoudenJLoss(
            num_classes=num_classes,
            epsilon=epsilon,
            include_background=include_background,
        )
        self.dice_loss_fn = MultiClassDiceLoss(
            num_classes=num_classes,
            epsilon=epsilon,
            include_background=include_background,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: [B, C, H, W]
            targets: [B, H, W] long tensor with class indices
            mask: Optional [B, H, W] spatial validity mask

        Returns:
            Dictionary containing 'total_loss', 'ce_loss', 'j_loss', and 'dice_loss'.
        """
        # 1. Cross-Entropy Loss
        if mask is None:
            ce_loss = self.ce(logits, targets)
        else:
            ce_raw = F.cross_entropy(logits, targets, reduction='none')
            ce_loss = (ce_raw * mask).sum() / (mask.sum() + 1e-6)

        # 2. Precompute Softmax and One-Hot ONCE in pure FP32
        b, c, h, w = logits.shape
        probs = F.softmax(logits, dim=1)
        if targets.ndim == 3:
            one_hot = F.one_hot(targets.clamp(0, c - 1), num_classes=c).permute(0, 3, 1, 2).float()
        else:
            one_hot = targets.float()

        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            probs = probs * mask
            one_hot = one_hot * mask

        # Flatten spatial dimensions: [B, C, N] where N = H * W
        probs_flat = probs.view(b, c, -1)
        one_hot_flat = one_hot.view(b, c, -1)

        # 3. Soft Youden's J Loss
        tp = torch.sum(probs_flat * one_hot_flat, dim=(0, 2))
        p_total = torch.sum(one_hot_flat, dim=(0, 2))
        fp = torch.sum(probs_flat * (1.0 - one_hot_flat), dim=(0, 2))
        n_total = torch.sum(1.0 - one_hot_flat, dim=(0, 2))
        tn = n_total - fp

        eps = self.j_loss_fn.epsilon
        sensitivity = (tp + eps) / (p_total + eps)
        specificity = (tn + eps) / (n_total + eps)
        j_loss_per_class = (1.0 - sensitivity) + (1.0 - specificity)
        start_idx = 0 if self.j_loss_fn.include_background else 1
        j_loss = torch.mean(j_loss_per_class[start_idx:])

        # 4. Multi-Class Dice Loss (reuse tp and p_total!)
        cardinality = p_total + torch.sum(probs_flat, dim=(0, 2))
        dice = (2.0 * tp + self.dice_loss_fn.epsilon) / (cardinality + self.dice_loss_fn.epsilon)
        dice_loss = torch.mean((1.0 - dice)[start_idx:])

        total_loss = (
            self.ce_weight * ce_loss
            + self.j_weight * j_loss
            + self.dice_weight * dice_loss
        )

        return {
            "total_loss": total_loss,
            "ce_loss": ce_loss,
            "j_loss": j_loss,
            "dice_loss": dice_loss,
        }


__all__ = ["SoftYoudenJLoss", "MultiClassDiceLoss", "JRegularizedMultiClassLoss"]
