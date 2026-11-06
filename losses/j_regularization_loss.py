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
    Computes differentiable Pairwise Soft Youden's J Loss:
      Reference:
        Guerrero-Pena et al., "J Regularization Improves Imbalanced Multiclass Segmentation"
        IEEE International Symposium on Biomedical Imaging (ISBI), 2020. arXiv:1910.09783.

    Formulation:
      For each ordered pair of classes (i, k) where i is the positive class and k is the negative class:
        alpha_i = sum_p z_i(p) * y_i(p) / n_i            (soft True Positive Rate, TPR)
        beta_{i, k} = sum_p (1 - z_i(p)) * y_k(p) / n_k  (soft True Negative Rate, TNR)
        J_{i, k} = alpha_i + beta_{i, k} - 1

      The normalized pairwise score is:
        (1 + J_{i, k}) / 2 = (alpha_i + beta_{i, k}) / 2 = 1/2 + 1/2 * (TPR_i - FPR_{i, k})

      The multiclass loss is the negative log of pairwise scores over all distinct pairs (i != k):
        L_J = - (1 / N_pairs) * sum_{i != k} log( (alpha_i + beta_{i, k}) / 2 )

      When predictions are perfect, TPR = 1, TNR = 1 -> score = 1.0 -> L_J = -log(1) = 0.
      When predictions are random (uniform), score = 0.5 -> L_J = -log(0.5) = log(2) ~= 0.6931.
    """
    def __init__(
        self,
        num_classes: int = 4,
        epsilon: float = 1e-6,
        include_background: bool = True,
        pairwise_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.include_background = include_background
        if pairwise_weights is not None:
            self.register_buffer("pairwise_weights", pairwise_weights)
        else:
            self.pairwise_weights = None

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
            Scalar tensor representing the Pairwise Soft Youden's J Loss.
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

        # Cross-class soft overlap matrix T [C, C]:
        # T[i, k] = sum_{b, p} probs[b, i, p] * one_hot[b, k, p]
        # (Total predicted probability of class i on true pixels of class k)
        T = torch.einsum("bcn,bkn->ck", probs_flat, one_hot_flat)  # [C, C]
        n = torch.sum(one_hot_flat, dim=(0, 2))                   # [C] pixel counts per class

        # Soft TPR alpha_i [C, 1] for positive class i
        alpha = ((torch.diagonal(T) + self.epsilon) / (n + self.epsilon)).unsqueeze(1)  # [C, 1]

        # Soft TNR beta_{i, k} [C, C] for negative class k against class i:
        # beta_{i, k} = (n_k - T_{i, k} + eps) / (n_k + eps) = 1.0 - T_{i, k} / (n_k + eps)
        n_col = n.unsqueeze(0)  # [1, C]
        beta = 1.0 - (T / (n_col + self.epsilon))  # [C, C]

        # Pairwise normalized J score: (1 + J_{i, k}) / 2 = (alpha_i + beta_{i, k}) / 2
        pairwise_score = 0.5 * (alpha + beta)  # [C, C]
        pairwise_score = torch.clamp(pairwise_score, min=self.epsilon, max=1.0)

        # Pairwise binary surrogate loss: -log( (alpha_i + beta_{i, k}) / 2 )
        pairwise_loss = -torch.log(pairwise_score)  # [C, C]

        # Identify valid non-diagonal pairs (i != k) where both classes are present in ground truth
        valid_pairs = (n.unsqueeze(1) > 0) & (n.unsqueeze(0) > 0)
        valid_pairs.fill_diagonal_(False)

        if not self.include_background and c > 1:
            valid_pairs[0, :] = False
            valid_pairs[:, 0] = False

        if self.pairwise_weights is not None:
            pairwise_loss = pairwise_loss * self.pairwise_weights
            if valid_pairs.any():
                w_sum = self.pairwise_weights[valid_pairs].sum().clamp(min=self.epsilon)
                return pairwise_loss[valid_pairs].sum() / w_sum
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        if valid_pairs.any():
            return pairwise_loss[valid_pairs].mean()

        return torch.tensor(0.0, device=logits.device, requires_grad=True)


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

        # 2. Pairwise Soft Youden's J Loss
        j_loss = self.j_loss_fn(logits, targets, mask=mask)

        # 3. Multi-Class Dice Loss
        dice_loss = self.dice_loss_fn(logits, targets, mask=mask)

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
