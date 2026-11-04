# losses/segmentation_classification_topology_loss.py

from typing import Optional, Dict, Any, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from losses.dice_loss import DiceLoss


class SegmentationClassificationTopologyLoss(nn.Module):
    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=0.0,
        topology_weight=0.1,
        cls_ce_weight=1.0,
        cls_dice_weight=1.0,
        alpha=0.5,
        beta=0.5,
        smooth=1e-6,
        num_classes=5,
        **kwargs,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.topology_weight = topology_weight
        self.cls_ce_weight = cls_ce_weight
        self.cls_dice_weight = cls_dice_weight
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.num_classes = num_classes

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.ce_loss = nn.CrossEntropyLoss()

    # ---------------------------------------------------------
    # Topology loss on binary foreground
    # ---------------------------------------------------------

    def topology_loss(
        self,
        dt: torch.Tensor,
        pred_logits: torch.Tensor,
        gt_binary: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates topology loss using signed distance transform."""
        pred_prob = torch.sigmoid(pred_logits)
        gt_binary = gt_binary.to(device=pred_logits.device, dtype=pred_logits.dtype)
        dt = dt.to(device=pred_logits.device, dtype=pred_logits.dtype)

        # Soft false positive and false negative
        fp = pred_prob - (gt_binary * pred_prob)
        fn = gt_binary - (gt_binary * pred_prob)

        inside_dist = torch.clamp(-dt, min=0.0)
        outside_dist = torch.clamp(dt, min=0.0)

        loss_pixel = (
            self.alpha * inside_dist * fn
            + self.beta * outside_dist * fp
        )

        return (loss_pixel + self.smooth).mean()

    def per_class_dice_loss(
        self,
        pred_cls: torch.Tensor,
        gt_cls: torch.Tensor
    ) -> torch.Tensor:

        probs = F.softmax(pred_cls, dim=1)
        gt_cls = gt_cls.to(dtype=probs.dtype)

        # Sum over batch and spatial dimensions
        dims = (0, 2, 3)

        intersection = (probs * gt_cls).sum(dim=dims)

        denominator = (
            probs.sum(dim=dims)
            + gt_cls.sum(dim=dims)
        )

        dice = (
            (2.0 * intersection + 1e-3)
            / (denominator + 1e-3)
        )

        # Include all 5 classes
        return (1.0 - dice).sum()

    def forward(
        self,
        pred_bin: torch.Tensor,
        pred_cls: torch.Tensor,
        gt_bin: torch.Tensor,
        gt_cls: torch.Tensor,
        sdf: torch.Tensor,
    ):
        """
        Tightly coupled forward pass:
          pred_bin: [B, 1, H, W] binary segmentation logits
          pred_cls: [B, 5, H, W] classification logits
          gt_bin:   [B, 1, H, W] binary segmentation ground truth
          gt_cls:   [B, 5, H, W] 5-channel classification ground truth
          sdf:      [B, 1, H, W] signed distance field
        """
        device = pred_bin.device
        dtype = pred_bin.dtype

        gt_bin = gt_bin.to(device=device, dtype=dtype)
        gt_cls = gt_cls.to(device=device, dtype=dtype)

        # 1. Binary segmentation branch losses
        loss_bin_ce = self.bce_loss(pred_bin, gt_bin)
        loss_bin_dice = (
            self.dice_loss(pred_bin, gt_bin)
            if self.dice_weight > 0
            else torch.tensor(0.0, device=device, dtype=dtype)
        )
        loss_bin_topo = self.topology_loss(sdf, pred_bin, gt_bin)

        # 2. Multi-class classification branch losses (5 classes: 0=Background, 1=Epi, 2=Lym, 3=Neu, 4=Mac)
        gt_cls_idx = torch.argmax(gt_cls, dim=1).long()
        loss_cls_ce = self.ce_loss(pred_cls, gt_cls_idx)
        loss_cls_dice = self.per_class_dice_loss(pred_cls, gt_cls)

        total_loss = (
            self.bce_weight * loss_bin_ce
            + self.dice_weight * loss_bin_dice
            + self.topology_weight * loss_bin_topo
            + self.cls_ce_weight * loss_cls_ce
            + self.cls_dice_weight * loss_cls_dice
        )

        components = {
            'bin_ce': loss_bin_ce.item(),
            'bin_dice': loss_bin_dice.item(),
            'bin_topo': loss_bin_topo.item(),
            'cls_ce': loss_cls_ce.item(),
            'cls_dice': loss_cls_dice.item(),
            'bce': loss_bin_ce.item(),
            'dice': loss_bin_dice.item(),
            'topology': loss_bin_topo.item(),
            'total': total_loss.item(),
        }

        return total_loss, components
