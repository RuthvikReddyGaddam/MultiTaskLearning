# scripts/build_losses.py
"""
Dedicated Loss Builder for Multi-Task Learning.
Constructs the decoupled multi-task loss with GLaS negative background supervision,
cross-entropy, generalized Dice, and J-regularization topology preservation.
"""

from typing import Dict, Any
import torch.nn as nn
from losses.multitask_loss import MultiTaskLoss


def build_multitask_loss(config: Dict[str, Any]) -> MultiTaskLoss:
    """Builds decoupled MultiTaskLoss for GLaS, MoNuSeg, and MoNuSAC."""
    loss_cfg = config.get("loss", {})
    dataset_cfg = config.get("dataset", {})
    heads_cfg = config.get("model", {}).get("heads", {})

    num_gland_classes = (
        heads_cfg.get("segmentation", {}).get("num_gland_classes")
        or dataset_cfg.get("num_gland_classes", 4)
    )
    num_cell_classes = (
        heads_cfg.get("segmentation", {}).get("num_cell_classes")
        or dataset_cfg.get("num_cell_classes", 4)
    )
    num_classification_classes = (
        heads_cfg.get("classification", {}).get("num_classes")
        or dataset_cfg.get("num_classes", 5)
    )

    return MultiTaskLoss(
        glas_weight=float(loss_cfg.get("glas_weight", 1.0)),
        monuseg_weight=float(loss_cfg.get("monuseg_weight", 1.0)),
        monusac_weight=float(loss_cfg.get("monusac_weight", 1.0)),
        j_reg_weight=float(loss_cfg.get("j_reg_weight", 1.0)),
        ce_weight=float(loss_cfg.get("ce_weight", 1.0)),
        dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
        negative_supervision_weight=float(loss_cfg.get("negative_supervision_weight", 0.5)),
        num_gland_classes=num_gland_classes,
        num_cell_classes=num_cell_classes,
        num_classification_classes=num_classification_classes,
    )


def build_loss(config: Dict[str, Any]) -> nn.Module:
    """Universal loss builder entrypoint for multi-task training."""
    return build_multitask_loss(config)


__all__ = ["build_loss", "build_multitask_loss"]

