# losses/monusac_loss.py

from typing import Dict, Any
from losses.segmentation_classification_topology_loss import SegmentationClassificationTopologyLoss

MoNuSACTopologyLoss = SegmentationClassificationTopologyLoss


def build_monusac_loss(config: Dict[str, Any]) -> SegmentationClassificationTopologyLoss:
    """
    Builds the joint dual-head segmentation & classification loss for MoNuSAC.
    Parses configuration parameters for BCE, Dice, Topology, and Multi-class CE/Dice losses.
    """
    loss_cfg = config["loss"]["segmentation"]
    return SegmentationClassificationTopologyLoss(
        bce_weight=float(loss_cfg.get("bce_weight", 1.0)),
        dice_weight=float(loss_cfg.get("dice_weight", 0.0)),
        topology_weight=float(loss_cfg.get("topology_weight", 0.1)),
        cls_ce_weight=float(loss_cfg.get("cls_ce_weight", 1.0)),
        cls_dice_weight=float(loss_cfg.get("cls_dice_weight", 1.0)),
        alpha=float(loss_cfg.get("alpha", 0.5)),
        beta=float(loss_cfg.get("beta", 0.5)),
        smooth=float(loss_cfg.get("smooth", 1e-6)),
        num_classes=int(config["dataset"].get("num_classes", 5)),
    )


__all__ = [
    "SegmentationClassificationTopologyLoss",
    "MoNuSACTopologyLoss",
    "build_monusac_loss",
]
