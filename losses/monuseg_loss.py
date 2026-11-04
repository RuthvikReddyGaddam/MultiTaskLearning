# losses/monuseg_loss.py
from losses.topology_loss import TopologyAwareLoss

def build_monuseg_loss(config: dict):
    """
    Builds the loss function for MoNuSeg single-head nuclear segmentation.
    Supported loss types:
      - 'topology_touching_cell': TopologyTouchingCellLoss
      - 'topology' / 'topology_aware': TopologyAwareLoss
      - 'bce_dice': SegmentationLoss
      - 'bce_jaccard': BCEJaccard
    """
    loss_cfg = config["loss"]["segmentation"]
    return TopologyAwareLoss(
        alpha=float(loss_cfg.get("alpha", 0.5)),
        beta=float(loss_cfg.get("beta", 0.5)),
        bce_weight=float(loss_cfg.get("bce_weight", 1.0)),
        dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
        topology_weight=float(loss_cfg.get("topology_weight", 0.1)),
    )


__all__ = [
    "build_monuseg_loss",
]

