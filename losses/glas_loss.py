# losses/glas_loss.py

from losses.topology_loss import TopologyAwareLoss


def build_glas_loss(config: dict):
    """
    Builds the loss function for GlaS gland segmentation.
    Supported loss types:
      - 'topology' / 'topology_aware' / 'topology_touching_cell': TopologyAwareLoss
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
    "build_glas_loss",
]

