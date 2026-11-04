import torch
from torch import nn

from losses.dice_loss import DiceLoss


class TopologyAwareLoss(nn.Module):

    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        topology_weight=0.1,
        alpha=0.5,
        beta=0.5,
        smooth=1e-8,
    ):
        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.topology_weight = topology_weight

        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def topology_loss(
        self,
        dt,
        pred,
        gt,
    ):
        # logits -> probabilities
        pred = torch.sigmoid(pred)

        gt = gt.to(
            device=pred.device,
            dtype=pred.dtype,
        )

        dt = dt.to(
            device=pred.device,
            dtype=pred.dtype,
        )

        # Soft false positive
        fp = pred - (gt * pred)

        # Soft false negative
        fn = gt - (gt * pred)

        inside_distance = torch.clamp(-dt, min=0.0)
        outside_distance = torch.clamp(dt, min=0.0)

        loss_pixel = self.alpha * inside_distance * fn + self.beta * outside_distance * fp

        loss_topo = loss_pixel + self.smooth
        return loss_topo.mean()

    def forward(
        self,
        gt: torch.Tensor,
        pred: torch.Tensor,
        sdf: torch.Tensor,
    ):
        gt = gt.to(device=pred.device, dtype=pred.dtype)

        loss_bce = self.bce_loss(pred, gt)
        loss_dice = self.dice_loss(pred, gt)
        loss_topo = self.topology_loss(sdf, pred, gt)

        total_loss = (
            self.bce_weight * loss_bce
            + self.dice_weight * loss_dice
            + self.topology_weight * loss_topo
        )

        components = {
            "bce": loss_bce.detach().cpu(),
            "dice": loss_dice.detach().cpu(),
            "topology": loss_topo.detach().cpu(),
            "total": total_loss.detach().cpu(),
        }
        return total_loss, components
