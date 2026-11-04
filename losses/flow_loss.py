# losses/flow_loss.py

from typing import Dict, Tuple, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowVectorLoss(nn.Module):
    """
    Loss function for training models to predict 2D flow vectors:
      F = (Fx, Fy) in [-1.0, 1.0]^2
    created by normalizing the two gradient channels.

    Penalizes:
      1. MSE coordinate error on foreground nuclei/cell pixels.
      2. Cosine angular difference (1 - cos(theta)) on foreground pixels to enforce correct orientation.
      3. Background flow regularization: penalizes non-zero flow on background where no nuclei exist.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        cosine_weight: float = 0.5,
        bg_weight: float = 0.1,
        eps: float = 1e-7,
        active_mag_threshold: float = 0.05,
    ):
        super().__init__()
        self.mse_weight = float(mse_weight)
        self.cosine_weight = float(cosine_weight)
        self.bg_weight = float(bg_weight)
        self.eps = float(eps)
        self.active_mag_threshold = float(active_mag_threshold)

    def forward(
        self,
        pred_flow: torch.Tensor,
        gt_flow: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
          pred_flow: [B, 2, H, W] predicted flow vector (Fx, Fy).
          gt_flow:   [B, 2, H, W] ground truth flow vector (Fx, Fy) in [-1.0, 1.0].
          mask:      [B, 1, H, W] or [B, H, W] binary foreground mask (1.0 for nuclei, 0.0 for bg).

        Returns:
          total_loss: scalar tensor.
          loss_dict: dictionary with individual loss components.
        """
        if mask is not None and mask.ndim == 3:
            mask = mask.unsqueeze(1)

        # 1. Coordinate MSE / L2 loss
        sq_diff = (pred_flow - gt_flow) ** 2  # [B, 2, H, W]
        l2_err = torch.sum(sq_diff, dim=1, keepdim=True)  # [B, 1, H, W]

        if mask is not None:
            fg = (mask > 0.5).float()
            bg = (mask <= 0.5).float()

            num_fg = torch.clamp(fg.sum(), min=1.0)
            num_bg = torch.clamp(bg.sum(), min=1.0)

            mse_fg_loss = (l2_err * fg).sum() / num_fg

            # Background flow should be 0
            pred_sq = torch.sum(pred_flow ** 2, dim=1, keepdim=True)
            bg_loss = (pred_sq * bg).sum() / num_bg
        else:
            mse_fg_loss = l2_err.mean()
            bg_loss = torch.tensor(0.0, device=pred_flow.device)

        # 2. Cosine Directional Loss (enforces physical orientation towards center/skeleton)
        gt_mag = torch.sqrt(torch.clamp(torch.sum(gt_flow ** 2, dim=1, keepdim=True), min=0.0))
        pred_mag = torch.sqrt(torch.clamp(torch.sum(pred_flow ** 2, dim=1, keepdim=True), min=0.0))

        # Dot product
        dot = torch.sum(pred_flow * gt_flow, dim=1, keepdim=True)
        denom = (pred_mag * gt_mag) + self.eps
        cos_sim = dot / denom  # [-1, 1]
        cos_loss_pixel = 1.0 - cos_sim  # [0, 2]

        if mask is not None:
            # Only compute cosine loss where gt has meaningful flow magnitude and inside foreground
            active_fg = fg * (gt_mag > self.active_mag_threshold).float()
            num_active = torch.clamp(active_fg.sum(), min=1.0)
            cos_loss = (cos_loss_pixel * active_fg).sum() / num_active
        else:
            cos_loss = cos_loss_pixel.mean()

        total_loss = (
            self.mse_weight * mse_fg_loss
            + self.cosine_weight * cos_loss
            + self.bg_weight * bg_loss
        )

        loss_dict = {
            "flow_total": total_loss.detach(),
            "flow_mse": mse_fg_loss.detach(),
            "flow_cos": cos_loss.detach(),
            "flow_bg": bg_loss.detach(),
        }

        return total_loss, loss_dict


def build_flow_loss(config: Optional[Dict] = None) -> FlowVectorLoss:
    """Helper factory for FlowVectorLoss."""
    if config is None:
        return FlowVectorLoss()
    flow_cfg = config.get("loss", {}).get("flow", {})
    return FlowVectorLoss(
        mse_weight=float(flow_cfg.get("mse_weight", 1.0)),
        cosine_weight=float(flow_cfg.get("cosine_weight", 0.5)),
        bg_weight=float(flow_cfg.get("bg_weight", 0.1)),
    )


__all__ = [
    "FlowVectorLoss",
    "build_flow_loss",
]

