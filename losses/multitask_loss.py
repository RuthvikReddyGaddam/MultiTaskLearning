# losses/multitask_loss.py

from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.j_regularization_loss import JRegularizedMultiClassLoss


class MultiTaskLoss(nn.Module):
    """
    Decoupled Multi-Task Loss with 4-Channel Youden's J-Regularization for:
      - GLaS:    4-class gland segmentation (bg, inside, boundary, gap)
      - MoNuSeg: 4-class cell nuclei segmentation (bg, inside, boundary, gap)
      - MoNuSAC: Shared 4-class cell nuclei segmentation + 5-class cell phenotype classification

    Key Advantages over Flow Regression:
      1. Intrinsic class balancing via Youden's J-statistic (Sensitivity + Specificity - 1)
         eliminates manual weighting and prevents optimization stalls on rare gap pixels.
      2. Bottom-Hat transform explicitly extracts inter-cellular clefts and touching contact seams.
      3. Zero flow prediction dependencies (faster training, fewer hyper-parameters, no Eikonal shock issues).
    """
    def __init__(
        self,
        glas_weight: float = 1.0,
        monuseg_weight: float = 1.0,
        monusac_weight: float = 1.0,
        j_reg_weight: float = 1.0,
        ce_weight: float = 1.0,
        dice_weight: float = 0.5,
        num_gland_classes: int = 4,
        num_cell_classes: int = 4,
        num_classification_classes: int = 5,
        negative_supervision_weight: float = 0.5,
        normalize_active_patches: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.glas_weight = float(glas_weight)
        self.monuseg_weight = float(monuseg_weight)
        self.monusac_weight = float(monusac_weight)
        self.j_reg_weight = float(j_reg_weight)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        self.negative_supervision_weight = float(negative_supervision_weight)
        self.normalize_active_patches = bool(normalize_active_patches)
        self.monusac_seg_weight = float(kwargs.get("monusac_seg_weight", 0.5 if self.normalize_active_patches else 1.0))
        self.monusac_cls_weight = float(kwargs.get("monusac_cls_weight", 0.5 if self.normalize_active_patches else 1.0))

        # 1. GLaS 4-Class Loss (bg, gland inside, boundary, bottom-hat gap)
        self.glas_seg_loss = JRegularizedMultiClassLoss(
            num_classes=num_gland_classes,
            ce_weight=ce_weight,
            j_weight=j_reg_weight,
            dice_weight=dice_weight,
        )

        # 2. Shared Cell 4-Class Loss (bg, nucleus inside, boundary, bottom-hat gap)
        self.cell_seg_loss = JRegularizedMultiClassLoss(
            num_classes=num_cell_classes,
            ce_weight=ce_weight,
            j_weight=j_reg_weight,
            dice_weight=dice_weight,
        )

        # 3. MoNuSAC Multi-Class Phenotype Classification Loss (5 classes)
        self.cell_cls_loss = JRegularizedMultiClassLoss(
            num_classes=num_classification_classes,
            ce_weight=ce_weight,
            j_weight=j_reg_weight,
            dice_weight=dice_weight,
        )

    def forward(
        self,
        preds: Dict[str, Optional[torch.Tensor]],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
          preds: Dictionary containing head predictions:
            - 'gland_seg' (or 'glas_seg')
            - 'cell_seg' (or 'monuseg_seg', 'monusac_seg')
            - 'cell_class' (or 'monusac_class')
          batch: Batch dictionary from MultiTaskCollateFn containing:
            - 'target_4class', 'class_map', 'is_gland', 'is_monuseg', 'is_monusac'
        """
        target_seg = batch.get("target_4class", batch.get("target_3class"))
        task_ids = batch.get("task_id")
        B = target_seg.shape[0]

        # =========================================================
        # Mask-based multi-task loss computation (supports shuffled batches)
        # =========================================================
        is_gland = batch.get("is_gland")
        if is_gland is None:
            is_gland = (task_ids == 0) if task_ids is not None else torch.zeros(B, dtype=torch.bool, device=target_seg.device)
        is_monuseg = batch.get("is_monuseg")
        if is_monuseg is None:
            is_monuseg = (task_ids == 1) if task_ids is not None else torch.zeros(B, dtype=torch.bool, device=target_seg.device)
        is_monusac = batch.get("is_monusac")
        if is_monusac is None:
            is_monusac = (task_ids == 2) if task_ids is not None else torch.zeros(B, dtype=torch.bool, device=target_seg.device)

        if not isinstance(is_gland, torch.Tensor):
            is_gland = torch.as_tensor(is_gland, dtype=torch.bool, device=target_seg.device)
        else:
            is_gland = is_gland.to(target_seg.device)
        if not isinstance(is_monuseg, torch.Tensor):
            is_monuseg = torch.as_tensor(is_monuseg, dtype=torch.bool, device=target_seg.device)
        else:
            is_monuseg = is_monuseg.to(target_seg.device)
        if not isinstance(is_monusac, torch.Tensor):
            is_monusac = torch.as_tensor(is_monusac, dtype=torch.bool, device=target_seg.device)
        else:
            is_monusac = is_monusac.to(target_seg.device)

        is_cell = is_monuseg | is_monusac

        n_gland = int(is_gland.sum().item())
        n_monuseg = int(is_monuseg.sum().item())
        n_monusac = int(is_monusac.sum().item())
        n_total = max(1, n_gland + n_monuseg + n_monusac)

        if self.normalize_active_patches:
            eff_glas_w = self.glas_weight * (n_gland / n_total) if n_gland > 0 else 0.0
            eff_monuseg_w = self.monuseg_weight * (n_monuseg / n_total) if n_monuseg > 0 else 0.0
            eff_monusac_seg_w = self.monusac_weight * (n_monusac / n_total) * self.monusac_seg_weight if n_monusac > 0 else 0.0
            eff_monusac_cls_w = self.monusac_weight * (n_monusac / n_total) * self.monusac_cls_weight if n_monusac > 0 else 0.0
        else:
            eff_glas_w = self.glas_weight if n_gland > 0 else 0.0
            eff_monuseg_w = self.monuseg_weight if n_monuseg > 0 else 0.0
            eff_monusac_seg_w = self.monusac_weight * self.monusac_seg_weight if n_monusac > 0 else 0.0
            eff_monusac_cls_w = self.monusac_weight * self.monusac_cls_weight if n_monusac > 0 else 0.0

        total_loss = torch.tensor(0.0, device=target_seg.device, requires_grad=True)
        logs = {
            "weight_eff_glas": float(eff_glas_w),
            "weight_eff_monuseg": float(eff_monuseg_w),
            "weight_eff_monusac_seg": float(eff_monusac_seg_w),
            "weight_eff_monusac_cls": float(eff_monusac_cls_w),
        }

        # 1. GLaS (gland segmentation + negative background supervision)
        if is_gland.any() and eff_glas_w > 0.0:
            target_gland = target_seg[is_gland]
            p_gland = preds.get("gland_seg", preds.get("glas_seg"))
            if p_gland is not None:
                p_gland_slice = p_gland[is_gland] if p_gland.shape[0] == B else p_gland
                l_gland_dict = self.glas_seg_loss(p_gland_slice, target_gland)
                l_gland = l_gland_dict["total_loss"]

                l_neg = torch.tensor(0.0, device=target_seg.device)
                if "gland_neg" in preds and self.negative_supervision_weight > 0.0:
                    p_gland_neg = preds["gland_neg"]
                    target_neg = torch.zeros(
                        (p_gland_neg.shape[0], p_gland_neg.shape[-2], p_gland_neg.shape[-1]),
                        dtype=torch.long,
                        device=p_gland_neg.device,
                    )
                    l_neg = F.cross_entropy(p_gland_neg, target_neg)
                    l_gland = l_gland + self.negative_supervision_weight * l_neg

                total_loss = total_loss + eff_glas_w * l_gland
                logs["loss_glas_total"] = l_gland.detach()
                logs["loss_glas_ce"] = l_gland_dict["ce_loss"].detach()
                logs["loss_glas_j"] = l_gland_dict["j_loss"].detach()
                logs["loss_glas_dice"] = l_gland_dict["dice_loss"].detach()
                logs["loss_glas_neg"] = l_neg.detach()

        # 2. MoNuSeg (cell nuclei segmentation)
        if is_monuseg.any() and eff_monuseg_w > 0.0:
            target_monuseg = target_seg[is_monuseg]
            p_monuseg = preds.get("monuseg_seg")
            if p_monuseg is not None and p_monuseg.shape[0] == B:
                p_monuseg = p_monuseg[is_monuseg]
            elif p_monuseg is None and "cell_seg" in preds:
                p_cell = preds["cell_seg"]
                if p_cell.shape[0] == B:
                    p_monuseg = p_cell[is_monuseg]
                elif is_cell.any() and p_cell.shape[0] == is_cell.sum().item() and task_ids is not None:
                    p_monuseg = p_cell[task_ids[is_cell] == 1]
                else:
                    p_monuseg = p_cell

            if p_monuseg is not None:
                l_monuseg_dict = self.cell_seg_loss(p_monuseg, target_monuseg)
                l_monuseg = l_monuseg_dict["total_loss"]
                total_loss = total_loss + eff_monuseg_w * l_monuseg
                logs["loss_monuseg_total"] = l_monuseg.detach()
                logs["loss_monuseg_ce"] = l_monuseg_dict["ce_loss"].detach()
                logs["loss_monuseg_j"] = l_monuseg_dict["j_loss"].detach()
                logs["loss_monuseg_dice"] = l_monuseg_dict["dice_loss"].detach()

        # 3. MoNuSAC (cell nuclei segmentation + 5-class cell phenotype classification)
        if is_monusac.any():
            target_monusac = target_seg[is_monusac]
            if eff_monusac_seg_w > 0.0:
                p_monusac_seg = preds.get("monusac_seg")
                if p_monusac_seg is not None and p_monusac_seg.shape[0] == B:
                    p_monusac_seg = p_monusac_seg[is_monusac]
                elif p_monusac_seg is None and "cell_seg" in preds:
                    p_cell = preds["cell_seg"]
                    if p_cell.shape[0] == B:
                        p_monusac_seg = p_cell[is_monusac]
                    elif is_cell.any() and p_cell.shape[0] == is_cell.sum().item() and task_ids is not None:
                        p_monusac_seg = p_cell[task_ids[is_cell] == 2]
                    else:
                        p_monusac_seg = p_cell

                if p_monusac_seg is not None:
                    l_monusac_seg_dict = self.cell_seg_loss(p_monusac_seg, target_monusac)
                    l_monusac_seg = l_monusac_seg_dict["total_loss"]
                    total_loss = total_loss + eff_monusac_seg_w * l_monusac_seg
                    logs["loss_monusac_seg_total"] = l_monusac_seg.detach()
                    logs["loss_monusac_seg_j"] = l_monusac_seg_dict["j_loss"].detach() if "j_loss" in l_monusac_seg_dict else torch.tensor(0.0)

            if eff_monusac_cls_w > 0.0:
                p_cls = preds.get("cell_class", preds.get("monusac_class"))
                if p_cls is not None:
                    p_cls_slice = p_cls[is_monusac] if p_cls.shape[0] == B else p_cls
                    sac_cls_gt = batch["class_map"][is_monusac]
                    valid_mask = (sac_cls_gt >= 0).float()
                    safe_targets = sac_cls_gt.clamp(min=0)
                    l_cls_dict = self.cell_cls_loss(p_cls_slice, safe_targets, mask=valid_mask)
                    l_cls = l_cls_dict["total_loss"]
                    total_loss = total_loss + eff_monusac_cls_w * l_cls
                    logs["loss_monusac_cls_total"] = l_cls.detach()
                    logs["loss_monusac_cls_ce"] = l_cls_dict["ce_loss"].detach()
                    logs["loss_monusac_cls_j"] = l_cls_dict["j_loss"].detach()
                    logs["loss_monusac_cls_dice"] = l_cls_dict["dice_loss"].detach()

        logs["total_loss"] = total_loss.detach()
        return total_loss, logs


__all__ = ["MultiTaskLoss"]
