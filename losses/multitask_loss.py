# losses/multitask_loss.py

from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.j_regularization_loss import JRegularizedMultiClassLoss, MultiClassDiceLoss


class MultiTaskLoss(nn.Module):
    """
    Decoupled Multi-Task Loss with J4 Topological Regularization:
      - GLaS:    4-class gland segmentation (bg, cell, touching, gap) via J-Regularization
      - MoNuSeg: 4-class cell nuclei segmentation (bg, cell, touching, gap) via J-Regularization
      - MoNuSAC: Shared 4-class cell segmentation + 5-class whole-patch cell phenotype classification
                 via unweighted Cross-Entropy + MultiClass Dice (without J-regularization).
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
        self.normalize_active_patches = bool(normalize_active_patches)
        self.monusac_seg_weight = float(kwargs.get("monusac_seg_weight", 0.5 if self.normalize_active_patches else 1.0))
        self.monusac_cls_weight = float(kwargs.get("monusac_cls_weight", 0.5 if self.normalize_active_patches else 1.0))

        # 1. GLaS 4-Class Segmentation Loss (bg, gland cell, touching, gap)
        self.glas_seg_loss = JRegularizedMultiClassLoss(
            num_classes=num_gland_classes,
            ce_weight=ce_weight,
            j_weight=j_reg_weight,
            dice_weight=dice_weight,
        )

        # 2. Shared Cell 4-Class Segmentation Loss (bg, nucleus cell, touching, gap)
        self.cell_seg_loss = JRegularizedMultiClassLoss(
            num_classes=num_cell_classes,
            ce_weight=ce_weight,
            j_weight=j_reg_weight,
            dice_weight=dice_weight,
        )

        # 3. MoNuSAC 5-Class Whole-Patch Phenotype Classification Loss:
        # Strictly unweighted Cross-Entropy (weight=None) + MultiClass Dice (NO J-regularization)
        self.cls_ce_loss = nn.CrossEntropyLoss(weight=None, ignore_index=-1)
        self.cls_dice_loss = MultiClassDiceLoss(
            num_classes=num_classification_classes,
            include_background=True,
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

        # 1. GLaS (gland segmentation)
        if is_gland.any() and eff_glas_w > 0.0:
            target_gland = target_seg[is_gland]
            p_gland = preds.get("gland_seg", preds.get("glas_seg"))
            if p_gland is not None:
                p_gland_slice = p_gland[is_gland] if p_gland.shape[0] == B else p_gland
                l_gland_dict = self.glas_seg_loss(p_gland_slice, target_gland)
                l_gland = l_gland_dict["total_loss"]

                total_loss = total_loss + eff_glas_w * l_gland
                logs["loss_glas_total"] = l_gland.detach()
                logs["loss_glas_ce"] = l_gland_dict["ce_loss"].detach()
                logs["loss_glas_j"] = l_gland_dict["j_loss"].detach()
                logs["loss_glas_dice"] = l_gland_dict["dice_loss"].detach()

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

        # 3. MoNuSAC (cell nuclei segmentation + 5-class whole-patch cell phenotype classification)
        if is_monusac.any():
            target_monusac = target_seg[is_monusac]
            if eff_monusac_seg_w > 0.0:
                p_monusac_seg = preds.get("monusac_seg", preds.get("monuseg_seg"))
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
                    logs["loss_monusac_seg_ce"] = l_monusac_seg_dict["ce_loss"].detach()
                    logs["loss_monusac_seg_j"] = l_monusac_seg_dict["j_loss"].detach()
                    logs["loss_monusac_seg_dice"] = l_monusac_seg_dict["dice_loss"].detach()

            if eff_monusac_cls_w > 0.0:
                p_cls = preds.get("cell_class", preds.get("monusac_class"))
                if p_cls is not None:
                    p_cls_slice = p_cls[is_monusac] if p_cls.shape[0] == B else p_cls
                    sac_cls_gt = batch["class_map"][is_monusac]  # [N, H, W] with 0..4 (and -1 if padded)

                    # Check if there are any valid supervised pixels (sac_cls_gt >= 0)
                    valid_mask_bool = (sac_cls_gt >= 0)
                    if valid_mask_bool.any():
                        # 1. Unweighted Cross-Entropy loss over the whole patch (no class weighting)
                        l_cls_ce = self.cls_ce_loss(p_cls_slice, sac_cls_gt)

                        # 2. MultiClass Dice loss over the whole patch (valid where sac_cls_gt >= 0)
                        valid_mask = valid_mask_bool.float()
                        safe_targets = sac_cls_gt.clamp(min=0)
                        l_cls_dice = self.cls_dice_loss(p_cls_slice, safe_targets, mask=valid_mask)

                        l_cls = self.ce_weight * l_cls_ce + self.dice_weight * l_cls_dice
                    else:
                        l_cls_ce = torch.tensor(0.0, device=target_seg.device, requires_grad=True)
                        l_cls_dice = torch.tensor(0.0, device=target_seg.device, requires_grad=True)
                        l_cls = torch.tensor(0.0, device=target_seg.device, requires_grad=True)

                    total_loss = total_loss + eff_monusac_cls_w * l_cls

                    logs["loss_monusac_cls_total"] = l_cls.detach()
                    logs["loss_monusac_cls_ce"] = l_cls_ce.detach()
                    logs["loss_monusac_cls_dice"] = l_cls_dice.detach()

        logs["total_loss"] = total_loss.detach()
        return total_loss, logs


__all__ = ["MultiTaskLoss"]
