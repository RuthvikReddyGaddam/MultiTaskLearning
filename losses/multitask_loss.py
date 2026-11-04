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

        half = B // 2
        q3 = half + B // 4

        # Fast path: deterministic multiple-of-8 contract (2:1:1 balanced batch)
        if (
            B % 8 == 0
            and task_ids is not None
            and (task_ids[:half] == 0).all()
            and (task_ids[half:q3] == 1).all()
            and (task_ids[q3:] == 2).all()
        ):
            p_gland = preds["gland_seg"]
            p_cell = preds["cell_seg"]
            p_cls = preds["cell_class"]

            l_gland_dict = self.glas_seg_loss(p_gland, target_seg[:half])
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

            l_monuseg_dict = self.cell_seg_loss(p_cell[: B // 4], target_seg[half:q3])
            l_monusac_seg_dict = self.cell_seg_loss(p_cell[B // 4 :], target_seg[q3:])

            sac_cls_gt = batch["class_map"][q3:]
            valid_mask = (sac_cls_gt >= 0).float()
            l_monusac_cls_dict = self.cell_cls_loss(p_cls, sac_cls_gt.clamp(min=0), mask=valid_mask)

            l_monuseg = l_monuseg_dict["total_loss"]
            l_monusac_seg = l_monusac_seg_dict["total_loss"]
            l_monusac_cls = l_monusac_cls_dict["total_loss"]

            total_loss = (
                self.glas_weight * l_gland
                + self.monuseg_weight * l_monuseg
                + self.monusac_weight * (l_monusac_seg + l_monusac_cls)
            )

            logs = {
                "loss_glas_total": l_gland.detach(),
                "loss_glas_ce": l_gland_dict["ce_loss"].detach(),
                "loss_glas_j": l_gland_dict["j_loss"].detach(),
                "loss_glas_dice": l_gland_dict["dice_loss"].detach(),
                "loss_glas_neg": l_neg.detach(),
                "loss_monuseg_total": l_monuseg.detach(),
                "loss_monuseg_ce": l_monuseg_dict["ce_loss"].detach(),
                "loss_monuseg_j": l_monuseg_dict["j_loss"].detach(),
                "loss_monuseg_dice": l_monuseg_dict["dice_loss"].detach(),
                "loss_monusac_seg_total": l_monusac_seg.detach(),
                "loss_monusac_seg_j": l_monusac_seg_dict["j_loss"].detach(),
                "loss_monusac_cls_total": l_monusac_cls.detach(),
                "loss_monusac_cls_ce": l_monusac_cls_dict["ce_loss"].detach(),
                "loss_monusac_cls_j": l_monusac_cls_dict["j_loss"].detach(),
                "loss_monusac_cls_dice": l_monusac_cls_dict["dice_loss"].detach(),
                "total_loss": total_loss.detach(),
            }
            return total_loss, logs

        # Dynamic fallback path for non-standard batches or unit tests
        is_gland = batch.get("is_gland", task_ids == 0 if task_ids is not None else torch.zeros(B, dtype=torch.bool))
        is_monuseg = batch.get("is_monuseg", task_ids == 1 if task_ids is not None else torch.zeros(B, dtype=torch.bool))
        is_monusac = batch.get("is_monusac", task_ids == 2 if task_ids is not None else torch.zeros(B, dtype=torch.bool))

        total_loss = torch.tensor(0.0, device=target_seg.device, requires_grad=True)
        logs = {}

        if is_gland.any():
            gland_gt = target_seg[is_gland]
            p_gland = preds.get("gland_seg", preds.get("glas_seg"))
            if p_gland is not None:
                p_gland_slice = p_gland[is_gland] if p_gland.shape[0] == is_gland.shape[0] else p_gland
                l_gland_dict = self.glas_seg_loss(p_gland_slice, gland_gt)
                l_gland = l_gland_dict["total_loss"]
                total_loss = total_loss + self.glas_weight * l_gland
                logs["loss_glas_total"] = l_gland.detach()
                logs["loss_glas_ce"] = l_gland_dict["ce_loss"].detach()
                logs["loss_glas_j"] = l_gland_dict["j_loss"].detach()
                logs["loss_glas_dice"] = l_gland_dict["dice_loss"].detach()

        if is_monuseg.any():
            monuseg_gt = target_seg[is_monuseg]
            p_cell = preds.get("monuseg_seg", preds.get("cell_seg"))
            if p_cell is not None:
                p_cell_slice = p_cell[is_monuseg] if p_cell.shape[0] == is_monuseg.shape[0] else p_cell
                l_cell_dict = self.cell_seg_loss(p_cell_slice, monuseg_gt)
                l_cell = l_cell_dict["total_loss"]
                total_loss = total_loss + self.monuseg_weight * l_cell
                logs["loss_monuseg_total"] = l_cell.detach()
                logs["loss_monuseg_ce"] = l_cell_dict["ce_loss"].detach()
                logs["loss_monuseg_j"] = l_cell_dict["j_loss"].detach()
                logs["loss_monuseg_dice"] = l_cell_dict["dice_loss"].detach()

        if is_monusac.any():
            monusac_gt = target_seg[is_monusac]
            monusac_gt_cls = batch["class_map"][is_monusac]

            p_cell = preds.get("monusac_seg", preds.get("cell_seg"))
            if p_cell is not None:
                p_cell_slice = p_cell[is_monusac] if p_cell.shape[0] == is_monusac.shape[0] else p_cell
                l_cell_dict = self.cell_seg_loss(p_cell_slice, monusac_gt)
                l_cell = l_cell_dict["total_loss"]
                total_loss = total_loss + self.monusac_weight * l_cell
                logs["loss_monusac_seg_total"] = l_cell.detach()
                logs["loss_monusac_seg_j"] = l_cell_dict["j_loss"].detach()

            p_cls = preds.get("cell_class", preds.get("monusac_class"))
            if p_cls is not None:
                p_cls_slice = p_cls[is_monusac] if p_cls.shape[0] == is_monusac.shape[0] else p_cls
                valid_mask = (monusac_gt_cls >= 0).float()
                safe_targets = monusac_gt_cls.clamp(min=0)
                l_cls_dict = self.cell_cls_loss(p_cls_slice, safe_targets, mask=valid_mask)
                l_cls = l_cls_dict["total_loss"]
                total_loss = total_loss + self.monusac_weight * l_cls
                logs["loss_monusac_cls_total"] = l_cls.detach()
                logs["loss_monusac_cls_ce"] = l_cls_dict["ce_loss"].detach()
                logs["loss_monusac_cls_j"] = l_cls_dict["j_loss"].detach()
                logs["loss_monusac_cls_dice"] = l_cls_dict["dice_loss"].detach()

        logs["total_loss"] = total_loss.detach()
        return total_loss, logs


__all__ = ["MultiTaskLoss"]
