# datasets/multitask_collate.py

from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F


def compute_4class_target(
    instance_masks: torch.Tensor,
    kernel_size: int = 5,
    touching_radius: int = 1,
) -> torch.Tensor:
    """
    Vectorized GPU-compatible computation converting [B, 1, H, W] (or [B, H, W])
    instance ID masks into the exact J4 4-class semantic masks:
      0: Background (Outside cells/glands)
      1: Cell Interior (M_cell, non-touching cell body pixels)
      2: Touching Region (Cell pixels directly adjacent to / within touching_radius of a different instance)
      3: Gap (Background pixels in narrow spaces between cells via Bottom-Hat: Closing(M) - M)

    Args:
        instance_masks: [B, 1, H, W] or [B, H, W] integer instance masks.
        kernel_size: Morphological closing kernel size for bottom-hat transform (default: 5).
        touching_radius: Pixel radius to expand the touching contact interface into contacting instances (default: 1).

    Returns:
        target_4class: [B, H, W] long tensor with values in {0, 1, 2, 3}.
    """
    if instance_masks.ndim == 4:
        inst = instance_masks.long()
    else:
        inst = instance_masks.unsqueeze(1).long()

    fg = (inst > 0).float()

    # 1. Morphological Bottom-Hat Transform on background:
    # Gap pixels are BACKGROUND pixels in narrow valleys/crevices between cells:
    # BTH(M) = Closing(M) - M (strictly where inst == 0)
    pad = kernel_size // 2
    dilated = F.max_pool2d(fg, kernel_size=kernel_size, stride=1, padding=pad)
    closed = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=pad)
    gap_mask = ((closed - fg) > 0.5) & (inst == 0)

    # 2. Touching Region (Class 2):
    # FOREGROUND pixels that touch a DIFFERENT cell instance in their 8-neighborhood.
    # Note: Isolated cells have NO touching region.
    shifts = [
        F.pad(inst[:, :, 1:, :], (0, 0, 0, 1), value=0),      # up
        F.pad(inst[:, :, :-1, :], (0, 0, 1, 0), value=0),     # down
        F.pad(inst[:, :, :, 1:], (0, 1, 0, 0), value=0),      # left
        F.pad(inst[:, :, :, :-1], (1, 0, 0, 0), value=0),     # right
        F.pad(inst[:, :, 1:, 1:], (0, 1, 0, 1), value=0),     # up-left
        F.pad(inst[:, :, 1:, :-1], (1, 0, 0, 1), value=0),    # up-right
        F.pad(inst[:, :, :-1, 1:], (0, 1, 1, 0), value=0),    # down-left
        F.pad(inst[:, :, :-1, :-1], (1, 0, 1, 0), value=0),   # down-right
    ]

    seam_touching = torch.zeros_like(inst, dtype=torch.bool)
    is_fg = (inst > 0)
    for s in shifts:
        seam_touching |= (is_fg & (s > 0) & (inst != s))

    # Expand touching region by touching_radius within the foreground
    if touching_radius > 1:
        k_touch = 2 * touching_radius + 1
        pad_t = touching_radius
        touching_expanded = F.max_pool2d(seam_touching.float(), kernel_size=k_touch, stride=1, padding=pad_t) > 0.5
        touching_mask = touching_expanded & is_fg
    else:
        touching_mask = seam_touching

    # 3. Cell Interior (Class 1):
    # Foreground pixels that are NOT touching another instance.
    interior_mask = is_fg & (~touching_mask)

    # Safety check: ensure small instances touching others do not completely disappear
    target_4class = torch.zeros_like(inst, dtype=torch.long)
    target_4class[gap_mask] = 3
    target_4class[touching_mask] = 2
    target_4class[interior_mask] = 1

    return target_4class.squeeze(1)


# Alias for backward compatibility
compute_3class_target = compute_4class_target


class MultiTaskCollateFn:
    """
    Collate function for multi-task learning across GLaS, MoNuSeg, and MoNuSAC,
    providing 4-class boundary/gap target maps via Bottom-Hat transform:
      0: Background
      1: Interior
      2: Boundary
      3: Gap

    Handles missing target maps gracefully:
      - 'class_map':  Padded with ignore_index (-1) for GLaS and MoNuSeg.
      - 'class_mask': Padded with 0.0 for GLaS and MoNuSeg.

    Supplies boolean indicator masks for routing features to decoupled model heads:
      - 'is_gland':    [B] bool (True for GLaS)
      - 'is_cell_seg': [B] bool (True for MoNuSeg and MoNuSAC)
      - 'is_cell_cls': [B] bool (True for MoNuSAC)
      - 'is_monuseg':  [B] bool (True for MoNuSeg)
      - 'is_monusac':  [B] bool (True for MoNuSAC)
    """
    def __init__(self, ignore_index: int = -1, num_classes: int = 5, check_equal_split: bool = True):
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.check_equal_split = check_equal_split

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        B = len(batch)
        if B == 0:
            return {}

        # 1. Images: [B, 3, H, W]
        images = torch.stack([b["image"] for b in batch], dim=0)

        # 2. Binary masks: [B, 1, H, W]
        binary_masks = torch.stack([b["binary_mask"] for b in batch], dim=0)

        # 3. 4-Class target masks: [B, H, W]
        if "target_4class" in batch[0]:
            target_4class = torch.stack([b["target_4class"] for b in batch], dim=0)
        elif "instance_mask" in batch[0]:
            inst_masks = torch.stack([b["instance_mask"] for b in batch], dim=0)
            target_4class = compute_4class_target(inst_masks)
        else:
            target_4class = binary_masks.squeeze(1).long()

        # 4. Class maps: [B, H, W]
        class_maps = torch.stack([b["class_map"] for b in batch], dim=0)

        # 5. Task IDs: [B]
        task_ids = torch.tensor([b.get("task_id", 0) for b in batch], dtype=torch.long)

        # 6. Boolean indicator masks
        is_gland = (task_ids == 0)
        is_monuseg = (task_ids == 1)
        is_monusac = (task_ids == 2)
        is_cell_seg = is_monuseg | is_monusac
        is_cell_cls = is_monusac

        # 7. 5-Channel class mask: [B, 5, H, W]
        if any("class_mask" in b and b["class_mask"] is not None for b in batch):
            _, _, H, W = images.shape
            class_masks = torch.zeros((B, self.num_classes, H, W), dtype=torch.float32)
            for i, b in enumerate(batch):
                if "class_mask" in b and b["class_mask"] is not None:
                    class_masks[i] = b["class_mask"]
        else:
            class_masks = None

        return {
            "image": images,                     # [B, 3, H, W]
            "target_4class": target_4class,     # [B, H, W] {0: bg, 1: inside, 2: boundary, 3: gap}
            "target_3class": target_4class,     # Alias for backwards compatibility
            "binary_mask": binary_masks,         # [B, 1, H, W]
            "mask": binary_masks,                # Alias for compatibility
            "class_map": class_maps,             # [B, H, W] (-1 for non-MoNuSAC)
            "class_mask": class_masks,           # [B, 5, H, W]
            "task_id": task_ids,                 # [B]
            "is_gland": is_gland,               # [B] bool
            "is_cell_seg": is_cell_seg,         # [B] bool
            "is_cell_cls": is_cell_cls,         # [B] bool
            "is_monuseg": is_monuseg,           # [B] bool
            "is_monusac": is_monusac,           # [B] bool
            "patch_names": [b.get("patch_name", f"patch_{i}") for i, b in enumerate(batch)],
        }


__all__ = ["compute_4class_target", "compute_3class_target", "MultiTaskCollateFn"]
