# datasets/multitask_collate.py

from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F


def compute_4class_target(instance_masks: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """
    Vectorized GPU-compatible computation converting [B, 1, H, W] (or [B, H, W])
    instance ID masks into 4-class semantic masks using the Bottom-Hat transform:
      0: Background
      1: Instance Interior (Core)
      2: Instance Boundary (Contour ring)
      3: Inter-Instance Gap (Valleys and touching seams via Bottom-Hat transform: Closing(M) - M)

    Args:
        instance_masks: [B, 1, H, W] or [B, H, W] integer instance masks.
        kernel_size: Morphological closing kernel size for bottom-hat transform (default: 5).

    Returns:
        target_4class: [B, H, W] long tensor with values in {0, 1, 2, 3}.
    """
    if instance_masks.ndim == 4:
        inst = instance_masks.float()
    else:
        inst = instance_masks.unsqueeze(1).float()

    fg = (inst > 0).float()

    # 1. Bottom-Hat Transform via PyTorch max_pool closing:
    # Closing(M) = Erosion(Dilation(M))
    # BTH(M) = Closing(M) - M
    pad = kernel_size // 2
    dilated = F.max_pool2d(fg, kernel_size=kernel_size, stride=1, padding=pad)
    closed = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=pad)
    bottom_hat = ((closed - fg) > 0.5)

    # 2. Contact seams between different touching instances
    s_up = F.pad(inst[:, :, 1:, :], (0, 0, 0, 1), value=0)
    s_down = F.pad(inst[:, :, :-1, :], (0, 0, 1, 0), value=0)
    s_left = F.pad(inst[:, :, :, 1:], (0, 1, 0, 0), value=0)
    s_right = F.pad(inst[:, :, :, :-1], (1, 0, 0, 0), value=0)

    touch_other = (
        ((inst > 0) & (s_up > 0) & (inst != s_up)) |
        ((inst > 0) & (s_down > 0) & (inst != s_down)) |
        ((inst > 0) & (s_left > 0) & (inst != s_left)) |
        ((inst > 0) & (s_right > 0) & (inst != s_right))
    )
    gap_mask = bottom_hat | touch_other

    # 3. Interior and Boundary separation
    diff_up = (inst != s_up)
    diff_down = (inst != s_down)
    diff_left = (inst != s_left)
    diff_right = (inst != s_right)
    is_boundary = (diff_up | diff_down | diff_left | diff_right) & (inst > 0)
    is_interior = (inst > 0) & (~is_boundary)

    target_4class = torch.zeros_like(inst, dtype=torch.long)
    target_4class[is_interior] = 1
    target_4class[is_boundary] = 2
    target_4class[gap_mask] = 3

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
