# datasets/multitask_augment.py

from typing import Dict, Any, Optional
import random
import torch
from torchvision.transforms import v2


class MultiTaskSpatialAugmentor:
    """
    Synchronous spatial and appearance augmentation for multi-task pathology patches.
    Transforms:
      - 'image':          [3, H, W] float32 in [0, 1]
      - 'mask':           [1, H, W] float32 binary mask
      - 'instance_mask':  [1, H, W] or [H, W] int64 instance mask
      - 'sdf':            [1, H, W] float32 signed distance map
      - 'class_mask':     Optional [5, H, W] float32 multi-class mask
      - 'class_map':      Optional [H, W] int64 multi-class semantic map
    """
    def __init__(
        self,
        p_hflip: float = 0.5,
        p_vflip: float = 0.5,
        p_rot90: float = 0.5,
        color_jitter: bool = True,
    ):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rot90 = p_rot90
        self.color_jitter = color_jitter
        if color_jitter:
            self.jitter = v2.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            )
        else:
            self.jitter = None

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # 1. Random Horizontal Flip
        if random.random() < self.p_hflip:
            sample["image"] = torch.flip(sample["image"], dims=[-1])
            sample["binary_mask"] = torch.flip(sample["binary_mask"], dims=[-1])
            sample["target_4class"] = torch.flip(sample["target_4class"], dims=[-1])
            sample["class_map"] = torch.flip(sample["class_map"], dims=[-1])
            if "mask" in sample:
                sample["mask"] = sample["binary_mask"]
            if "target_3class" in sample:
                sample["target_3class"] = sample["target_4class"]
            if "class_mask" in sample and sample["class_mask"] is not None:
                sample["class_mask"] = torch.flip(sample["class_mask"], dims=[-1])

        # 2. Random Vertical Flip
        if random.random() < self.p_vflip:
            sample["image"] = torch.flip(sample["image"], dims=[-2])
            sample["binary_mask"] = torch.flip(sample["binary_mask"], dims=[-2])
            sample["target_4class"] = torch.flip(sample["target_4class"], dims=[-2])
            sample["class_map"] = torch.flip(sample["class_map"], dims=[-2])
            if "mask" in sample:
                sample["mask"] = sample["binary_mask"]
            if "target_3class" in sample:
                sample["target_3class"] = sample["target_4class"]
            if "class_mask" in sample and sample["class_mask"] is not None:
                sample["class_mask"] = torch.flip(sample["class_mask"], dims=[-2])

        # 3. Random 90-degree Rotation
        if random.random() < self.p_rot90:
            k = random.choice([1, 2, 3])
            sample["image"] = torch.rot90(sample["image"], k=k, dims=[-2, -1])
            sample["binary_mask"] = torch.rot90(sample["binary_mask"], k=k, dims=[-2, -1])
            sample["target_4class"] = torch.rot90(sample["target_4class"], k=k, dims=[-2, -1])
            sample["class_map"] = torch.rot90(sample["class_map"], k=k, dims=[-2, -1])
            if "mask" in sample:
                sample["mask"] = sample["binary_mask"]
            if "target_3class" in sample:
                sample["target_3class"] = sample["target_4class"]
            if "class_mask" in sample and sample["class_mask"] is not None:
                sample["class_mask"] = torch.rot90(sample["class_mask"], k=k, dims=[-2, -1])

        # 4. Color Jitter
        if self.jitter is not None and random.random() < 0.3:
            sample["image"] = self.jitter(sample["image"])

        return sample
