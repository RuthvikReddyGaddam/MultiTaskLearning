# datasets/multitask_augment.py

from typing import Dict, Any, Optional
import random
import torch
import torch.nn.functional as F
from torchvision.transforms import v2


def apply_median_blur(img: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Applies median blur to a [C, H, W] or [B, C, H, W] float32 image tensor.
    Uses sliding window unfold along spatial dimensions for pure PyTorch tensor processing.
    """
    orig_dim = img.ndim
    if orig_dim == 3:
        img_4d = img.unsqueeze(0)
    else:
        img_4d = img

    B, C, H, W = img_4d.shape
    pad = kernel_size // 2
    # Reflect pad to avoid border artifacts on tissue boundaries
    padded = F.pad(img_4d, (pad, pad, pad, pad), mode="reflect")
    patches = padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    # patches shape: [B, C, H, W, kernel_size, kernel_size]
    flattened = patches.contiguous().view(B, C, H, W, kernel_size * kernel_size)
    filtered = flattened.median(dim=-1).values

    if orig_dim == 3:
        return filtered.squeeze(0)
    return filtered


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
        p_gaussian_blur: float = 0.2,
        p_median_blur: float = 0.15,
    ):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_rot90 = p_rot90
        self.color_jitter = color_jitter
        self.p_gaussian_blur = p_gaussian_blur
        self.p_median_blur = p_median_blur

        if color_jitter:
            self.jitter = v2.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            )
        else:
            self.jitter = None

        self.gaussian_blur = v2.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 1.0),
        )

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

        # 4. Color Jitter (applied to image only)
        if self.jitter is not None and random.random() < 0.3:
            sample["image"] = self.jitter(sample["image"])

        # 5. Gaussian Blur (applied to image only)
        if self.p_gaussian_blur > 0.0 and random.random() < self.p_gaussian_blur:
            sample["image"] = self.gaussian_blur(sample["image"])

        # 6. Median Blur (applied to image only, edge-preserving denoising)
        if self.p_median_blur > 0.0 and random.random() < self.p_median_blur:
            sample["image"] = apply_median_blur(sample["image"], kernel_size=3)

        return sample
