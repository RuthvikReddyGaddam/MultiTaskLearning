# datasets/transforms.py

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode


def get_train_transforms():
    """
    Standard patch-level training augmentations using torchvision v2.
    Synchronously transforms Image and Mask tensors.
    """
    train_transforms = v2.Compose([
        # --------------------------------------------------
        # Basic geometric augmentation
        # --------------------------------------------------
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomChoice([
            v2.RandomRotation((0, 0)),
            v2.RandomRotation((90, 90)),
            v2.RandomRotation((180, 180)),
            v2.RandomRotation((270, 270)),
        ]),

        # --------------------------------------------------
        # Translation, scaling, rotation, and shear
        # --------------------------------------------------
        v2.RandomApply(
            [
                v2.RandomAffine(
                    degrees=10,
                    translate=(0.05, 0.05),
                    scale=(0.90, 1.10),
                    shear=(-5, 5),
                    interpolation=InterpolationMode.BILINEAR,
                    fill={
                        tv_tensors.Image: 255,
                        tv_tensors.Mask: 0,
                    },
                )
            ],
            p=0.30,
        ),

        # --------------------------------------------------
        # Mild tissue deformation
        # --------------------------------------------------
        v2.RandomApply(
            [
                v2.ElasticTransform(
                    alpha=20.0,
                    sigma=5.0,
                    interpolation=InterpolationMode.BILINEAR,
                    fill={
                        tv_tensors.Image: 255,
                        tv_tensors.Mask: 0,
                    },
                )
            ],
            p=0.15,
        ),

        # --------------------------------------------------
        # Image appearance augmentation (applied to Image only)
        # --------------------------------------------------
        v2.RandomApply(
            [
                v2.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.02,
                )
            ],
            p=0.30,
        ),

        v2.RandomAdjustSharpness(
            sharpness_factor=1.5,
            p=0.20,
        ),

        v2.RandomAutocontrast(p=0.10),

        v2.RandomApply(
            [
                v2.GaussianBlur(
                    kernel_size=3,
                    sigma=(0.1, 1.0),
                )
            ],
            p=0.15,
        ),

        # --------------------------------------------------
        # Convert datatype
        # --------------------------------------------------
        v2.ToDtype(
            dtype={
                tv_tensors.Image: torch.float32,
                tv_tensors.Mask: torch.float32,
                "others": None,
            },
            scale=True,
        ),
    ])

    return train_transforms


def get_test_transforms():
    """
    Validation/test transforms without random augmentations.
    """
    validation_transforms = v2.Compose([
        v2.ToDtype(
            dtype={
                tv_tensors.Image: torch.float32,
                tv_tensors.Mask: torch.float32,
                "others": None,
            },
            scale=True,
        ),
    ])

    return validation_transforms