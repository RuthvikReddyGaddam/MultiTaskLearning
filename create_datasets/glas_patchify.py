# create_datasets/glas_patchify.py

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from typing import Optional
import numpy as np
from glas_utils import (
    get_patch_starts,
    compute_instance_sdf,
    extract_inside_distance,
    compute_inside_distance_map,
    compute_centroid_distance_map,
    compute_inside_distance_and_hv,
    compute_hv_gradients,
    compute_3class_target,
    compute_4class_target_with_bottom_hat,
    save_patch_heptaplet,
    save_patch_pentaplet,
    save_patch_triplet,
    save_patch_quadruplet,
    save_full_image_preview,
)


def patchify_glas_dataset(
    data_dir: Path,
    output_dir: Path,
    patch_size: int = 256,
    touching_radius: int = 4,
    gap_kernel_size: int = 15,
    save_hv: bool = True,
    save_3class: bool = True,
    save_4class: bool = True,
    normalize_hv: bool = True,
    save_previews: bool = True,
    preview_dir: Optional[Path] = None,
) -> None:
    dirs = {
        "images": output_dir / "images",
        "binary_masks": output_dir / "binary_masks",
        "instance_masks": output_dir / "instance_masks",
        "sdfs": output_dir / "sdfs",
    }
    if save_3class:
        dirs["targets_3class"] = output_dir / "targets_3class"
    if save_4class:
        dirs["targets_4class"] = output_dir / "targets_4class"
    if save_hv:
        dirs["hv_maps"] = output_dir / "hv_maps"

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    if save_previews:
        if preview_dir is None:
            preview_dir = PROJECT_ROOT / "data" / "glas_full_previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        print(f"Full-size PNG previews will be saved to: {preview_dir}")

    image_files = sorted([
        f for f in data_dir.glob("train_*.npy")
        if "_anno" not in f.stem and "_binary" not in f.stem
    ])
    print(f"Found {len(image_files)} full images in {data_dir}")
    print(f"Patch size: {patch_size}x{patch_size}")

    total_patches = 0
    for idx, img_path in enumerate(image_files, start=1):
        base_name = img_path.stem
        mask_path = data_dir / f"{base_name}_anno.npy"

        # 1. Load full image and instance mask
        image = np.load(img_path)
        instance_mask = np.load(mask_path)
        h, w = image.shape[:2]

        # Ensure image is at least patch_size in both dimensions (pad with white background and 0 mask)
        if h < patch_size or w < patch_size:
            pad_h = max(0, patch_size - h)
            pad_w = max(0, patch_size - w)
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=255)
            instance_mask = np.pad(instance_mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
            h, w = image.shape[:2]

        # 2. Compute full-size SDF, 3-class target, 4-class target, and HV flow
        sdf = compute_instance_sdf(instance_mask)
        target_3class = compute_3class_target(instance_mask) if save_3class else None
        target_4class = compute_4class_target_with_bottom_hat(
            instance_mask,
            kernel_size=gap_kernel_size,
            touching_radius=touching_radius,
        ) if save_4class else None

        # Optional HV flow map calculation
        hv_map = None
        dist_map = None
        gx = None
        gy = None
        if save_hv:
            dist_map, gx, gy = compute_inside_distance_and_hv(instance_mask, method="sobel")
            hv_map = np.stack([gx, gy], axis=0)  # [2, H, W]

        binary_mask = (instance_mask > 0).astype(np.uint8)

        # 3. Plot full-size images, masks, and maps before patching
        if save_previews:
            preview_file = preview_dir / f"{base_name}_full_preview.png"
            save_full_image_preview(
                preview_file,
                image=image,
                instance_mask=instance_mask,
                sdf=sdf,
                dist_map=dist_map,
                gx=gx,
                gy=gy,
                target_3class=target_3class,
                target_4class=target_4class,
                binary_mask=binary_mask,
                title=f"{base_name} ({h}x{w})",
            )

        # 4. Extract patches: images, binary masks, instance masks, sdfs, 3-class, 4-class, flows
        y_starts = get_patch_starts(h, patch_size)
        x_starts = get_patch_starts(w, patch_size)

        patch_idx = 0
        for y in y_starts:
            for x in x_starts:
                img_patch = image[y:y + patch_size, x:x + patch_size]
                mask_patch = instance_mask[y:y + patch_size, x:x + patch_size]
                bin_patch = (mask_patch > 0).astype(np.float32)
                sdf_patch = sdf[y:y + patch_size, x:x + patch_size]
                t3_patch = target_3class[y:y + patch_size, x:x + patch_size] if target_3class is not None else None
                t4_patch = target_4class[y:y + patch_size, x:x + patch_size] if target_4class is not None else None
                hv_patch = hv_map[:, y:y + patch_size, x:x + patch_size] if hv_map is not None else None

                patch_name = f"{base_name}_patch_{patch_idx:04d}"
                save_patch_heptaplet(
                    dirs=dirs,
                    patch_name=patch_name,
                    image_patch=img_patch,
                    binary_mask_patch=bin_patch,
                    instance_mask_patch=mask_patch,
                    sdf_patch=sdf_patch,
                    target_3class_patch=t3_patch,
                    target_4class_patch=t4_patch,
                    hv_patch=hv_patch,
                )

                patch_idx += 1
                total_patches += 1

        print(f"[{idx}/{len(image_files)}] {base_name}: extracted {patch_idx} patches (starts Y={len(y_starts)}, X={len(x_starts)})")

    print(f"Successfully patchified GlaS dataset into {output_dir} ({total_patches} total patches)")


def main():
    parser = argparse.ArgumentParser(description="Patchify GlaS images, instance masks, and compute full-image SDFs and HV maps.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "glas_train",
        help="Path to folder containing full GlaS train .npy images and masks.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Dynamic patch size (default: 256).",
    )
    parser.add_argument(
        "--touching_radius",
        type=int,
        default=4,
        help="Pixel radius to expand touching gland boundary interface (default: 4).",
    )
    parser.add_argument(
        "--gap_kernel_size",
        type=int,
        default=15,
        help="Diameter of Bottom-Hat morphological kernel in pixels to capture wider inter-gland stromal gap clefts (default: 15).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Path to output patch directory. Defaults to data/glas_train_patch_<patch_size>.",
    )
    parser.add_argument(
        "--save_hv",
        action="store_true",
        default=True,
        help="Derive and save horizontal and vertical (HV) maps from full-size inside cell distance transform.",
    )
    parser.add_argument(
        "--save_3class",
        action="store_true",
        default=True,
        help="Compute and save 3-class J-loss targets (0: BG, 1: Interior, 2: Boundary).",
    )
    parser.add_argument(
        "--save_4class",
        action="store_true",
        default=True,
        help="Compute and save 4-class J-loss targets (0: BG, 1: Interior, 2: Boundary, 3: Gap).",
    )
    parser.add_argument(
        "--normalize_hv",
        action="store_true",
        default=True,
        help="Normalize HV gradients per nucleus to [-1, 1] (default: True).",
    )
    parser.add_argument(
        "--save_previews",
        action="store_true",
        default=True,
        help="Save PNG previews of whole-size images, masks, and HV maps before patchifying (default: True).",
    )
    parser.add_argument(
        "--preview_dir",
        type=Path,
        default=None,
        help="Directory to save whole-size PNG previews (defaults to data/glas_full_previews).",
    )
    args = parser.parse_args()

    out_dir = args.output_dir or (PROJECT_ROOT / "data" / f"glas_train_patch_{args.patch_size}")
    patchify_glas_dataset(
        args.data_dir,
        out_dir,
        args.patch_size,
        touching_radius=args.touching_radius,
        gap_kernel_size=args.gap_kernel_size,
        save_hv=args.save_hv,
        save_3class=args.save_3class,
        save_4class=args.save_4class,
        normalize_hv=args.normalize_hv,
        save_previews=args.save_previews,
        preview_dir=args.preview_dir,
    )


if __name__ == "__main__":
    main()
