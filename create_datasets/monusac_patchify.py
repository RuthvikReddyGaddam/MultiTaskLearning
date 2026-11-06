# create_datasets/monusac_patchify.py

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import json
import numpy as np
from typing import Optional
from monusac_utils import (
    pad_if_smaller,
    get_patch_starts,
    merge_to_unified_instance_mask,
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


def patchify_monusac_dataset(
    data_dir: Path,
    output_dir: Path,
    patch_size: int = 256,
    save_hv: bool = True,
    save_3class: bool = True,
    save_4class: bool = True,
    normalize_hv: bool = True,
    save_previews: bool = True,
    preview_dir: Optional[Path] = None,
    touching_radius: int = 1,
    gap_kernel_size: int = 5,
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
            preview_dir = PROJECT_ROOT / "data" / "monusac_full_previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        print(f"Full-size PNG previews will be saved to: {preview_dir}")

    patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"Found {len(patient_dirs)} patient directories in {data_dir}")
    print(f"Patch size: {patch_size}x{patch_size}")

    total_subimages = 0
    total_patches = 0
    patch_stats = {}

    for p_idx, p_dir in enumerate(patient_dirs, start=1):
        patient_name = p_dir.name
        sub_dirs = sorted([d for d in p_dir.iterdir() if d.is_dir()])

        for sub_dir in sub_dirs:
            sub_name = sub_dir.name
            img_file = sub_dir / "image.npy"
            mask_file = sub_dir / "mask.npy"

            # 1. Load full image and 4-channel instance mask
            image = np.load(img_file)
            mask_4ch = np.load(mask_file)

            # 2. Pad to at least patch_size if sub-image dimensions are smaller
            image, mask_4ch = pad_if_smaller(image, mask_4ch, patch_size)
            h, w = image.shape[:2]

            # 3. Compute per-instance SDF, 3-class target, 4-class target, and HV flow
            unified_mask = merge_to_unified_instance_mask(mask_4ch)
            sdf = compute_instance_sdf(unified_mask)
            target_3class = compute_3class_target(unified_mask) if save_3class else None
            target_4class = (
                compute_4class_target_with_bottom_hat(
                    unified_mask,
                    kernel_size=gap_kernel_size,
                    touching_radius=touching_radius,
                )
                if save_4class
                else None
            )

            # Optional HV flow map calculation
            hv_map = None
            dist_map = None
            gx = None
            gy = None
            if save_hv:
                dist_map, gx, gy = compute_inside_distance_and_hv(unified_mask, method="sobel")
                hv_map = np.stack([gx, gy], axis=0)  # [2, H, W]

            binary_mask = (unified_mask > 0).astype(np.uint8)

            # Multiclass cell phenotype map (0: BG, 1: Ep, 2: Lym, 3: Neu, 4: Mac)
            cell_cls_map = np.zeros((h, w), dtype=np.int64)
            for ch_idx in range(min(4, mask_4ch.shape[0])):
                cell_cls_map[mask_4ch[ch_idx] > 0] = ch_idx + 1

            patch_base = sub_name if sub_name.startswith(patient_name) else f"{patient_name}_{sub_name}"

            # 4. Plot full-size images, masks, and maps before patching
            if save_previews:
                preview_file = preview_dir / f"{patch_base}_full_preview.png"
                save_full_image_preview(
                    preview_file,
                    image=image,
                    instance_mask=unified_mask,
                    sdf=sdf,
                    dist_map=dist_map,
                    gx=gx,
                    gy=gy,
                    target_3class=target_3class,
                    target_4class=target_4class,
                    binary_mask=binary_mask,
                    cell_class_map=cell_cls_map,
                    title=f"{patch_base} ({h}x{w})",
                )

            # 5. Extract patches: images, binary masks, instance masks, sdfs, 3-class, 4-class, flows
            y_starts = get_patch_starts(h, patch_size)
            x_starts = get_patch_starts(w, patch_size)

            patch_idx = 0
            for y in y_starts:
                for x in x_starts:
                    img_patch = image[y:y + patch_size, x:x + patch_size]
                    mask_patch = mask_4ch[:, y:y + patch_size, x:x + patch_size]
                    bin_patch = (mask_patch > 0).any(axis=0).astype(np.float32)
                    sdf_patch = sdf[y:y + patch_size, x:x + patch_size]
                    t3_patch = target_3class[y:y + patch_size, x:x + patch_size] if target_3class is not None else None
                    t4_patch = target_4class[y:y + patch_size, x:x + patch_size] if target_4class is not None else None
                    hv_patch = hv_map[:, y:y + patch_size, x:x + patch_size] if hv_map is not None else None

                    patch_name = f"{patch_base}_patch_{patch_idx:04d}"
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

                    # Compute class statistics and percentages for this patch
                    total_pixels = int(patch_size * patch_size)
                    class_pixels = [int(np.count_nonzero(mask_patch[i])) for i in range(4)]
                    fg_pixels = int(np.count_nonzero((mask_patch > 0).any(axis=0)))
                    bg_pixels = total_pixels - fg_pixels
                    bg_percent = (bg_pixels / total_pixels) * 100.0
                    cell_percents = [(px / total_pixels) * 100.0 for px in class_pixels]
                    class_percents_5ch = [bg_percent] + cell_percents
                    class_ratios = [float(px / fg_pixels) if fg_pixels > 0 else 0.0 for px in class_pixels]
                    classes_present = [i for i, px in enumerate(class_pixels) if px > 0]

                    patch_stats[patch_name] = {
                        "patch_name": patch_name,
                        "total_pixels": total_pixels,
                        "bg_pixels": bg_pixels,
                        "fg_pixels": fg_pixels,
                        "class_pixels": class_pixels,
                        "class_percents_5ch": class_percents_5ch,
                        "class_percents": cell_percents,
                        "all_class_percents": {
                            "background": bg_percent,
                            "epithelial": cell_percents[0],
                            "lymphocyte": cell_percents[1],
                            "neutrophil": cell_percents[2],
                            "macrophage": cell_percents[3],
                        },
                        "class_ratios": class_ratios,
                        "classes_present": classes_present,
                        "has_macrophage": bool(class_pixels[3] > 0),
                        "has_cells": bool(fg_pixels > 0),
                    }

                    patch_idx += 1
                    total_patches += 1

            total_subimages += 1

        print(f"[{p_idx}/{len(patient_dirs)}] {patient_name}: processed {len(sub_dirs)} sub-images")

    stats_file = output_dir / "patch_class_stats.json"
    with open(stats_file, "w") as f:
        json.dump(patch_stats, f, indent=2)
    print(f"Saved patch class percentages to {stats_file}")
    print(f"Successfully patchified MoNuSAC dataset into {output_dir} ({total_subimages} sub-images, {total_patches} total patches)")


def compute_and_save_stats_for_existing(output_dir: Path) -> None:
    from datasets.utils import compute_or_load_patch_class_stats
    print(f"Computing patch class statistics for {output_dir}...")
    stats = compute_or_load_patch_class_stats(output_dir, force_recompute=True)
    print(f"Computed stats for {len(stats)} patches, saved to {output_dir / 'patch_class_stats.json'}")


def main():
    parser = argparse.ArgumentParser(description="Patchify MoNuSAC images, instance masks, and compute full-image SDFs.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "monusac_train",
        help="Path to folder containing per-patient MoNuSAC train sub-images.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Dynamic patch size (default: 256).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Path to output patch directory. Defaults to data/monusac_train_patch_<patch_size>.",
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
        help="Directory to save whole-size PNG previews (defaults to data/monusac_full_previews).",
    )
    parser.add_argument(
        "--touching_radius",
        type=int,
        default=1,
        help="Radius for expanding touching boundary seams within cell bodies (default: 1).",
    )
    parser.add_argument(
        "--gap_kernel_size",
        type=int,
        default=5,
        help="Structuring element diameter for bottom-hat transform (default: 5).",
    )
    parser.add_argument(
        "--compute_stats_only",
        action="store_true",
        help="Only compute and save patch_class_stats.json for existing patch directory without regenerating patches.",
    )
    args = parser.parse_args()

    out_dir = args.output_dir or (PROJECT_ROOT / "data" / f"monusac_train_patch_{args.patch_size}")

    if args.compute_stats_only:
        compute_and_save_stats_for_existing(out_dir)
    else:
        patchify_monusac_dataset(
            args.data_dir,
            out_dir,
            args.patch_size,
            save_hv=args.save_hv,
            save_3class=args.save_3class,
            save_4class=args.save_4class,
            normalize_hv=args.normalize_hv,
            save_previews=args.save_previews,
            preview_dir=args.preview_dir,
            touching_radius=args.touching_radius,
            gap_kernel_size=args.gap_kernel_size,
        )


if __name__ == "__main__":
    main()
