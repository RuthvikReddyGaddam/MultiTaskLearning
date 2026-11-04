# create_datasets/glas_train.py

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from glas_utils import (
    load_bmp_image,
    load_bmp_mask,
    save_glas_sample,
    compute_instance_sdf,
    compute_inside_distance_and_hv,
    compute_3class_target,
    compute_4class_target_with_bottom_hat,
)
import numpy as np


def process_sample(
    image_path: Path,
    mask_path: Path,
    output_dir: Path,
    save_sdf: bool = False,
    save_flow: bool = False,
    save_3class: bool = False,
    save_4class: bool = False,
) -> None:
    base_name = image_path.stem
    image = load_bmp_image(image_path)
    mask = load_bmp_mask(mask_path)

    sdf = compute_instance_sdf(mask) if save_sdf else None
    hv_map = None
    if save_flow:
        _, gx, gy = compute_inside_distance_and_hv(mask, method="sobel")
        hv_map = np.stack([gx, gy], axis=0)
    t3 = compute_3class_target(mask) if save_3class else None
    t4 = compute_4class_target_with_bottom_hat(mask) if save_4class else None

    save_glas_sample(
        output_dir,
        base_name,
        image=image,
        mask=mask,
        sdf=sdf,
        hv_map=hv_map,
        target_3class=t3,
        target_4class=t4,
    )


def create_glas_dataset(
    raw_dir: Path,
    output_dir: Path,
    save_sdf: bool = False,
    save_flow: bool = False,
    save_3class: bool = False,
    save_4class: bool = False,
) -> None:
    data_source = raw_dir / "Warwick_QU_Dataset" if (raw_dir / "Warwick_QU_Dataset").is_dir() else raw_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([
        f for f in data_source.glob("train_*.bmp")
        if "_anno" not in f.stem
    ])
    print(f"Found {len(image_files)} training images in {data_source}")

    for idx, image_path in enumerate(image_files, start=1):
        mask_path = data_source / f"{image_path.stem}_anno.bmp"
        process_sample(
            image_path,
            mask_path,
            output_dir,
            save_sdf=save_sdf,
            save_flow=save_flow,
            save_3class=save_3class,
            save_4class=save_4class,
        )
        print(f"[{idx}/{len(image_files)}] Processed {image_path.stem}")

    print(f"Successfully created GlaS train dataset at {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create GlaS train dataset with .npy files in a single flat directory.")
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "glas",
        help="Path to GlaS folder (containing Warwick_QU_Dataset).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "glas_train",
        help="Path to output dataset folder.",
    )
    parser.add_argument("--save_sdf", action="store_true", help="Precompute and save full-size SDF.")
    parser.add_argument("--save_flow", action="store_true", help="Precompute and save full-size HV flow map.")
    parser.add_argument("--save_3class", action="store_true", help="Precompute and save 3-class J-loss target.")
    parser.add_argument("--save_4class", action="store_true", help="Precompute and save 4-class J-loss target.")
    args = parser.parse_args()

    create_glas_dataset(
        args.raw_dir,
        args.output_dir,
        save_sdf=args.save_sdf,
        save_flow=args.save_flow,
        save_3class=args.save_3class,
        save_4class=args.save_4class,
    )


if __name__ == "__main__":
    main()
