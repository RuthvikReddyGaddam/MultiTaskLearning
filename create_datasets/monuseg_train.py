# create_datasets/monuseg_train.py

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from monuseg_utils import (
    load_tiff_image,
    parse_xml_to_instance_mask,
    save_patient_data,
    compute_instance_sdf,
    compute_inside_distance_and_hv,
    compute_3class_target,
    compute_4class_target_with_bottom_hat,
)
import numpy as np


def process_patient(
    image_path: Path,
    xml_path: Path,
    output_dir: Path,
    save_sdf: bool = False,
    save_flow: bool = False,
    save_3class: bool = False,
    save_4class: bool = False,
) -> None:
    patient_name = image_path.stem
    image = load_tiff_image(image_path)
    height, width = image.shape[:2]
    mask = parse_xml_to_instance_mask(xml_path, height, width)

    sdf = compute_instance_sdf(mask) if save_sdf else None
    hv_map = None
    if save_flow:
        _, gx, gy = compute_inside_distance_and_hv(mask, method="sobel")
        hv_map = np.stack([gx, gy], axis=0)
    t3 = compute_3class_target(mask) if save_3class else None
    t4 = compute_4class_target_with_bottom_hat(mask) if save_4class else None

    save_patient_data(
        output_dir / patient_name,
        image=image,
        mask=mask,
        sdf=sdf,
        hv_map=hv_map,
        target_3class=t3,
        target_4class=t4,
    )


def create_monuseg_dataset(
    raw_dir: Path,
    output_dir: Path,
    save_sdf: bool = False,
    save_flow: bool = False,
    save_3class: bool = False,
    save_4class: bool = False,
) -> None:
    image_dir = raw_dir / "Tissue Images"
    xml_dir = raw_dir / "Annotations"

    image_files = sorted(image_dir.glob("*.tif"))
    print(f"Found {len(image_files)} images in {image_dir}")

    for idx, image_path in enumerate(image_files, start=1):
        patient_name = image_path.stem
        xml_path = xml_dir / f"{patient_name}.xml"
        process_patient(
            image_path,
            xml_path,
            output_dir,
            save_sdf=save_sdf,
            save_flow=save_flow,
            save_3class=save_3class,
            save_4class=save_4class,
        )
        print(f"[{idx}/{len(image_files)}] Processed {patient_name}")

    print(f"Successfully created dataset at {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create MoNuSeg train dataset with per-patient .npy files.")
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "MoNuSegTrain",
        help="Path to MoNuSegTrain folder containing 'Tissue Images' and 'Annotations'.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "monuseg_train",
        help="Path to output dataset folder.",
    )
    parser.add_argument("--save_sdf", action="store_true", help="Precompute and save full-size SDF.")
    parser.add_argument("--save_flow", action="store_true", help="Precompute and save full-size HV flow map.")
    parser.add_argument("--save_3class", action="store_true", help="Precompute and save 3-class J-loss target.")
    parser.add_argument("--save_4class", action="store_true", help="Precompute and save 4-class J-loss target.")
    args = parser.parse_args()

    create_monuseg_dataset(
        args.raw_dir,
        args.output_dir,
        save_sdf=args.save_sdf,
        save_flow=args.save_flow,
        save_3class=args.save_3class,
        save_4class=args.save_4class,
    )


if __name__ == "__main__":
    main()
