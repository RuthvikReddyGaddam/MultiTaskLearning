# create_datasets/monusac_train.py

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from monusac_utils import (
    load_monusac_image,
    parse_monusac_xml,
    save_subimage_data,
    merge_to_unified_instance_mask,
    compute_instance_sdf,
    compute_inside_distance_and_hv,
    compute_3class_target,
    compute_4class_target_with_bottom_hat,
)
import numpy as np


def process_subimage(
    image_path: Path,
    xml_path: Path,
    output_subimage_dir: Path,
    save_sdf: bool = False,
    save_flow: bool = False,
    save_3class: bool = False,
    save_4class: bool = False,
) -> None:
    image = load_monusac_image(image_path)
    height, width = image.shape[:2]
    mask_4ch = parse_monusac_xml(xml_path, height, width)

    unified = merge_to_unified_instance_mask(mask_4ch) if (save_sdf or save_flow or save_3class or save_4class) else None
    sdf = compute_instance_sdf(unified) if (save_sdf and unified is not None) else None
    hv_map = None
    if save_flow and unified is not None:
        _, gx, gy = compute_inside_distance_and_hv(unified, method="sobel")
        hv_map = np.stack([gx, gy], axis=0)
    t3 = compute_3class_target(unified) if (save_3class and unified is not None) else None
    t4 = compute_4class_target_with_bottom_hat(unified) if (save_4class and unified is not None) else None

    save_subimage_data(
        output_subimage_dir,
        image=image,
        mask=mask_4ch,
        sdf=sdf,
        hv_map=hv_map,
        target_3class=t3,
        target_4class=t4,
    )


def create_monusac_dataset(
    raw_dir: Path,
    output_dir: Path,
    save_sdf: bool = False,
    save_flow: bool = False,
    save_3class: bool = False,
    save_4class: bool = False,
) -> None:
    patient_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    print(f"Found {len(patient_dirs)} patient directories in {raw_dir}")

    total_subimages = 0
    for p_idx, patient_dir in enumerate(patient_dirs, start=1):
        patient_name = patient_dir.name
        xml_files = sorted(patient_dir.glob("*.xml"))

        for xml_path in xml_files:
            sub_name = xml_path.stem
            tif_path = patient_dir / f"{sub_name}.tif"
            img_path = tif_path if tif_path.exists() else (patient_dir / f"{sub_name}.svs")

            out_sub_dir = output_dir / patient_name / sub_name
            process_subimage(
                img_path,
                xml_path,
                out_sub_dir,
                save_sdf=save_sdf,
                save_flow=save_flow,
                save_3class=save_3class,
                save_4class=save_4class,
            )
            total_subimages += 1

        print(f"[{p_idx}/{len(patient_dirs)}] Processed patient {patient_name} ({len(xml_files)} sub-images)")

    print(f"Successfully created MoNuSAC train dataset at {output_dir} ({total_subimages} total sub-images)")


def main():
    parser = argparse.ArgumentParser(description="Create MoNuSAC train dataset with per-subimage 4-channel instance masks.")
    default_raw = (
        PROJECT_ROOT / "data" / "MoNuSAC_images_and_annotations"
        if (PROJECT_ROOT / "data" / "MoNuSAC_images_and_annotations").is_dir()
        else PROJECT_ROOT / "data" / "monusac_images_and_annotations"
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=default_raw,
        help="Path to raw MoNuSAC images and annotations directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "monusac_train",
        help="Path to output dataset folder.",
    )
    parser.add_argument("--save_sdf", action="store_true", help="Precompute and save full-size SDF.")
    parser.add_argument("--save_flow", action="store_true", help="Precompute and save full-size HV flow map.")
    parser.add_argument("--save_3class", action="store_true", help="Precompute and save 3-class J-loss target.")
    parser.add_argument("--save_4class", action="store_true", help="Precompute and save 4-class J-loss target.")
    args = parser.parse_args()

    create_monusac_dataset(
        args.raw_dir,
        args.output_dir,
        save_sdf=args.save_sdf,
        save_flow=args.save_flow,
        save_3class=args.save_3class,
        save_4class=args.save_4class,
    )


if __name__ == "__main__":
    main()
