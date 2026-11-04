# metrics/pq.py

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn


CELL_TYPES = ["Epithelial", "Lymphocyte", "Neutrophil", "Macrophage"]


def Panoptic_quality(
    ground_truth_image: np.ndarray,
    predicted_image: np.ndarray,
    match_iou_threshold: float = 0.5,
) -> float:
    """
    Compute Panoptic Quality (PQ) metric for a single 2D instance mask pair.

    Args:
        ground_truth_image: 2D array [H, W] where 0=background, 1..N=ground truth instance IDs.
        predicted_image: 2D array [H, W] where 0=background, 1..M=predicted instance IDs.
        match_iou_threshold: Minimum IoU required for an instance match (default: 0.5).

    Returns:
        float: Panoptic Quality (PQ = sum_IOU / (TP + 0.5*FP + 0.5*FN)).
               Returns 1.0 if both ground truth and predicted images have 0 instances.
               Returns 0.0 if one has instances and the other has none.
    """
    gt = np.asarray(ground_truth_image)
    pred = np.asarray(predicted_image)

    gt_ids = np.unique(gt[gt > 0])
    pred_ids = np.unique(pred[pred > 0])

    num_gt = len(gt_ids)
    num_pred = len(pred_ids)

    # Corner cases: empty ground truth or prediction
    if num_gt == 0 and num_pred == 0:
        return 1.0
    if num_gt == 0 or num_pred == 0:
        return 0.0

    # Find matched instances
    matched_instances = {}  # gt_id -> (pred_id, iou)

    for i in gt_ids:
        temp_image = (gt == i)
        # Only check predicted instances that actually overlap with this GT instance
        overlapping_pred = pred[temp_image]
        cand_pred_ids = np.unique(overlapping_pred[overlapping_pred > 0])

        for j in cand_pred_ids:
            pred_temp = (pred == j)
            intersection = np.count_nonzero(temp_image & pred_temp)
            union = np.count_nonzero(temp_image | pred_temp)
            if union == 0:
                continue
            iou = intersection / union
            if iou > match_iou_threshold:
                matched_instances[i] = (j, iou)
                break

    TP = 0
    sum_IOU = 0.0
    matched_pred_ids = set()

    for indx in gt_ids:
        if indx in matched_instances:
            pred_id, iou = matched_instances[indx]
            if pred_id not in matched_pred_ids:
                matched_pred_ids.add(pred_id)
                TP += 1
                sum_IOU += iou

    FN = num_gt - TP
    FP = num_pred - len(matched_pred_ids)

    denominator = TP + 0.5 * FP + 0.5 * FN
    if denominator == 0:
        return 1.0

    return float(sum_IOU / denominator)


def compute_class_wise_pq(
    gt_masks: Dict[str, np.ndarray],
    pred_masks: Dict[str, np.ndarray],
    cell_types: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Computes Panoptic Quality (PQ) for each cell type individually.

    Args:
        gt_masks: Dict mapping cell_type -> [H, W] instance mask.
        pred_masks: Dict mapping cell_type -> [H, W] instance mask.
        cell_types: List of cell types to evaluate. Defaults to CELL_TYPES.

    Returns:
        Dict mapping cell_type -> PQ float score.
    """
    if cell_types is None:
        cell_types = CELL_TYPES

    results = {}
    for c in cell_types:
        gt_m = gt_masks.get(c, np.zeros((1, 1), dtype=np.uint16))
        pred_m = pred_masks.get(c, np.zeros((1, 1), dtype=np.uint16))
        results[c] = Panoptic_quality(gt_m, pred_m)

    return results


def evaluate_monusac_dataset_pq(
    gt_mask_root: Union[str, Path] = "data/MoNuSAC_masks",
    pred_mask_root: Union[str, Path] = "data/MoNuSAC_inference_masks",
    cell_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Evaluates Panoptic Quality across all patient sub-images in gt_mask_root and pred_mask_root.
    Both root folders follow the structure:
      <root>/<patient_name>/<sub_image_name>/<Cell_Type>/mask.png

    Aggregates per-class PQ per patient and computes overall dataset average.

    Returns:
        pandas.DataFrame with columns:
          ['Patient', 'SubImage', 'Epithelial_PQ', 'Lymphocyte_PQ', 'Neutrophil_PQ', 'Macrophage_PQ', 'Mean_PQ']
    """
    gt_root = Path(gt_mask_root)
    pred_root = Path(pred_mask_root)
    if cell_types is None:
        cell_types = CELL_TYPES

    if not gt_root.exists():
        raise FileNotFoundError(f"Ground truth mask root not found: {gt_root}")
    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction mask root not found: {pred_root}")

    rows = []

    # Iterate over predicted patient directories
    for patient_dir in sorted(pred_root.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient_name = patient_dir.name
        gt_patient_dir = gt_root / patient_name
        if not gt_patient_dir.exists():
            continue

        for sub_dir in sorted(patient_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            sub_name = sub_dir.name
            gt_sub_dir = gt_patient_dir / sub_name
            if not gt_sub_dir.exists():
                continue

            row = {
                "Patient": patient_name,
                "SubImage": sub_name,
            }

            class_pqs = []
            for c in cell_types:
                # Load GT mask
                gt_c_dir = gt_sub_dir / c
                gt_file = gt_c_dir / "mask.png"
                if not gt_file.exists():
                    gt_file = gt_c_dir / f"{sub_name}_mask.png"

                # Load Pred mask
                pred_c_dir = sub_dir / c
                pred_file = pred_c_dir / "mask.png"
                if not pred_file.exists():
                    pred_file = pred_c_dir / f"{sub_name}_mask.png"

                gt_mask = cv2.imread(str(gt_file), cv2.IMREAD_UNCHANGED) if gt_file.exists() else None
                pred_mask = cv2.imread(str(pred_file), cv2.IMREAD_UNCHANGED) if pred_file.exists() else None

                if gt_mask is None:
                    gt_mask = np.zeros((1, 1), dtype=np.uint16)
                if pred_mask is None:
                    pred_mask = np.zeros(gt_mask.shape[:2], dtype=np.uint16)

                pq = Panoptic_quality(gt_mask, pred_mask)
                row[f"{c}_PQ"] = pq
                class_pqs.append(pq)

            row["Mean_PQ"] = float(np.mean(class_pqs))
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def summarize_patient_pq(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes patient-level average PQ for each cell type.
    """
    if df.empty:
        return df

    numeric_cols = [c for c in df.columns if c.endswith("_PQ") or c == "Mean_PQ"]
    patient_df = df.groupby("Patient")[numeric_cols].mean().reset_index()
    return patient_df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Panoptic Quality (PQ) for MoNuSAC.")
    parser.add_argument(
        "--gt-dir",
        "--gt_dir",
        type=str,
        default="data/MoNuSAC_masks",
        help="Path to ground truth masks root directory structured as MoNuSAC_masks.",
    )
    parser.add_argument(
        "--pred-dir",
        "--pred_dir",
        type=str,
        default="data/MoNuSAC_inference_masks",
        help="Path to predicted masks root directory structured as MoNuSAC_masks.",
    )
    parser.add_argument(
        "--output-csv",
        "--output_csv",
        type=str,
        default=None,
        help="Optional path to save per-image and per-patient PQ metrics CSV.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Evaluating Panoptic Quality (PQ) for MoNuSAC")
    print(f"Ground Truth Masks : {args.gt_dir}")
    print(f"Predicted Masks    : {args.pred_dir}")
    print("=" * 70)

    df_subimages = evaluate_monusac_dataset_pq(
        gt_mask_root=args.gt_dir,
        pred_mask_root=args.pred_dir,
    )

    if df_subimages.empty:
        print("No matching prediction sub-images found in predicted masks directory.")
        return

    print("\n--- Per-SubImage PQ Metrics ---")
    print(df_subimages.to_string(index=False))

    patient_summary = summarize_patient_pq(df_subimages)
    print("\n--- Per-Patient Average PQ Metrics ---")
    print(patient_summary.to_string(index=False))

    overall_mean = df_subimages[[c for c in df_subimages.columns if c.endswith("_PQ") or c == "Mean_PQ"]].mean()
    print("\n--- Dataset Overall PQ Averages ---")
    for k, v in overall_mean.items():
        print(f"  {k:18s}: {v:.4f}")

    if args.output_csv:
        out_p = Path(args.output_csv)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df_subimages.to_csv(out_p, index=False)
        print(f"\nPer-image metrics saved to: {out_p}")


if __name__ == "__main__":
    main() 