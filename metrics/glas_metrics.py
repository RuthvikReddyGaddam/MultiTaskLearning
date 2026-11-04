# metrics/glas_metrics.py
"""
Evaluation metrics for GlaS (Gland Segmentation) dataset.

Metric Explanations:
-------------------
1. Binary Dice Coefficient (F1-score):
   Measures spatial overlap agreement between binarized predicted gland masks and ground truth.
   Formula:
       Dice = (2 * |P ∩ G|) / (|P| + |G|)
   Where P is the binarized prediction (probability >= threshold) and G is the ground truth mask.
   Range: [0.0, 1.0], where 1.0 indicates perfect overlap. If both P and G are empty, Dice = 1.0.

2. Intersection over Union (IoU / Jaccard Index):
   Measures intersection area divided by union area of predicted and ground truth glands.
   Formula:
       IoU = |P ∩ G| / |P ∪ G| = |P ∩ G| / (|P| + |G| - |P ∩ G|)
   Range: [0.0, 1.0], where 1.0 indicates perfect overlap. If both P and G are empty, IoU = 1.0.

Reporting Breakdown:
--------------------
- Per-Image Metrics: Dice and IoU evaluated on full-size reconstructed images (Warwick QU test set).
- Subgroup Metrics:
    - testA: 60 benign and malignant gland images with normal/standard morphology.
    - testB: 20 high-grade / poorly differentiated gland images with challenging complex morphology.
- Overall Metrics: Macro-average across all evaluation images (testA + testB).
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd


def print_glas_metric_explanations() -> None:
    """Print explicit explanations of GlaS evaluation metrics before reporting."""
    print("=" * 80)
    print("GLAS EVALUATION METRIC FORMULATIONS & CALCULATION DETAILS:")
    print("=" * 80)
    print("1. Binary Dice Coefficient:")
    print("   Formula: Dice = (2 * |P ∩ G| + eps) / (|P| + |G| + eps)")
    print("   Calculated on whole slide reconstructed predictions thresholded at 0.5 against ground truth.")
    print("2. Intersection over Union (IoU / Jaccard Index):")
    print("   Formula: IoU  = (|P ∩ G| + eps) / (|P ∪ G| + eps)")
    print("3. Split Reporting:")
    print("   - testA: Standard & benign gland structures (60 images)")
    print("   - testB: High-grade cancerous / poorly differentiated glands (20 images)")
    print("   - Overall: Aggregate mean and standard deviation across all 80 images.")
    print("=" * 80 + "\n")


def compute_glas_metrics(
    full_pred: np.ndarray,
    full_gt: np.ndarray,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute binary Dice and IoU for a full-size reconstructed GlaS image.

    Args:
        full_pred: [H, W] binary predicted mask (0 or 1).
        full_gt: [H, W] binary ground truth mask (0 or 1).
        smooth: Numerical stability epsilon.

    Returns:
        Dict containing:
            "Dice": float in [0.0, 1.0]
            "IoU":  float in [0.0, 1.0]
    """
    p = (np.asarray(full_pred) > 0).astype(np.float32)
    g = (np.asarray(full_gt) > 0).astype(np.float32)

    intersection = float(np.sum(p * g))
    pred_area = float(np.sum(p))
    gt_area = float(np.sum(g))

    cardinality = pred_area + gt_area
    union = pred_area + gt_area - intersection

    if cardinality == 0:
        dice = 1.0
        iou = 1.0
    else:
        dice = (2.0 * intersection + smooth) / (cardinality + smooth)
        iou = (intersection + smooth) / (union + smooth) if union > 0 else 1.0

    return {
        "Dice": float(dice),
        "IoU": float(iou),
    }


def aggregate_glas_summary(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Aggregate per-image GlaS evaluation metrics across testA, testB, and overall.

    Args:
        results: List of dicts, each containing:
            "image": str, "group": "testA"|"testB", "Dice": float, "IoU": float

    Returns:
        pd.DataFrame with columns:
            ["Split", "Count", "Dice_Mean", "Dice_Std", "IoU_Mean", "IoU_Std"]
    """
    if not results:
        return pd.DataFrame(columns=["Split", "Count", "Dice_Mean", "Dice_Std", "IoU_Mean", "IoU_Std"])

    df = pd.DataFrame(results)
    summary_rows = []

    for grp in ["testA", "testB"]:
        sub = df[df["group"].str.lower() == grp.lower()] if "group" in df.columns else pd.DataFrame()
        if len(sub) > 0:
            summary_rows.append({
                "Split": grp,
                "Count": len(sub),
                "Dice_Mean": float(sub["Dice"].mean()),
                "Dice_Std": float(sub["Dice"].std()) if len(sub) > 1 else 0.0,
                "IoU_Mean": float(sub["IoU"].mean()),
                "IoU_Std": float(sub["IoU"].std()) if len(sub) > 1 else 0.0,
            })

    # Overall summary
    summary_rows.append({
        "Split": "Overall",
        "Count": len(df),
        "Dice_Mean": float(df["Dice"].mean()),
        "Dice_Std": float(df["Dice"].std()) if len(df) > 1 else 0.0,
        "IoU_Mean": float(df["IoU"].mean()),
        "IoU_Std": float(df["IoU"].std()) if len(df) > 1 else 0.0,
    })

    return pd.DataFrame(summary_rows)


__all__ = [
    "compute_glas_metrics",
    "aggregate_glas_summary",
    "print_glas_metric_explanations",
]
