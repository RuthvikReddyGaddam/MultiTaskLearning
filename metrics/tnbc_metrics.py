# metrics/tnbc_metrics.py
"""
Evaluation metrics for TNBC (Triple Negative Breast Cancer) dataset.

Provides:
- compute_tnbc_metrics
- aggregate_tnbc_summary
- binary_to_instances
- print_tnbc_metric_explanations
"""

from typing import Dict, Any, List, Optional
import cv2
import numpy as np
import pandas as pd

from metrics.aji import compute_aji, compute_aji_plus, remap_label


def print_tnbc_metric_explanations() -> None:
    """Print explicit explanations of TNBC evaluation metrics before reporting."""
    print("=" * 90)
    print("TNBC ZERO-SHOT EVALUATION METRIC FORMULATIONS & DETAILS:")
    print("=" * 90)
    print("1. Pixel-Level Segmentation:")
    print("   - Binary Dice = 2 * |P_bin ∩ G_bin| / (|P_bin| + |G_bin|)")
    print("   - Binary IoU  = |P_bin ∩ G_bin| / (|P_bin ∪ G_bin|)")
    print("2. Instance-Level Jaccard Indices:")
    print("   - AJI (Aggregated Jaccard Index):")
    print("     AJI = sum_i |G_i ∩ P_j*(i)| / (sum_i |G_i ∪ P_j*(i)| + sum_{k unmatched} |P_k|)")
    print("   - AJI+ (AJI Plus):")
    print("     Strict 1-to-1 bipartite matched AJI preventing multiple GT from claiming the same pred.")
    print("3. Panoptic Metrics (Match IoU threshold = 0.5):")
    print("   - DQ (Detection Quality / F1)  = |TP| / (|TP| + 0.5*|FP| + 0.5*|FN|)")
    print("   - SQ (Segmentation Quality)   = sum_{(i,j) in TP} IoU(G_i, P_j) / |TP|")
    print("   - PQ (Panoptic Quality)       = DQ * SQ")
    print("4. Instance Dice:")
    print("   - Instance Dice = 2 * sum_{(i,j) in TP} |G_i ∩ P_j| / (sum |G_i| + sum |P_j|)")
    print("=" * 90 + "\n")


def binary_to_instances(
    mask: np.ndarray,
    min_size: int = 10,
) -> np.ndarray:
    """
    Convert a binary mask into an integer instance mask using connected components.

    Args:
        mask: 2D binary array [H, W] (0 or >0).
        min_size: Minimum pixel area to keep an instance.

    Returns:
        [H, W] int32 array (0=background, 1..N=instance IDs).
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    instance_map = np.zeros_like(labels, dtype=np.int32)
    new_id = 1

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_size:
            instance_map[labels == label_id] = new_id
            new_id += 1

    return instance_map


def compute_tnbc_metrics(
    pred_binary: np.ndarray,
    gt_binary: np.ndarray,
    pred_instance: np.ndarray,
    gt_instance: np.ndarray,
    pq_match_iou: float = 0.5,
    min_instance_size: int = 10,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute comprehensive pixel-level and instance-level metrics for TNBC.

    Returns:
        Dict containing:
            "Dice": float,
            "IoU": float,
            "AJI": float,
            "AJI_Plus": float,
            "PQ": float,
            "DQ": float,
            "SQ": float,
            "Instance_Dice": float,
    """
    # 1. Pixel-level binary metrics
    p_bin = (np.asarray(pred_binary) > 0).astype(np.float32)
    g_bin = (np.asarray(gt_binary) > 0).astype(np.float32)

    inter = float(np.sum(p_bin * g_bin))
    total_area = float(np.sum(p_bin) + np.sum(g_bin))
    union = total_area - inter

    if total_area == 0:
        dice = 1.0
        iou = 1.0
    else:
        dice = (2.0 * inter + smooth) / (total_area + smooth)
        iou = (inter + smooth) / (union + smooth) if union > 0 else 1.0

    # 2. Instance maps remapping
    gt_inst = remap_label(gt_instance)
    pred_inst = remap_label(pred_instance)

    # 3. AJI and AJI+
    aji = compute_aji(gt_inst, pred_inst)
    aji_plus = compute_aji_plus(gt_inst, pred_inst)

    # 4. Panoptic Quality (PQ, DQ, SQ)
    gt_ids = np.unique(gt_inst[gt_inst > 0])
    pred_ids = np.unique(pred_inst[pred_inst > 0])

    num_gt = len(gt_ids)
    num_pred = len(pred_ids)

    if num_gt == 0 and num_pred == 0:
        pq, dq, sq = 1.0, 1.0, 1.0
        inst_dice = 1.0
    elif num_gt == 0 or num_pred == 0:
        pq, dq, sq = 0.0, 0.0, 0.0
        inst_dice = 0.0
    else:
        pred_areas = {p: int(np.count_nonzero(pred_inst == p)) for p in pred_ids}
        gt_areas = {g: int(np.count_nonzero(gt_inst == g)) for g in gt_ids}

        tp = 0
        sum_iou = 0.0
        matched_inter_sum = 0

        for g in gt_ids:
            g_mask = (gt_inst == g)
            overlapping_pred = pred_inst[g_mask]
            cand_p_ids = np.unique(overlapping_pred[overlapping_pred > 0])

            for p in cand_p_ids:
                intersection = int(np.count_nonzero(overlapping_pred == p))
                u = gt_areas[g] + pred_areas[p] - intersection
                iou_val = intersection / u if u > 0 else 0.0

                if iou_val > pq_match_iou:
                    tp += 1
                    sum_iou += iou_val
                    matched_inter_sum += intersection
                    break

        fp = num_pred - tp
        fn = num_gt - tp

        denom = tp + 0.5 * fp + 0.5 * fn
        dq = float(tp / denom) if denom > 0 else 1.0
        sq = float(sum_iou / tp) if tp > 0 else 0.0
        pq = float(sum_iou / denom) if denom > 0 else 1.0

        total_inst_area = sum(gt_areas.values()) + sum(pred_areas.values())
        inst_dice = float((2.0 * matched_inter_sum) / total_inst_area) if total_inst_area > 0 else 1.0

    return {
        "Dice": float(dice),
        "IoU": float(iou),
        "AJI": float(aji),
        "AJI_Plus": float(aji_plus),
        "PQ": float(pq),
        "DQ": float(dq),
        "SQ": float(sq),
        "Instance_Dice": float(inst_dice),
        # Lowercase aliases for interoperability with monuseg evaluator format
        "dice": float(dice),
        "iou": float(iou),
        "aji": float(aji),
        "aji_plus": float(aji_plus),
        "pq": float(pq),
        "dq": float(dq),
        "sq": float(sq),
        "instance_dice": float(inst_dice),
    }


def aggregate_tnbc_summary(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate per-image TNBC results into summary DataFrame with mean and std."""
    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    metric_cols = ["Dice", "IoU", "AJI", "AJI_Plus", "PQ", "DQ", "SQ", "Instance_Dice"]

    summary_dict = {"count": len(df)}
    for col in metric_cols:
        if col in df.columns:
            summary_dict[f"{col}_Mean"] = float(df[col].mean())
            summary_dict[f"{col}_Std"] = float(df[col].std()) if len(df) > 1 else 0.0
        elif col.lower() in df.columns:
            summary_dict[f"{col}_Mean"] = float(df[col.lower()].mean())
            summary_dict[f"{col}_Std"] = float(df[col.lower()].std()) if len(df) > 1 else 0.0

    return pd.DataFrame([summary_dict])


__all__ = [
    "binary_to_instances",
    "compute_tnbc_metrics",
    "aggregate_tnbc_summary",
    "print_tnbc_metric_explanations",
]
