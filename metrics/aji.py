# metrics/aji.py
"""
Aggregated Jaccard Index (AJI) and AJI Plus (AJI+) computation for nuclear segmentation.

Metric Definitions:
------------------
1. Aggregated Jaccard Index (AJI):
   Proposed by Kumar et al. (IEEE TMI 2017).
   For each ground-truth nucleus G_i, find the predicted nucleus P_j that maximizes IoU(G_i, P_j).
   If IoU(G_i, P_j*) > 0, P_j* is considered the matched prediction.
   - Aggregate Intersection: Sum of |G_i ∩ P_j*| over all matched pairs.
   - Aggregate Union: Sum of |G_i ∪ P_j*| over matched pairs + unmatched GT nuclei + unmatched Pred nuclei.
   AJI = Intersection / Union

2. AJI Plus (AJI+):
   Enforces a strict 1-to-1 matching constraint: each predicted nucleus can only be matched to
   at most ONE ground truth nucleus (selected by maximal IoU). Any subsequent GT nucleus attempting
   to claim an already-matched predicted nucleus is treated as unmatched, preventing over-counting.
"""

from typing import Union, Tuple
import numpy as np


def remap_label(mask: np.ndarray) -> np.ndarray:
    """
    Remap arbitrary instance IDs in an integer mask to contiguous integers 1..N.
    0 is preserved as background.

    Args:
        mask: 2D integer numpy array.

    Returns:
        2D int32 numpy array with contiguous instance labels 1..N.
    """
    mask = np.asarray(mask, dtype=np.int32)
    out = np.zeros_like(mask, dtype=np.int32)
    unique_ids = np.unique(mask[mask > 0])
    for new_id, old_id in enumerate(unique_ids, start=1):
        out[mask == old_id] = new_id
    return out


def compute_aji(
    gt_instance: np.ndarray,
    pred_instance: np.ndarray,
) -> float:
    """
    Compute Aggregated Jaccard Index (AJI).

    Formula:
      AJI = sum_i |G_i ∩ P_j*(i)| / (sum_i |G_i ∪ P_j*(i)| + sum_{k unmatched} |P_k|)

    Args:
        gt_instance: [H, W] ground truth instance mask (0=bg, 1..N=instances).
        pred_instance: [H, W] predicted instance mask (0=bg, 1..M=instances).

    Returns:
        float: AJI score in [0.0, 1.0].
    """
    gt = remap_label(gt_instance)
    pred = remap_label(pred_instance)

    gt_ids = np.unique(gt[gt > 0])
    pred_ids = np.unique(pred[pred > 0])

    num_gt = len(gt_ids)
    num_pred = len(pred_ids)

    if num_gt == 0 and num_pred == 0:
        return 1.0
    if num_gt == 0 or num_pred == 0:
        return 0.0

    # Precompute areas
    pred_areas = {p_id: int(np.count_nonzero(pred == p_id)) for p_id in pred_ids}

    overall_intersection = 0
    overall_union = 0
    used_pred_ids = set()

    for g_id in gt_ids:
        g_mask = (gt == g_id)
        g_area = int(np.count_nonzero(g_mask))

        # Only examine overlapping predicted instances
        overlapping_pred = pred[g_mask]
        cand_p_ids = np.unique(overlapping_pred[overlapping_pred > 0])

        best_iou = 0.0
        best_p_id = None
        best_inter = 0
        best_union = g_area

        for p_id in cand_p_ids:
            inter = int(np.count_nonzero(overlapping_pred == p_id))
            union = g_area + pred_areas[p_id] - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_p_id = p_id
                best_inter = inter
                best_union = union

        if best_p_id is not None and best_iou > 0.0:
            overall_intersection += best_inter
            overall_union += best_union
            used_pred_ids.add(best_p_id)
        else:
            overall_union += g_area

    # Add unmatched predicted instances to union
    for p_id in pred_ids:
        if p_id not in used_pred_ids:
            overall_union += pred_areas[p_id]

    if overall_union == 0:
        return 1.0
    return float(overall_intersection / overall_union)


def compute_aji_plus(
    gt_instance: np.ndarray,
    pred_instance: np.ndarray,
) -> float:
    """
    Compute Aggregated Jaccard Index Plus (AJI+).
    Enforces strict 1-to-1 matching: each predicted instance can be matched to at most one GT instance.

    Args:
        gt_instance: [H, W] ground truth instance mask (0=bg, 1..N=instances).
        pred_instance: [H, W] predicted instance mask (0=bg, 1..M=instances).

    Returns:
        float: AJI+ score in [0.0, 1.0].
    """
    gt = remap_label(gt_instance)
    pred = remap_label(pred_instance)

    gt_ids = np.unique(gt[gt > 0])
    pred_ids = np.unique(pred[pred > 0])

    num_gt = len(gt_ids)
    num_pred = len(pred_ids)

    if num_gt == 0 and num_pred == 0:
        return 1.0
    if num_gt == 0 or num_pred == 0:
        return 0.0

    pred_areas = {p_id: int(np.count_nonzero(pred == p_id)) for p_id in pred_ids}
    gt_areas = {g_id: int(np.count_nonzero(gt == g_id)) for g_id in gt_ids}

    # Find candidate pairwise IoU
    candidates = []
    for g_id in gt_ids:
        g_mask = (gt == g_id)
        overlapping_pred = pred[g_mask]
        cand_p_ids = np.unique(overlapping_pred[overlapping_pred > 0])

        for p_id in cand_p_ids:
            inter = int(np.count_nonzero(overlapping_pred == p_id))
            union = gt_areas[g_id] + pred_areas[p_id] - inter
            iou = inter / union if union > 0 else 0.0
            if iou > 0.0:
                candidates.append((iou, inter, union, g_id, p_id))

    # Sort descending by IoU for greedy 1-to-1 matching
    candidates.sort(key=lambda x: x[0], reverse=True)

    matched_gt = set()
    matched_pred = set()
    overall_intersection = 0
    overall_union = 0

    for iou, inter, union, g_id, p_id in candidates:
        if g_id not in matched_gt and p_id not in matched_pred:
            matched_gt.add(g_id)
            matched_pred.add(p_id)
            overall_intersection += inter
            overall_union += union

    # Add unmatched GT
    for g_id in gt_ids:
        if g_id not in matched_gt:
            overall_union += gt_areas[g_id]

    # Add unmatched Pred
    for p_id in pred_ids:
        if p_id not in matched_pred:
            overall_union += pred_areas[p_id]

    if overall_union == 0:
        return 1.0
    return float(overall_intersection / overall_union)


__all__ = [
    "remap_label",
    "compute_aji",
    "compute_aji_plus",
]
