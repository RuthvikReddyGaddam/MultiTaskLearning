"""
Pure Vector-Field Instance Segmentation Post-Processing (Watershed-Free).

Provides two robust vector-field post-processing methods:
1. Method A: Cellpose-Style Pixel Marching (Euler streamline integration toward flow attractors/sinks).
2. Method B: Directional Opposition Cut & Refinement (detecting opposing vectors at contact interfaces,
   cutting boundaries, and refining via flow-directed alignment).
"""

from typing import Optional, Tuple
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def _fallback_connected_components(binary_mask: np.ndarray, min_size: int = 10) -> np.ndarray:
    """Standard connected components fallback."""
    bin_u8 = (np.asarray(binary_mask) > 0).astype(np.uint8)
    if not bin_u8.any():
        return np.zeros_like(bin_u8, dtype=np.int32)
    num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(bin_u8, connectivity=8)
    out = np.zeros_like(labels, dtype=np.int32)
    new_id = 1
    for i in range(1, num_cc):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            out[labels == i] = new_id
            new_id += 1
    return out


def flow_marching_instances(
    binary_mask: np.ndarray,
    flow_vector: Optional[np.ndarray],
    n_iter: int = 100,
    step: float = 1.0,
    min_size: int = 10,
    peak_min_dist: int = 7,
) -> np.ndarray:
    """
    Cellpose-style Euler flow integration toward instance attractors/sinks.

    Args:
        binary_mask: 2D binary array [H, W].
        flow_vector: 3D flow array [2, H, W] where flow[0] is Fx and flow[1] is Fy.
        n_iter: Maximum number of Euler integration steps.
        step: Step size along normalized flow direction.
        min_size: Minimum instance area threshold in pixels.
        peak_min_dist: Minimum distance between adjacent attractor peaks.

    Returns:
        instance_map: 2D array [H, W] with integer instance IDs (1..K, 0=bg).
    """
    bin_u8 = (np.asarray(binary_mask) > 0).astype(np.uint8)
    if not bin_u8.any():
        return np.zeros_like(bin_u8, dtype=np.int32)

    if flow_vector is None or flow_vector.ndim != 3 or flow_vector.shape[0] != 2:
        return _fallback_connected_components(bin_u8, min_size=min_size)

    H, W = bin_u8.shape
    fx = flow_vector[0].astype(np.float32)
    fy = flow_vector[1].astype(np.float32)

    # Zero out flow outside binary mask
    fx[bin_u8 == 0] = 0.0
    fy[bin_u8 == 0] = 0.0

    mag = np.sqrt(fx**2 + fy**2)
    if float(mag.max()) < 1e-4:
        return _fallback_connected_components(bin_u8, min_size=min_size)

    # Normalize vectors inside foreground
    nonzero = mag > 1e-4
    fx_norm = np.zeros_like(fx)
    fy_norm = np.zeros_like(fy)
    fx_norm[nonzero] = fx[nonzero] / mag[nonzero]
    fy_norm[nonzero] = fy[nonzero] / mag[nonzero]

    # Foreground pixel coordinates
    fg_y, fg_x = np.where(bin_u8 > 0)
    N = len(fg_y)
    if N == 0:
        return np.zeros_like(bin_u8, dtype=np.int32)

    # Trajectory integration (Euler method)
    p_y = fg_y.astype(np.float32).copy()
    p_x = fg_x.astype(np.float32).copy()

    for _ in range(n_iter):
        iy = np.clip(np.round(p_y).astype(np.int32), 0, H - 1)
        ix = np.clip(np.round(p_x).astype(np.int32), 0, W - 1)

        vy = fy_norm[iy, ix]
        vx = fx_norm[iy, ix]

        disp = np.sqrt(vy**2 + vx**2)
        if np.mean(disp) < 0.02:
            break

        p_y = np.clip(p_y + step * vy, 0, H - 1)
        p_x = np.clip(p_x + step * vx, 0, W - 1)

    final_y = np.clip(np.round(p_y).astype(np.int32), 0, H - 1)
    final_x = np.clip(np.round(p_x).astype(np.int32), 0, W - 1)

    # 2D convergence density map
    hist = np.zeros((H, W), dtype=np.float32)
    np.add.at(hist, (final_y, final_x), 1.0)
    hist[bin_u8 == 0] = 0.0

    # Smooth density map to fuse nearby terminal trajectories
    hist_smooth = cv2.GaussianBlur(hist, (0, 0), sigmaX=1.2, sigmaY=1.2)

    # Peak detection using local maximum filter
    ksize = max(3, peak_min_dist | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    dilated = cv2.dilate(hist_smooth, kernel)
    peaks = (hist_smooth == dilated) & (hist_smooth > 2.0) & (bin_u8 > 0)

    num_peaks, peak_labels, peak_stats, peak_centroids = cv2.connectedComponentsWithStats(
        peaks.astype(np.uint8), connectivity=8
    )

    if num_peaks <= 1:
        # If no distinct peaks, fallback to connected components
        return _fallback_connected_components(bin_u8, min_size=min_size)

    # Extract peak coordinates (y, x)
    peak_coords = []
    for i in range(1, num_peaks):
        cy, cx = peak_centroids[i][1], peak_centroids[i][0]
        peak_coords.append((cy, cx))
    peak_coords = np.array(peak_coords, dtype=np.float32)  # [K, 2]

    # Assign each foreground pixel to the peak closest to its terminal coordinate
    # terminal coords: [N, 2]
    term_coords = np.stack([p_y, p_x], axis=1)  # [N, 2]

    # Compute pairwise squared distances: [N, K]
    # (term_y - peak_y)^2 + (term_x - peak_x)^2
    diff_y = term_coords[:, 0:1] - peak_coords[:, 0].reshape(1, -1)
    diff_x = term_coords[:, 1:2] - peak_coords[:, 1].reshape(1, -1)
    dists_sq = diff_y**2 + diff_x**2
    assigned_peaks = np.argmin(dists_sq, axis=1) + 1  # 1-indexed

    instance_map = np.zeros((H, W), dtype=np.int32)
    instance_map[fg_y, fg_x] = assigned_peaks

    # Filter small components and renumber sequentially
    final_instance_map = np.zeros((H, W), dtype=np.int32)
    unique_ids = np.unique(instance_map)
    new_id = 1
    for uid in unique_ids:
        if uid == 0:
            continue
        mask_u = (instance_map == uid)
        if int(np.count_nonzero(mask_u)) >= min_size:
            final_instance_map[mask_u] = new_id
            new_id += 1

    if final_instance_map.max() == 0:
        return _fallback_connected_components(bin_u8, min_size=min_size)

    # If any foreground pixels were dropped due to min_size, fill them via nearest neighbor
    missing_fg = (bin_u8 > 0) & (final_instance_map == 0)
    if missing_fg.any() and final_instance_map.max() > 0:
        _, idx = cv2.distanceTransformWithLabels((final_instance_map == 0).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        labels_flat = final_instance_map.flatten()
        final_instance_map[missing_fg] = labels_flat[idx[missing_fg] - 1]

    return final_instance_map


def flow_directional_cut_instances(
    binary_mask: np.ndarray,
    flow_vector: Optional[np.ndarray],
    cos_thresh: float = 0.0,
    min_size: int = 10,
    oppose_margin: float = 0.05,
    refine_step: float = 1.5,
) -> np.ndarray:
    """
    Directional Opposition Cut & Flow Refinement (Watershed-Free).

    Identifies touching cell/gland boundaries by detecting diverging flow vectors
    where neighboring pixels point away from each other into their respective centers.
    Cuts the diverging boundary line to isolate distinct component cores, then refines
    all boundary pixels along their predicted inward flow trajectories and nearest
    core geodesics.

    Args:
        binary_mask: 2D binary array [H, W].
        flow_vector: 3D flow array [2, H, W] where flow[0] is Fx and flow[1] is Fy.
        cos_thresh: Cosine similarity threshold for opposing flow pairs (kept for API compatibility).
        min_size: Minimum instance area threshold in pixels.
        oppose_margin: Minimum outward flow projection margin along neighbor step.
        refine_step: Step distance along normalized flow to project boundary pixels into cores.

    Returns:
        instance_map: 2D array [H, W] with integer instance IDs (1..K, 0=bg).
    """
    bin_u8 = (np.asarray(binary_mask) > 0).astype(np.uint8)
    if not bin_u8.any():
        return np.zeros_like(bin_u8, dtype=np.int32)

    if flow_vector is None or flow_vector.ndim != 3 or flow_vector.shape[0] != 2:
        return _fallback_connected_components(bin_u8, min_size=min_size)

    H, W = bin_u8.shape
    fx = flow_vector[0].astype(np.float32)
    fy = flow_vector[1].astype(np.float32)

    # Zero out flow outside binary mask
    fx[bin_u8 == 0] = 0.0
    fy[bin_u8 == 0] = 0.0

    mag = np.sqrt(fx**2 + fy**2)
    if float(mag.max()) < 1e-4:
        return _fallback_connected_components(bin_u8, min_size=min_size)

    # Normalize vectors inside foreground (only consider pixels with meaningful flow)
    nonzero = (mag > 0.02) & (bin_u8 > 0)
    fx_norm = np.zeros_like(fx)
    fy_norm = np.zeros_like(fy)
    fx_norm[nonzero] = fx[nonzero] / mag[nonzero]
    fy_norm[nonzero] = fy[nonzero] / mag[nonzero]

    cut_mask = np.zeros((H, W), dtype=bool)

    # 1. Horizontal neighbor opposition (left pixel points left, right pixel points right)
    both_x = nonzero[:, :-1] & nonzero[:, 1:]
    oppose_x = (fx_norm[:, :-1] < -oppose_margin) & (fx_norm[:, 1:] > oppose_margin) & both_x
    cut_mask[:, :-1] |= oppose_x
    cut_mask[:, 1:] |= oppose_x

    # 2. Vertical neighbor opposition (top pixel points up, bottom pixel points down)
    both_y = nonzero[:-1, :] & nonzero[1:, :]
    oppose_y = (fy_norm[:-1, :] < -oppose_margin) & (fy_norm[1:, :] > oppose_margin) & both_y
    cut_mask[:-1, :] |= oppose_y
    cut_mask[1:, :] |= oppose_y

    # 3. Diagonal 1 opposition (down-right: top-left points up-left, bottom-right points down-right)
    both_d1 = nonzero[:-1, :-1] & nonzero[1:, 1:]
    v_d1_tl = (fx_norm[:-1, :-1] + fy_norm[:-1, :-1]) / 1.4142
    v_d1_br = (fx_norm[1:, 1:] + fy_norm[1:, 1:]) / 1.4142
    oppose_d1 = (v_d1_tl < -oppose_margin) & (v_d1_br > oppose_margin) & both_d1
    cut_mask[:-1, :-1] |= oppose_d1
    cut_mask[1:, 1:] |= oppose_d1

    # 4. Diagonal 2 opposition (down-left: top-right points up-right, bottom-left points down-left)
    both_d2 = nonzero[:-1, 1:] & nonzero[1:, :-1]
    v_d2_tr = (-fx_norm[:-1, 1:] + fy_norm[:-1, 1:]) / 1.4142
    v_d2_bl = (-fx_norm[1:, :-1] + fy_norm[1:, :-1]) / 1.4142
    oppose_d2 = (v_d2_tr < -oppose_margin) & (v_d2_bl > oppose_margin) & both_d2
    cut_mask[:-1, 1:] |= oppose_d2
    cut_mask[1:, :-1] |= oppose_d2

    cut_mask &= (bin_u8 > 0)

    # 5. Split binary mask into disconnected cores
    split_mask = (bin_u8 > 0) & (~cut_mask)

    num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(split_mask.astype(np.uint8), connectivity=8)

    if num_cc <= 1:
        # If cuts didn't split, fallback
        return _fallback_connected_components(bin_u8, min_size=min_size)

    # Assign initial core labels (filtering noise smaller than min_size // 2)
    core_min_size = max(3, min_size // 2)
    core_map = np.zeros((H, W), dtype=np.int32)
    new_id = 1
    for i in range(1, num_cc):
        if stats[i, cv2.CC_STAT_AREA] >= core_min_size:
            core_map[labels == i] = new_id
            new_id += 1

    if core_map.max() == 0:
        return _fallback_connected_components(bin_u8, min_size=min_size)

    # 6. Flow-directed boundary refinement
    unassigned_y, unassigned_x = np.where((bin_u8 > 0) & (core_map == 0))
    if len(unassigned_y) > 0:
        proj_y = np.clip(np.round(unassigned_y + refine_step * fy_norm[unassigned_y, unassigned_x]).astype(np.int32), 0, H - 1)
        proj_x = np.clip(np.round(unassigned_x + refine_step * fx_norm[unassigned_y, unassigned_x]).astype(np.int32), 0, W - 1)

        proj_labels = core_map[proj_y, proj_x]
        valid_proj = proj_labels > 0
        core_map[unassigned_y[valid_proj], unassigned_x[valid_proj]] = proj_labels[valid_proj]

    # 7. Re-assign any remaining boundary pixels via exact nearest labeled core
    remaining = (bin_u8 > 0) & (core_map == 0)
    if remaining.any() and core_map.max() > 0:
        dist, indices = distance_transform_edt(core_map == 0, return_indices=True)
        nearest_labels = core_map[tuple(indices)]
        core_map[remaining] = nearest_labels[remaining]

    # 8. Final filtering by min_size and sequential renumbering
    final_instance_map = np.zeros((H, W), dtype=np.int32)
    unique_ids = np.unique(core_map)
    new_id = 1
    for uid in unique_ids:
        if uid == 0:
            continue
        mask_u = (core_map == uid)
        if int(np.count_nonzero(mask_u)) >= min_size:
            final_instance_map[mask_u] = new_id
            new_id += 1

    if final_instance_map.max() == 0:
        return _fallback_connected_components(bin_u8, min_size=min_size)

    # Restore any foreground pixels dropped by min_size to nearest valid instance
    missing_fg = (bin_u8 > 0) & (final_instance_map == 0)
    if missing_fg.any() and final_instance_map.max() > 0:
        dist, indices = distance_transform_edt(final_instance_map == 0, return_indices=True)
        nearest_labels = final_instance_map[tuple(indices)]
        final_instance_map[missing_fg] = nearest_labels[missing_fg]

    return final_instance_map

