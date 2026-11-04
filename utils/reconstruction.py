# utils/reconstruction.py
"""
Sliding-window patchification, whole-image reconstruction with weighted/count blending,
and visualization artifact saving.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def generate_patch_positions(
    length: int,
    patch_size: int,
    stride: Optional[int] = None,
    overlap: Optional[float] = None,
) -> List[int]:
    """
    Generate starting indices along a 1D dimension.
    Guarantees:
      - First patch always starts at 0.
      - Last patch ends exactly at length (start = max(0, length - patch_size)).
      - No gaps between patches.
    """
    if length <= patch_size:
        return [0]

    max_start = length - patch_size

    if stride is not None and stride > 0:
        starts = list(range(0, max_start + 1, stride))
        if starts[-1] != max_start:
            starts.append(max_start)
        return sorted(list(dict.fromkeys(starts)))

    if overlap is not None and overlap > 0.0:
        if not (0.0 < overlap < 1.0):
            overlap = 0.5
        step = max(1, int(round(patch_size * (1.0 - overlap))))
        starts = list(range(0, max_start + 1, step))
        if starts[-1] != max_start:
            starts.append(max_start)
        return sorted(list(dict.fromkeys(starts)))

    # Minimal-overlap even spacing
    n = int(np.ceil(length / patch_size))
    if n <= 1:
        return [0]
    step = max_start / (n - 1)
    starts = [int(round(i * step)) for i in range(n)]
    return sorted(list(dict.fromkeys(starts)))


def patchify_image(
    image: Union[torch.Tensor, np.ndarray],
    patch_size: int = 256,
    stride: Optional[int] = None,
    overlap: Optional[float] = 0.5,
) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]], Tuple[int, int], Tuple[int, int]]:
    """
    Split full-size image into overlapping or contiguous patches.

    Args:
        image: [C, H, W] or [H, W, C] tensor/array.
        patch_size: Square patch size.
        stride: Explicit step stride (optional).
        overlap: Overlap fraction in [0.0, 1.0) (optional).

    Returns:
        patches: [N, C, patch_size, patch_size] tensor.
        coordinates: List of (y1, y2, x1, x2) integer bounds for each patch.
        padded_shape: (padded_h, padded_w)
        orig_shape: (orig_h, orig_w)
    """
    if isinstance(image, np.ndarray):
        img_t = torch.from_numpy(image)
        if img_t.ndim == 2:
            img_t = img_t.unsqueeze(0)
        elif img_t.ndim == 3 and img_t.shape[-1] in (1, 3):
            img_t = img_t.permute(2, 0, 1)
    else:
        img_t = image.clone()
        if img_t.ndim == 2:
            img_t = img_t.unsqueeze(0)
        elif img_t.ndim == 4:
            img_t = img_t.squeeze(0)

    c, h, w = img_t.shape
    orig_shape = (h, w)

    # Pad if smaller than patch size
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)

    if pad_h > 0 or pad_w > 0:
        img_t = F.pad(img_t, (0, pad_w, 0, pad_h), mode="reflect")

    _, padded_h, padded_w = img_t.shape
    padded_shape = (padded_h, padded_w)

    y_starts = generate_patch_positions(padded_h, patch_size, stride=stride, overlap=overlap)
    x_starts = generate_patch_positions(padded_w, patch_size, stride=stride, overlap=overlap)

    patches = []
    coordinates = []

    for y in y_starts:
        for x in x_starts:
            y1, y2 = y, y + patch_size
            x1, x2 = x, x + patch_size
            patch = img_t[:, y1:y2, x1:x2]
            patches.append(patch)
            coordinates.append((y1, y2, x1, x2))

    patches_tensor = torch.stack(patches, dim=0).float()
    return patches_tensor, coordinates, padded_shape, orig_shape


def reconstruct_image(
    patch_probs: List[np.ndarray],
    coordinates: List[Tuple[int, int, int, int]],
    original_shape: Tuple[int, int],
    padded_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Reconstruct full-size probability map by averaging overlapping predicted patches.

    Args:
        patch_probs: List of 2D numpy arrays [patch_size, patch_size].
        coordinates: List of (y1, y2, x1, x2) for each patch.
        original_shape: (orig_h, orig_w).
        padded_shape: (padded_h, padded_w).

    Returns:
        [orig_h, orig_w] float32 array in [0.0, 1.0].
    """
    padded_h, padded_w = padded_shape
    orig_h, orig_w = original_shape

    canvas = np.zeros((padded_h, padded_w), dtype=np.float32)
    count = np.zeros((padded_h, padded_w), dtype=np.float32)

    for patch, (y1, y2, x1, x2) in zip(patch_probs, coordinates):
        p = np.asarray(patch).squeeze().astype(np.float32)
        canvas[y1:y2, x1:x2] += p
        count[y1:y2, x1:x2] += 1.0

    count = np.maximum(count, 1e-8)
    full_prob = canvas / count
    full_prob = np.nan_to_num(full_prob, nan=0.0, posinf=1.0, neginf=0.0)

    # Crop out padding to return exact original image resolution
    return full_prob[:orig_h, :orig_w]


def save_reconstructed_results(
    image_name: str,
    probability: np.ndarray,
    prediction_binary: np.ndarray,
    ground_truth_binary: np.ndarray,
    save_dir: Union[str, Path],
    prediction_instance: Optional[np.ndarray] = None,
    ground_truth_instance: Optional[np.ndarray] = None,
    original_image: Optional[np.ndarray] = None,
) -> None:
    """
    Save reconstructed outputs (predictions, ground truth, probabilities, instance maps).
    """
    save_path = Path(save_dir)
    pred_dir = save_path / "predictions"
    prob_dir = save_path / "probabilities"
    gt_dir = save_path / "ground_truth"
    for d in (pred_dir, prob_dir, gt_dir):
        d.mkdir(parents=True, exist_ok=True)

    base = image_name if not image_name.endswith(".png") else image_name[:-4]

    # 1. Binary prediction (pred mask)
    pred_img = (prediction_binary.astype(np.uint8) * 255)
    cv2.imwrite(str(pred_dir / f"{base}_pred.png"), pred_img)

    # 2. Probability heatmap
    probability = np.nan_to_num(probability, nan=0.0, posinf=1.0, neginf=0.0)
    prob_normalized = (np.clip(probability, 0.0, 1.0) * 255).astype(np.uint8)
    prob_color = cv2.applyColorMap(prob_normalized, cv2.COLORMAP_JET)
    cv2.imwrite(str(prob_dir / f"{base}_prob.png"), prob_color)

    # 3. Ground truth / mask
    gt_img = (ground_truth_binary.astype(np.uint8) * 255)
    cv2.imwrite(str(gt_dir / f"{base}_gt.png"), gt_img)
    mask_dir = save_path / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_dir / f"{base}_mask.png"), gt_img)

    # 4. Optional instances (instance mask)
    if prediction_instance is not None or ground_truth_instance is not None:
        inst_dir = save_path / "instances"
        inst_dir.mkdir(parents=True, exist_ok=True)

        if prediction_instance is not None:
            inst_pred = prediction_instance.astype(np.int32)
            inst_pred_vis = cv2.applyColorMap(((inst_pred % 255) * 25).astype(np.uint8), cv2.COLORMAP_JET)
            inst_pred_vis[inst_pred == 0] = 0
            cv2.imwrite(str(inst_dir / f"{base}_pred_instance.png"), inst_pred_vis)
            cv2.imwrite(str(inst_dir / f"{base}_instance.png"), inst_pred_vis)

        if ground_truth_instance is not None:
            inst_gt = ground_truth_instance.astype(np.int32)
            inst_gt_vis = cv2.applyColorMap(((inst_gt % 255) * 25).astype(np.uint8), cv2.COLORMAP_JET)
            inst_gt_vis[inst_gt == 0] = 0
            cv2.imwrite(str(inst_dir / f"{base}_gt_instance.png"), inst_gt_vis)

    # 5. Original image and overlays
    if original_image is not None:
        images_dir = save_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        overlay_dir = save_path / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)

        orig = original_image.copy()
        if orig.ndim == 3 and orig.shape[0] in (1, 3):
            orig = np.transpose(orig, (1, 2, 0))
        if orig.max() <= 1.0:
            orig = (orig * 255).astype(np.uint8)
        else:
            orig = orig.astype(np.uint8)

        if orig.shape[-1] == 1:
            orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)

        # Save raw original image
        cv2.imwrite(str(images_dir / f"{base}.png"), cv2.cvtColor(orig, cv2.COLOR_RGB2BGR))

        overlay = orig.copy()
        # Green contour for ground truth, Red contour for prediction
        gt_contours, _ = cv2.findContours(ground_truth_binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pred_contours, _ = cv2.findContours(prediction_binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), 2)
        cv2.drawContours(overlay, pred_contours, -1, (0, 0, 255), 2)
        cv2.imwrite(str(overlay_dir / f"{base}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


__all__ = [
    "generate_patch_positions",
    "patchify_image",
    "reconstruct_image",
    "save_reconstructed_results",
]
