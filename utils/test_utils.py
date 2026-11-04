# utils/test_utils.py
"""
Common testing utilities for no-dataloader full-image patch inference,
coordinate generation, sliding-window stitching, and metric evaluation
(Dice, IoU, AJI, PQ) across MoNuSeg, GlaS, and MoNuSAC.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

from metrics.aji import compute_aji, remap_label
from metrics.pq import Panoptic_quality
from scripts.build_models import MODEL_BUILDERS
from utils.checkpoint import load_checkpoint
from utils.config import load_config
from utils.device import get_device

Image.MAX_IMAGE_PIXELS = None

MODEL_REGISTRY = dict(MODEL_BUILDERS)
MONUSAC_CELL_NAMES = ["Epithelial", "Lymphocyte", "Neutrophil", "Macrophage"]


def get_patch_starts(
    length: int,
    patch_size: int,
    stride: Optional[int] = None,
    overlap: Optional[float] = None,
) -> List[int]:
    """
    Generate starting indices along a 1D dimension.
    Options:
      1. Explicit stride (>0): Steps by stride, ensuring last patch ends at length.
      2. Explicit overlap (>0.0 and <1.0): Computes step = patch_size * (1 - overlap).
      3. No stride and no overlap (default): Generates evenly spaced patch starting
         coordinates spanning the dimension with minimal uniform overlap.
    """
    if length <= patch_size:
        return [0]

    max_start = length - patch_size

    # 1. Explicit stride
    if stride is not None and stride > 0:
        starts = list(range(0, max_start + 1, int(stride)))
        if starts[-1] != max_start:
            starts.append(max_start)
        return sorted(list(dict.fromkeys(starts)))

    # 2. Explicit overlap fraction (e.g. 0.5)
    if overlap is not None and overlap > 0.0:
        if not (0.0 < overlap < 1.0):
            overlap = 0.5
        step = max(1, int(round(patch_size * (1.0 - overlap))))
        starts = list(range(0, max_start + 1, step))
        if starts[-1] != max_start:
            starts.append(max_start)
        return sorted(list(dict.fromkeys(starts)))

    # 3. No stride or overlap: minimal uniform overlap evenly spaced patches
    n = int(np.ceil(length / patch_size))
    if n <= 1:
        return [0]
    step = (length - patch_size) / (n - 1)
    starts = [int(round(i * step)) for i in range(n)]
    return sorted(list(dict.fromkeys(starts)))


def pad_image(
    image: np.ndarray,
    patch_size: int,
    pad_val: int = 255,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    Pad image to at least patch_size in both dimensions.
    Returns (padded_image, original_shape, padded_shape).
    """
    h, w = image.shape[:2]
    orig_shape = (h, w)
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)

    if pad_h == 0 and pad_w == 0:
        return image, orig_shape, (h, w)

    if image.ndim == 3:
        padded = np.full((h + pad_h, w + pad_w, image.shape[2]), pad_val, dtype=image.dtype)
        padded[:h, :w, :] = image
    else:
        padded = np.full((h + pad_h, w + pad_w), pad_val, dtype=image.dtype)
        padded[:h, :w] = image

    return padded, orig_shape, (h + pad_h, w + pad_w)


def patchify_image(
    image: np.ndarray,
    patch_size: int = 256,
    stride: Optional[int] = None,
    overlap: Optional[float] = None,
) -> Tuple[torch.Tensor, List[Tuple[int, int, int, int]], Tuple[int, int], Tuple[int, int]]:
    """
    Split a full-size image into patches without a dataloader.

    Args:
        image: [H, W, 3] or [H, W] uint8/float numpy array.
        patch_size: Size of square patch.
        stride: Optional pixel stride.
        overlap: Optional overlap ratio.

    Returns:
        patches: [N, 3, patch_size, patch_size] FloatTensor normalized to [0, 1].
        coordinates: List of (y1, y2, x1, x2) for each patch.
        orig_shape: (H, W)
        padded_shape: (pad_H, pad_W)
    """
    # Ensure RGB [H, W, 3]
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    # Pad with white (255) for histology background
    padded_np, orig_shape, padded_shape = pad_image(image, patch_size=patch_size, pad_val=255)
    pad_h, pad_w = padded_shape

    y_starts = get_patch_starts(pad_h, patch_size, stride=stride, overlap=overlap)
    x_starts = get_patch_starts(pad_w, patch_size, stride=stride, overlap=overlap)

    patch_list = []
    coordinates = []

    for y in y_starts:
        for x in x_starts:
            y1, y2 = y, y + patch_size
            x1, x2 = x, x + patch_size
            p = np.array(padded_np[y1:y2, x1:x2], copy=True)

            # Convert to [3, patch_size, patch_size] float in [0, 1]
            if p.dtype == np.uint8:
                p_t = torch.from_numpy(p).permute(2, 0, 1).float() / 255.0
            else:
                p_t = torch.from_numpy(p).permute(2, 0, 1).float()
                if p_t.max() > 1.0:
                    p_t = p_t / 255.0

            patch_list.append(p_t)
            coordinates.append((y1, y2, x1, x2))

    patches_tensor = torch.stack(patch_list, dim=0)
    return patches_tensor, coordinates, orig_shape, padded_shape


def stitch_probability_map(
    patch_probs: List[np.ndarray],
    coordinates: List[Tuple[int, int, int, int]],
    orig_shape: Tuple[int, int],
    padded_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Reconstruct full-size 2D probability map [orig_h, orig_w] from patch probabilities
    using uniform averaging over overlapping regions.
    """
    padded_h, padded_w = padded_shape
    orig_h, orig_w = orig_shape

    canvas = np.zeros((padded_h, padded_w), dtype=np.float32)
    count = np.zeros((padded_h, padded_w), dtype=np.float32)

    for patch, (y1, y2, x1, x2) in zip(patch_probs, coordinates):
        p = np.asarray(patch).squeeze().astype(np.float32)
        canvas[y1:y2, x1:x2] += p
        count[y1:y2, x1:x2] += 1.0

    count = np.maximum(count, 1e-8)
    full_prob = canvas / count
    full_prob = np.nan_to_num(full_prob, nan=0.0, posinf=1.0, neginf=0.0)
    return full_prob[:orig_h, :orig_w]


def stitch_flow_map(
    patch_flows: List[np.ndarray],
    coordinates: List[Tuple[int, int, int, int]],
    orig_shape: Tuple[int, int],
    padded_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Reconstruct full-size 2D flow field [2, orig_h, orig_w] from patch flows
    using uniform averaging over overlapping regions.
    """
    padded_h, padded_w = padded_shape
    orig_h, orig_w = orig_shape

    canvas = np.zeros((2, padded_h, padded_w), dtype=np.float32)
    count = np.zeros((1, padded_h, padded_w), dtype=np.float32)

    for patch, (y1, y2, x1, x2) in zip(patch_flows, coordinates):
        p = np.asarray(patch).astype(np.float32)
        if p.ndim == 2:
            p = np.expand_dims(p, 0)
        canvas[:, y1:y2, x1:x2] += p
        count[:, y1:y2, x1:x2] += 1.0

    count = np.maximum(count, 1e-8)
    full_flow = canvas / count
    full_flow = np.nan_to_num(full_flow, nan=0.0)
    return full_flow[:, :orig_h, :orig_w]


def stitch_multiclass_probability_map(
    patch_probs: List[np.ndarray],
    coordinates: List[Tuple[int, int, int, int]],
    orig_shape: Tuple[int, int],
    padded_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Reconstruct full-size multi-class probability map [num_classes, orig_h, orig_w]
    from patch probabilities using uniform count averaging.
    """
    padded_h, padded_w = padded_shape
    orig_h, orig_w = orig_shape
    num_classes = patch_probs[0].shape[0] if len(patch_probs) > 0 else 5

    canvas = np.zeros((num_classes, padded_h, padded_w), dtype=np.float32)
    count = np.zeros((padded_h, padded_w), dtype=np.float32)

    for patch, (y1, y2, x1, x2) in zip(patch_probs, coordinates):
        p = np.asarray(patch).astype(np.float32)
        canvas[:, y1:y2, x1:x2] += p
        count[y1:y2, x1:x2] += 1.0

    count = np.maximum(count, 1e-8)
    full_cls_prob = canvas / count[None, :, :]
    full_cls_prob = np.nan_to_num(full_cls_prob, nan=0.0, posinf=1.0, neginf=0.0)
    return full_cls_prob[:, :orig_h, :orig_w]


def binary_to_instances(binary_mask: np.ndarray, min_size: int = 10) -> np.ndarray:
    """
    Convert a 2D binary mask into an integer instance mask (0=bg, 1..N=instances)
    using 8-connected components, filtering instances smaller than min_size.
    """
    bin_u8 = (np.asarray(binary_mask) > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_u8, connectivity=8)
    instance_map = np.zeros_like(labels, dtype=np.int32)
    new_id = 1
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_size:
            instance_map[labels == label_id] = new_id
            new_id += 1
    return instance_map


def compute_binary_metrics(
    pred_binary: np.ndarray,
    gt_binary: np.ndarray,
    pred_instance: Optional[np.ndarray] = None,
    gt_instance: Optional[np.ndarray] = None,
    min_instance_size: int = 10,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute binary Dice, IoU, AJI, and PQ on full-size images.
    """
    p_bin = (np.asarray(pred_binary) > 0).astype(np.float32)
    g_bin = (np.asarray(gt_binary) > 0).astype(np.float32)

    intersection = float(np.sum(p_bin * g_bin))
    pred_area = float(np.sum(p_bin))
    gt_area = float(np.sum(g_bin))
    union = pred_area + gt_area - intersection

    if (pred_area + gt_area) == 0:
        dice = 1.0
        iou = 1.0
    else:
        dice = (2.0 * intersection + smooth) / (pred_area + gt_area + smooth)
        iou = (intersection + smooth) / (union + smooth) if union > 0 else 1.0

    # Ensure instance masks for AJI and PQ
    if pred_instance is None:
        p_inst = binary_to_instances(pred_binary, min_size=min_instance_size)
    else:
        p_inst = np.asarray(pred_instance, dtype=np.int32)

    if gt_instance is None:
        g_inst = binary_to_instances(gt_binary, min_size=min_instance_size)
    else:
        g_inst = np.asarray(gt_instance, dtype=np.int32)

    aji = float(compute_aji(g_inst, p_inst))
    pq = float(Panoptic_quality(g_inst, p_inst, match_iou_threshold=0.5))

    return {
        "Dice": float(dice),
        "IoU": float(iou),
        "AJI": float(aji),
        "PQ": float(pq),
    }


def compute_monusac_cell_metrics(
    pred_class_masks: np.ndarray,
    gt_class_masks: np.ndarray,
    gt_class_instances: Optional[np.ndarray] = None,
    min_instance_size: int = 10,
) -> Dict[str, float]:
    """
    Compute per-class Dice, IoU, AJI, and PQ for MoNuSAC 4 cell classes:
    Epithelial, Lymphocyte, Neutrophil, Macrophage.
    """
    metrics = {}
    dices, ious, ajis, pqs = [], [], [], []

    for c_idx, c_name in enumerate(MONUSAC_CELL_NAMES):
        tag = c_name.lower()
        p_c = pred_class_masks[c_idx] > 0
        g_c = (gt_class_masks[c_idx] > 0) if c_idx < gt_class_masks.shape[0] else np.zeros_like(p_c)

        if gt_class_instances is not None and c_idx < gt_class_instances.shape[0]:
            g_inst_c = gt_class_instances[c_idx]
        else:
            g_inst_c = binary_to_instances(g_c, min_size=min_instance_size)

        p_inst_c = binary_to_instances(p_c, min_size=min_instance_size)

        c_scores = compute_binary_metrics(
            pred_binary=p_c,
            gt_binary=g_c,
            pred_instance=p_inst_c,
            gt_instance=g_inst_c,
            min_instance_size=min_instance_size,
        )

        metrics[f"dice_{tag}"] = c_scores["Dice"]
        metrics[f"iou_{tag}"] = c_scores["IoU"]
        metrics[f"aji_{tag}"] = c_scores["AJI"]
        metrics[f"pq_{tag}"] = c_scores["PQ"]

        dices.append(c_scores["Dice"])
        ious.append(c_scores["IoU"])
        ajis.append(c_scores["AJI"])
        pqs.append(c_scores["PQ"])

    metrics["mean_class_dice"] = float(np.mean(dices))
    metrics["mean_class_iou"] = float(np.mean(ious))
    metrics["mean_class_aji"] = float(np.mean(ajis))
    metrics["mean_class_pq"] = float(np.mean(pqs))

    return metrics


def extract_model_outputs(
    outputs: Any,
    task: Optional[str] = None,
    return_flow: bool = False,
) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
    """
    Extract segmentation logits, optional classification logits, and optional flow from model output.
    Supports both traditional tuple/dict returns and multi-task dictionary outputs.
    """
    seg = None
    cls_ = None
    flow = None

    def _get_first(d: dict, keys: list):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return None

    if isinstance(outputs, (tuple, list)):
        seg = outputs[0]
        if len(outputs) >= 2:
            cls_ = outputs[1]
        if len(outputs) >= 3:
            flow = outputs[2]
    elif isinstance(outputs, dict):
        # 1. Resolve task-specific keys if task is explicitly specified
        if task in ("glas", "gland"):
            seg = _get_first(outputs, ["gland_seg", "glas_seg", "gland_binary", "glas_binary"])
            flow = _get_first(outputs, ["gland_flow", "glas_flow"])
            cls_ = None
        elif task in ("monuseg", "cell", "nuclei"):
            seg = _get_first(outputs, ["cell_seg", "monuseg_seg", "cell_binary", "monuseg_binary"])
            flow = _get_first(outputs, ["cell_flow", "monuseg_flow"])
            cls_ = None
        elif task == "monusac":
            seg = _get_first(outputs, ["cell_seg", "monusac_seg", "cell_binary", "monusac_binary"])
            flow = _get_first(outputs, ["cell_flow", "monusac_flow"])
            cls_ = _get_first(outputs, ["cell_class", "monusac_class"])

        # 2. General fallback resolution
        if seg is None:
            seg = _get_first(outputs, [
                "cell_seg", "gland_seg", "segmentation", "cell_binary",
                "gland_binary", "monusac_binary", "monuseg_binary", "glas_binary", "logits"
            ])
        if cls_ is None:
            cls_ = _get_first(outputs, ["classification", "cell_class", "monusac_class"])
        if flow is None:
            flow = _get_first(outputs, [
                "flow", "cell_flow", "gland_flow", "monusac_flow", "monuseg_flow", "glas_flow"
            ])

    if return_flow:
        return seg, cls_, flow
    return seg, cls_


def class4_to_instances(
    seg_logits: Union[np.ndarray, torch.Tensor],
    min_size: int = 10,
    expand_boundary: bool = True,
    max_boundary_dist: Optional[float] = None,
    is_gland: bool = False,
    fill_lumen: bool = True,
    close_kernel: int = 25,
    original_rgb: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Converts 4-class segmentation logits/map [4, H, W] or [H, W] into separated instance labels:
      0: Background
      1: Interior (Core)
      2: Boundary (Contour ring)
      3: Gap (Inter-cellular clefts/valleys via Bottom-Hat transform)

    Instance separation:
      1. Connected components on Interior cores (class 1).
      2. For glands (is_gland=True): safely reclaims interior white lumina using
         per-instance boundary closing and optical-white saturation gating (preventing
         false-positive filling of pink stroma or inter-gland background).
      3. Gap (class 3) and Background (class 0) act as non-instance barriers.
      4. Expansion of interior cores into adjacent Boundary pixels (class 2) via distance transform.
         Gap pixels (class 3) are never absorbed, preserving physical separation between touching cells.
    """
    from scipy.ndimage import distance_transform_edt, binary_fill_holes

    if isinstance(seg_logits, torch.Tensor):
        if seg_logits.ndim == 3 and seg_logits.shape[0] >= 3:
            pred_class = seg_logits.argmax(dim=0).cpu().numpy()
        elif seg_logits.ndim == 4 and seg_logits.shape[1] >= 3:
            pred_class = seg_logits[0].argmax(dim=0).cpu().numpy()
        else:
            pred_class = seg_logits.squeeze().cpu().numpy()
    else:
        seg_np = np.asarray(seg_logits)
        if seg_np.ndim == 3 and seg_np.shape[0] >= 3:
            pred_class = np.argmax(seg_np, axis=0)
        else:
            pred_class = seg_np

    interior = (pred_class == 1).astype(np.uint8)

    # Gland-specific safe lumen reclamation
    if is_gland and fill_lumen:
        fg_mask = ((pred_class == 1) | (pred_class == 2)).astype(np.uint8)
        if close_kernel > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
            closed_fg = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k)
        else:
            closed_fg = fg_mask

        # Identify background holes enclosed by the gland boundary
        enclosed_holes = binary_fill_holes(closed_fg > 0) & (pred_class == 0)

        # Apply optical-white gating if original image is provided:
        # True lumens are optical white (high brightness V > 210, low saturation S < 35).
        # Pink stroma (eosin stain: high saturation S > 50) is strictly rejected!
        if original_rgb is not None and np.any(enclosed_holes):
            orig_hsv = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2HSV)
            sat = orig_hsv[:, :, 1]
            val = orig_hsv[:, :, 2]
            is_optical_white = (val > 210) & (sat < 35)
            valid_lumen = enclosed_holes & is_optical_white
        else:
            valid_lumen = enclosed_holes

        interior = (interior | valid_lumen.astype(np.uint8))

    num_labels, labels = cv2.connectedComponents(interior)

    if num_labels <= 1:
        return np.zeros_like(pred_class, dtype=np.int32)

    unique_ids, counts = np.unique(labels, return_counts=True)
    inst_mask = np.zeros_like(labels, dtype=np.int32)
    next_id = 1
    for uid, count in zip(unique_ids, counts):
        if uid == 0:
            continue
        if count >= min_size:
            inst_mask[labels == uid] = next_id
            next_id += 1

    if not expand_boundary or next_id == 1:
        return inst_mask

    boundary = (pred_class == 2)
    if not np.any(boundary):
        return inst_mask

    has_labels = (inst_mask > 0)
    if not np.any(has_labels):
        return inst_mask

    _, (nearest_y, nearest_x) = distance_transform_edt(~has_labels, return_indices=True)
    if max_boundary_dist is not None:
        dist = np.sqrt(
            (np.arange(pred_class.shape[0])[:, None] - nearest_y) ** 2
            + (np.arange(pred_class.shape[1])[None, :] - nearest_x) ** 2
        )
        valid_expand = boundary & (dist <= max_boundary_dist)
    else:
        valid_expand = boundary

    inst_mask[valid_expand] = inst_mask[nearest_y[valid_expand], nearest_x[valid_expand]]

    # Final per-instance lumen seal (only for individual instances, never merging across instances)
    if is_gland and fill_lumen:
        for inst_id in range(1, next_id):
            curr_inst = (inst_mask == inst_id)
            single_holes = binary_fill_holes(curr_inst) & (~curr_inst)
            if np.any(single_holes):
                if original_rgb is not None:
                    orig_hsv = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2HSV)
                    is_white = (orig_hsv[:, :, 2] > 210) & (orig_hsv[:, :, 1] < 35)
                    single_holes = single_holes & is_white
                inst_mask[single_holes] = inst_id

    return inst_mask


# Backward compatibility alias
class3_to_instances = class4_to_instances


def flow_to_instances(
    binary_mask: np.ndarray,
    flow_vector: Optional[np.ndarray] = None,
    min_size: int = 10,
    method: str = "class4",
    **kwargs,
) -> np.ndarray:
    """
    Converts 4-class/3-class segmentation map, binary mask, or flow field into separated instance labels.
    If 4-class or 3-class segmentation is provided (or flow_vector is None and values include 2 or 3),
    uses class4_to_instances to extract instances cleanly separated by the bottom-hat gap class.
    """
    mask_np = np.asarray(binary_mask)
    if (mask_np.ndim == 3 and mask_np.shape[0] in (3, 4)) or (flow_vector is None and np.max(mask_np) in (2, 3)):
        return class4_to_instances(mask_np, min_size=min_size, **kwargs)

    bin_u8 = (mask_np > 0).astype(np.uint8)
    if not bin_u8.any():
        return np.zeros_like(bin_u8, dtype=np.int32)

    if flow_vector is None:
        return binary_to_instances(bin_u8, min_size=min_size)

    from utils.flow_postprocessing import flow_marching_instances, flow_directional_cut_instances

    if method == "directional_cut":
        return flow_directional_cut_instances(
            bin_u8,
            flow_vector=flow_vector,
            min_size=min_size,
            cos_thresh=kwargs.get("cos_thresh", 0.0),
            oppose_margin=kwargs.get("oppose_margin", 0.05),
            refine_step=kwargs.get("refine_step", 1.5),
        )
    else:  # default 'flow_marching'
        return flow_marching_instances(
            bin_u8,
            flow_vector=flow_vector,
            min_size=min_size,
            n_iter=kwargs.get("n_iter", 100),
            step=kwargs.get("step", 1.0),
            peak_min_dist=kwargs.get("peak_min_dist", 7),
        )


def load_model_for_evaluation(
    config: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """
    Instantiate model from config and load trained checkpoint weights.
    """
    if device is None:
        device = get_device()

    model_name = config.get("model", {}).get("name", "sdenetpointgrad")
    clean_name = str(model_name).lower().strip()
    if clean_name not in MODEL_REGISTRY and model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' not recognized. Available: {list(MODEL_REGISTRY.keys())}"
        )

    builder = MODEL_REGISTRY.get(clean_name, MODEL_REGISTRY.get(model_name))
    model = builder(config)
    model.to(device)

    # Resolve checkpoint
    chk = checkpoint_path
    if not chk:
        exp_name = config.get("experiment", {}).get("name", "")
        chk_dir = config.get("checkpoint", {}).get("path", "checkpoints")
        cand_best = os.path.join(chk_dir, exp_name, "best.pt")
        cand_latest = os.path.join(chk_dir, exp_name, "latest.pt")
        if os.path.exists(cand_best):
            chk = cand_best
        elif os.path.exists(cand_latest):
            chk = cand_latest

    if chk and os.path.exists(chk):
        print(f"Loading checkpoint weights from: {chk}")
        load_checkpoint(model=model, path=chk, device=device)
    else:
        print(f"[Warning] No checkpoint found at '{chk}'. Evaluating with initial weights.")

    model.eval()
    return model


def colorize_instance_mask(instance_mask: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Convert an integer instance mask (0=background, 1..N=instances)
    into a colorful RGB image where each instance has a distinct color.
    """
    inst = np.asarray(instance_mask, dtype=np.int32)
    h, w = inst.shape[:2]
    max_id = int(inst.max()) if inst.size > 0 else 0
    if max_id <= 0:
        return np.zeros((h, w, 3), dtype=np.uint8)

    rng = np.random.RandomState(seed)
    colors = rng.randint(60, 255, size=(max_id + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background is black

    return colors[inst]


def colorize_multiclass_mask(
    class_masks: np.ndarray,
    colors: Optional[List[Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """
    Convert a 4-channel binary stack [4, H, W] or 1-channel argmax map [H, W]
    into a distinct RGB map for the 4 MoNuSAC cell classes:
      0: Epithelial  -> Red    [230, 40, 40]
      1: Lymphocyte  -> Blue   [30, 144, 255]
      2: Neutrophil  -> Green  [50, 205, 50]
      3: Macrophage  -> Yellow [255, 215, 0]
    """
    if colors is None:
        colors = [
            (230, 40, 40),    # Epithelial: Red
            (30, 144, 255),   # Lymphocyte: Dodger Blue
            (50, 205, 50),    # Neutrophil: Lime Green
            (255, 215, 0),    # Macrophage: Gold/Yellow
        ]

    arr = np.asarray(class_masks)
    if arr.ndim == 3 and arr.shape[0] == 4:
        h, w = arr.shape[1], arr.shape[2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for c_idx in range(4):
            mask = arr[c_idx] > 0
            rgb[mask] = colors[c_idx]
        return rgb
    elif arr.ndim == 2:
        h, w = arr.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for c_idx in range(4):
            mask = (arr == (c_idx + 1)) | (arr == c_idx)
            rgb[mask] = colors[c_idx]
        return rgb
    return np.zeros((arr.shape[-2], arr.shape[-1], 3), dtype=np.uint8)


def colorize_4class_map(class4_map: np.ndarray) -> np.ndarray:
    """
    Convert a 4-class semantic map (0: background, 1: interior core, 2: boundary ring, 3: bottom-hat gap)
    into a vivid, publication-ready RGB visualization:
      0: Background -> Black       [0, 0, 0]
      1: Interior   -> Light Green [46, 204, 113]
      2: Boundary   -> Amber Gold  [241, 196, 15]
      3: Gap        -> Crimson Red [231, 76, 60]
    """
    c4 = np.asarray(class4_map)
    if c4.ndim == 3 and c4.shape[0] == 4:
        c4 = np.argmax(c4, axis=0)
    h, w = c4.shape[-2:]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    palette = [
        [0, 0, 0],         # 0: BG (Black)
        [46, 204, 113],    # 1: Interior (Emerald Green)
        [241, 196, 15],    # 2: Boundary (Sun Gold)
        [231, 76, 60],     # 3: Gap (Crimson Red)
    ]
    for c_idx, color in enumerate(palette):
        rgb[c4 == c_idx] = color
    return rgb


def create_tp_fp_fn_overlay(
    image: np.ndarray,
    gt_binary: np.ndarray,
    pred_binary: np.ndarray,
    alpha: float = 0.5,
    add_legend: bool = True,
) -> np.ndarray:
    """
    Create a pixel-level error overlay on the original RGB image:
      - True Positive (TP):  Green  [0, 255, 0]   (Correctly identified nucleus/gland)
      - False Positive (FP): Red    [255, 0, 0]   (False alarm / extra prediction)
      - False Negative (FN): Yellow [255, 220, 0] (Missed ground truth)

    Args:
        image: RGB uint8 image [H, W, 3].
        gt_binary: Ground truth binary mask [H, W] in {0, 1}.
        pred_binary: Predicted binary mask [H, W] in {0, 1}.
        alpha: Blending transparency factor.
        add_legend: If True, prepends an informative top legend banner.

    Returns:
        RGB uint8 image with TP/FP/FN colored overlay.
    """
    h, w = image.shape[:2]
    gt_bin = (np.asarray(gt_binary) > 0)
    pred_bin = (np.asarray(pred_binary) > 0)

    tp = gt_bin & pred_bin
    fp = (~gt_bin) & pred_bin
    fn = gt_bin & (~pred_bin)

    overlay = image.astype(np.float32).copy()

    # Colors in RGB format
    TP_COLOR = np.array([0, 255, 0], dtype=np.float32)       # Green
    FP_COLOR = np.array([255, 0, 0], dtype=np.float32)       # Red
    FN_COLOR = np.array([255, 220, 0], dtype=np.float32)     # Yellow

    if np.any(tp):
        overlay[tp] = (1.0 - alpha) * overlay[tp] + alpha * TP_COLOR
    if np.any(fp):
        overlay[fp] = (1.0 - alpha) * overlay[fp] + alpha * FP_COLOR
    if np.any(fn):
        overlay[fn] = (1.0 - alpha) * overlay[fn] + alpha * FN_COLOR

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    if not add_legend:
        return overlay

    # Create top banner with legend
    banner_h = max(36, int(round(h * 0.045)))
    banner = np.full((banner_h, w, 3), 32, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.42, min(0.75, w / 1500.0))
    thickness = 1 if font_scale < 0.65 else 2

    legend_items = [
        ("TP: Hit", (0, 255, 0)),
        ("FP: False Alarm", (255, 0, 0)),
        ("FN: Missed", (255, 220, 0)),
    ]

    box_size = max(14, int(round(banner_h * 0.45)))
    spacing = w // len(legend_items)
    y_center = banner_h // 2

    for idx, (label, color) in enumerate(legend_items):
        x_start = idx * spacing + max(10, int(w * 0.02))
        y1 = y_center - box_size // 2
        y2 = y_center + box_size // 2
        x1 = x_start
        x2 = x_start + box_size

        # Swatch
        banner[y1:y2, x1:x2] = color
        cv2.rectangle(banner, (x1, y1), (x2, y2), (255, 255, 255), 1)

        # Text label
        text_x = x2 + 8
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_y = y_center + text_size[1] // 2
        cv2.putText(banner, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return np.vstack([banner, overlay])


def create_comparison_panel(
    image: np.ndarray,
    gt_binary: np.ndarray,
    pred_binary: np.ndarray,
    overlay: np.ndarray,
    max_dim: int = 1024,
) -> np.ndarray:
    """
    Create a clean 2x2 comparison panel:
      Top-Left: Original Image      | Top-Right: TP/FP/FN Overlay
      Bottom-Left: Ground Truth Mask | Bottom-Right: Prediction Mask
    """
    h, w = image.shape[:2]

    # Convert binary masks to 3-channel RGB for tiling
    gt_rgb = np.stack([(gt_binary > 0).astype(np.uint8) * 255] * 3, axis=-1)
    pred_rgb = np.stack([(pred_binary > 0).astype(np.uint8) * 255] * 3, axis=-1)

    if overlay.shape[0] != h:
        overlay_crop = overlay[-h:, :w]
    else:
        overlay_crop = overlay[:h, :w]

    def add_label(img: np.ndarray, text: str) -> np.ndarray:
        out = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, min(1.0, w / 1000.0))
        th = 2
        bh = max(30, int(round(h * 0.045)))
        out[:bh, :, :] = (out[:bh, :, :] * 0.3).astype(np.uint8)
        cv2.putText(out, text, (10, bh - 8), font, scale, (255, 255, 255), th, cv2.LINE_AA)
        return out

    p1 = add_label(image, "Original Image")
    p2 = add_label(overlay_crop, "TP (Green) | FP (Red) | FN (Yellow)")
    p3 = add_label(gt_rgb, "Ground Truth Mask")
    p4 = add_label(pred_rgb, "Predicted Mask")

    row1 = np.hstack([p1, p2])
    row2 = np.hstack([p3, p4])
    grid = np.vstack([row1, row2])

    if max(grid.shape[:2]) > max_dim:
        scale = max_dim / float(max(grid.shape[:2]))
        new_w = int(round(grid.shape[1] * scale))
        new_h = int(round(grid.shape[0] * scale))
        grid = cv2.resize(grid, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return grid


def create_all_heads_panel(
    image: np.ndarray,
    gland_inst: np.ndarray,
    cell_inst: np.ndarray,
    pheno_rgb: np.ndarray,
    max_dim: int = 1024,
) -> np.ndarray:
    """
    Creates a publication-ready 2x2 multi-head comparison panel:
      Top-Left: Original Image                 | Top-Right: GLaS Gland Head (Instances)
      Bottom-Left: Shared Cell Head (Instances)| Bottom-Right: MoNuSAC Phenotype Head
    """
    h, w = image.shape[:2]
    gland_rgb = colorize_instance_mask(gland_inst)
    cell_rgb = colorize_instance_mask(cell_inst)

    def add_label(img: np.ndarray, text: str) -> np.ndarray:
        out = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.42, min(0.85, w / 1100.0))
        th = 2
        bh = max(32, int(round(h * 0.05)))
        out[:bh, :, :] = (out[:bh, :, :] * 0.25).astype(np.uint8)
        cv2.putText(out, text, (10, bh - 9), font, scale, (255, 255, 255), th, cv2.LINE_AA)
        return out

    p1 = add_label(image, "Original Image")
    p2 = add_label(gland_rgb, "GLaS Gland Head (Instances)")
    p3 = add_label(cell_rgb, "Shared Cell Nuclei Head (Instances)")
    p4 = add_label(pheno_rgb, "MoNuSAC Phenotype Head (Classes)")

    row1 = np.hstack([p1, p2])
    row2 = np.hstack([p3, p4])
    grid = np.vstack([row1, row2])

    if max(grid.shape[:2]) > max_dim:
        scale = max_dim / float(max(grid.shape[:2]))
        new_w = int(round(grid.shape[1] * scale))
        new_h = int(round(grid.shape[0] * scale))
        grid = cv2.resize(grid, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return grid


def create_dataset_panel(
    image: np.ndarray,
    gt_inst: np.ndarray,
    pred_inst: np.ndarray,
    overlay: np.ndarray,
    max_dim: int = 1024,
) -> np.ndarray:
    """
    Creates a 2x2 dataset-specific comparison panel:
      Top-Left: Original Image                 | Top-Right: TP/FP/FN Error Overlay
      Bottom-Left: Ground Truth Instances      | Bottom-Right: Predicted Instances
    """
    h, w = image.shape[:2]
    gt_rgb = colorize_instance_mask(gt_inst)
    pred_rgb = colorize_instance_mask(pred_inst)

    if overlay.shape[0] != h:
        overlay_crop = overlay[-h:, :w]
    else:
        overlay_crop = overlay[:h, :w]

    def add_label(img: np.ndarray, text: str) -> np.ndarray:
        out = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.42, min(0.85, w / 1100.0))
        th = 2
        bh = max(32, int(round(h * 0.05)))
        out[:bh, :, :] = (out[:bh, :, :] * 0.25).astype(np.uint8)
        cv2.putText(out, text, (10, bh - 9), font, scale, (255, 255, 255), th, cv2.LINE_AA)
        return out

    p1 = add_label(image, "Original Image")
    p2 = add_label(overlay_crop, "TP (Green) | FP (Red) | FN (Yellow)")
    p3 = add_label(gt_rgb, "Ground Truth Instances")
    p4 = add_label(pred_rgb, "Predicted Instances")

    row1 = np.hstack([p1, p2])
    row2 = np.hstack([p3, p4])
    grid = np.vstack([row1, row2])

    if max(grid.shape[:2]) > max_dim:
        scale = max_dim / float(max(grid.shape[:2]))
        new_w = int(round(grid.shape[1] * scale))
        new_h = int(round(grid.shape[0] * scale))
        grid = cv2.resize(grid, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return grid


def create_monusac_dataset_panel(
    image: np.ndarray,
    gt_inst: np.ndarray,
    pred_inst: np.ndarray,
    overlay: np.ndarray,
    gt_pheno_rgb: np.ndarray,
    pred_pheno_rgb: np.ndarray,
    max_dim: int = 1200,
) -> np.ndarray:
    """
    Creates a 2x3 dataset-specific comparison panel for MoNuSAC:
      Top Row:    Original Image | Ground Truth Instances | Predicted Instances
      Bottom Row: TP/FP/FN Error Overlay | GT Phenotypes | Predicted Phenotypes
    """
    h, w = image.shape[:2]
    gt_rgb = colorize_instance_mask(gt_inst)
    pred_rgb = colorize_instance_mask(pred_inst)

    if overlay.shape[0] != h:
        overlay_crop = overlay[-h:, :w]
    else:
        overlay_crop = overlay[:h, :w]

    def add_label(img: np.ndarray, text: str) -> np.ndarray:
        out = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.42, min(0.85, w / 1100.0))
        th = 2
        bh = max(32, int(round(h * 0.05)))
        out[:bh, :, :] = (out[:bh, :, :] * 0.25).astype(np.uint8)
        cv2.putText(out, text, (10, bh - 9), font, scale, (255, 255, 255), th, cv2.LINE_AA)
        return out

    p1 = add_label(image, "Original Image")
    p2 = add_label(gt_rgb, "Ground Truth Instances")
    p3 = add_label(pred_rgb, "Predicted Instances")
    p4 = add_label(overlay_crop, "TP (Grn) | FP (Red) | FN (Yel)")
    p5 = add_label(gt_pheno_rgb, "Ground Truth Phenotypes")
    p6 = add_label(pred_pheno_rgb, "Predicted Phenotypes")

    row1 = np.hstack([p1, p2, p3])
    row2 = np.hstack([p4, p5, p6])
    grid = np.vstack([row1, row2])

    if max(grid.shape[:2]) > max_dim:
        scale = max_dim / float(max(grid.shape[:2]))
        new_w = int(round(grid.shape[1] * scale))
        new_h = int(round(grid.shape[0] * scale))
        grid = cv2.resize(grid, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return grid


def predict_whole_image_multitask(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    device: torch.device,
    patch_size: int = 512,
    stride: Optional[int] = None,
    overlap: Optional[float] = None,
    patch_batch_size: int = 4,
    min_instance_size: int = 10,
) -> Dict[str, Any]:
    """
    Runs multi-task sliding-window patch inference across a full-size image,
    reconstructing full-resolution predictions for ALL heads:
      1. GLaS Gland Head: 4-class logits, instances, binary mask, probability map
      2. Shared Cell Head: 4-class logits, instances, binary mask, probability map
      3. MoNuSAC Phenotype Head: 5-class logits, per-class binary masks, colorized phenotypes

    Returns:
      Dictionary containing stitched full-resolution numpy arrays for each head.
    """
    orig_h, orig_w = image_rgb.shape[:2]

    # 1. Patchify image
    patches, coords, orig_shape, padded_shape = patchify_image(
        image=image_rgb,
        patch_size=patch_size,
        stride=stride,
        overlap=overlap,
    )

    # 2. Batched patch inference across all heads
    patch_gland_probs = []
    patch_cell_probs = []
    patch_class_probs = []

    model.eval()
    with torch.inference_mode():
        for i in range(0, len(patches), patch_batch_size):
            chunk = patches[i : i + patch_batch_size].to(device, non_blocking=True)
            raw_out = model(chunk)

            # Gland head
            g_logits = raw_out.get("gland_seg", raw_out.get("glas_seg"))
            if g_logits is not None:
                if g_logits.shape[-2:] != chunk.shape[-2:]:
                    g_logits = F.interpolate(g_logits, size=chunk.shape[-2:], mode="bilinear", align_corners=False)
                g_prob = F.softmax(g_logits, dim=1).detach().cpu().numpy()
                patch_gland_probs.extend(g_prob)

            # Cell head
            c_logits = raw_out.get("cell_seg", raw_out.get("monuseg_seg", raw_out.get("monusac_seg")))
            if c_logits is not None:
                if c_logits.shape[-2:] != chunk.shape[-2:]:
                    c_logits = F.interpolate(c_logits, size=chunk.shape[-2:], mode="bilinear", align_corners=False)
                c_prob = F.softmax(c_logits, dim=1).detach().cpu().numpy()
                patch_cell_probs.extend(c_prob)

            # Phenotype classification head
            cls_logits = raw_out.get("cell_class", raw_out.get("monusac_class"))
            if cls_logits is not None:
                if cls_logits.shape[-2:] != chunk.shape[-2:]:
                    cls_logits = F.interpolate(cls_logits, size=chunk.shape[-2:], mode="bilinear", align_corners=False)
                cls_prob = F.softmax(cls_logits, dim=1).detach().cpu().numpy()
                patch_class_probs.extend(cls_prob)

    results: Dict[str, Any] = {}

    # 3. Stitch GLaS Gland Head
    if len(patch_gland_probs) == len(coords):
        full_gland_prob = stitch_multiclass_probability_map(
            patch_probs=patch_gland_probs,
            coordinates=coords,
            orig_shape=orig_shape,
            padded_shape=padded_shape,
        )
        gland_inst = class4_to_instances(
            full_gland_prob,
            min_size=min_instance_size,
            is_gland=True,
            fill_lumen=True,
            close_kernel=25,
            original_rgb=image_rgb,
        )
        gland_bin = (gland_inst > 0).astype(np.uint8)
        gland_prob_2d = full_gland_prob[1] + full_gland_prob[2] if full_gland_prob.shape[0] >= 3 else (1.0 - full_gland_prob[0])
        gland_4class = np.argmax(full_gland_prob, axis=0)

        results["gland"] = {
            "instances": gland_inst,
            "binary_mask": gland_bin,
            "class4_map": gland_4class,
            "probability": gland_prob_2d,
            "full_prob": full_gland_prob,
            "colorized_instances": colorize_instance_mask(gland_inst),
            "colorized_4class": colorize_4class_map(gland_4class),
        }

    # 4. Stitch Shared Cell Head
    if len(patch_cell_probs) == len(coords):
        full_cell_prob = stitch_multiclass_probability_map(
            patch_probs=patch_cell_probs,
            coordinates=coords,
            orig_shape=orig_shape,
            padded_shape=padded_shape,
        )
        cell_inst = class4_to_instances(full_cell_prob, min_size=min_instance_size)
        cell_bin = (cell_inst > 0).astype(np.uint8)
        cell_prob_2d = full_cell_prob[1] + full_cell_prob[2] if full_cell_prob.shape[0] >= 3 else (1.0 - full_cell_prob[0])
        cell_4class = np.argmax(full_cell_prob, axis=0)

        results["cell"] = {
            "instances": cell_inst,
            "binary_mask": cell_bin,
            "class4_map": cell_4class,
            "probability": cell_prob_2d,
            "full_prob": full_cell_prob,
            "colorized_instances": colorize_instance_mask(cell_inst),
            "colorized_4class": colorize_4class_map(cell_4class),
        }

    # 5. Stitch MoNuSAC Phenotype Classification Head
    if len(patch_class_probs) == len(coords):
        full_cls_prob = stitch_multiclass_probability_map(
            patch_probs=patch_class_probs,
            coordinates=coords,
            orig_shape=orig_shape,
            padded_shape=padded_shape,
        )
        if full_cls_prob.shape[0] >= 5:
            fg_argmax = np.argmax(full_cls_prob[1:5], axis=0)  # 0: Epi, 1: Lym, 2: Neu, 3: Mac
        else:
            fg_argmax = np.argmax(full_cls_prob, axis=0)

        cell_bin = results["cell"]["binary_mask"] if "cell" in results else np.ones((orig_h, orig_w), dtype=np.uint8)
        pred_class_masks = np.zeros((4, orig_h, orig_w), dtype=np.uint8)
        for c in range(4):
            pred_class_masks[c] = ((cell_bin > 0) & (fg_argmax == c)).astype(np.uint8)

        pheno_color = colorize_multiclass_mask(pred_class_masks)
        results["phenotype"] = {
            "full_prob": full_cls_prob,
            "class_masks": pred_class_masks,
            "colorized_classes": pheno_color,
        }

    return results


def save_image_safe(path: Union[str, Path], img: np.ndarray) -> None:
    """
    Save an image safely to disk, creating parent directories if needed.
    Correctly handles RGB vs BGR (expects RGB for 3D arrays), uint16 for instance IDs,
    and uint8/bool for binary masks.
    """
    path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arr = np.asarray(img)

    if arr.ndim == 2:
        if arr.dtype == bool or (arr.dtype == np.uint8 and arr.max() <= 1):
            cv2.imwrite(path, (arr.astype(np.uint8) * 255))
        elif np.issubdtype(arr.dtype, np.integer) and arr.max() > 255:
            cv2.imwrite(path, arr.astype(np.uint16))
        else:
            cv2.imwrite(path, arr.astype(np.uint8))
    elif arr.ndim == 3:
        if arr.shape[2] == 3:
            bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr)
        elif arr.shape[2] == 4:
            bgra = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(path, bgra)
        else:
            Image.fromarray(arr).save(path)


__all__ = [
    "get_patch_starts",
    "pad_image",
    "patchify_image",
    "stitch_probability_map",
    "stitch_flow_map",
    "stitch_multiclass_probability_map",
    "binary_to_instances",
    "class4_to_instances",
    "class3_to_instances",
    "flow_to_instances",
    "compute_binary_metrics",
    "compute_monusac_cell_metrics",
    "extract_model_outputs",
    "load_model_for_evaluation",
    "colorize_instance_mask",
    "colorize_multiclass_mask",
    "colorize_4class_map",
    "create_tp_fp_fn_overlay",
    "create_comparison_panel",
    "create_dataset_panel",
    "create_monusac_dataset_panel",
    "create_all_heads_panel",
    "predict_whole_image_multitask",
    "save_image_safe",
    "MONUSAC_CELL_NAMES",
]
