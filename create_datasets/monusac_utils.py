# create_datasets/monusac_utils.py

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, find_objects, gaussian_filter
from skimage.morphology import skeletonize
import xml.etree.ElementTree as ET

Image.MAX_IMAGE_PIXELS = None

CLASS_NAMES = ["Epithelial", "Lymphocyte", "Neutrophil", "Macrophage"]
CLASS_TO_IDX = {name.lower(): idx for idx, name in enumerate(CLASS_NAMES)}


def load_monusac_image(image_path: Path) -> np.ndarray:
    """Reads a MoNuSAC TIFF/SVS sub-image into an RGB uint8 array [H, W, 3]."""
    with Image.open(image_path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def parse_monusac_xml(xml_path: Path, height: int, width: int) -> np.ndarray:
    """
    Parses MoNuSAC Aperio XML annotations into a 4-channel instance mask [4, H, W].
    Channels: 0: Epithelial, 1: Lymphocyte, 2: Neutrophil, 3: Macrophage.
    Each channel contains unique integer instance IDs 1..N (0 = background).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    mask_4ch = np.zeros((4, height, width), dtype=np.int32)

    for annotation in root.findall(".//Annotation"):
        # Identify target cell class from Attributes
        class_idx = None
        for attr in annotation.findall(".//Attribute"):
            attr_name = attr.attrib.get("Name", "").strip().lower()
            if attr_name in CLASS_TO_IDX:
                class_idx = CLASS_TO_IDX[attr_name]
                break

        if class_idx is None:
            continue

        instance_id = 1
        for region in annotation.findall(".//Region"):
            points = []
            for vertex in region.iter("Vertex"):
                x = int(round(float(vertex.attrib["X"])))
                y = int(round(float(vertex.attrib["Y"])))
                points.append([x, y])

            if len(points) >= 3:
                pts = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask_4ch[class_idx], [pts], color=instance_id)
                instance_id += 1

    return mask_4ch


def save_subimage_data(
    subimage_dir: Path,
    image: np.ndarray,
    mask: np.ndarray,
    sdf: Optional[np.ndarray] = None,
    hv_map: Optional[np.ndarray] = None,
    target_3class: Optional[np.ndarray] = None,
    target_4class: Optional[np.ndarray] = None,
) -> None:
    """Saves image.npy, mask.npy, binary_mask.npy, and optional full-size targets into subimage_dir."""
    subimage_dir.mkdir(parents=True, exist_ok=True)
    np.save(subimage_dir / "image.npy", image)
    np.save(subimage_dir / "mask.npy", mask)
    bin_mask = (mask > 0).any(axis=0).astype(np.float32) if mask.ndim == 3 else (mask > 0).astype(np.float32)
    np.save(subimage_dir / "binary_mask.npy", bin_mask)
    if sdf is not None:
        np.save(subimage_dir / "sdf.npy", sdf)
    if hv_map is not None:
        np.save(subimage_dir / "hv_map.npy", hv_map)
    if target_3class is not None:
        np.save(subimage_dir / "target_3class.npy", target_3class)
    if target_4class is not None:
        np.save(subimage_dir / "target_4class.npy", target_4class)


def pad_if_smaller(image: np.ndarray, mask: np.ndarray, patch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pads image (with 255) and mask (with 0) if dimensions are smaller than patch_size."""
    h, w = image.shape[:2]
    out_h = max(h, patch_size)
    out_w = max(w, patch_size)
    if out_h == h and out_w == w:
        return image, mask

    padded_img = np.full((out_h, out_w, 3), 255, dtype=image.dtype)
    padded_img[:h, :w] = image

    padded_mask = np.zeros((mask.shape[0], out_h, out_w), dtype=mask.dtype)
    padded_mask[:, :h, :w] = mask

    return padded_img, padded_mask


def get_patch_starts(dim_size: int, patch_size: int) -> List[int]:
    """
    Derives patch starting coordinates with minimal uniform overlap to span dim_size.
    Formula:
      n = ceil(dim_size / patch_size)  # number of patches
      step = ceil(dim_size - patch_size) / (n - 1)  # uniform step between starts
      starts = [round(i * step) for i in range(n)]
    """
    if dim_size <= patch_size:
        return [0]

    n = int(np.ceil(dim_size / patch_size))
    if n <= 1:
        return [0]

    max_start = dim_size - patch_size
    step = np.ceil(max_start) / (n - 1)
    starts = [int(round(i * step)) for i in range(n)]
    return sorted(list(dict.fromkeys(starts)))


def merge_to_unified_instance_mask(mask_4ch: np.ndarray) -> np.ndarray:
    """Combines 4-channel instance mask [4, H, W] into a single [H, W] instance mask with unique global IDs."""
    h, w = mask_4ch.shape[1:3]
    unified = np.zeros((h, w), dtype=np.int32)
    current_max_id = 0

    for ch_idx in range(mask_4ch.shape[0]):
        ch_data = mask_4ch[ch_idx]
        unique_ids = np.unique(ch_data[ch_data > 0])
        for uid in unique_ids:
            current_max_id += 1
            unified[ch_data == uid] = current_max_id

    return unified


def compute_instance_sdf(instance_mask: np.ndarray) -> np.ndarray:
    """
    Computes per-instance Euclidean signed distance field (SDF) on the full image:
      - Outside distance is positive (> 0) from foreground boundaries.
      - Inside distance is negative (< 0) computed per individual nucleus.
      - Boundary pixels of each nucleus are zero (= 0.0).
    """
    instance_mask = np.asarray(instance_mask)
    h, w = instance_mask.shape[:2]
    sdf = np.zeros((h, w), dtype=np.float32)

    foreground = instance_mask > 0
    if not np.any(foreground):
        return np.full((h, w), float(max(h, w)), dtype=np.float32)

    # 1. Outside positive distance from all foreground boundaries
    outside_dist = distance_transform_edt(~foreground).astype(np.float32)
    sdf[~foreground] = outside_dist[~foreground]

    # 2. Inside negative distance computed per individual instance
    kernel = np.ones((3, 3), dtype=np.uint8)
    slices = find_objects(instance_mask)

    for inst_id, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        obj_crop = (instance_mask[sl] == inst_id)
        if not np.any(obj_crop):
            continue

        inside_crop = distance_transform_edt(obj_crop).astype(np.float32)

        # Inner 1-pixel boundary of this instance
        eroded = cv2.erode(obj_crop.astype(np.uint8), kernel, iterations=1)
        boundary_crop = obj_crop & (eroded == 0)

        sdf_crop = sdf[sl]
        # Assign negative inside distance
        sdf_crop[obj_crop] = -inside_crop[obj_crop]
        # Set boundary pixels to exactly 0.0
        sdf_crop[boundary_crop] = 0.0
        sdf[sl] = sdf_crop

    return sdf.astype(np.float32)


def compute_4class_target_with_bottom_hat(
    instance_mask: np.ndarray,
    kernel_size: int = 5,
    boundary_width: int = 1,
) -> np.ndarray:
    """
    Computes a 4-channel target map using per-instance boundary/interior extraction
    and Bottom-Hat transform for inter-cellular gap cleft isolation:
      0: Background
      1: Interior (Core seed)
      2: Boundary (Cell contour ring)
      3: Gap (Inter-cellular clefts/valleys via Bottom-Hat: Closing(M) - M)

    Args:
        instance_mask: [H, W] (or [4, H, W]) integer instance mask.
        kernel_size: Structuring element diameter for bottom-hat transform (default: 5).
        boundary_width: Width of instance contour boundary in pixels (default: 1).

    Returns:
        target_4class: [H, W] integer array with values in {0, 1, 2, 3}.
    """
    if instance_mask.ndim == 3:
        if instance_mask.shape[0] == 4:
            unified = np.zeros(instance_mask.shape[1:], dtype=np.int32)
            for c in range(4):
                c_mask = instance_mask[c]
                unified[c_mask > 0] = c_mask[c_mask > 0]
            instance_mask = unified
        else:
            instance_mask = instance_mask.squeeze(0)

    h, w = instance_mask.shape[:2]
    target = np.zeros((h, w), dtype=np.int64)

    bin_mask = (instance_mask > 0).astype(np.uint8)
    if not np.any(bin_mask):
        return target

    # 1. Bottom-Hat Transform on binary mask: BTH(M) = Closing(M) - M
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    bottom_hat = cv2.morphologyEx(bin_mask, cv2.MORPH_BLACKHAT, kernel)
    gap_mask = (bottom_hat > 0)

    # 2. Add touching contact seams between distinct instances to gap_mask
    slices = find_objects(instance_mask)
    if len(slices) > 1:
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            y_src = slice(max(0, -dy), min(h, h - dy))
            x_src = slice(max(0, -dx), min(w, w - dx))
            y_dst = slice(max(0, dy), min(h, h + dy))
            x_dst = slice(max(0, dx), min(w, w + dx))

            src_vals = instance_mask[y_src, x_src]
            dst_vals = instance_mask[y_dst, x_dst]

            seam = (src_vals > 0) & (dst_vals > 0) & (src_vals != dst_vals)
            gap_mask[y_src, x_src] |= seam

    # 3. For each instance, compute interior (class 1) and boundary (class 2) per-instance
    k_erode = np.ones((3, 3), np.uint8) if boundary_width == 1 else cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * boundary_width + 1, 2 * boundary_width + 1))

    for inst_id, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        obj_crop = (instance_mask[sl] == inst_id)
        if not np.any(obj_crop):
            continue

        target_sl = target[sl]

        if boundary_width > 0:
            eroded = cv2.erode(obj_crop.astype(np.uint8), k_erode, iterations=1)
            boundary = obj_crop & (eroded == 0)

            # Degeneracy check: ensure small instances retain at least 1 interior seed pixel
            if not np.any(eroded):
                edt = distance_transform_edt(obj_crop)
                peak_y, peak_x = np.unravel_index(np.argmax(edt), obj_crop.shape)
                eroded[peak_y, peak_x] = 1
                boundary[peak_y, peak_x] = False

            target_sl[boundary] = 2
            target_sl[eroded > 0] = 1
        else:
            target_sl[obj_crop] = 1

        target[sl] = target_sl

    # 4. Gaps take priority at contact/valley locations (class 3)
    target[gap_mask] = 3

    # Ensure background is 0
    target[(instance_mask == 0) & (~gap_mask)] = 0

    return target


def compute_3class_target(
    instance_mask: np.ndarray,
    boundary_width: int = 1,
) -> np.ndarray:
    """
    Computes a 3-channel target map using per-instance boundary/interior extraction:
      0: Background
      1: Interior (Core seed)
      2: Boundary (Instance contour ring)

    Args:
        instance_mask: [H, W] integer instance mask (0 = bg, 1..N = instance IDs),
                       or [4, H, W] 4-channel instance mask.
        boundary_width: Width of instance contour boundary in pixels (default: 1).

    Returns:
        target_3class: [H, W] int64 array with values in {0, 1, 2}.
    """
    if instance_mask.ndim == 3:
        if instance_mask.shape[0] == 4:
            unified = np.zeros(instance_mask.shape[1:], dtype=np.int32)
            current_max_id = 0
            for c in range(4):
                c_mask = instance_mask[c]
                unique_ids = np.unique(c_mask[c_mask > 0])
                for uid in unique_ids:
                    current_max_id += 1
                    unified[c_mask == uid] = current_max_id
            instance_mask = unified
        else:
            instance_mask = instance_mask.squeeze(0)

    h, w = instance_mask.shape[:2]
    target = np.zeros((h, w), dtype=np.int64)

    bin_mask = (instance_mask > 0).astype(np.uint8)
    if not np.any(bin_mask):
        return target

    slices = find_objects(instance_mask)
    k_erode = np.ones((3, 3), np.uint8) if boundary_width == 1 else cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * boundary_width + 1, 2 * boundary_width + 1))

    for inst_id, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        obj_crop = (instance_mask[sl] == inst_id)
        if not np.any(obj_crop):
            continue

        target_sl = target[sl]

        if boundary_width > 0:
            eroded = cv2.erode(obj_crop.astype(np.uint8), k_erode, iterations=1)
            boundary = obj_crop & (eroded == 0)

            # Degeneracy check: ensure small instances retain at least 1 interior seed pixel
            if not np.any(eroded):
                edt = distance_transform_edt(obj_crop)
                peak_y, peak_x = np.unravel_index(np.argmax(edt), obj_crop.shape)
                eroded[peak_y, peak_x] = 1
                boundary[peak_y, peak_x] = False

            target_sl[boundary] = 2
            target_sl[eroded > 0] = 1
        else:
            target_sl[obj_crop] = 1

        target[sl] = target_sl

    target[instance_mask == 0] = 0
    return target


def compute_skeleton_distance_and_hv(
    instance_mask: np.ndarray,
    normalize_per_cell: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the skeleton of each individual instance mask, calculates the Euclidean
    distance of each pixel inside the instance to that instance's skeleton, and derives
    the horizontal (Gx) and vertical (Gy) gradients.

    Args:
        instance_mask: [H, W] integer instance mask (0 = background, 1..N = instance IDs),
                       or [C, H, W] multi-channel instance mask.
        normalize_per_cell: If True, normalizes gradient magnitude per nucleus to [-1, 1].

    Returns:
        (dist_map, gx, gy):
            dist_map: [H, W] (or [C, H, W]) float32 distance to skeleton (0 on background).
            gx: [H, W] (or [C, H, W]) float32 horizontal gradient in [-1.0, 1.0].
            gy: [H, W] (or [C, H, W]) float32 vertical gradient in [-1.0, 1.0].
    """
    if instance_mask.ndim == 3:
        dist_map = np.zeros_like(instance_mask, dtype=np.float32)
        gx = np.zeros_like(instance_mask, dtype=np.float32)
        gy = np.zeros_like(instance_mask, dtype=np.float32)
        for c in range(instance_mask.shape[0]):
            dist_map[c], gx[c], gy[c] = compute_skeleton_distance_and_hv(
                instance_mask[c],
                normalize_per_cell=normalize_per_cell,
            )
        return dist_map, gx, gy

    h, w = instance_mask.shape[:2]
    dist_map = np.zeros((h, w), dtype=np.float32)
    gx = np.zeros((h, w), dtype=np.float32)
    gy = np.zeros((h, w), dtype=np.float32)

    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids > 0]
    if len(unique_ids) == 0:
        return dist_map, gx, gy

    slices = find_objects(instance_mask)
    for inst_id in unique_ids:
        sl = slices[inst_id - 1] if (inst_id - 1 < len(slices)) else None
        if sl is None:
            inst_crop = (instance_mask == inst_id)
            if not np.any(inst_crop):
                continue
            y_coords, x_coords = np.where(inst_crop)
            y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords)) + 1
            x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords)) + 1
            sl = (slice(y_min, y_max), slice(x_min, x_max))
            inst_crop = (instance_mask[sl] == inst_id)
        else:
            inst_crop = (instance_mask[sl] == inst_id)
            if not np.any(inst_crop):
                continue

        # Extract skeleton of instance
        skel = skeletonize(inst_crop)
        if not np.any(skel):
            # Fallback for tiny/degenerate instances (1-2 pixels)
            edt = distance_transform_edt(inst_crop)
            center = np.unravel_index(np.argmax(edt), inst_crop.shape)
            skel[center] = True

        # Compute Euclidean distance from each pixel to skeleton
        dist_c = distance_transform_edt(~skel).astype(np.float32)

        # Gradients of the distance field
        if dist_c.shape[0] < 2 or dist_c.shape[1] < 2:
            dist_pad = np.pad(dist_c, ((1, 1), (1, 1)), mode="edge")
            gy_pad, gx_pad = np.gradient(dist_pad)
            gy_c = gy_pad[1:-1, 1:-1]
            gx_c = gx_pad[1:-1, 1:-1]
        else:
            gy_c, gx_c = np.gradient(dist_c)

        # Assign distance inside the instance
        d_sub = dist_map[sl]
        d_sub[inst_crop] = dist_c[inst_crop]
        dist_map[sl] = d_sub

        # Assign gradients inside the instance
        gx_sub = gx[sl]
        gy_sub = gy[sl]
        if normalize_per_cell:
            max_v = max(1e-7, float(np.max(np.abs(gx_c[inst_crop]))), float(np.max(np.abs(gy_c[inst_crop]))))
            gx_sub[inst_crop] = np.clip(gx_c[inst_crop] / max_v, -1.0, 1.0)
            gy_sub[inst_crop] = np.clip(gy_c[inst_crop] / max_v, -1.0, 1.0)
        else:
            gx_sub[inst_crop] = np.clip(gx_c[inst_crop], -1.0, 1.0)
            gy_sub[inst_crop] = np.clip(gy_c[inst_crop], -1.0, 1.0)
        gx[sl] = gx_sub
        gy[sl] = gy_sub

    return dist_map, gx, gy


def compute_skeleton_distance_map(
    instance_mask: np.ndarray,
) -> np.ndarray:
    """
    Computes Euclidean distance of each pixel to the instance's skeleton.
    Distance is 0 at the skeleton and outside instances, and positive inside instances.
    """
    dist_map, _, _ = compute_skeleton_distance_and_hv(instance_mask)
    return dist_map


def extract_inside_distance(
    sdf: np.ndarray,
    instance_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extracts only the inside-cell distance transform from the signed distance field (SDF).
    Inside distance is positive (>= 0), reaching peak at cell center and 0 at cell boundary.
    Outside cell distance is 0.
    """
    inside_dist = np.maximum(-sdf, 0.0).astype(np.float32)
    if instance_mask is not None:
        inside_dist = np.where(instance_mask > 0, inside_dist, 0.0).astype(np.float32)
    return inside_dist


def compute_inside_distance_map(
    instance_mask: np.ndarray,
) -> np.ndarray:
    """
    Computes the inside-cell Euclidean distance transform for individual instances:
    For each cell instance, calculates the distance of each cell pixel to the cell boundary
    (distance is 0 outside the cell, >= 0 inside reaching peak at the cell medial axis/center).

    Args:
        instance_mask: [H, W] integer instance mask (0 = background, 1..N = instance IDs),
                       or [C, H, W] multi-channel instance mask.

    Returns:
        dist_map: [H, W] (or [C, H, W]) float32 distance map.
    """
    dist_map, _, _ = compute_inside_distance_and_hv(instance_mask)
    return dist_map


def compute_centroid_distance_map(
    instance_mask: np.ndarray,
) -> np.ndarray:
    """
    Computes a distance map from individual nuclei centroids:
    1. First calculates the geometric centroid (cy, cx) of each individual nucleus in the mask.
    2. Then, for each cell pixel belonging to a nucleus instance, calculates its Euclidean
       distance to that nucleus's centroid: d = sqrt((y - cy)^2 + (x - cx)^2).
    Background pixels are set to 0.0.

    Args:
        instance_mask: [H, W] integer instance mask (0 = background, 1..N = instance IDs),
                       or [C, H, W] multi-channel instance mask.

    Returns:
        dist_map: [H, W] (or [C, H, W]) float32 distance map.
    """
    if instance_mask.ndim == 3:
        dist_map = np.zeros_like(instance_mask, dtype=np.float32)
        for c in range(instance_mask.shape[0]):
            dist_map[c] = compute_centroid_distance_map(instance_mask[c])
        return dist_map

    h, w = instance_mask.shape[:2]
    dist_map = np.zeros((h, w), dtype=np.float32)

    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids > 0]
    if len(unique_ids) == 0:
        return dist_map

    slices = find_objects(instance_mask)
    for inst_id in unique_ids:
        sl = slices[inst_id - 1] if (inst_id - 1 < len(slices)) else None
        if sl is None:
            inst_crop = (instance_mask == inst_id)
            if not np.any(inst_crop):
                continue
            y_coords, x_coords = np.where(inst_crop)
            cy = float(np.mean(y_coords))
            cx = float(np.mean(x_coords))
            dists = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)
            dist_map[y_coords, x_coords] = dists
        else:
            inst_crop = (instance_mask[sl] == inst_id)
            if not np.any(inst_crop):
                continue
            y_local, x_local = np.where(inst_crop)
            cy_local = float(np.mean(y_local))
            cx_local = float(np.mean(x_local))
            dists = np.sqrt((y_local - cy_local) ** 2 + (x_local - cx_local) ** 2)
            dist_crop = dist_map[sl]
            dist_crop[y_local, x_local] = dists.astype(np.float32)
            dist_map[sl] = dist_crop

    return dist_map.astype(np.float32)


def compute_inside_distance_and_hv(
    instance_mask: np.ndarray,
    method: str = "sobel",
    sigma: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes inside distance and HV gradients following the exact flow:
    1. Calculate the inside distance for each instance separately.
    2. Apply Gaussian smoothing to remove medial axis jump discontinuities,
       strictly zeroing out background to prevent spatial leakage.
    3. Calculate HV gradients and normalize them to [-1, 1] per instance to
       guarantee consistent dynamic range and eliminate varying instance flow strength.
    4. After doing this for all nuclei in the full size input image, combine them into single maps.

    Args:
        instance_mask: [H, W] (or [C, H, W]) integer instance mask.
        method: 'sobel' (cv2.Sobel with 1/8 normalization) or 'gradient' (central differences).
        sigma: Standard deviation for Gaussian smoothing of the distance transform (default: 1.5).

    Returns:
        (dist_map, gx, gy): Full-size distance map [H, W] and horizontal/vertical gradients [H, W].
    """
    if instance_mask.ndim == 3:
        c, h, w = instance_mask.shape
        full_dist = np.zeros((c, h, w), dtype=np.float32)
        full_gx = np.zeros((c, h, w), dtype=np.float32)
        full_gy = np.zeros((c, h, w), dtype=np.float32)
        for i in range(c):
            full_dist[i], full_gx[i], full_gy[i] = compute_inside_distance_and_hv(
                instance_mask[i], method=method, sigma=sigma
            )
        return full_dist, full_gx, full_gy

    h, w = instance_mask.shape[:2]
    dist_map = np.zeros((h, w), dtype=np.float32)
    gx_map = np.zeros((h, w), dtype=np.float32)
    gy_map = np.zeros((h, w), dtype=np.float32)

    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids > 0]
    if len(unique_ids) == 0:
        return dist_map, gx_map, gy_map

    slices = find_objects(instance_mask)

    for inst_id in unique_ids:
        sl = slices[inst_id - 1] if (inst_id - 1 < len(slices)) else None
        if sl is None:
            inst_crop = (instance_mask == inst_id)
            if not np.any(inst_crop):
                continue
            y_coords, x_coords = np.where(inst_crop)
            y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords)) + 1
            x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords)) + 1
            sl = (slice(y_min, y_max), slice(x_min, x_max))
            inst_crop = (instance_mask[sl] == inst_id)
        else:
            inst_crop = (instance_mask[sl] == inst_id)
            if not np.any(inst_crop):
                continue

        # Border touching checks for this instance
        touch_t = (sl[0].start == 0)
        touch_b = (sl[0].stop >= h)
        touch_l = (sl[1].start == 0)
        touch_r = (sl[1].stop >= w)

        pad_t = 0 if touch_t else 1
        pad_b = 0 if touch_b else 1
        pad_l = 0 if touch_l else 1
        pad_r = 0 if touch_r else 1

        # 1. Calculate inside distance for each instance separately
        m_pad = np.pad(inst_crop, ((pad_t, pad_b), (pad_l, pad_r)), mode="constant", constant_values=0)
        d_pad = distance_transform_edt(m_pad).astype(np.float32)

        end_y = d_pad.shape[0] - pad_b if pad_b > 0 else d_pad.shape[0]
        end_x = d_pad.shape[1] - pad_r if pad_r > 0 else d_pad.shape[1]
        dist_inst = d_pad[pad_t:end_y, pad_l:end_x]

        # 2. Smooth distance transform to remove medial axis jump discontinuities,
        # but strictly constrain to foreground (zero out any background leakage).
        if sigma > 0:
            smooth_mode = "nearest" if (touch_t or touch_b or touch_l or touch_r) else "constant"
            d_smooth = gaussian_filter(dist_inst, sigma=sigma, mode=smooth_mode, cval=0.0)
            d_smooth[~inst_crop] = 0.0
        else:
            d_smooth = dist_inst.copy()

        # 3. Calculate HV gradients for each instance separately
        d_sobel_pad = np.pad(d_smooth, 1, mode="constant", constant_values=0.0)
        if touch_t:
            d_sobel_pad[0, :] = d_sobel_pad[1, :]
        if touch_b:
            d_sobel_pad[-1, :] = d_sobel_pad[-2, :]
        if touch_l:
            d_sobel_pad[:, 0] = d_sobel_pad[:, 1]
        if touch_r:
            d_sobel_pad[:, -1] = d_sobel_pad[:, -2]

        if method == "sobel":
            gx_crop = cv2.Sobel(d_sobel_pad, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0)[1:-1, 1:-1]
            gy_crop = cv2.Sobel(d_sobel_pad, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0)[1:-1, 1:-1]
        else:
            gy_p, gx_p = np.gradient(d_sobel_pad)
            gx_crop = gx_p[1:-1, 1:-1]
            gy_crop = gy_p[1:-1, 1:-1]

        # Strictly enforce zero background gradient
        gx_crop[~inst_crop] = 0.0
        gy_crop[~inst_crop] = 0.0

        # Normalize gradients to [-1, 1] range per instance to avoid varying strength
        max_gx = float(np.max(np.abs(gx_crop[inst_crop]))) if np.any(inst_crop) else 0.0
        max_gy = float(np.max(np.abs(gy_crop[inst_crop]))) if np.any(inst_crop) else 0.0
        if max_gx > 1e-7:
            gx_crop[inst_crop] /= max_gx
        if max_gy > 1e-7:
            gy_crop[inst_crop] /= max_gy

        gx_crop[~inst_crop] = 0.0
        gy_crop[~inst_crop] = 0.0

        # 4. Combine into single maps
        dist_map[sl][inst_crop] = dist_inst[inst_crop]
        gx_map[sl][inst_crop] = gx_crop[inst_crop]
        gy_map[sl][inst_crop] = gy_crop[inst_crop]

    gx_map = np.clip(gx_map, -1.0, 1.0)
    gy_map = np.clip(gy_map, -1.0, 1.0)
    gx_map[instance_mask == 0] = 0.0
    gy_map[instance_mask == 0] = 0.0

    return dist_map, gx_map.astype(np.float32), gy_map.astype(np.float32)


def compute_hv_gradients(
    distance_map: Optional[np.ndarray] = None,
    instance_mask: Optional[np.ndarray] = None,
    normalize_per_cell: bool = True,
    method: str = "sobel",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes horizontal (Gx) and vertical (Gy) gradients following the 3-step per-instance flow:
    1. Calculate inside distance for each instance separately.
    2. Calculate HV gradients for each instance separately and normalize to [-1, 1] range.
    3. Combine into a single map.

    Args:
        distance_map: Optional [H, W] float array of inside-cell distance map (or [C, H, W]).
        instance_mask: Optional [H, W] integer instance mask (or [C, H, W]). If provided, computes
                       per-instance inside distance and gradients normalized to [-1, 1].
        normalize_per_cell: If True, normalizes gradient magnitude to [-1, 1] per nucleus.
        method: 'sobel' (cv2.Sobel with 1/8 normalization) or 'gradient' (central differences).

    Returns:
        (gx, gy): Horizontal and vertical gradients [H, W] (or [C, H, W]), 0 outside cells.
    """
    if instance_mask is not None:
        _, gx, gy = compute_inside_distance_and_hv(instance_mask, method=method)
        return gx, gy

    if distance_map is None:
        raise ValueError("Either distance_map or instance_mask must be provided.")

    if distance_map.ndim == 3:
        gx = np.zeros_like(distance_map, dtype=np.float32)
        gy = np.zeros_like(distance_map, dtype=np.float32)
        for c in range(distance_map.shape[0]):
            gx[c], gy[c] = compute_hv_gradients(
                distance_map=distance_map[c],
                normalize_per_cell=normalize_per_cell,
                method=method,
            )
        return gx, gy

    h, w = distance_map.shape[:2]
    fg = distance_map > 0
    dist_padded = np.pad(distance_map, 1, mode="edge")
    if method == "sobel":
        gx_pad = cv2.Sobel(dist_padded, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0)
        gy_pad = cv2.Sobel(dist_padded, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0)
    else:
        gy_pad, gx_pad = np.gradient(dist_padded)
    gx = gx_pad[1:-1, 1:-1]
    gy = gy_pad[1:-1, 1:-1]

    gx[~fg] = 0.0
    gy[~fg] = 0.0

    if normalize_per_cell and np.any(fg):
        max_gx = float(np.max(np.abs(gx[fg])))
        max_gy = float(np.max(np.abs(gy[fg])))
        if max_gx > 1e-7:
            gx[fg] /= max_gx
        if max_gy > 1e-7:
            gy[fg] /= max_gy

    gx = np.clip(gx, -1.0, 1.0)
    gy = np.clip(gy, -1.0, 1.0)
    return gx.astype(np.float32), gy.astype(np.float32)


def save_patch_heptaplet(
    dirs: Dict[str, Path],
    patch_name: str,
    image_patch: np.ndarray,
    binary_mask_patch: np.ndarray,
    instance_mask_patch: np.ndarray,
    sdf_patch: np.ndarray,
    target_3class_patch: Optional[np.ndarray] = None,
    target_4class_patch: Optional[np.ndarray] = None,
    hv_patch: Optional[np.ndarray] = None,
) -> None:
    """Saves up to 7 patch representations (image, binary, instance, sdf, target_3class, target_4class, hv_map) as .npy files."""
    img_dir = dirs.get("images")
    bin_dir = dirs.get("binary_masks")
    mask_dir = dirs.get("instance_masks", dirs.get("instance masks"))
    sdf_dir = dirs.get("sdfs", dirs.get("signed_distance_function_maps"))
    t3_dir = dirs.get("targets_3class")
    t4_dir = dirs.get("targets_4class")
    hv_dir = dirs.get("hv_maps", dirs.get("hv_gradients"))

    if img_dir is not None and image_patch is not None:
        np.save(img_dir / f"{patch_name}.npy", image_patch)
    if bin_dir is not None and binary_mask_patch is not None:
        np.save(bin_dir / f"{patch_name}.npy", binary_mask_patch)
    if mask_dir is not None and instance_mask_patch is not None:
        np.save(mask_dir / f"{patch_name}.npy", instance_mask_patch)
    if sdf_dir is not None and sdf_patch is not None:
        np.save(sdf_dir / f"{patch_name}.npy", sdf_patch)
    if t3_dir is not None and target_3class_patch is not None:
        np.save(t3_dir / f"{patch_name}.npy", target_3class_patch)
    if t4_dir is not None and target_4class_patch is not None:
        np.save(t4_dir / f"{patch_name}.npy", target_4class_patch)
    if hv_dir is not None and hv_patch is not None:
        np.save(hv_dir / f"{patch_name}.npy", hv_patch)


def save_patch_pentaplet(
    dirs: Dict[str, Path],
    patch_name: str,
    image_patch: np.ndarray,
    binary_mask_patch: np.ndarray,
    mask_patch: np.ndarray,
    sdf_patch: np.ndarray,
    target_4class_patch: Optional[np.ndarray] = None,
    target_3class_patch: Optional[np.ndarray] = None,
    hv_patch: Optional[np.ndarray] = None,
) -> None:
    """Saves patch triplet plus target_4class, target_3class, and/or hv_map."""
    save_patch_heptaplet(
        dirs=dirs,
        patch_name=patch_name,
        image_patch=image_patch,
        binary_mask_patch=binary_mask_patch,
        instance_mask_patch=mask_patch,
        sdf_patch=sdf_patch,
        target_3class_patch=target_3class_patch,
        target_4class_patch=target_4class_patch,
        hv_patch=hv_patch,
    )


def save_patch_triplet(
    dirs: Dict[str, Path],
    patch_name: str,
    image_patch: np.ndarray,
    mask_patch: np.ndarray,
    sdf_patch: np.ndarray,
) -> None:
    """Saves image, 4-channel instance mask, and signed distance map patches as .npy files."""
    sdf_dir = dirs.get("sdfs", dirs.get("signed_distance_function_maps"))
    mask_dir = dirs.get("instance_masks", dirs.get("instance masks"))
    img_dir = dirs.get("images")
    bin_dir = dirs.get("binary_masks")
    np.save(img_dir / f"{patch_name}.npy", image_patch)
    if bin_dir is not None:
        bin_patch = (mask_patch > 0).astype(np.float32) if mask_patch.ndim == 2 else (mask_patch > 0).any(axis=0).astype(np.float32)
        np.save(bin_dir / f"{patch_name}.npy", bin_patch)
    np.save(mask_dir / f"{patch_name}.npy", mask_patch)
    np.save(sdf_dir / f"{patch_name}.npy", sdf_patch)


def save_patch_quadruplet(
    dirs: Dict[str, Path],
    patch_name: str,
    image_patch: np.ndarray,
    mask_patch: np.ndarray,
    sdf_patch: np.ndarray,
    hv_patch: np.ndarray,
) -> None:
    """Saves image, 4-channel instance mask, signed distance map, and HV maps patches as .npy files."""
    save_patch_triplet(dirs, patch_name, image_patch, mask_patch, sdf_patch)
    hv_dir = dirs.get("hv_maps", dirs.get("hv_gradients", None))
    if hv_dir is not None:
        np.save(hv_dir / f"{patch_name}.npy", hv_patch)


def colorize_instance_mask(instance_mask: np.ndarray) -> np.ndarray:
    """Assigns distinct random colors to instances for visualization."""
    h, w = instance_mask.shape[:2]
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids > 0]
    if len(unique_ids) == 0:
        return colored

    np.random.seed(42)
    colors = np.random.randint(60, 255, size=(int(unique_ids.max()) + 1, 3), dtype=np.uint8)
    for inst_id in unique_ids:
        colored[instance_mask == inst_id] = colors[inst_id]
    return colored


def create_boundary_overlay(image: np.ndarray, instance_mask: np.ndarray) -> np.ndarray:
    """Overlays instance boundaries on top of the original image in yellow."""
    kernel = np.ones((3, 3), np.uint8)
    overlay = image.copy()
    if overlay.dtype != np.uint8:
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    unique_ids = np.unique(instance_mask)
    unique_ids = unique_ids[unique_ids > 0]
    boundary_total = np.zeros(instance_mask.shape[:2], dtype=bool)

    for inst_id in unique_ids:
        inst_mask = (instance_mask == inst_id).astype(np.uint8)
        eroded = cv2.erode(inst_mask, kernel, iterations=1)
        boundary_total |= (inst_mask == 1) & (eroded == 0)

    overlay[boundary_total] = [255, 255, 0]
    return overlay


def create_combined_hv_rgb(
    gx: np.ndarray,
    gy: np.ndarray,
    mask: Optional[np.ndarray] = None,
    bg_color: str = "white",
) -> np.ndarray:
    """
    Creates an RGB image visualizing 2D flow vectors (Gx, Gy) using the standard optical flow HSV color wheel:
      - Hue: encodes vector flow direction / angle theta = arctan2(Gy, Gx) in [0, 2*pi] mapped to [0, 1]
      - Saturation: 1.0 inside foreground (vivid saturated color)
      - Value: 1.0 inside foreground (pure vivid color showing direction)
      - Background: white or black depending on bg_color.
    """
    import matplotlib.colors as mcolors

    mag = np.sqrt(gx**2 + gy**2)
    # Angle theta in [0, 2*pi]
    ang = (np.arctan2(gy, gx) + 2 * np.pi) % (2 * np.pi)
    hue = ang / (2 * np.pi)

    if mask is None:
        fg = (mag > 1e-4)
    else:
        fg = (mask > 0)

    sat = np.ones_like(hue)
    val = np.ones_like(hue)

    hsv = np.stack([hue, sat, val], axis=-1)
    rgb = (mcolors.hsv_to_rgb(hsv) * 255.0).astype(np.uint8)

    if bg_color == "white":
        rgb[~fg] = [255, 255, 255]
    else:
        rgb[~fg] = [0, 0, 0]

    return rgb


def create_3channel_rgb(target_3class: np.ndarray) -> np.ndarray:
    """Converts 2D 3-class target [H, W] into RGB image using semantic palette."""
    COLOR_3CH = np.array([
        [25, 30, 45],       # 0: Background
        [30, 160, 240],     # 1: Interior Core (Cyan)
        [50, 220, 90],      # 2: Boundary (Lime Green)
    ], dtype=np.uint8)
    rgb = np.zeros((target_3class.shape[0], target_3class.shape[1], 3), dtype=np.uint8)
    for c in range(min(3, COLOR_3CH.shape[0])):
        rgb[target_3class == c] = COLOR_3CH[c]
    return rgb


def create_4channel_rgb(target_4class: np.ndarray) -> np.ndarray:
    """Converts 2D 4-class target [H, W] into RGB image using semantic palette."""
    COLOR_4CH = np.array([
        [25, 30, 45],       # 0: Background
        [30, 160, 240],     # 1: Interior Core (Cyan)
        [50, 220, 90],      # 2: Boundary (Lime Green)
        [245, 35, 45],      # 3: Gap Cleft (Bright Red)
    ], dtype=np.uint8)
    rgb = np.zeros((target_4class.shape[0], target_4class.shape[1], 3), dtype=np.uint8)
    for c in range(min(4, COLOR_4CH.shape[0])):
        rgb[target_4class == c] = COLOR_4CH[c]
    return rgb


def overlay_4channel_on_rgb(image_rgb: np.ndarray, target_4class: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Overlays 4-channel semantic mask onto RGB image with alpha blending."""
    color_map = create_4channel_rgb(target_4class)
    fg = (target_4class > 0)
    blended = image_rgb.copy()
    blended[fg] = (blended[fg] * (1.0 - alpha) + color_map[fg] * alpha).astype(np.uint8)
    return blended


def save_full_image_preview(
    preview_path: Path,
    image: np.ndarray,
    instance_mask: np.ndarray,
    sdf: np.ndarray,
    dist_map: Optional[np.ndarray] = None,
    gx: Optional[np.ndarray] = None,
    gy: Optional[np.ndarray] = None,
    target_3class: Optional[np.ndarray] = None,
    target_4class: Optional[np.ndarray] = None,
    binary_mask: Optional[np.ndarray] = None,
    cell_class_map: Optional[np.ndarray] = None,
    title: str = "",
) -> None:
    """
    Saves a comprehensive 12-panel PNG preview of the WHOLE SIZE image, masks, targets, and flows:
      Row 1: Primary Spatial Representations
        - 1. Full RGB Image
        - 2. Full Binary Mask (Foreground vs Background)
        - 3. Full Instance Mask (Colorized with instance count)
        - 4. Boundary Overlay (Yellow Contours on RGB)
      Row 2: Topological & J-Regularization Target Maps
        - 5. Signed Distance Field (SDF) Map (bwr colormap)
        - 6. Inside Cell Distance Transform (viridis colormap)
        - 7. 3-Class J-Reg Target Map (Background, Interior, Boundary)
        - 8. 4-Class J-Reg Target Map (Background, Interior, Boundary, Bottom-Hat Gap)
      Row 3: Gradient Flow Fields & Semantic Overlays
        - 9. Horizontal Flow Field (Gx, Sobel X, bwr colormap)
        - 10. Vertical Flow Field (Gy, Sobel Y, bwr colormap)
        - 11. Combined 2D Flow Field (HSV Color Wheel: Direction via Hue)
        - 12. Cell Phenotypes (or 4-Class Semantic) Overlay on Image
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    h, w = image.shape[:2]
    if binary_mask is None:
        binary_mask = (instance_mask > 0).astype(np.uint8)

    # Derive dist_map, gx, gy if not provided
    if dist_map is None or gx is None or gy is None:
        d_map, g_x, g_y = compute_inside_distance_and_hv(instance_mask, method="sobel")
        if dist_map is None:
            dist_map = d_map
        if gx is None:
            gx = g_x
        if gy is None:
            gy = g_y

    # Derive target_3class if not provided
    if target_3class is None:
        target_3class = compute_3class_target(instance_mask)

    # Derive target_4class if not provided
    if target_4class is None:
        target_4class = compute_4class_target_with_bottom_hat(instance_mask)

    fig, axes = plt.subplots(3, 4, figsize=(24, 16.5), dpi=150)

    if title:
        fig.suptitle(f"Whole-Size Image & Ground Truth Multi-Representation Preview: {title}", fontsize=15, fontweight="bold", y=0.99)

    # -------------------------------------------------------------
    # ROW 1: PRIMARY SPATIAL MASKS
    # -------------------------------------------------------------
    # 1. Full RGB Image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f"1. Full RGB Image ({h}x{w})", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    # 2. Binary Mask
    axes[0, 1].imshow(binary_mask, cmap="gray", vmin=0, vmax=1)
    fg_pct = float(np.mean(binary_mask > 0) * 100)
    axes[0, 1].set_title(f"2. Binary Mask (FG={fg_pct:.1f}%)", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    # 3. Instance Mask
    colored_mask = colorize_instance_mask(instance_mask)
    n_inst = len(np.unique(instance_mask[instance_mask > 0]))
    axes[0, 2].imshow(colored_mask)
    axes[0, 2].set_title(f"3. Full Instance Mask ({n_inst} instances)", fontsize=11, fontweight="bold")
    axes[0, 2].axis("off")

    # 4. Boundary Overlay
    overlay = create_boundary_overlay(image, instance_mask)
    axes[0, 3].imshow(overlay)
    axes[0, 3].set_title("4. Boundary Overlay (Yellow Contours)", fontsize=11, fontweight="bold")
    axes[0, 3].axis("off")

    # -------------------------------------------------------------
    # ROW 2: TOPOLOGICAL & J-REGULARIZATION MAPS
    # -------------------------------------------------------------
    # 5. SDF Map
    max_sdf = max(1.0, float(np.max(np.abs(sdf))))
    im_sdf = axes[1, 0].imshow(sdf, cmap="bwr", vmin=-max_sdf, vmax=max_sdf)
    axes[1, 0].set_title(f"5. Signed Distance Field [{sdf.min():.1f}, {sdf.max():.1f}]", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")
    fig.colorbar(im_sdf, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 6. Inside Cell Distance Transform
    max_dt = float(np.max(dist_map)) if dist_map is not None and dist_map.size > 0 else 1.0
    im_dt = axes[1, 1].imshow(dist_map, cmap="viridis", vmin=0, vmax=max(1.0, max_dt))
    axes[1, 1].set_title(f"6. Inside Distance Transform (max={max_dt:.1f} px)", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")
    fig.colorbar(im_dt, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # 7. 3-Class Target Map
    rgb_3ch = create_3channel_rgb(target_3class)
    axes[1, 2].imshow(rgb_3ch)
    axes[1, 2].set_title("7. 3-Class Target (BG, Inside, Boundary)", fontsize=11, fontweight="bold")
    axes[1, 2].axis("off")
    legend_3ch = [
        Patch(facecolor="#191e2d", edgecolor="gray", label="0: Background"),
        Patch(facecolor="#1ea0f0", edgecolor="none", label="1: Interior"),
        Patch(facecolor="#32dc5a", edgecolor="none", label="2: Boundary"),
    ]
    axes[1, 2].legend(handles=legend_3ch, loc="lower right", fontsize=8, framealpha=0.85)

    # 8. 4-Class Target Map (with Gap)
    rgb_4ch = create_4channel_rgb(target_4class)
    axes[1, 3].imshow(rgb_4ch)
    n_gap_pixels = int(np.sum(target_4class == 3))
    axes[1, 3].set_title(f"8. 4-Class Target (Gap Clefts={n_gap_pixels} px)", fontsize=11, fontweight="bold")
    axes[1, 3].axis("off")
    legend_4ch = [
        Patch(facecolor="#191e2d", edgecolor="gray", label="0: Background"),
        Patch(facecolor="#1ea0f0", edgecolor="none", label="1: Interior"),
        Patch(facecolor="#32dc5a", edgecolor="none", label="2: Boundary"),
        Patch(facecolor="#f5232d", edgecolor="none", label="3: Gap Cleft"),
    ]
    axes[1, 3].legend(handles=legend_4ch, loc="lower right", fontsize=8, framealpha=0.85)

    # -------------------------------------------------------------
    # ROW 3: FLOW FIELDS & SEMANTIC OVERLAYS
    # -------------------------------------------------------------
    # 9. Horizontal Flow (Gx)
    im_gx = axes[2, 0].imshow(gx, cmap="bwr", vmin=-1.0, vmax=1.0)
    axes[2, 0].set_title("9. Horizontal Flow Gx [-1.0, +1.0]", fontsize=11, fontweight="bold")
    axes[2, 0].axis("off")
    fig.colorbar(im_gx, ax=axes[2, 0], fraction=0.046, pad=0.04)

    # 10. Vertical Flow (Gy)
    im_gy = axes[2, 1].imshow(gy, cmap="bwr", vmin=-1.0, vmax=1.0)
    axes[2, 1].set_title("10. Vertical Flow Gy [-1.0, +1.0]", fontsize=11, fontweight="bold")
    axes[2, 1].axis("off")
    fig.colorbar(im_gy, ax=axes[2, 1], fraction=0.046, pad=0.04)

    # 11. Combined 2D Flow Field (HSV Color Wheel)
    combined_hv = create_combined_hv_rgb(gx, gy, mask=binary_mask > 0, bg_color="white")
    axes[2, 2].imshow(combined_hv)
    axes[2, 2].set_title("11. Combined Flow Direction (HSV Color Wheel)", fontsize=11, fontweight="bold")
    axes[2, 2].axis("off")
    try:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        import matplotlib.colors as mcolors
        ax_wheel = inset_axes(axes[2, 2], width="20%", height="20%", loc="lower right", borderpad=0.4)
        x_g = np.linspace(-1, 1, 80)
        y_g = np.linspace(-1, 1, 80)
        xx, yy = np.meshgrid(x_g, y_g)
        r_cart = np.sqrt(xx**2 + yy**2)
        theta_cart = (np.arctan2(yy, xx) + 2 * np.pi) % (2 * np.pi)
        wheel_h = theta_cart / (2 * np.pi)
        wheel_s = np.clip(r_cart, 0, 1)
        wheel_v = np.ones_like(r_cart)
        wheel_rgb = mcolors.hsv_to_rgb(np.stack([wheel_h, wheel_s, wheel_v], axis=-1))
        wheel_rgb[r_cart > 1.0] = [1.0, 1.0, 1.0]
        ax_wheel.imshow(wheel_rgb)
        ax_wheel.set_title("Direction", fontsize=7.5, pad=1)
        ax_wheel.axis("off")
    except Exception:
        pass

    # 12. Cell Phenotypes (or 4-Class) Overlay on RGB
    if cell_class_map is not None and np.max(cell_class_map) > 0:
        COLOR_MONUSAC = np.array([
            [25, 30, 45],       # 0: BG
            [40, 180, 80],      # 1: Epithelial
            [255, 180, 20],     # 2: Lymphocyte
            [230, 40, 40],      # 3: Neutrophil
            [200, 50, 220],     # 4: Macrophage
        ], dtype=np.uint8)
        cls_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(5):
            cls_rgb[cell_class_map == c] = COLOR_MONUSAC[c]
        alpha = 0.55
        fg = (cell_class_map > 0)
        pheno_overlay = image.copy()
        pheno_overlay[fg] = (pheno_overlay[fg] * (1.0 - alpha) + cls_rgb[fg] * alpha).astype(np.uint8)
        axes[2, 3].imshow(pheno_overlay)
        axes[2, 3].set_title("12. Cell Phenotypes Overlay on RGB", fontsize=11, fontweight="bold")
        legend_pheno = [
            Patch(facecolor="#28b450", edgecolor="none", label="Epithelial"),
            Patch(facecolor="#ffb414", edgecolor="none", label="Lymphocyte"),
            Patch(facecolor="#e62828", edgecolor="none", label="Neutrophil"),
            Patch(facecolor="#c832dc", edgecolor="none", label="Macrophage"),
        ]
        axes[2, 3].legend(handles=legend_pheno, loc="lower right", fontsize=7.5, framealpha=0.85)
    else:
        overlay_4ch = overlay_4channel_on_rgb(image, target_4class, alpha=0.5)
        axes[2, 3].imshow(overlay_4ch)
        axes[2, 3].set_title("12. 4-Class Target Overlaid on RGB", fontsize=11, fontweight="bold")

    axes[2, 3].axis("off")

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(preview_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
