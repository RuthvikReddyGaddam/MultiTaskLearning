# utils/instance_utils.py
"""
Utilities for parsing Aperio XML annotations and generating instance / binary masks.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
import xml.etree.ElementTree as ET


def xml_to_instance_mask(
    xml_path: Union[str, Path],
    height: int,
    width: int,
) -> np.ndarray:
    """
    Parse Aperio XML polygon annotations into an instance mask.

    Args:
        xml_path: Path to the XML annotation file.
        height: Target image height.
        width: Target image width.

    Returns:
        [H, W] uint16 array (0 = background, 1..N = instance IDs).
    """
    instance_mask = np.zeros((height, width), dtype=np.uint16)
    if not os.path.exists(str(xml_path)):
        return instance_mask

    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except Exception as e:
        print(f"[Warning] Failed to parse XML {xml_path}: {e}")
        return instance_mask

    instance_id = 1
    for region in root.iter("Region"):
        points = []
        for vertex in region.iter("Vertex"):
            x = int(round(float(vertex.attrib["X"])))
            y = int(round(float(vertex.attrib["Y"])))
            points.append([x, y])

        if len(points) < 3:
            continue

        pts = np.array(points, dtype=np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)

        cv2.fillPoly(instance_mask, [pts], color=int(instance_id))
        instance_id += 1

    return instance_mask


def xml_to_binary_mask(
    xml_path: Union[str, Path],
    height: int,
    width: int,
) -> np.ndarray:
    """
    Parse Aperio XML polygon annotations directly into a binary mask.

    Args:
        xml_path: Path to the XML annotation file.
        height: Target image height.
        width: Target image width.

    Returns:
        [H, W] uint8 binary array (0 = background, 1 = foreground).
    """
    inst = xml_to_instance_mask(xml_path, height, width)
    return (inst > 0).astype(np.uint8)


def load_instance_mask(
    path_or_dir: Union[str, Path],
    height: int,
    width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ground truth instance mask and corresponding binary mask from various formats:
      1. Directory containing per-instance binary mask images (e.g. 1.png, 2.png, mask_001.png).
      2. 3D numpy array (.npy/.npz) [N_instances, H, W] of stacked binary instance masks.
      3. 2D numpy array (.npy/.npz) [H, W] of integer instance labels.
      4. MATLAB file (.mat) with 'inst_map', 'instance_map', 'mask', or 'labels'.
      5. Image file (.png/.tif/.bmp) with integer instance IDs (or binary mask).
      6. Aperio XML file (.xml) with polygon annotations.

    Returns:
        (instance_mask: np.ndarray [H, W] int32, binary_mask: np.ndarray [H, W] uint8 in {0, 1})
    """
    path = Path(path_or_dir)
    if not path.exists():
        inst_empty = np.zeros((height, width), dtype=np.int32)
        return inst_empty, np.zeros((height, width), dtype=np.uint8)

    # 1. Directory of individual binary instance masks
    if path.is_dir():
        mask_files = sorted([
            f for f in path.iterdir()
            if f.is_file() and f.suffix.lower() in (".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg")
        ])
        if len(mask_files) > 0:
            inst_mask = np.zeros((height, width), dtype=np.int32)
            for inst_id, mf in enumerate(mask_files, start=1):
                try:
                    m = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
                    if m is None:
                        continue
                    if m.shape != (height, width):
                        m = cv2.resize(m, (width, height), interpolation=cv2.INTER_NEAREST)
                    inst_mask[m > 0] = inst_id
                except Exception as e:
                    print(f"[Warning] Failed to read instance mask file {mf}: {e}")
            bin_mask = (inst_mask > 0).astype(np.uint8)
            return inst_mask, bin_mask

    # 2. Aperio XML file
    if path.suffix.lower() == ".xml":
        inst = xml_to_instance_mask(path, height, width).astype(np.int32)
        bin_mask = (inst > 0).astype(np.uint8)
        return inst, bin_mask

    # 3. Numpy array (.npy, .npz)
    if path.suffix.lower() in (".npy", ".npz"):
        try:
            data = np.load(path)
            arr = data["arr_0"] if isinstance(data, np.lib.npyio.NpzFile) else data
            if arr.ndim == 3:  # [N, H, W] stacked binary instance masks
                inst_mask = np.zeros((height, width), dtype=np.int32)
                for inst_id in range(arr.shape[0]):
                    m = arr[inst_id]
                    if m.shape != (height, width):
                        m = cv2.resize(m.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                    inst_mask[m > 0] = inst_id + 1
                bin_mask = (inst_mask > 0).astype(np.uint8)
                return inst_mask, bin_mask
            elif arr.ndim == 2:
                inst_mask = arr.astype(np.int32)
                if inst_mask.shape != (height, width):
                    inst_mask = cv2.resize(inst_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                bin_mask = (inst_mask > 0).astype(np.uint8)
                return inst_mask, bin_mask
        except Exception as e:
            print(f"[Warning] Failed to load numpy instance mask {path}: {e}")

    # 4. MATLAB file (.mat)
    if path.suffix.lower() == ".mat":
        try:
            import scipy.io as sio
            mat = sio.loadmat(str(path))
            for key in ("inst_map", "instance_map", "inst", "mask", "labels", "cell_mask"):
                if key in mat:
                    arr = mat[key]
                    if arr.ndim == 3:
                        inst_mask = np.zeros((height, width), dtype=np.int32)
                        for inst_id in range(arr.shape[0]):
                            inst_mask[arr[inst_id] > 0] = inst_id + 1
                    else:
                        inst_mask = arr.astype(np.int32)
                    if inst_mask.shape != (height, width):
                        inst_mask = cv2.resize(inst_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    bin_mask = (inst_mask > 0).astype(np.uint8)
                    return inst_mask, bin_mask
        except Exception as e:
            print(f"[Warning] Failed to load mat instance mask {path}: {e}")

    # 5. Image file (.png, .tif, .tiff, .bmp)
    try:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is not None:
            if raw.ndim == 3 and raw.shape[-1] in (3, 4):
                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            if raw.shape != (height, width):
                raw = cv2.resize(raw, (width, height), interpolation=cv2.INTER_NEAREST)

            unique_vals = np.unique(raw)
            # If image contains multiple unique instance IDs (>2 values and not standard 0/255/128 grayscale)
            if len(unique_vals) > 2 and not (len(unique_vals) == 3 and 0 in unique_vals and 255 in unique_vals):
                inst_mask = raw.astype(np.int32)
                bin_mask = (inst_mask > 0).astype(np.uint8)
                return inst_mask, bin_mask
            else:
                # Binary mask: extract instances via connected components
                bin_mask = (raw > 0).astype(np.uint8)
                num_labels, labels = cv2.connectedComponents(bin_mask, connectivity=8)
                return labels.astype(np.int32), bin_mask
    except Exception as e:
        print(f"[Warning] Failed to read image mask {path}: {e}")

    inst_empty = np.zeros((height, width), dtype=np.int32)
    return inst_empty, np.zeros((height, width), dtype=np.uint8)


__all__ = [
    "xml_to_instance_mask",
    "xml_to_binary_mask",
    "load_instance_mask",
]
