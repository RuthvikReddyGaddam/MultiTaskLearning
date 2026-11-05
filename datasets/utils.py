# datasets/utils.py

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASS_NAMES = ["Epithelial", "Lymphocyte", "Neutrophil", "Macrophage"]
ALL_CLASS_NAMES = ["Background", "Epithelial", "Lymphocyte", "Neutrophil", "Macrophage"]


class PatchSample(dict):
    """
    Dictionary container for patch sample data supporting:
      1. Dict key access: sample["image"], sample["mask"], sample["sdf"], sample["class_mask"], sample["class_percents"]
      2. Tuple integer indexing: sample[0], sample[1], sample[2]
    """
    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


def load_config(config: Union[Dict[str, Any], str, Path]) -> Dict[str, Any]:
    """Loads a YAML configuration from a path or returns an existing dictionary."""
    if isinstance(config, (str, Path)):
        with open(config, "r") as f:
            return yaml.safe_load(f)
    return config


def resolve_patch_dir(config: Dict[str, Any], dataset_name: str, patch_size: int = 256) -> Path:
    """
    Resolves data_dir from config["dataset"]["data_dir"].
    If specified directory does not exist, falls back to data/{dataset_name}_train_patch_{patch_size}.
    """
    dataset_cfg = config.get("dataset", {})
    cfg_data_dir = dataset_cfg.get("data_dir")
    if cfg_data_dir:
        p = Path(cfg_data_dir)
        if p.exists() and (p / "images").is_dir():
            return p

    # Default fallback to newly created patch dataset
    p_size = dataset_cfg.get("patch_size", dataset_cfg.get("image_size", patch_size))
    default_dir = PROJECT_ROOT / "data" / f"{dataset_name}_train_patch_{p_size}"
    if default_dir.exists():
        return default_dir

    return Path(cfg_data_dir) if cfg_data_dir else default_dir


def load_patch_paths(data_dir: Path) -> List[Tuple[Path, Path, Path]]:
    """
    Finds all valid patch triplets (image, instance_mask, sdf) in data_dir:
      - data_dir/images/<name>.npy
      - data_dir/instance_masks/<name>.npy
      - data_dir/sdfs/<name>.npy
    """
    img_dir = data_dir / "images"
    mask_dir = data_dir / "instance_masks"
    sdf_dir = data_dir / "sdfs"

    image_files = sorted(list(img_dir.glob("*.npy")))
    triplets = []
    for img_p in image_files:
        name = img_p.name
        mask_p = mask_dir / name
        sdf_p = sdf_dir / name
        triplets.append((img_p, mask_p, sdf_p))

    return triplets


def compute_or_load_patch_class_stats(
    data_dir: Path,
    force_recompute: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Computes or loads cached per-patch class statistics and percentages for MoNuSAC:
      - 'patch_name':         name of the patch
      - 'total_pixels':       H * W
      - 'bg_pixels':          background pixel count
      - 'fg_pixels':          foreground cell pixel count
      - 'class_pixels':       [n_epi, n_lym, n_neu, n_mac]
      - 'class_percents_5ch': [bg_pct, epi_pct, lym_pct, neu_pct, mac_pct] (0-100% of patch)
      - 'class_percents':     [epi_pct, lym_pct, neu_pct, mac_pct] (0-100% of patch)
      - 'all_class_percents': dict of {class_name: percent of patch}
      - 'class_ratios':       proportions of each class among foreground nuclei (sums to 1.0)
      - 'classes_present':    list of class indices present in the patch
      - 'has_macrophage':     bool (true if macrophage pixels > 0)
      - 'has_cells':          bool (true if any nucleus present)
    """
    stats_file = data_dir / "patch_class_stats.json"
    if stats_file.exists() and not force_recompute:
        try:
            with open(stats_file, "r") as f:
                return json.load(f)
        except Exception:
            pass

    mask_dir = data_dir / "instance_masks"
    mask_files = sorted(list(mask_dir.glob("*.npy")))
    stats = {}

    for mask_path in mask_files:
        name = mask_path.stem
        # Use mmap_mode to avoid loading entire array into memory
        mask = np.load(mask_path, mmap_mode="r")
        c = mask.shape[0] if mask.ndim == 3 else 1
        h, w = mask.shape[1:3] if mask.ndim == 3 else mask.shape[:2]
        total_pixels = int(h * w)

        if mask.ndim == 3:
            class_pixels = [int(np.count_nonzero(mask[i])) for i in range(c)]
            fg_pixels = int(np.count_nonzero((mask > 0).any(axis=0)))
        else:
            class_pixels = [int(np.count_nonzero(mask))]
            fg_pixels = class_pixels[0]

        bg_pixels = total_pixels - fg_pixels
        bg_percent = (bg_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
        cell_percents = [
            (px / total_pixels) * 100.0 if total_pixels > 0 else 0.0
            for px in class_pixels
        ]
        class_percents_5ch = [bg_percent] + cell_percents

        class_ratios = [
            float(px / fg_pixels) if fg_pixels > 0 else 0.0
            for px in class_pixels
        ]
        classes_present = [i for i, px in enumerate(class_pixels) if px > 0]
        has_macrophage = bool(len(class_pixels) > 3 and class_pixels[3] > 0)

        all_class_percents = {"background": bg_percent}
        for i, c_name in enumerate(CLASS_NAMES):
            all_class_percents[c_name.lower()] = cell_percents[i] if i < len(cell_percents) else 0.0

        stats[name] = {
            "patch_name": name,
            "total_pixels": total_pixels,
            "bg_pixels": bg_pixels,
            "fg_pixels": fg_pixels,
            "class_pixels": class_pixels,
            "class_percents_5ch": class_percents_5ch,
            "class_percents": cell_percents,
            "all_class_percents": all_class_percents,
            "class_ratios": class_ratios,
            "classes_present": classes_present,
            "has_macrophage": has_macrophage,
            "has_cells": bool(fg_pixels > 0),
        }

    # Cache to disk for instant future loads
    try:
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save {stats_file}: {e}")

    return stats


def parse_class_weights(
    class_weights: Union[None, List[float], Dict[str, float], torch.Tensor],
    default_bg_weight: float = 0.05,
) -> Tuple[List[float], float]:
    """
    Normalizes class weights into:
      - cell_weights: 4-element list [w_epi, w_lym, w_neu, w_mac]
      - bg_weight: float weight for background/empty patches
    Supports dict by name ('macrophage': 10.0, 'background': 0.05), list, tensor, or None.
    """
    default_cell_weights = [1.0, 1.0, 1.0, 1.0]
    if class_weights is None:
        return default_cell_weights, default_bg_weight

    bg_weight = default_bg_weight

    if isinstance(class_weights, (list, tuple)):
        vals = [float(w) for w in class_weights]
        if len(vals) == 5:
            # 5-element format: [bg, epi, lym, neu, mac]
            return vals[1:5], vals[0]
        return vals[:4], default_bg_weight

    elif isinstance(class_weights, torch.Tensor):
        vals = [float(w) for w in class_weights.detach().cpu().flatten().tolist()]
        if len(vals) == 5:
            return vals[1:5], vals[0]
        return vals[:4], default_bg_weight

    elif isinstance(class_weights, dict):
        cell_weights = list(default_cell_weights)
        alias_map = {
            "epithelial": 0, "epi": 0, "0": 0,
            "lymphocyte": 1, "lym": 1, "1": 1,
            "neutrophil": 2, "neu": 2, "2": 2,
            "macrophage": 3, "mac": 3, "3": 3,
        }
        bg_keys = {"background", "bg", "empty", "0_bg"}

        for k, v in class_weights.items():
            k_clean = str(k).lower().strip()
            if k_clean in alias_map:
                cell_weights[alias_map[k_clean]] = float(v)
            elif k_clean in bg_keys:
                bg_weight = float(v)
        return cell_weights, bg_weight

    return default_cell_weights, default_bg_weight


def assign_patch_hierarchical_class(stat: Dict[str, Any]) -> int:
    """
    Hierarchical single-class assignment for a patch:
      - Class 3 (Macrophage): If patch contains >= 1 macrophage nucleus (pixels > 0)
      - Class 2 (Neutrophil): Else if patch contains >= 1 neutrophil nucleus (pixels > 0)
      - Else majority between Class 0 (Epithelial) and Class 1 (Lymphocyte)
      - If no nuclei are present at all, returns -1 (Background / Empty)
    
    Returns integer in {-1, 0, 1, 2, 3}:
      -1: Background / No cells
       0: Epithelial
       1: Lymphocyte
       2: Neutrophil
       3: Macrophage
    """
    if not stat or not stat.get("has_cells", False):
        return -1

    class_pixels = stat.get("class_pixels", [])
    # 1. Macrophage priority (Index 3)
    if len(class_pixels) > 3 and class_pixels[3] > 0:
        return 3

    # 2. Neutrophil priority (Index 2)
    if len(class_pixels) > 2 and class_pixels[2] > 0:
        return 2

    # 3. Majority between Epithelial (0) and Lymphocyte (1)
    epi_px = class_pixels[0] if len(class_pixels) > 0 else 0
    lym_px = class_pixels[1] if len(class_pixels) > 1 else 0

    if epi_px == 0 and lym_px == 0:
        # Check if any remaining pixel is present
        classes_present = stat.get("classes_present", [])
        if classes_present:
            return classes_present[0]
        return -1

    return 0 if epi_px >= lym_px else 1


def compute_patch_sampling_weights(
    samples: List[Tuple[Path, Path, Path]],
    patch_stats: Dict[str, Dict[str, Any]],
    class_weights: Union[List[float], Dict[str, float], torch.Tensor],
    empty_weight: Optional[float] = None,
    presence_boost: bool = True,
) -> torch.Tensor:
    """
    Computes per-patch sampling weights based on class weights and each patch's class percentages.
    
    Formula:
      - If empty patch (no nuclei): empty_weight (or bg_weight)
      - If nuclei present:
          weighted_sum = sum(w_c * ratio_c for c in range(4))
          If presence_boost:
            max_presence_w = max(w_c for c in classes_present)
            weight = max(weighted_sum, max_presence_w * 0.5)
    """
    cell_weights, bg_weight = parse_class_weights(class_weights)
    if empty_weight is None:
        empty_weight = bg_weight

    weights = []

    for s in samples:
        name = s[0].stem
        stat = patch_stats.get(name)
        if stat is None:
            weights.append(1.0)
            continue

        classes_present = stat.get("classes_present", [])
        if not classes_present or not stat.get("has_cells", False):
            weights.append(empty_weight)
            continue

        ratios = stat.get("class_ratios", [0.0, 0.0, 0.0, 0.0])
        weighted_sum = sum(cell_weights[c] * ratios[c] for c in range(min(4, len(ratios))))

        if presence_boost:
            max_presence = max(cell_weights[c] for c in classes_present)
            w = max(weighted_sum, max_presence * 0.5)
        else:
            w = weighted_sum

        weights.append(max(1e-4, float(w)))

    return torch.tensor(weights, dtype=torch.float32)


def create_monusac_class_maps(mask_4ch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Derives MoNuSAC masks from 4-channel instance mask [4, H, W]:
      1. binary_mask: [H, W] float32 (1.0 for any nucleus foreground, 0.0 for background).
      2. class_mask_5ch: [5, H, W] float32:
         - Channel 0: Background (1.0 where no cell is present, else 0.0)
         - Channel 1: Epithelial (1.0 where Epithelial, else 0.0)
         - Channel 2: Lymphocyte (1.0 where Lymphocyte, else 0.0)
         - Channel 3: Neutrophil (1.0 where Neutrophil, else 0.0)
         - Channel 4: Macrophage (1.0 where Macrophage, else 0.0)
      3. class_map_2d: [H, W] int64 with integer values in {0, 1, 2, 3, 4}:
         - 0: Background
         - 1: Epithelial
         - 2: Lymphocyte
         - 3: Neutrophil
         - 4: Macrophage
    """
    c, h, w = mask_4ch.shape
    binary_mask = (mask_4ch > 0).any(axis=0).astype(np.float32)

    class_mask_5ch = np.zeros((5, h, w), dtype=np.float32)
    class_mask_5ch[0] = 1.0 - binary_mask
    for i in range(4):
        class_mask_5ch[i + 1] = (mask_4ch[i] > 0).astype(np.float32)

    class_map_2d = np.zeros((h, w), dtype=np.int64)
    for i in range(4):
        class_map_2d[mask_4ch[i] > 0] = i + 1

    return binary_mask, class_mask_5ch, class_map_2d


def apply_patch_transforms(
    image: np.ndarray,
    binary_mask: np.ndarray,
    sdf: np.ndarray,
    class_mask: Optional[np.ndarray] = None,
    transform: Optional[Any] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Converts numpy patches to tensors and applies torchvision v2 transforms synchronously.
    Returns:
      - If class_mask is None: (img_t, mask_t, sdf_t)
      - If class_mask is not None: (img_t, mask_t, sdf_t, class_mask_t, class_map_t)
    """
    # 1. Base tensor conversions
    img_t = torch.from_numpy(image).permute(2, 0, 1).contiguous()  # [3, H, W] uint8
    mask_t = torch.from_numpy(binary_mask).unsqueeze(0).float()     # [1, H, W] float32
    sdf_t = torch.from_numpy(sdf).unsqueeze(0).float()             # [1, H, W] float32

    if class_mask is not None:
        cls_mask_t = torch.from_numpy(class_mask).float()          # [5, H, W] float32
    else:
        cls_mask_t = None

    # 2. If transform is provided, wrap in tv_tensors and apply
    if transform is not None:
        img_tv = tv_tensors.Image(img_t)
        mask_tv = tv_tensors.Mask(mask_t)
        sdf_tv = tv_tensors.Mask(sdf_t)

        if cls_mask_t is not None:
            cls_tv = tv_tensors.Mask(cls_mask_t)
            t_img, t_mask, t_sdf, t_cls = transform(img_tv, mask_tv, sdf_tv, cls_tv)
            t_cls = t_cls.as_subclass(torch.Tensor)
            # Derive integer class map with values in {0, 1, 2, 3, 4}
            class_map_t = torch.argmax(t_cls, dim=0).long()
            # Reconstruct clean 5-channel one-hot mask and aligned binary mask
            class_mask_t = F.one_hot(class_map_t, num_classes=5).permute(2, 0, 1).float()
            mask_t = (class_map_t > 0).float().unsqueeze(0)
        else:
            t_img, t_mask, t_sdf = transform(img_tv, mask_tv, sdf_tv)
            mask_t = (t_mask.as_subclass(torch.Tensor) > 0.5).float()
            class_mask_t = None
            class_map_t = None

        img_t = t_img.as_subclass(torch.Tensor)
        if img_t.max() > 1.0:
            img_t = img_t.float() / 255.0
        else:
            img_t = img_t.float()
        sdf_t = t_sdf.as_subclass(torch.Tensor)
    else:
        img_t = img_t.float() / 255.0
        if cls_mask_t is not None:
            class_mask_t = cls_mask_t
            class_map_t = torch.argmax(cls_mask_t, dim=0).long()
        else:
            class_mask_t = None
            class_map_t = None

    if class_mask_t is not None:
        return img_t, mask_t, sdf_t, class_mask_t, class_map_t
    return img_t, mask_t, sdf_t


def build_dataloader(
    dataset: Dataset,
    config: Dict[str, Any],
    shuffle: bool = True,
    **override_kwargs,
) -> DataLoader:
    """
    Creates a PyTorch DataLoader using dataset and training parameters from config.
    Automatically attaches a WeightedRandomSampler if dataset supports it and class_weights
    are defined in the config or passed as an argument.
    """
    training_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {})

    batch_size = override_kwargs.get("batch_size", training_cfg.get("batch_size", 4))
    num_workers = override_kwargs.get("num_workers", dataset_cfg.get("num_workers", 2))
    pin_memory = override_kwargs.get("pin_memory", dataset_cfg.get("pin_memory", True))
    drop_last = override_kwargs.get("drop_last", training_cfg.get("drop_last", False))

    sampler = override_kwargs.get("sampler", None)
    if sampler is None and hasattr(dataset, "get_sampler"):
        class_weights = override_kwargs.get("class_weights", dataset_cfg.get("class_weights", None))
        if class_weights is not None:
            sampler = dataset.get_sampler(class_weights)

    actual_shuffle = False if sampler is not None else override_kwargs.get("shuffle", shuffle)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=actual_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
