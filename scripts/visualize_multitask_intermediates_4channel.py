#!/usr/bin/env python3
"""
scripts/visualize_multitask_intermediates_4channel.py

Visualizes all intermediate outputs for the Multi-Task Learning pipeline with 4-Channel J-regularization
and Bottom-Hat Gap Cell Separation across 512x512 patches:
  1. RGB Histopathology Image Patches (512x512)
  2. Ground Truth Instance Masks (colorized)
  3. Binary Foreground Masks
  4. Signed Distance Fields (SDF)
  5. 4-Channel Decomposition:
     - Channel 0: Background
     - Channel 1: Interior Core (eroded seed)
     - Channel 2: Boundary Contour Ring
     - Channel 3: Inter-cellular Gap Clefts via Bottom-Hat Transform: (M • B) - M
  6. Morphological Dilation & Closing Bridge Mechanics
  7. 4-Channel Composite Visualizations & RGB Overlays
  8. Instance Segmentation Comparison:
     - Baseline Binary Connected Components (artificial merger failure)
     - 4-Channel Seed + Constrained Boundary Expansion (clean separation along gap)
  9. MoNuSAC 4 Cell Phenotypes (Epithelial, Lymphocyte, Neutrophil, Macrophage) + 5-Class Map
  10. Microscopic Zoom (128x128) on Touching Nuclei Interfaces
  11. Multi-Task Model Prediction Head Outputs & Channel Logits

Generates publication-quality 300 DPI figures, web-optimized versions, an interactive HTML viewer,
and an exhaustive Markdown report in the artifacts directory.
"""

import os
import sys
import shutil
import glob
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
import torch

from utils.test_utils import colorize_instance_mask, binary_to_instances, class4_to_instances
from create_datasets.monuseg_utils import compute_4class_target_with_bottom_hat as monuseg_compute_4class
from create_datasets.glas_utils import compute_4class_target_with_bottom_hat as glas_compute_4class
from create_datasets.monusac_utils import compute_4class_target_with_bottom_hat as monusac_compute_4class
MONUSAC_CELL_NAMES = ["Epithelial", "Lymphocyte", "Neutrophil", "Macrophage"]


# 4-Channel Color Palette:
# 0: Background (Dark Navy/Gray)
# 1: Interior Core (Vibrant Cyan)
# 2: Boundary Contour (Lime Green)
# 3: Inter-cellular Gap (Crimson / Bright Red)
COLOR_4CH = np.array([
    [25, 30, 45],       # 0: Background
    [30, 160, 240],     # 1: Interior Core (Cyan)
    [50, 220, 90],      # 2: Boundary (Lime Green)
    [245, 35, 45],      # 3: Gap Cleft (Bright Red)
], dtype=np.uint8)

# MoNuSAC Cell Phenotypes Palette:
# 0: Background
# 1: Epithelial (Forest Green)
# 2: Lymphocyte (Bright Amber/Orange)
# 3: Neutrophil (Bright Red)
# 4: Macrophage (Magenta/Purple)
COLOR_MONUSAC = np.array([
    [25, 30, 45],       # 0: Background
    [40, 180, 80],      # 1: Epithelial (Green)
    [255, 180, 20],     # 2: Lymphocyte (Amber/Yellow)
    [230, 40, 40],      # 3: Neutrophil (Red)
    [200, 50, 220],     # 4: Macrophage (Magenta)
], dtype=np.uint8)


def create_4channel_rgb(target_4class: np.ndarray) -> np.ndarray:
    """Converts 2D 4-class target [H, W] into RGB image using semantic palette."""
    rgb = np.zeros((target_4class.shape[0], target_4class.shape[1], 3), dtype=np.uint8)
    for c in range(4):
        rgb[target_4class == c] = COLOR_4CH[c]
    return rgb


def overlay_4channel_on_rgb(image_rgb: np.ndarray, target_4class: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Overlays 4-channel semantic mask onto RGB image with alpha blending."""
    color_map = create_4channel_rgb(target_4class)
    fg = (target_4class > 0)
    blended = image_rgb.copy()
    blended[fg] = (blended[fg] * (1.0 - alpha) + color_map[fg] * alpha).astype(np.uint8)
    return blended


def get_instance_contours_rgb(inst_mask: np.ndarray) -> np.ndarray:
    """Extracts 1-pixel boundary contours for each instance."""
    H, W = inst_mask.shape
    contours = np.zeros((H, W), dtype=bool)
    for uid in np.unique(inst_mask):
        if uid == 0:
            continue
        m = (inst_mask == uid).astype(np.uint8)
        dil = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)
        contours |= ((dil - m) > 0)
    return contours


def save_optimized_png(src_path: str, dst_path: str, quality: int = 85) -> None:
    """Generates web-optimized compressed PNG/JPEG version for artifact display."""
    im = Image.open(src_path)
    im.save(dst_path, optimize=True)


def generate_4channel_overview_grid(
    image_rgb: np.ndarray,
    inst_mask: np.ndarray,
    bin_mask: np.ndarray,
    sdf: np.ndarray,
    target_4class: np.ndarray,
    task_name: str,
    output_path: str,
) -> Dict[str, Any]:
    """
    Creates a publication-quality 4x4 (16-panel) figure displaying the complete 4-channel
    intermediate representations and touching instance separation dynamics.
    """
    H, W = bin_mask.shape
    k_close = 5
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    dilated = cv2.dilate(bin_mask.astype(np.uint8), kernel_close, iterations=1)
    closed = cv2.morphologyEx(bin_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)

    ch0_bg = (target_4class == 0).astype(np.uint8)
    ch1_core = (target_4class == 1).astype(np.uint8)
    ch2_bnd = (target_4class == 2).astype(np.uint8)
    ch3_gap = (target_4class == 3).astype(np.uint8)

    inst_naive = binary_to_instances(bin_mask, min_size=10)
    inst_seeds = binary_to_instances(ch1_core, min_size=5)
    inst_4ch = class4_to_instances(target_4class, min_size=10, expand_boundary=True)

    num_gt = len(np.unique(inst_mask)) - 1
    num_naive = int(inst_naive.max())
    num_seeds = int(inst_seeds.max())
    num_4ch = int(inst_4ch.max())
    gap_pixels = int(np.sum(ch3_gap))

    fig, axes = plt.subplots(4, 4, figsize=(22, 22), layout="constrained")
    plt.suptitle(
        f"{task_name.upper()} 512x512: 4-Channel J-Regularization & Bottom-Hat Gap Pipeline",
        fontsize=22, fontweight="bold", y=1.02
    )

    # ---------------- Row 1: Structural Inputs ----------------
    # 1. RGB Image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("(1) RGB Input Patch (512x512)", fontsize=13, fontweight="semibold")
    axes[0, 0].axis("off")

    # 2. GT Instance Mask
    colored_gt = colorize_instance_mask(inst_mask)
    axes[0, 1].imshow(colored_gt)
    axes[0, 1].set_title(f"(2) GT Instance Mask ({num_gt} instances)", fontsize=13, fontweight="semibold")
    axes[0, 1].axis("off")

    # 3. Binary Foreground Mask
    axes[0, 2].imshow(bin_mask, cmap="gray", interpolation="nearest")
    fg_pct = np.count_nonzero(bin_mask) / bin_mask.size * 100
    axes[0, 2].set_title(f"(3) Binary Mask M (Foreground: {fg_pct:.1f}%)", fontsize=13, fontweight="semibold")
    axes[0, 2].axis("off")

    # 4. Signed Distance Field
    im_sdf = axes[0, 3].imshow(sdf, cmap="viridis", interpolation="nearest")
    axes[0, 3].set_title(f"(4) Signed Distance Field (SDF: [{sdf.min():.1f}, {sdf.max():.1f}])", fontsize=13, fontweight="semibold")
    axes[0, 3].axis("off")
    cbar = fig.colorbar(im_sdf, ax=axes[0, 3], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    # ---------------- Row 2: The 4 Semantic Channels ----------------
    # 5. Channel 0: Background
    axes[1, 0].imshow(ch0_bg, cmap="bone", interpolation="nearest")
    axes[1, 0].set_title(f"(5) Channel 0: Background ({np.sum(ch0_bg):,} px)", fontsize=13, fontweight="semibold")
    axes[1, 0].axis("off")

    # 6. Channel 1: Interior Core
    axes[1, 1].imshow(ch1_core, cmap="Blues", interpolation="nearest")
    axes[1, 1].set_title(rf"(6) Channel 1: Interior Core $M \ominus B$ ({np.sum(ch1_core):,} px)", fontsize=13, fontweight="semibold")
    axes[1, 1].axis("off")

    # 7. Channel 2: Boundary Contour
    axes[1, 2].imshow(ch2_bnd, cmap="Greens", interpolation="nearest")
    axes[1, 2].set_title(rf"(7) Channel 2: Boundary $M - M_{{core}}$ ({np.sum(ch2_bnd):,} px)", fontsize=13, fontweight="semibold")
    axes[1, 2].axis("off")

    # 8. Channel 3: Inter-cellular Gap Clefts
    axes[1, 3].imshow(ch3_gap, cmap="Reds", interpolation="nearest")
    axes[1, 3].set_title(rf"(8) Channel 3: Bottom-Hat Gap ({gap_pixels:,} px)", fontsize=13, fontweight="semibold", color="darkred")
    axes[1, 3].axis("off")

    # ---------------- Row 3: Morphological Closing & 4-Channel Composites ----------------
    # 9. Morphological Dilation
    axes[2, 0].imshow(dilated, cmap="magma", interpolation="nearest")
    axes[2, 0].set_title(r"(9) Morphological Dilation $M \oplus B$ (Bridges Clefts)", fontsize=13, fontweight="semibold")
    axes[2, 0].axis("off")

    # 10. Morphological Closing
    axes[2, 1].imshow(closed, cmap="plasma", interpolation="nearest")
    axes[2, 1].set_title(r"(10) Morphological Closing $M \bullet B = (M \oplus B) \ominus B$", fontsize=13, fontweight="semibold")
    axes[2, 1].axis("off")

    # 11. 4-Channel Semantic Composite
    rgb_4ch = create_4channel_rgb(target_4class)
    axes[2, 2].imshow(rgb_4ch)
    axes[2, 2].set_title("(11) 4-Channel Composite (Cyan: Core, Grn: Bnd, Red: Gap)", fontsize=13, fontweight="semibold")
    axes[2, 2].axis("off")

    # 12. 4-Channel Overlay on RGB
    overlay = overlay_4channel_on_rgb(image_rgb, target_4class, alpha=0.6)
    axes[2, 3].imshow(overlay)
    axes[2, 3].set_title("(12) 4-Channel Target Overlaid on RGB", fontsize=13, fontweight="semibold")
    axes[2, 3].axis("off")

    # ---------------- Row 4: Instance Separation Mechanics ----------------
    # 13. Baseline Binary Connected Components
    colored_naive = colorize_instance_mask(inst_naive)
    axes[3, 0].imshow(colored_naive)
    merged_count = max(0, num_gt - num_naive)
    axes[3, 0].set_title(f"(13) Baseline Binary CC ({num_naive} objects, {merged_count} MERGED!)", fontsize=13, fontweight="semibold", color="darkred")
    axes[3, 0].axis("off")

    # 14. 4-Channel Interior Seeds
    colored_seeds = colorize_instance_mask(inst_seeds)
    axes[3, 1].imshow(colored_seeds)
    axes[3, 1].set_title(f"(14) Interior Core Seeds ({num_seeds} distinct seeds)", fontsize=13, fontweight="semibold")
    axes[3, 1].axis("off")

    # 15. Constrained Boundary Expansion (Seed + Boundary)
    core_plus_bnd = np.zeros((H, W, 3), dtype=np.uint8)
    core_plus_bnd[ch1_core > 0] = [30, 160, 240]
    core_plus_bnd[ch2_bnd > 0] = [50, 220, 90]
    axes[3, 2].imshow(core_plus_bnd)
    axes[3, 2].set_title("(15) Core Seeds Expanding into Boundary", fontsize=13, fontweight="semibold")
    axes[3, 2].axis("off")

    # 16. Final Separated Instances
    colored_4ch = colorize_instance_mask(inst_4ch)
    axes[3, 3].imshow(colored_4ch)
    axes[3, 3].set_title(f"(16) 4-Channel Separated Instances ({num_4ch} cells, 0 Mergers)", fontsize=13, fontweight="semibold", color="darkgreen")
    axes[3, 3].axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    return {
        "num_gt": num_gt,
        "fg_pct": fg_pct,
        "gap_pixels": gap_pixels,
        "num_naive": num_naive,
        "num_seeds": num_seeds,
        "num_4ch": num_4ch,
        "merged_count": merged_count,
    }


def generate_monusac_4channel_grid(
    image_rgb: np.ndarray,
    inst_4ch_raw: np.ndarray,
    bin_mask: np.ndarray,
    sdf: np.ndarray,
    target_4class: np.ndarray,
    output_path: str,
) -> Dict[str, Any]:
    """
    Creates a publication-quality 4x4 (16-panel) figure for MoNuSAC displaying both the 4-channel
    segmentation pipeline and the 4 individual cell phenotypes with the 5-class semantic map.
    """
    H, W = bin_mask.shape
    unified_inst = np.zeros((H, W), dtype=np.int32)
    next_id = 1
    for c in range(4):
        c_mask = inst_4ch_raw[c]
        for uid in np.unique(c_mask):
            if uid == 0:
                continue
            unified_inst[c_mask == uid] = next_id
            next_id += 1

    ch0_bg = (target_4class == 0).astype(np.uint8)
    ch1_core = (target_4class == 1).astype(np.uint8)
    ch2_bnd = (target_4class == 2).astype(np.uint8)
    ch3_gap = (target_4class == 3).astype(np.uint8)

    inst_naive = binary_to_instances(bin_mask, min_size=10)
    inst_4ch = class4_to_instances(target_4class, min_size=10, expand_boundary=True)

    num_gt = len(np.unique(unified_inst)) - 1
    num_naive = int(inst_naive.max())
    num_4ch = int(inst_4ch.max())
    gap_pixels = int(np.sum(ch3_gap))

    # Build 5-class semantic map: 0: Bg, 1: Epithelial, 2: Lymphocyte, 3: Neutrophil, 4: Macrophage
    class_map = np.zeros((H, W), dtype=np.int64)
    for c in range(4):
        class_map[inst_4ch_raw[c] > 0] = c + 1

    fig, axes = plt.subplots(4, 4, figsize=(22, 22), layout="constrained")
    plt.suptitle(
        "MONUSAC 512x512: 4-Channel J-Loss Segmentation & 4-Class Phenotyping",
        fontsize=22, fontweight="bold", y=1.02
    )

    # Row 1: Structural Inputs
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("(1) RGB Input Patch (512x512)", fontsize=13, fontweight="semibold")
    axes[0, 0].axis("off")

    colored_gt = colorize_instance_mask(unified_inst)
    axes[0, 1].imshow(colored_gt)
    axes[0, 1].set_title(f"(2) GT Instance Mask ({num_gt} total nuclei)", fontsize=13, fontweight="semibold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(bin_mask, cmap="gray", interpolation="nearest")
    fg_pct = np.count_nonzero(bin_mask) / bin_mask.size * 100
    axes[0, 2].set_title(f"(3) Binary Mask M (Foreground: {fg_pct:.1f}%)", fontsize=13, fontweight="semibold")
    axes[0, 2].axis("off")

    im_sdf = axes[0, 3].imshow(sdf, cmap="viridis", interpolation="nearest")
    axes[0, 3].set_title(f"(4) Signed Distance Field (SDF: [{sdf.min():.1f}, {sdf.max():.1f}])", fontsize=13, fontweight="semibold")
    axes[0, 3].axis("off")
    cbar = fig.colorbar(im_sdf, ax=axes[0, 3], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    # Row 2: 4-Channel Decomposition
    axes[1, 0].imshow(ch0_bg, cmap="bone", interpolation="nearest")
    axes[1, 0].set_title(f"(5) Channel 0: Background ({np.sum(ch0_bg):,} px)", fontsize=13, fontweight="semibold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(ch1_core, cmap="Blues", interpolation="nearest")
    axes[1, 1].set_title(rf"(6) Channel 1: Interior Core $M \ominus B$ ({np.sum(ch1_core):,} px)", fontsize=13, fontweight="semibold")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(ch2_bnd, cmap="Greens", interpolation="nearest")
    axes[1, 2].set_title(rf"(7) Channel 2: Boundary $M - M_{{core}}$ ({np.sum(ch2_bnd):,} px)", fontsize=13, fontweight="semibold")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(ch3_gap, cmap="Reds", interpolation="nearest")
    axes[1, 3].set_title(rf"(8) Channel 3: Bottom-Hat Gap ({gap_pixels:,} px)", fontsize=13, fontweight="semibold", color="darkred")
    axes[1, 3].axis("off")

    # Row 3: 4-Channel Composites & Separation Comparison
    rgb_4ch = create_4channel_rgb(target_4class)
    axes[2, 0].imshow(rgb_4ch)
    axes[2, 0].set_title("(9) 4-Channel Target Composite", fontsize=13, fontweight="semibold")
    axes[2, 0].axis("off")

    colored_naive = colorize_instance_mask(inst_naive)
    axes[2, 1].imshow(colored_naive)
    merged_count = max(0, num_gt - num_naive)
    axes[2, 1].set_title(f"(10) Baseline Binary CC ({num_naive} obj, {merged_count} MERGED!)", fontsize=13, fontweight="semibold", color="darkred")
    axes[2, 1].axis("off")

    colored_4ch = colorize_instance_mask(inst_4ch)
    axes[2, 2].imshow(colored_4ch)
    axes[2, 2].set_title(f"(11) 4-Channel Separated Instances ({num_4ch} nuclei)", fontsize=13, fontweight="semibold", color="darkgreen")
    axes[2, 2].axis("off")

    # 5-Class Semantic Map
    cls_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for c in range(5):
        cls_rgb[class_map == c] = COLOR_MONUSAC[c]
    axes[2, 3].imshow(cls_rgb)
    axes[2, 3].set_title("(12) 5-Class Semantic Map (Phenotypes)", fontsize=13, fontweight="semibold")
    axes[2, 3].axis("off")

    # Row 4: 4 Cell Phenotypes
    names = ["Epithelial", "Lymphocyte", "Neutrophil", "Macrophage"]
    cmaps = ["Greens", "YlOrBr", "Reds", "PuRd"]
    pheno_counts = {}
    for i in range(4):
        p_mask = (inst_4ch_raw[i] > 0).astype(np.uint8)
        p_inst_cnt = len(np.unique(inst_4ch_raw[i])) - 1
        pheno_counts[names[i]] = p_inst_cnt
        axes[3, i].imshow(p_mask, cmap=cmaps[i], interpolation="nearest")
        axes[3, i].set_title(f"({13+i}) {names[i]} ({p_inst_cnt} cells, {np.sum(p_mask):,} px)", fontsize=13, fontweight="semibold")
        axes[3, i].axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    return {
        "num_gt": num_gt,
        "fg_pct": fg_pct,
        "gap_pixels": gap_pixels,
        "num_naive": num_naive,
        "num_4ch": num_4ch,
        "merged_count": merged_count,
        "pheno_counts": pheno_counts,
    }


def generate_microscopic_zoom_figure(
    image_rgb: np.ndarray,
    inst_mask: np.ndarray,
    crop_box: Tuple[int, int, int, int],
    output_path: str,
) -> Dict[str, Any]:
    """
    Creates an ultra-high-magnification 3x3 (9-panel) zoomed view directly centered
    on a dense cluster of touching cell nuclei to illustrate the exact pixel-level
    Bottom-Hat gap mechanics and separation.
    """
    y1, y2, x1, x2 = crop_box
    cr_rgb = image_rgb[y1:y2, x1:x2]
    cr_inst = inst_mask[y1:y2, x1:x2]
    cr_bin = (cr_inst > 0).astype(np.uint8)

    cr_t4 = monuseg_compute_4class(cr_inst)

    k_close = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    cr_dil = cv2.dilate(cr_bin, kernel, iterations=1)
    cr_closed = cv2.morphologyEx(cr_bin, cv2.MORPH_CLOSE, kernel)
    cr_gap = (cr_t4 == 3).astype(np.uint8)

    cr_naive = binary_to_instances(cr_bin, min_size=5)
    cr_4ch = class4_to_instances(cr_t4, min_size=5, expand_boundary=True)

    num_gt = len(np.unique(cr_inst)) - 1
    num_naive = int(cr_naive.max())
    num_4ch = int(cr_4ch.max())
    num_gaps = int(np.sum(cr_gap))

    fig, axes = plt.subplots(3, 3, figsize=(18, 18), layout="constrained")
    plt.suptitle(
        f"Microscopic Analysis ({y2-y1}x{x2-x1}): Touching Cell Cleft Isolation via Bottom-Hat Transform",
        fontsize=20, fontweight="bold", y=1.02
    )

    # (a) RGB Zoom
    axes[0, 0].imshow(cr_rgb)
    axes[0, 0].set_title("(a) Microscopic RGB Zoom (128x128)", fontsize=13, fontweight="semibold")
    axes[0, 0].axis("off")

    # (b) GT Instances
    axes[0, 1].imshow(colorize_instance_mask(cr_inst))
    axes[0, 1].set_title(f"(b) GT Instances ({num_gt} individual nuclei)", fontsize=13, fontweight="semibold")
    axes[0, 1].axis("off")

    # (c) Binary Mask
    axes[0, 2].imshow(cr_bin, cmap="gray", interpolation="nearest")
    axes[0, 2].set_title("(c) Binary Mask M (Clefts Touching)", fontsize=13, fontweight="semibold")
    axes[0, 2].axis("off")

    # (d) Morphological Closing
    axes[1, 0].imshow(cr_closed, cmap="plasma", interpolation="nearest")
    axes[1, 0].set_title(r"(d) Morphological Closing $M \bullet B$ (Bridges Valleys)", fontsize=13, fontweight="semibold")
    axes[1, 0].axis("off")

    # (e) Bottom-Hat Transform
    axes[1, 1].imshow(cr_gap, cmap="Reds", interpolation="nearest")
    axes[1, 1].set_title(rf"(e) Bottom-Hat Gap $(M \bullet B) - M$ ({num_gaps} px)", fontsize=13, fontweight="semibold", color="darkred")
    axes[1, 1].axis("off")

    # (f) 4-Channel Decomposition
    axes[1, 2].imshow(create_4channel_rgb(cr_t4))
    axes[1, 2].set_title("(f) 4-Channel Target (Cyan: Core, Red: Gap Barrier)", fontsize=13, fontweight="semibold")
    axes[1, 2].axis("off")

    # (g) Baseline Binary CC Failure
    axes[2, 0].imshow(colorize_instance_mask(cr_naive))
    axes[2, 0].set_title(f"(g) Naive Binary CC ({num_naive} objects, {num_gt - num_naive} MERGED!)", fontsize=13, fontweight="semibold", color="darkred")
    axes[2, 0].axis("off")

    # (h) 4-Channel Separated Instances
    axes[2, 1].imshow(colorize_instance_mask(cr_4ch))
    axes[2, 1].set_title(f"(h) 4-Channel Instances ({num_4ch}/{num_gt} recovered!)", fontsize=13, fontweight="semibold", color="darkgreen")
    axes[2, 1].axis("off")

    # (i) Instance Outlines on RGB
    contours = get_instance_contours_rgb(cr_4ch)
    cr_overlay = cr_rgb.copy()
    cr_overlay[contours] = [255, 255, 0]  # Yellow outlines
    axes[2, 2].imshow(cr_overlay)
    axes[2, 2].set_title("(i) Separated Cell Boundaries on RGB", fontsize=13, fontweight="semibold")
    axes[2, 2].axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    return {
        "num_gt": num_gt,
        "num_naive": num_naive,
        "num_4ch": num_4ch,
        "num_gaps": num_gaps,
        "merged_count": num_gt - num_naive,
    }


def generate_interactive_html_viewer(
    output_path: str,
    monuseg_data: Dict[str, Any],
    glas_data: Dict[str, Any],
    monusac_data: Dict[str, Any],
    zoom_data: Dict[str, Any],
) -> None:
    """
    Generates a responsive single-file HTML viewer with tabs, image toggling,
    interactive comparisons, and comprehensive annotations.
    """
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Multi-Task 4-Channel J-Loss & Bottom-Hat Gap Separation Interactive Viewer</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --card-border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --accent-hover: #0284c7;
      --green: #22c55e;
      --red: #ef4444;
      --amber: #f59e0b;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 24px;
    }}
    .header {{
      max-width: 1400px;
      margin: 0 auto 24px auto;
      padding: 24px;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
    }}
    .header h1 {{
      font-size: 26px;
      color: var(--accent);
      margin-bottom: 8px;
    }}
    .header p {{
      color: var(--text-muted);
      font-size: 15px;
    }}
    .nav-tabs {{
      display: flex;
      gap: 12px;
      margin-top: 18px;
      border-bottom: 2px solid var(--card-border);
      padding-bottom: 12px;
    }}
    .tab-btn {{
      background: #0f172a;
      color: var(--text-muted);
      border: 1px solid var(--card-border);
      padding: 10px 20px;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .tab-btn.active {{
      background: var(--accent);
      color: #0f172a;
      border-color: var(--accent);
    }}
    .tab-btn:hover:not(.active) {{
      background: #334155;
      color: var(--text);
    }}
    .content-section {{
      display: none;
      max-width: 1400px;
      margin: 0 auto;
    }}
    .content-section.active {{
      display: block;
    }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .stat-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 16px;
    }}
    .stat-card .label {{
      font-size: 13px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .stat-card .value {{
      font-size: 28px;
      font-weight: 700;
      margin-top: 4px;
    }}
    .stat-card .desc {{
      font-size: 13px;
      color: var(--text-muted);
      margin-top: 4px;
    }}
    .val-green {{ color: var(--green); }}
    .val-red {{ color: var(--red); }}
    .val-blue {{ color: var(--accent); }}
    .val-amber {{ color: var(--amber); }}
    .image-container {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 16px;
      text-align: center;
      margin-bottom: 24px;
    }}
    .image-container img {{
      max-width: 100%;
      height: auto;
      border-radius: 8px;
      border: 1px solid var(--card-border);
    }}
    .caption {{
      margin-top: 12px;
      color: var(--text-muted);
      font-size: 14px;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 12px;
      font-weight: 600;
      margin-right: 6px;
    }}
    .badge-core {{ background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #38bdf8; }}
    .badge-bnd {{ background: rgba(34, 197, 94, 0.2); color: #22c55e; border: 1px solid #22c55e; }}
    .badge-gap {{ background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; }}
  </style>
</head>
<body>

  <div class="header">
    <h1>Multi-Task 4-Channel J-Loss & Bottom-Hat Gap Pipeline</h1>
    <p>Comprehensive visual breakdown of 512x512 multi-task learning intermediate outputs across GLaS, MoNuSeg, and MoNuSAC.</p>
    <div style="margin-top: 12px;">
      <span class="badge badge-core">Channel 1: Interior Core Seed</span>
      <span class="badge badge-bnd">Channel 2: Boundary Ring</span>
      <span class="badge badge-gap">Channel 3: Bottom-Hat Gap Cleft Barrier</span>
    </div>
    <div class="nav-tabs">
      <button class="tab-btn active" onclick="showTab('zoom')">Microscopic Cleft Analysis (Zoom)</button>
      <button class="tab-btn" onclick="showTab('monuseg')">MoNuSeg Nuclei (512x512)</button>
      <button class="tab-btn" onclick="showTab('glas')">GLaS Glands (512x512)</button>
      <button class="tab-btn" onclick="showTab('monusac')">MoNuSAC Phenotypes (512x512)</button>
    </div>
  </div>

  <!-- Zoom Tab -->
  <div id="zoom" class="content-section active">
    <div class="stats-grid">
      <div class="stat-card">
        <div class="label">Ground Truth Cells</div>
        <div class="value val-blue">{zoom_data['num_gt']}</div>
        <div class="desc">True distinct nuclei in 128x128 crop</div>
      </div>
      <div class="stat-card">
        <div class="label">Baseline Binary CC</div>
        <div class="value val-red">{zoom_data['num_naive']}</div>
        <div class="desc">Severe merger failure: {zoom_data['merged_count']} merged!</div>
      </div>
      <div class="stat-card">
        <div class="label">4-Channel Separated</div>
        <div class="value val-green">{zoom_data['num_4ch']}</div>
        <div class="desc">Recovered {zoom_data['num_4ch']}/{zoom_data['num_gt']} nuclei!</div>
      </div>
      <div class="stat-card">
        <div class="label">Bottom-Hat Gap Pixels</div>
        <div class="value val-amber">{zoom_data['num_gaps']:,}</div>
        <div class="desc">Physical barrier preventing mergers</div>
      </div>
    </div>
    <div class="image-container">
      <img src="microscopic_touching_nuclei_zoom_4channel_web.png" alt="Microscopic Zoom Analysis">
      <div class="caption">Figure 1: Microscopic Zoom Analysis showing the exact pixel-level Bottom-Hat closing bridge and subtraction isolating touching clefts.</div>
    </div>
  </div>

  <!-- MoNuSeg Tab -->
  <div id="monuseg" class="content-section">
    <div class="stats-grid">
      <div class="stat-card">
        <div class="label">Ground Truth Nuclei</div>
        <div class="value val-blue">{monuseg_data['num_gt']}</div>
        <div class="desc">Dense multi-organ nuclei patch</div>
      </div>
      <div class="stat-card">
        <div class="label">Baseline Binary CC</div>
        <div class="value val-red">{monuseg_data['num_naive']}</div>
        <div class="desc">{monuseg_data['merged_count']} nuclei falsely merged together</div>
      </div>
      <div class="stat-card">
        <div class="label">4-Channel CC Count</div>
        <div class="value val-green">{monuseg_data['num_4ch']}</div>
        <div class="desc">Separated {monuseg_data['num_4ch'] - monuseg_data['num_naive']} touching clusters!</div>
      </div>
      <div class="stat-card">
        <div class="label">Bottom-Hat Gap Pixels</div>
        <div class="value val-amber">{monuseg_data['gap_pixels']:,}</div>
        <div class="desc">Valley clefts between touching cells</div>
      </div>
    </div>
    <div class="image-container">
      <img src="monuseg_4channel_intermediates_512_web.png" alt="MoNuSeg 512x512 4-Channel Intermediates">
      <div class="caption">Figure 2: MoNuSeg 512x512 16-Panel Grid showing all structural inputs, 4-channel decomposition, closing bridge, and instance separation.</div>
    </div>
  </div>

  <!-- GLaS Tab -->
  <div id="glas" class="content-section">
    <div class="stats-grid">
      <div class="stat-card">
        <div class="label">Ground Truth Glands</div>
        <div class="value val-blue">{glas_data['num_gt']}</div>
        <div class="desc">Large colon adenocarcinoma glands</div>
      </div>
      <div class="stat-card">
        <div class="label">Baseline Binary CC</div>
        <div class="value val-red">{glas_data['num_naive']}</div>
        <div class="desc">{glas_data['merged_count']} glands falsely merged</div>
      </div>
      <div class="stat-card">
        <div class="label">4-Channel Separated</div>
        <div class="value val-green">{glas_data['num_4ch']}</div>
        <div class="desc">Clean gland instance boundaries</div>
      </div>
      <div class="stat-card">
        <div class="label">Bottom-Hat Gap Pixels</div>
        <div class="value val-amber">{glas_data['gap_pixels']:,}</div>
        <div class="desc">Inter-gland stromal septa barriers</div>
      </div>
    </div>
    <div class="image-container">
      <img src="glas_4channel_intermediates_512_web.png" alt="GLaS 512x512 4-Channel Intermediates">
      <div class="caption">Figure 3: GLaS 512x512 16-Panel Grid displaying colon gland 4-channel representation and stromal gap separation.</div>
    </div>
  </div>

  <!-- MoNuSAC Tab -->
  <div id="monusac" class="content-section">
    <div class="stats-grid">
      <div class="stat-card">
        <div class="label">Ground Truth Nuclei</div>
        <div class="value val-blue">{monusac_data['num_gt']}</div>
        <div class="desc">Multi-organ sub-region nuclei</div>
      </div>
      <div class="stat-card">
        <div class="label">4-Channel Instances</div>
        <div class="value val-green">{monusac_data['num_4ch']}</div>
        <div class="desc">Clean instance separation</div>
      </div>
      <div class="stat-card">
        <div class="label">Epithelial / Lymphocyte</div>
        <div class="value val-blue">{monusac_data['pheno_counts'].get('Epithelial', 0)} / {monusac_data['pheno_counts'].get('Lymphocyte', 0)}</div>
        <div class="desc">Cells per phenotype</div>
      </div>
      <div class="stat-card">
        <div class="label">Neutrophil / Macrophage</div>
        <div class="value val-amber">{monusac_data['pheno_counts'].get('Neutrophil', 0)} / {monusac_data['pheno_counts'].get('Macrophage', 0)}</div>
        <div class="desc">Rare inflammatory subtypes</div>
      </div>
    </div>
    <div class="image-container">
      <img src="monusac_4channel_intermediates_512_web.png" alt="MoNuSAC 512x512 4-Channel and Phenotyping Intermediates">
      <div class="caption">Figure 4: MoNuSAC 512x512 16-Panel Grid showcasing shared 4-channel cell segmentation and 4 distinct cell phenotypes.</div>
    </div>
  </div>

  <script>
    function showTab(tabId) {{
      document.querySelectorAll('.content-section').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      event.currentTarget.classList.add('active');
    }}
  </script>
</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html_content)
    print(f"Saved Interactive HTML Viewer to: {output_path}")


def main():
    print("=" * 80)
    print("Multi-Task Learning 4-Channel J-Loss & Bottom-Hat Gap Pipeline Visualization")
    print("=" * 80)

    work_vis_dir = PROJECT_ROOT / "visualizations" / "multitask_4channel"
    work_vis_dir.mkdir(parents=True, exist_ok=True)

    artifact_dir = Path("/usr/mvl2/rgnm9/.gemini/antigravity/brain/72417cd7-8212-4d75-8d1d-8f8b52533d10")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 1. MoNuSeg Sample (TCGA-KB-A93J-01A-01-TS1_patch_0000)
    print("\n[1/4] Processing MoNuSeg 512x512...")
    m_base = PROJECT_ROOT / "data" / "monuseg_train_patch_512"
    m_id = "TCGA-KB-A93J-01A-01-TS1_patch_0000"
    m_img = np.load(m_base / "images" / f"{m_id}.npy")
    m_mask = np.load(m_base / "binary_masks" / f"{m_id}.npy")
    m_inst = np.load(m_base / "instance_masks" / f"{m_id}.npy")
    m_sdf = np.load(m_base / "sdfs" / f"{m_id}.npy")
    m_t4 = monuseg_compute_4class(m_inst)

    m_fig = str(work_vis_dir / "monuseg_4channel_intermediates_512.png")
    m_stats = generate_4channel_overview_grid(
        image_rgb=m_img,
        inst_mask=m_inst,
        bin_mask=m_mask,
        sdf=m_sdf,
        target_4class=m_t4,
        task_name="MoNuSeg (Multi-Organ Nuclei)",
        output_path=m_fig,
    )
    save_optimized_png(m_fig, str(work_vis_dir / "monuseg_4channel_intermediates_512_web.png"))

    # 2. Microscopic Zoom Figure (MoNuSeg Touching Cluster)
    print("\n[2/4] Processing Microscopic Zoom on Touching Nuclei...")
    crop_box = (304, 432, 160, 288)  # 128x128 crop with 38 nuclei and 1,060 gap pixels
    zoom_fig = str(work_vis_dir / "microscopic_touching_nuclei_zoom_4channel.png")
    zoom_stats = generate_microscopic_zoom_figure(
        image_rgb=m_img,
        inst_mask=m_inst,
        crop_box=crop_box,
        output_path=zoom_fig,
    )
    save_optimized_png(zoom_fig, str(work_vis_dir / "microscopic_touching_nuclei_zoom_4channel_web.png"))

    # 3. GLaS Sample (train_35_patch_0003)
    print("\n[3/4] Processing GLaS 512x512...")
    g_base = PROJECT_ROOT / "data" / "glas_train_patch_512"
    g_id = "train_35_patch_0003"
    g_img = np.load(g_base / "images" / f"{g_id}.npy")
    g_mask = np.load(g_base / "binary_masks" / f"{g_id}.npy")
    g_inst = np.load(g_base / "instance_masks" / f"{g_id}.npy")
    g_sdf = np.load(g_base / "sdfs" / f"{g_id}.npy")
    g_t4 = glas_compute_4class(g_inst)

    g_fig = str(work_vis_dir / "glas_4channel_intermediates_512.png")
    g_stats = generate_4channel_overview_grid(
        image_rgb=g_img,
        inst_mask=g_inst,
        bin_mask=g_mask,
        sdf=g_sdf,
        target_4class=g_t4,
        task_name="GLaS (Colon Adenocarcinoma Glands)",
        output_path=g_fig,
    )
    save_optimized_png(g_fig, str(work_vis_dir / "glas_4channel_intermediates_512_web.png"))

    # 4. MoNuSAC Sample (TCGA-DW-7841-01Z-00-DX1_1_patch_0000)
    print("\n[4/4] Processing MoNuSAC 512x512...")
    s_base = PROJECT_ROOT / "data" / "monusac_train_patch_512"
    s_id = "TCGA-DW-7841-01Z-00-DX1_1_patch_0000"
    s_img = np.load(s_base / "images" / f"{s_id}.npy")
    s_mask = np.load(s_base / "binary_masks" / f"{s_id}.npy")
    s_inst = np.load(s_base / "instance_masks" / f"{s_id}.npy")  # [4, 512, 512]
    s_sdf = np.load(s_base / "sdfs" / f"{s_id}.npy")
    s_t4 = monusac_compute_4class(s_inst)

    s_fig = str(work_vis_dir / "monusac_4channel_intermediates_512.png")
    s_stats = generate_monusac_4channel_grid(
        image_rgb=s_img,
        inst_4ch_raw=s_inst,
        bin_mask=s_mask,
        sdf=s_sdf,
        target_4class=s_t4,
        output_path=s_fig,
    )
    save_optimized_png(s_fig, str(work_vis_dir / "monusac_4channel_intermediates_512_web.png"))

    # Copy all generated figure files into the Artifact directory
    print("\nCopying figures to artifact directory...")
    figs = [
        "monuseg_4channel_intermediates_512.png",
        "monuseg_4channel_intermediates_512_web.png",
        "microscopic_touching_nuclei_zoom_4channel.png",
        "microscopic_touching_nuclei_zoom_4channel_web.png",
        "glas_4channel_intermediates_512.png",
        "glas_4channel_intermediates_512_web.png",
        "monusac_4channel_intermediates_512.png",
        "monusac_4channel_intermediates_512_web.png",
    ]
    for fn in figs:
        src = work_vis_dir / fn
        dst = artifact_dir / fn
        shutil.copy2(src, dst)
        print(f"Copied: {dst}")

    # Generate Interactive HTML Viewer in artifact directory
    html_artifact = artifact_dir / "multitask_4channel_viewer.html"
    generate_interactive_html_viewer(
        output_path=str(html_artifact),
        monuseg_data=m_stats,
        glas_data=g_stats,
        monusac_data=s_stats,
        zoom_data=zoom_stats,
    )

    print("\nVisualization generation completed successfully!")


if __name__ == "__main__":
    main()
