# create_datasets/plot_monusac_patches.py

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from create_datasets.monusac_utils import (
    extract_inside_distance,
    compute_inside_distance_map,
    compute_inside_distance_and_hv,
    compute_hv_gradients,
    create_combined_hv_rgb,
)

CLASS_NAMES = ["Epithelial", "Lymphocyte", "Neutrophil", "Macrophage"]
CLASS_COLORS = [
    [220, 20, 60],   # Class 0: Epithelial - Red
    [46, 204, 113],  # Class 1: Lymphocyte - Green
    [30, 144, 255],  # Class 2: Neutrophil - Blue
    [255, 193, 7],   # Class 3: Macrophage - Yellow
]
CLASS_HEX = ["#DC143C", "#2ECC71", "#1E90FF", "#FFC107"]


def colorize_single_class(mask_2d: np.ndarray, class_idx: int) -> np.ndarray:
    """Colorizes a binary 2D class mask with its designated color on black background."""
    h, w = mask_2d.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    colored[mask_2d > 0] = CLASS_COLORS[class_idx]
    return colored


def colorize_instance_mask(inst_mask: np.ndarray, seed: int = 42) -> np.ndarray:
    """Colorizes an instance mask with distinct random colors per nucleus."""
    h, w = inst_mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    unique_ids = np.unique(inst_mask)
    unique_ids = unique_ids[unique_ids > 0]
    if len(unique_ids) == 0:
        return colored

    cmap = plt.get_cmap("turbo", max(20, len(unique_ids) + 1))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(unique_ids))
    for i, uid in enumerate(unique_ids):
        rgba = cmap((perm[i] + 1) / (len(unique_ids) + 1))
        colored[inst_mask == uid] = (np.array(rgba[:3]) * 255).astype(np.uint8)
    return colored


def create_patch_overlay(image: np.ndarray, foreground_mask: np.ndarray) -> np.ndarray:
    """Overlays nucleus boundaries on the RGB image in bright yellow."""
    overlay = image.copy()
    fg = foreground_mask > 0
    if not np.any(fg):
        return overlay

    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(fg.astype(np.uint8), kernel, iterations=1)
    boundary = fg & (eroded == 0)
    overlay[boundary] = [255, 230, 0]
    return overlay


def select_diverse_monusac_patches(
    patch_dir: Path,
    num_samples: int = 6,
) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Selects informative MoNuSAC patches containing nuclei and diverse cell types.
    Returns: List of (patch_name, image, mask_4ch, sdf)
    """
    img_dir = patch_dir / "images"
    mask_dir = patch_dir / "instance_masks"
    sdf_dir = patch_dir / "sdfs"

    image_files = sorted(list(img_dir.glob("*.npy")))
    if not image_files:
        raise FileNotFoundError(f"No patches found in {img_dir}")

    scored_candidates = []
    for img_p in image_files:
        name = img_p.stem
        mask_p = mask_dir / f"{name}.npy"
        if not mask_p.exists():
            continue

        mask_4ch = np.load(mask_p)
        classes_present = sum(1 for c in range(4) if np.any(mask_4ch[c] > 0))
        total_nuclei_px = int((mask_4ch > 0).sum())

        if total_nuclei_px > 0:
            scored_candidates.append((classes_present, total_nuclei_px, img_p))

    # Sort descending by classes present, then by nuclei area
    scored_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

    # Pick samples distributed across patients if possible
    selected_samples = []
    seen_patients = set()
    for _, _, img_p in scored_candidates:
        name = img_p.stem
        patient = name.split("_")[0]
        if patient not in seen_patients or len(seen_patients) >= 4:
            seen_patients.add(patient)
            mask_4ch = np.load(mask_dir / f"{name}.npy")
            sdf = np.load(sdf_dir / f"{name}.npy")
            image = np.load(img_p)
            selected_samples.append((name, image, mask_4ch, sdf))
            if len(selected_samples) >= num_samples:
                break

    # Fallback if not enough samples
    if len(selected_samples) < num_samples:
        for img_p in image_files:
            name = img_p.stem
            if any(s[0] == name for s in selected_samples):
                continue
            selected_samples.append((
                name,
                np.load(img_p),
                np.load(mask_dir / f"{name}.npy"),
                np.load(sdf_dir / f"{name}.npy"),
            ))
            if len(selected_samples) >= num_samples:
                break

    return selected_samples


def plot_monusac_patches(
    patch_dir: Path,
    output_dir: Path,
    num_samples: int = 5,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = select_diverse_monusac_patches(patch_dir, num_samples=num_samples)
    if not samples:
        raise ValueError(f"No patches could be loaded from {patch_dir}")

    n_rows = len(samples)
    cols = 13
    fig = plt.figure(figsize=(38, 3.8 * n_rows), dpi=200)
    subfigs = fig.subfigures(n_rows, 1, hspace=0.15)
    if n_rows == 1:
        subfigs = [subfigs]

    col_titles = [
        "Image Patch",
        "Binary Mask",
        "Epithelial (Class 0)",
        "Lymphocyte (Class 1)",
        "Neutrophil (Class 2)",
        "Macrophage (Class 3)",
        "Instance Mask",
        "Boundary Overlay",
        "Signed Distance Field",
        "Inside Cell Dist",
        "Horizontal Grad Gx",
        "Vertical Grad Gy",
        "Combined (Red/Blue)",
    ]

    for row_idx, (sf, (name, img_patch, mask_4ch, sdf_patch)) in enumerate(zip(subfigs, samples)):
        sf.suptitle(
            f"MoNuSAC Sample [{row_idx + 1}/{n_rows}]: {name}",
            fontsize=12,
            fontweight="bold",
            y=0.98,
            color="#1F2937",
        )
        axes = sf.subplots(1, cols)

        bin_mask = (mask_4ch > 0).any(axis=0).astype(np.uint8)

        # Build unified instance mask for visualization
        inst_mask = np.zeros(bin_mask.shape, dtype=np.int32)
        cur_id = 0
        for c in range(4):
            ch = mask_4ch[c]
            u_ids = np.unique(ch[ch > 0])
            for uid in u_ids:
                cur_id += 1
                inst_mask[ch == uid] = cur_id

        inst_rgb = colorize_instance_mask(inst_mask, seed=row_idx * 17)
        overlay = create_patch_overlay(img_patch, bin_mask)

        hv_p = patch_dir / "hv_maps" / f"{name}.npy"
        if hv_p.exists():
            hv_map = np.load(hv_p)
            if hv_map.shape[0] == 2:
                gx_patch = hv_map[0]
                gy_patch = hv_map[1]
            else:
                gx_patch = hv_map[:, :, 0]
                gy_patch = hv_map[:, :, 1]
            dist_patch = compute_inside_distance_map(inst_mask)
        else:
            dist_patch, gx_patch, gy_patch = compute_inside_distance_and_hv(inst_mask, method="sobel")

        hv_rgb_patch = create_combined_hv_rgb(gx_patch, gy_patch, mask=inst_mask > 0, bg_color="white")

        # 0. Image Patch
        axes[0].imshow(img_patch)
        axes[0].set_xlabel("RGB", fontsize=8, color="#4B5563")

        # 1. Binary Mask
        axes[1].imshow(bin_mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_xlabel(f"{int(bin_mask.sum())} px", fontsize=8, color="#4B5563")

        # 2-5. 4 Cell Class Panels
        for c in range(4):
            c_mask = (mask_4ch[c] > 0).astype(np.uint8)
            c_rgb = colorize_single_class(c_mask, c)
            axes[2 + c].imshow(c_rgb)
            c_px = int(c_mask.sum())
            if c_px > 0:
                axes[2 + c].set_xlabel(f"Present ({c_px} px)", fontsize=8, color=CLASS_HEX[c], fontweight="bold")
            else:
                axes[2 + c].set_xlabel("Absent (0 px)", fontsize=8, color="#9CA3AF")

        # 6. Instance Mask
        axes[6].imshow(inst_rgb)
        axes[6].set_xlabel(f"{cur_id} nuclei", fontsize=8, color="#4B5563")

        # 7. Boundary Overlay
        axes[7].imshow(overlay)
        axes[7].set_xlabel("Yellow = Boundary", fontsize=8, color="#4B5563")

        # 8. SDF (bwr Colormap)
        max_abs = max(1.0, float(np.max(np.abs(sdf_patch))))
        im_sdf = axes[8].imshow(sdf_patch, cmap="bwr", vmin=-max_abs, vmax=max_abs)
        axes[8].set_xlabel(f"[{sdf_patch.min():.1f}, {sdf_patch.max():.1f}]", fontsize=8, color="#4B5563")
        cbar = sf.colorbar(im_sdf, ax=axes[8], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)

        # 9. Inside Cell Distance Transform
        im_dt = axes[9].imshow(dist_patch, cmap="viridis", vmin=0)
        axes[9].set_xlabel(f"[0, {dist_patch.max():.1f} px]", fontsize=8, color="#4B5563")
        cbar_dt = sf.colorbar(im_dt, ax=axes[9], fraction=0.046, pad=0.04)
        cbar_dt.ax.tick_params(labelsize=7)

        # 10. Horizontal Field Gx (Sobel X)
        vmax = max(0.2, min(1.0, float(max(np.max(np.abs(gx_patch)), np.max(np.abs(gy_patch))))))
        im_gx = axes[10].imshow(gx_patch, cmap="bwr", vmin=-vmax, vmax=vmax)
        axes[10].set_xlabel(f"[-{vmax:.2f}, +{vmax:.2f}]", fontsize=8, color="#4B5563")
        cbar_gx = sf.colorbar(im_gx, ax=axes[10], fraction=0.046, pad=0.04)
        cbar_gx.ax.tick_params(labelsize=7)

        # 11. Vertical Field Gy (Sobel Y)
        im_gy = axes[11].imshow(gy_patch, cmap="bwr", vmin=-vmax, vmax=vmax)
        axes[11].set_xlabel(f"[-{vmax:.2f}, +{vmax:.2f}]", fontsize=8, color="#4B5563")
        cbar_gy = sf.colorbar(im_gy, ax=axes[11], fraction=0.046, pad=0.04)
        cbar_gy.ax.tick_params(labelsize=7)

        # 12. Combined Gradient Map (Red=Gx, Blue=Gy, Purple=Combined)
        axes[12].imshow(hv_rgb_patch)
        axes[12].set_xlabel("Red=Gx, Blue=Gy", fontsize=8, color="#4B5563")

        for j in range(cols):
            axes[j].set_xticks([])
            axes[j].set_yticks([])

        if row_idx == 0:
            for j in range(cols):
                color = CLASS_HEX[j - 2] if 2 <= j <= 5 else "#111827"
                axes[j].set_title(col_titles[j], fontsize=9.5, fontweight="bold", pad=8, color=color)

    output_file = output_dir / "monusac_patches_visualization.png"
    plt.savefig(output_file, bbox_inches="tight", dpi=200)
    plt.close()

    print(f"MoNuSAC visualization saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Plot image, masks, and SDF patches for MoNuSAC dataset.")
    parser.add_argument(
        "--patch_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "monusac_train_patch_256",
        help="Path to pre-extracted MoNuSAC patch directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "dataset_visualizations" / "intermediate outputs",
        help="Path to intermediate outputs directory.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of patch samples to visualize (default: 5).",
    )
    args = parser.parse_args()

    plot_monusac_patches(
        patch_dir=args.patch_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
