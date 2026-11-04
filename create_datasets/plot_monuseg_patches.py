# create_datasets/plot_monuseg_patches.py

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


from create_datasets.monuseg_utils import (
    extract_inside_distance,
    compute_inside_distance_map,
    compute_inside_distance_and_hv,
    compute_hv_gradients,
    create_combined_hv_rgb,
)


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


def select_monuseg_patch_samples(
    patch_dir: Path,
    num_samples: int = 6,
) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Selects informative MoNuSeg patches containing cell nuclei and loads or computes inside distance and HV fields."""
    img_dir = patch_dir / "images"
    mask_dir = patch_dir / "instance_masks"
    sdf_dir = patch_dir / "sdfs"
    hv_dir = patch_dir / "hv_maps"

    image_files = sorted(list(img_dir.glob("*.npy")))
    if not image_files:
        raise FileNotFoundError(f"No patches found in {img_dir}")

    candidates = []
    for img_p in image_files:
        name = img_p.stem
        mask_p = mask_dir / f"{name}.npy"
        if not mask_p.exists():
            continue

        inst_mask = np.load(mask_p)
        num_nuclei = len(np.unique(inst_mask[inst_mask > 0]))
        if num_nuclei >= 5:
            candidates.append((num_nuclei, img_p))

    candidates.sort(key=lambda x: x[0], reverse=True)

    def load_item(img_p: Path):
        name = img_p.stem
        image = np.load(img_p)
        inst_mask = np.load(mask_dir / f"{name}.npy")
        sdf = np.load(sdf_dir / f"{name}.npy")

        # Inside distance transform
        dist_map = extract_inside_distance(sdf, instance_mask=inst_mask)

        # HV maps
        hv_p = hv_dir / f"{name}.npy"
        if hv_p.exists():
            hv_map = np.load(hv_p)
            if hv_map.shape[0] == 2:
                gx = hv_map[0]
                gy = hv_map[1]
            else:
                gx = hv_map[:, :, 0]
                gy = hv_map[:, :, 1]
        else:
            dist_map, gx, gy = compute_inside_distance_and_hv(inst_mask, method="sobel")

        hv_rgb = create_combined_hv_rgb(gx, gy, mask=inst_mask > 0, bg_color="white")
        return (name, image, inst_mask, sdf, dist_map, gx, gy, hv_rgb)

    samples = []
    seen_patients = set()
    for _, img_p in candidates:
        name = img_p.stem
        patient = name.split("_")[0]
        if patient not in seen_patients or len(seen_patients) >= 4:
            seen_patients.add(patient)
            samples.append(load_item(img_p))
            if len(samples) >= num_samples:
                break

    # Fallback
    if len(samples) < num_samples:
        for img_p in image_files:
            name = img_p.stem
            if any(s[0] == name for s in samples):
                continue
            samples.append(load_item(img_p))
            if len(samples) >= num_samples:
                break

    return samples


def plot_monuseg_patches(
    patch_dir: Path,
    output_dir: Path,
    num_samples: int = 6,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = select_monuseg_patch_samples(patch_dir, num_samples=num_samples)
    if not samples:
        raise ValueError(f"No patches found in {patch_dir}")

    n_rows = len(samples)
    cols = 9
    fig, axes = plt.subplots(n_rows, cols, figsize=(28, 3.8 * n_rows), dpi=200)
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, (name, img_patch, inst_patch, sdf_patch, dist_patch, gx_patch, gy_patch, hv_rgb_patch) in enumerate(samples):
        bin_mask = (inst_patch > 0).astype(np.uint8)
        colored_inst = colorize_instance_mask(inst_patch)
        overlay = create_patch_overlay(img_patch, bin_mask)

        # 0. Image Patch
        axes[row_idx, 0].imshow(img_patch)
        axes[row_idx, 0].set_title(f"{name}\nRGB Image", fontsize=9, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # 1. Binary Mask
        axes[row_idx, 1].imshow(bin_mask, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 1].set_title(f"Binary Mask\n({int(bin_mask.sum())} px)", fontsize=9, fontweight="bold")
        axes[row_idx, 1].axis("off")

        # 2. Instance Mask
        n_inst = len(np.unique(inst_patch[inst_patch > 0]))
        axes[row_idx, 2].imshow(colored_inst)
        axes[row_idx, 2].set_title(f"Instance Mask\n({n_inst} nuclei)", fontsize=9, fontweight="bold")
        axes[row_idx, 2].axis("off")

        # 3. Boundary Overlay
        axes[row_idx, 3].imshow(overlay)
        axes[row_idx, 3].set_title("Boundary Overlay\n(Yellow = Interface)", fontsize=9, fontweight="bold")
        axes[row_idx, 3].axis("off")

        # 4. SDF Map
        max_abs = max(1.0, float(np.max(np.abs(sdf_patch))))
        im_sdf = axes[row_idx, 4].imshow(sdf_patch, cmap="bwr", vmin=-max_abs, vmax=max_abs)
        axes[row_idx, 4].set_title(f"SDF\n[{sdf_patch.min():.1f}, {sdf_patch.max():.1f}]", fontsize=9, fontweight="bold")
        axes[row_idx, 4].axis("off")
        fig.colorbar(im_sdf, ax=axes[row_idx, 4], fraction=0.046, pad=0.04)

        # 5. Inside Cell Distance Transform
        im_dt = axes[row_idx, 5].imshow(dist_patch, cmap="viridis", vmin=0)
        axes[row_idx, 5].set_title(
            f"Inside Cell Dist\n[0, {dist_patch.max():.1f} px]",
            fontsize=9,
            fontweight="bold",
        )
        axes[row_idx, 5].axis("off")
        fig.colorbar(im_dt, ax=axes[row_idx, 5], fraction=0.046, pad=0.04)

        # 6. Horizontal Field Gx (Sobel X)
        vmax = max(0.2, min(1.0, float(max(np.max(np.abs(gx_patch)), np.max(np.abs(gy_patch))))))
        im_gx = axes[row_idx, 6].imshow(gx_patch, cmap="bwr", vmin=-vmax, vmax=vmax)
        axes[row_idx, 6].set_title(f"Horizontal Gx (Sobel X)\n[-{vmax:.2f}, +{vmax:.2f}]", fontsize=9, fontweight="bold")
        axes[row_idx, 6].axis("off")
        fig.colorbar(im_gx, ax=axes[row_idx, 6], fraction=0.046, pad=0.04)

        # 7. Vertical Field Gy (Sobel Y)
        im_gy = axes[row_idx, 7].imshow(gy_patch, cmap="bwr", vmin=-vmax, vmax=vmax)
        axes[row_idx, 7].set_title(f"Vertical Gy (Sobel Y)\n[-{vmax:.2f}, +{vmax:.2f}]", fontsize=9, fontweight="bold")
        axes[row_idx, 7].axis("off")
        fig.colorbar(im_gy, ax=axes[row_idx, 7], fraction=0.046, pad=0.04)

        # 8. Combined Gradient Map (Red=Gx, Blue=Gy, Purple=Combined)
        axes[row_idx, 8].imshow(hv_rgb_patch)
        axes[row_idx, 8].set_title("Combined Gradient\n(Red=Gx, Blue=Gy)", fontsize=9, fontweight="bold")
        axes[row_idx, 8].axis("off")

    plt.tight_layout()
    output_file = output_dir / "monuseg_patches_visualization.png"
    plt.savefig(output_file, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"MoNuSeg visualization saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Plot image, masks, and SDF patches for MoNuSeg dataset.")
    parser.add_argument(
        "--patch_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "monuseg_train_patch_256",
        help="Path to pre-extracted MoNuSeg patch directory.",
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
        default=6,
        help="Number of patch samples to visualize (default: 6).",
    )
    args = parser.parse_args()

    plot_monuseg_patches(
        patch_dir=args.patch_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
