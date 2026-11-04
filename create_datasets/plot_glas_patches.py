# create_datasets/plot_glas_patches.py

import argparse
import sys
from pathlib import Path

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

from glas_utils import (
    colorize_instance_mask,
    compute_instance_sdf,
    extract_inside_distance,
    compute_inside_distance_map,
    compute_inside_distance_and_hv,
    compute_hv_gradients,
    create_combined_hv_rgb,
    get_patch_starts,
)


def create_patch_overlay(image: np.ndarray, instance_mask: np.ndarray) -> np.ndarray:
    """Overlays instance boundaries on top of the RGB image."""
    overlay = image.copy()
    inst = np.asarray(instance_mask, dtype=np.int32)
    foreground = inst > 0
    if not np.any(foreground):
        return overlay

    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(foreground.astype(np.uint8), kernel, iterations=1)
    boundary = foreground & (eroded == 0)

    # Draw yellow contour on boundary
    overlay[boundary] = [255, 230, 0]
    return overlay


def load_or_generate_patches(
    patch_dir: Path,
    full_data_dir: Path,
    patch_size: int,
    num_samples: int,
):
    """
    Loads patches from patch_dir if available, or generates them on-the-fly
    from full_data_dir for visualization.
    """
    samples = []

    # Check if pre-extracted patch_dir exists and has images
    img_dir = patch_dir / "images"
    mask_dir = patch_dir / "instance_masks"
    sdf_dir = patch_dir / "sdfs"
    hv_dir = patch_dir / "hv_maps"

    def _get_sample_tuple(p_name, img, mask, sdf):
        hv_path = hv_dir / f"{p_name}.npy"
        if hv_path.exists():
            hv = np.load(hv_path)
            gx, gy = (hv[0], hv[1]) if hv.shape[0] == 2 else (hv[:, :, 0], hv[:, :, 1])
            dist = compute_inside_distance_map(mask)
        else:
            dist, gx, gy = compute_inside_distance_and_hv(mask, method="sobel")
        hv_rgb = create_combined_hv_rgb(gx, gy, mask=mask > 0, bg_color="white")
        return (p_name, img, mask, sdf, dist, gx, gy, hv_rgb)

    if img_dir.is_dir() and any(img_dir.glob("*.npy")):
        patch_files = sorted(list(img_dir.glob("*.npy")))
        # Choose patches containing glands (foreground > 0)
        for p_img in patch_files:
            p_name = p_img.stem
            p_mask_path = mask_dir / f"{p_name}.npy"
            p_sdf_path = sdf_dir / f"{p_name}.npy"

            mask = np.load(p_mask_path)
            if np.any(mask > 0):
                img = np.load(p_img)
                sdf = np.load(p_sdf_path)
                samples.append(_get_sample_tuple(p_name, img, mask, sdf))
                if len(samples) >= num_samples:
                    break

        # Fallback if few patches have foreground
        if len(samples) < num_samples:
            for p_img in patch_files:
                p_name = p_img.stem
                if any(s[0] == p_name for s in samples):
                    continue
                img = np.load(p_img)
                mask = np.load(mask_dir / f"{p_name}.npy")
                sdf = np.load(sdf_dir / f"{p_name}.npy")
                samples.append(_get_sample_tuple(p_name, img, mask, sdf))
                if len(samples) >= num_samples:
                    break

        return samples

    # Otherwise generate patches on the fly from full_data_dir
    print(f"Pre-extracted patch directory {patch_dir} not found. Generating sample patches from {full_data_dir}...")
    full_images = sorted([f for f in full_data_dir.glob("train_*.npy") if "_anno" not in f.stem])

    for f_img in full_images:
        base_name = f_img.stem
        f_mask = full_data_dir / f"{base_name}_anno.npy"

        image = np.load(f_img)
        instance_mask = np.load(f_mask)
        h, w = image.shape[:2]

        sdf = compute_instance_sdf(instance_mask)
        dist_full, gx_full, gy_full = compute_inside_distance_and_hv(instance_mask, method="sobel")

        y_starts = get_patch_starts(h, patch_size)
        x_starts = get_patch_starts(w, patch_size)

        for p_idx, y in enumerate(y_starts):
            for x in x_starts:
                img_p = image[y:y + patch_size, x:x + patch_size]
                mask_p = instance_mask[y:y + patch_size, x:x + patch_size]
                sdf_p = sdf[y:y + patch_size, x:x + patch_size]
                dist_p = dist_full[y:y + patch_size, x:x + patch_size]
                gx_p = gx_full[y:y + patch_size, x:x + patch_size]
                gy_p = gy_full[y:y + patch_size, x:x + patch_size]
                hv_rgb_p = create_combined_hv_rgb(gx_p, gy_p, mask=mask_p > 0, bg_color="white")

                # Prioritize patches containing glands
                if np.any(mask_p > 0) or len(samples) == 0:
                    p_name = f"{base_name}_p{p_idx:02d}_{y}_{x}"
                    samples.append((p_name, img_p, mask_p, sdf_p, dist_p, gx_p, gy_p, hv_rgb_p))
                    if len(samples) >= num_samples:
                        return samples

    return samples


def plot_glas_patches(
    patch_dir: Path,
    full_data_dir: Path,
    output_dir: Path,
    patch_size: int = 256,
    num_samples: int = 6,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_or_generate_patches(patch_dir, full_data_dir, patch_size, num_samples)

    if not samples:
        raise ValueError("No patches found or generated to plot.")

    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 8, figsize=(25, 3.4 * n_rows), dpi=200)
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    for row_idx, (name, img_patch, mask_patch, sdf_patch, dist_patch, gx_patch, gy_patch, hv_rgb_patch) in enumerate(samples):
        # 0. Image
        axes[row_idx, 0].imshow(img_patch)
        axes[row_idx, 0].set_title(f"{name}\nRGB Image", fontsize=9, fontweight="bold")
        axes[row_idx, 0].axis("off")

        # 1. Instance Mask
        colored_mask = colorize_instance_mask(mask_patch)
        axes[row_idx, 1].imshow(colored_mask)
        num_glands = len(np.unique(mask_patch[mask_patch > 0]))
        axes[row_idx, 1].set_title(f"Instance Mask\n({num_glands} glands)", fontsize=9, fontweight="bold")
        axes[row_idx, 1].axis("off")

        # 2. Overlay
        overlay = create_patch_overlay(img_patch, mask_patch)
        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].set_title("Boundary Overlay\n(Yellow = Interface)", fontsize=9, fontweight="bold")
        axes[row_idx, 2].axis("off")

        # 3. SDF Map (Diverging Colormap)
        max_abs = max(1.0, float(np.max(np.abs(sdf_patch))))
        im_sdf = axes[row_idx, 3].imshow(sdf_patch, cmap="bwr", vmin=-max_abs, vmax=max_abs)
        axes[row_idx, 3].set_title(
            f"Signed Distance Field\n[{sdf_patch.min():.1f}, {sdf_patch.max():.1f}]",
            fontsize=9,
            fontweight="bold",
        )
        axes[row_idx, 3].axis("off")
        fig.colorbar(im_sdf, ax=axes[row_idx, 3], fraction=0.046, pad=0.04)

        # 4. Inside Cell Distance Transform
        im_dt = axes[row_idx, 4].imshow(dist_patch, cmap="viridis", vmin=0)
        axes[row_idx, 4].set_title(
            f"Inside Cell Dist\n[0, {dist_patch.max():.1f} px]",
            fontsize=9,
            fontweight="bold",
        )
        axes[row_idx, 4].axis("off")
        fig.colorbar(im_dt, ax=axes[row_idx, 4], fraction=0.046, pad=0.04)

        # 5. Horizontal Field Gx (Sobel X)
        vmax = max(0.2, min(1.0, float(max(np.max(np.abs(gx_patch)), np.max(np.abs(gy_patch))))))
        im_gx = axes[row_idx, 5].imshow(gx_patch, cmap="bwr", vmin=-vmax, vmax=vmax)
        axes[row_idx, 5].set_title(f"Horizontal Gx (Sobel X)\n[-{vmax:.2f}, +{vmax:.2f}]", fontsize=9, fontweight="bold")
        axes[row_idx, 5].axis("off")
        fig.colorbar(im_gx, ax=axes[row_idx, 5], fraction=0.046, pad=0.04)

        # 6. Vertical Field Gy (Sobel Y)
        im_gy = axes[row_idx, 6].imshow(gy_patch, cmap="bwr", vmin=-vmax, vmax=vmax)
        axes[row_idx, 6].set_title(f"Vertical Gy (Sobel Y)\n[-{vmax:.2f}, +{vmax:.2f}]", fontsize=9, fontweight="bold")
        axes[row_idx, 6].axis("off")
        fig.colorbar(im_gy, ax=axes[row_idx, 6], fraction=0.046, pad=0.04)

        # 7. Combined Gradient Map (Red=Gx, Blue=Gy, Purple=Combined)
        axes[row_idx, 7].imshow(hv_rgb_patch)
        axes[row_idx, 7].set_title("Combined Gradient\n(Red=Gx, Blue=Gy)", fontsize=9, fontweight="bold")
        axes[row_idx, 7].axis("off")

    plt.tight_layout()
    output_file = output_dir / "glas_patches_visualization.png"
    plt.savefig(output_file, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"Visualization saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Plot image, instance mask, and SDF patches for GlaS dataset.")
    parser.add_argument(
        "--patch_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "glas_train_patch_256",
        help="Path to pre-extracted patch directory.",
    )
    parser.add_argument(
        "--full_data_dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "glas_train",
        help="Path to full-size GlaS train images and masks (used if patch_dir not found).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "dataset_visualizations" / "intermediate outputs",
        help="Path to intermediate outputs directory.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=256,
        help="Patch size (default: 256).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=6,
        help="Number of patch samples to visualize (default: 6).",
    )
    args = parser.parse_args()

    plot_glas_patches(
        patch_dir=args.patch_dir,
        full_data_dir=args.full_data_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
