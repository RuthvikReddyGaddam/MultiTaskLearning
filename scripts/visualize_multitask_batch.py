# scripts/visualize_multitask_batch.py

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, hsv_to_rgb
import matplotlib.patches as mpatches

from datasets import build_multitask_dataloader

CELL_CLASSES = ["Background", "Epithelial", "Lymphocyte", "Neutrophil", "Macrophage"]
CELL_COLORS = ["#000000", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]


def flow_to_hsv(flow: np.ndarray) -> np.ndarray:
    """
    Converts 2D flow field [2, H, W] to RGB image via HSV color wheel.
    - Hue: direction angle atan2(Fy, Fx) mapped to [0, 1]
    - Saturation: 1.0 on foreground, 0 on background
    - Value: vector magnitude clamped to [0, 1]
    Background (Fx=0, Fy=0) renders as pure black.
    """
    fx, fy = flow[0], flow[1]
    mag = np.sqrt(fx**2 + fy**2)
    angle = np.arctan2(fy, fx)  # [-pi, pi]

    h = (angle + np.pi) / (2 * np.pi)  # [0, 1]
    s = np.ones_like(h)
    s[mag < 1e-4] = 0.0
    v = np.clip(mag, 0.0, 1.0)

    hsv = np.stack([h, s, v], axis=-1)
    rgb = hsv_to_rgb(hsv)
    return rgb


def visualize_batch(save_path: str = "eval_results/smoothed_flow_512_preview.png"):
    print("Building multi-task 512 DataLoader...")
    loader = build_multitask_dataloader(
        patch_size=512,
        batch_size=6,
        num_workers=0,
        monusac_resample_size=600,
        augment=False,
    )

    batch = next(iter(loader))
    images = batch["image"].numpy()         # [6, 3, 512, 512]
    binary_masks = batch["binary_mask"].numpy()  # [6, 1, 512, 512]
    sdfs = batch["sdf"].numpy()             # [6, 1, 512, 512]
    flow_vectors = batch["flow_vector"].numpy()  # [6, 2, 512, 512]
    class_maps = batch["class_map"].numpy() # [6, 512, 512]
    task_ids = batch["task_id"].numpy()     # [6]
    task_names = ["GLaS (Gland)", "MoNuSeg (Nuclei)", "MoNuSAC (4-Class Nuclei)"]

    num_samples = len(task_ids)
    fig, axes = plt.subplots(num_samples, 5, figsize=(22, 4.2 * num_samples))
    plt.subplots_adjust(wspace=0.15, hspace=0.25)

    custom_cmap = ListedColormap(CELL_COLORS)

    for i in range(num_samples):
        tid = task_ids[i]
        tname = task_names[tid]

        # 1. RGB Image
        img = np.transpose(images[i], (1, 2, 0))
        img = np.clip(img, 0.0, 1.0)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"[{tname}] Sample #{i+1}\nRGB Image (512x512)", fontsize=11, fontweight="bold")
        axes[i, 0].axis("off")

        # 2. Binary Mask
        mask = binary_masks[i, 0]
        axes[i, 1].imshow(mask, cmap="gray")
        fg_pct = (np.count_nonzero(mask) / mask.size) * 100.0
        axes[i, 1].set_title(f"Binary Mask\nForeground: {fg_pct:.1f}%", fontsize=11)
        axes[i, 1].axis("off")

        # 3. Signed Distance Function (SDF)
        sdf = sdfs[i, 0]
        im_sdf = axes[i, 2].imshow(sdf, cmap="coolwarm")
        fig.colorbar(im_sdf, ax=axes[i, 2], fraction=0.046, pad=0.04)
        axes[i, 2].set_title(f"Signed Distance (SDF)\nMin: {sdf.min():.1f}, Max: {sdf.max():.1f}", fontsize=11)
        axes[i, 2].axis("off")

        # 4. Smoothed 2D Flow Field (HSV Orientation & Magnitude)
        flow = flow_vectors[i]
        flow_rgb = flow_to_hsv(flow)
        axes[i, 3].imshow(flow_rgb)
        max_mag = np.sqrt(flow[0]**2 + flow[1]**2).max()
        axes[i, 3].set_title(f"Smoothed Flow Field (HSV)\nMax Mag: {max_mag:.2f} (Zero at BG)", fontsize=11)
        axes[i, 3].axis("off")

        # 5. Class Map or Quiver Overlay
        if tid == 2:
            # MoNuSAC: Show Multi-Class Map
            cmap_img = class_maps[i]
            # Map -1 (ignore) to 0
            cmap_display = np.where(cmap_img < 0, 0, cmap_img)
            im_cls = axes[i, 4].imshow(cmap_display, cmap=custom_cmap, vmin=0, vmax=4)
            present = [CELL_CLASSES[c] for c in np.unique(cmap_display) if c > 0]
            axes[i, 4].set_title(f"MoNuSAC Cell Classes\nPresent: {', '.join(present) if present else 'None'}", fontsize=11)
            axes[i, 4].axis("off")
        else:
            # GLaS / MoNuSeg: Show Quiver Flow Overlay
            step = 16
            y_coords, x_coords = np.mgrid[0:512:step, 0:512:step]
            u = flow[0, ::step, ::step]
            v = flow[1, ::step, ::step]
            mag_sub = np.sqrt(u**2 + v**2)
            # Mask out background arrows
            u_masked = np.where(mag_sub > 0.05, u, np.nan)
            v_masked = np.where(mag_sub > 0.05, v, np.nan)

            axes[i, 4].imshow(mask, cmap="bone", alpha=0.5)
            axes[i, 4].quiver(
                x_coords, y_coords, u_masked, -v_masked,
                color="red", scale=25, width=0.004, headwidth=3, headlength=4,
            )
            axes[i, 4].set_title(f"Vector Quiver Overlay\n(Arrow points toward center)", fontsize=11)
            axes[i, 4].axis("off")

    # Add Class Legend at top right
    patches = [mpatches.Patch(color=CELL_COLORS[i], label=CELL_CLASSES[i]) for i in range(1, 5)]
    fig.legend(handles=patches, loc="upper right", bbox_to_anchor=(0.99, 0.995), fontsize=12, ncol=4)

    plt.suptitle("Multi-Task Balanced Batch (512x512): GLaS (2), MoNuSeg (2), MoNuSAC (2)", fontsize=16, fontweight="bold", y=0.998)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[*] Visual preview saved to: {save_path}")


if __name__ == "__main__":
    visualize_batch()

