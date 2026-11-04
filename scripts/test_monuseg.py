# scripts/test_monuseg.py
"""
Evaluation script for the MoNuSeg (Multi-Organ Nuclei Segmentation) test dataset.
Directly reads full-size images and ground truth masks (no DataLoader).
Patchifies images based on stride, overlap, or evenly-spaced minimal uniform overlap.
Reconstructs full-size predictions, and computes binary Dice, IoU, AJI, and PQ.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config
from utils.device import get_device
from utils.instance_utils import xml_to_instance_mask
from utils.test_utils import (
    patchify_image,
    stitch_probability_map,
    stitch_multiclass_probability_map,
    stitch_flow_map,
    binary_to_instances,
    flow_to_instances,
    class3_to_instances,
    class4_to_instances,
    compute_binary_metrics,
    extract_model_outputs,
    load_model_for_evaluation,
    colorize_instance_mask,
    colorize_4class_map,
    create_tp_fp_fn_overlay,
    create_comparison_panel,
    create_dataset_panel,
    create_all_heads_panel,
    predict_whole_image_multitask,
    save_image_safe,
)


def find_monuseg_test_images(data_dir: Union[str, Path]) -> List[Tuple[str, Optional[str], Optional[str], str]]:
    """
    Scans data_dir for MoNuSeg test images and pairs them with ground truth XML or mask files.
    Returns list of tuples: (image_path, xml_path_or_none, mask_path_or_none, sample_name).
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        cand = Path("data/MoNuSegTestData")
        if cand.is_dir():
            data_path = cand

    valid_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    # Filter files in the top-level directory (avoid subdirs like masks/ or instance_masks/)
    image_files = sorted([
        f for f in data_path.iterdir()
        if f.is_file() and f.suffix.lower() in valid_exts and not f.stem.endswith("_mask") and not f.stem.endswith("_pred")
    ])

    samples = []
    for f in image_files:
        stem = f.stem
        # 1. Look for XML
        xml_cand = f.parent / f"{stem}.xml"
        xml_path = str(xml_cand) if xml_cand.exists() else None

        # 2. Look for mask PNG
        mask_cands = [
            f.parent / "masks" / f"{stem}_mask.png",
            f.parent / "masks" / f"{stem}.png",
            f.parent / f"{stem}_mask.png",
            f.parent / "instance_masks" / f"{stem}_instance.png",
            f.parent / "instance_masks" / f"{stem}.png",
        ]
        mask_path = None
        for m in mask_cands:
            if m.exists():
                mask_path = str(m)
                break

        samples.append((str(f), xml_path, mask_path, stem))

    return samples


def run_monuseg_eval(
    model: torch.nn.Module,
    samples: List[Tuple[str, Optional[str], Optional[str], str]],
    device: torch.device,
    patch_size: int = 256,
    stride: Optional[int] = None,
    overlap: Optional[float] = None,
    patch_batch_size: int = 4,
    threshold: float = 0.5,
    min_instance_size: int = 10,
    save_dir: Optional[str] = None,
    save_predictions: bool = True,
    save_all_heads: bool = True,
    max_samples: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates MoNuSeg on full-size test images:
      - Patchifies with specified stride, overlap, or evenly-spaced minimal uniform overlap.
      - Batched patch inference.
      - Whole-image probability map stitching.
      - Computes Dice, IoU, AJI, and PQ.
      - Saves both individual PNGs and combined multi-panel PNGs for dataset-specific and all-heads predictions.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if save_predictions:
            os.makedirs(os.path.join(save_dir, "predictions"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "probabilities"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "ground_truth"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "overlays"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "panels"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "individual"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "combined"), exist_ok=True)

    print("\n" + "=" * 90)
    print("MONUSEG EVALUATION: FULL-IMAGE PATCH INFERENCE WITHOUT DATALOADERS")
    print(f"Total test samples found: {len(samples)}")
    print(f"Patch configuration: size={patch_size}, stride={stride}, overlap={overlap}, batch_size={patch_batch_size}")
    if stride is None and overlap is None:
        print("Patch spacing: Evenly-spaced minimal uniform overlap enabled.")
    print("=" * 90)

    results = []

    print(f"\n{'Sample':<28} {'Dice':>8} {'IoU':>8} {'AJI':>8} {'PQ':>8}")
    print("-" * 65)

    is_multitask = getattr(model, "multitask", False)

    with torch.inference_mode():
        for idx, (img_path, xml_path, mask_path, name) in enumerate(tqdm(samples, desc="MoNuSeg Testing")):
            if max_samples is not None and idx >= max_samples:
                break

            # 1. Read full-size image
            img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            # 2. Resolve ground truth annotations
            gt_inst = None
            if xml_path and os.path.exists(xml_path):
                gt_inst = xml_to_instance_mask(xml_path, h, w).astype(np.int32)
                gt_bin = (gt_inst > 0).astype(np.uint8)
            elif mask_path and os.path.exists(mask_path):
                m_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if m_img.ndim == 3:
                    m_img = m_img[..., 0]
                if m_img.max() > 1 and len(np.unique(m_img)) > 2:
                    gt_inst = m_img.astype(np.int32)
                    gt_bin = (gt_inst > 0).astype(np.uint8)
                else:
                    gt_bin = (m_img > 0).astype(np.uint8)
                    gt_inst = binary_to_instances(gt_bin, min_size=min_instance_size)
            else:
                print(f"[Warning] No ground truth found for {name}. Skipping.")
                continue

            multi_out = {}
            colorized_4class = None

            if is_multitask:
                # Whole-image multi-task inference across all heads
                multi_out = predict_whole_image_multitask(
                    model=model,
                    image_rgb=img_rgb,
                    device=device,
                    patch_size=patch_size,
                    stride=stride,
                    overlap=overlap,
                    patch_batch_size=patch_batch_size,
                    min_instance_size=min_instance_size,
                )
                cell_out = multi_out.get("cell", {})
                full_pred_bin = cell_out.get("binary_mask", np.zeros((h, w), dtype=np.uint8))
                full_pred_inst = cell_out.get("instances", np.zeros((h, w), dtype=np.int32))
                full_prob = cell_out.get("probability", np.zeros((h, w), dtype=np.float32))
                colorized_4class = cell_out.get("colorized_4class", None)
            else:
                # 3. Patchify image
                patches, coords, orig_shape, padded_shape = patchify_image(
                    image=img_rgb,
                    patch_size=patch_size,
                    stride=stride,
                    overlap=overlap,
                )

                # 4. Model inference on patches
                patch_probs = []
                patch_flows = []
                has_flow = False
                is_multiclass = False
                num_classes = 1
                for i in range(0, len(patches), patch_batch_size):
                    chunk = patches[i : i + patch_batch_size].to(device, non_blocking=True)
                    raw_out = model(chunk)
                    seg_logits, _, flow_out = extract_model_outputs(raw_out, task="monuseg", return_flow=True)
                    if seg_logits.ndim == 3:
                        seg_logits = seg_logits.unsqueeze(1)

                    is_multiclass = (seg_logits.shape[1] > 1)
                    num_classes = seg_logits.shape[1]

                    p_h, p_w = chunk.shape[-2:]
                    if seg_logits.shape[-2:] != (p_h, p_w):
                        seg_logits = F.interpolate(seg_logits, size=(p_h, p_w), mode="bilinear", align_corners=False)

                    if is_multiclass:
                        probs = F.softmax(seg_logits, dim=1).detach().cpu().numpy()
                        patch_probs.extend(probs)
                    else:
                        probs = torch.sigmoid(seg_logits)[:, 0].detach().cpu().numpy()
                        patch_probs.extend(probs)

                    if flow_out is not None:
                        has_flow = True
                        if flow_out.shape[-2:] != (p_h, p_w):
                            flow_out = F.interpolate(flow_out, size=(p_h, p_w), mode="bilinear", align_corners=False)
                        patch_flows.extend(flow_out.detach().cpu().numpy())

                # 5. Stitch back full probability and flow maps
                full_flow = None
                if has_flow and len(patch_flows) == len(coords):
                    full_flow = stitch_flow_map(
                        patch_flows=patch_flows,
                        coordinates=coords,
                        orig_shape=orig_shape,
                        padded_shape=padded_shape,
                    )

                # 6. Instance labeling (Multi-class J-loss vs Flow integration)
                if is_multiclass:
                    full_cls_prob = stitch_multiclass_probability_map(
                        patch_probs=patch_probs,
                        coordinates=coords,
                        orig_shape=orig_shape,
                        padded_shape=padded_shape,
                    )
                    if num_classes == 4:
                        full_pred_inst = class4_to_instances(full_cls_prob, min_size=min_instance_size)
                        full_4class = np.argmax(full_cls_prob, axis=0)
                        colorized_4class = colorize_4class_map(full_4class)
                        full_prob = full_cls_prob[1] + full_cls_prob[2]
                    elif num_classes == 3:
                        full_pred_inst = class3_to_instances(full_cls_prob, min_size=min_instance_size)
                        full_4class = np.argmax(full_cls_prob, axis=0)
                        colorized_4class = colorize_4class_map(full_4class)
                        full_prob = full_cls_prob[1] + full_cls_prob[2]
                    else:
                        pred_labels = np.argmax(full_cls_prob, axis=0)
                        full_pred_inst = binary_to_instances((pred_labels > 0).astype(np.uint8), min_size=min_instance_size)
                        full_prob = 1.0 - full_cls_prob[0]
                    full_pred_bin = (full_pred_inst > 0).astype(np.uint8)
                else:
                    full_prob = stitch_probability_map(
                        patch_probs=patch_probs,
                        coordinates=coords,
                        orig_shape=orig_shape,
                        padded_shape=padded_shape,
                    )
                    full_pred_bin = (full_prob >= threshold).astype(np.uint8)
                    full_pred_inst = flow_to_instances(full_pred_bin, flow_vector=full_flow, min_size=min_instance_size)

            # 7. Calculate full image metrics
            scores = compute_binary_metrics(
                pred_binary=full_pred_bin,
                gt_binary=gt_bin,
                pred_instance=full_pred_inst,
                gt_instance=gt_inst,
                min_instance_size=min_instance_size,
            )

            row = {
                "Sample": name,
                "Dice": scores["Dice"],
                "IoU": scores["IoU"],
                "AJI": scores["AJI"],
                "PQ": scores["PQ"],
            }
            results.append(row)

            print(
                f"{name:<28} "
                f"{scores['Dice']:>8.4f} {scores['IoU']:>8.4f} "
                f"{scores['AJI']:>8.4f} {scores['PQ']:>8.4f}"
            )

            # 8. Save visual artifacts (both individual PNGs and combined multi-panel PNGs)
            if save_dir and save_predictions:
                # Traditional directory outputs
                save_image_safe(os.path.join(save_dir, "ground_truth", f"{name}_gt_mask.png"), gt_bin * 255)
                save_image_safe(os.path.join(save_dir, "ground_truth", f"{name}_gt_instance.png"), colorize_instance_mask(gt_inst))
                save_image_safe(os.path.join(save_dir, "ground_truth", f"{name}_gt_instance_ids.png"), gt_inst.astype(np.uint16))

                save_image_safe(os.path.join(save_dir, "predictions", f"{name}_pred_mask.png"), full_pred_bin * 255)
                save_image_safe(os.path.join(save_dir, "predictions", f"{name}_pred_instance.png"), colorize_instance_mask(full_pred_inst))
                save_image_safe(os.path.join(save_dir, "predictions", f"{name}_pred_instance_ids.png"), full_pred_inst.astype(np.uint16))

                prob_heatmap = (np.clip(full_prob, 0.0, 1.0) * 255).astype(np.uint8)
                prob_color = cv2.applyColorMap(prob_heatmap, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(save_dir, "probabilities", f"{name}_prob.png"), prob_color)

                overlay = create_tp_fp_fn_overlay(img_rgb, gt_bin, full_pred_bin)
                save_image_safe(os.path.join(save_dir, "overlays", f"{name}_overlay.png"), overlay)

                panel = create_comparison_panel(img_rgb, gt_bin, full_pred_bin, overlay)
                save_image_safe(os.path.join(save_dir, "panels", f"{name}_panel.png"), panel)

                # --- 1. INDIVIDUAL PNGs ---
                indiv_dir = os.path.join(save_dir, "individual", name)
                save_image_safe(os.path.join(indiv_dir, "original_image.png"), img_rgb)
                save_image_safe(os.path.join(indiv_dir, "gt_mask.png"), gt_bin * 255)
                save_image_safe(os.path.join(indiv_dir, "gt_instance.png"), colorize_instance_mask(gt_inst))
                save_image_safe(os.path.join(indiv_dir, "pred_binary_mask.png"), full_pred_bin * 255)
                save_image_safe(os.path.join(indiv_dir, "pred_instance.png"), colorize_instance_mask(full_pred_inst))
                save_image_safe(os.path.join(indiv_dir, "pred_probability.png"), prob_color)
                save_image_safe(os.path.join(indiv_dir, "error_overlay.png"), overlay)
                if colorized_4class is not None:
                    save_image_safe(os.path.join(indiv_dir, "pred_4class.png"), colorized_4class)

                # --- 2. COMBINED PNGs ---
                comb_dir = os.path.join(save_dir, "combined")
                ds_panel = create_dataset_panel(img_rgb, gt_inst, full_pred_inst, overlay)
                save_image_safe(os.path.join(comb_dir, f"{name}_dataset_panel.png"), ds_panel)

                # --- 3. ALL HEADS OUTPUT (Individual + Combined) ---
                if is_multitask and save_all_heads and "gland" in multi_out and "phenotype" in multi_out:
                    all_heads_dir = os.path.join(indiv_dir, "all_heads")
                    save_image_safe(os.path.join(all_heads_dir, "gland_instances.png"), colorize_instance_mask(multi_out["gland"]["instances"]))
                    save_image_safe(os.path.join(all_heads_dir, "gland_4class.png"), multi_out["gland"]["colorized_4class"])
                    save_image_safe(os.path.join(all_heads_dir, "cell_instances.png"), colorize_instance_mask(multi_out["cell"]["instances"]))
                    save_image_safe(os.path.join(all_heads_dir, "cell_4class.png"), multi_out["cell"]["colorized_4class"])
                    save_image_safe(os.path.join(all_heads_dir, "phenotypes.png"), multi_out["phenotype"]["colorized_classes"])

                    all_heads_panel = create_all_heads_panel(
                        image=img_rgb,
                        gland_inst=multi_out["gland"]["instances"],
                        cell_inst=multi_out["cell"]["instances"],
                        pheno_rgb=multi_out["phenotype"]["colorized_classes"],
                    )
                    save_image_safe(os.path.join(comb_dir, f"{name}_all_heads_panel.png"), all_heads_panel)

    df_detailed = pd.DataFrame(results)

    # 9. Compute summary statistics
    metrics_to_agg = ["Dice", "IoU", "AJI", "PQ"]
    summary_rows = []
    for m in metrics_to_agg:
        vals = df_detailed[m].dropna()
        summary_rows.append({
            "Metric": m,
            "Mean": float(vals.mean()),
            "Std": float(vals.std()) if len(vals) > 1 else 0.0,
            "Median": float(vals.median()),
            "Min": float(vals.min()),
            "Max": float(vals.max()),
        })

    df_summary = pd.DataFrame(summary_rows)

    print("\n" + "=" * 80)
    print("MONUSEG EVALUATION SUMMARY RESULTS")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80 + "\n")

    if save_dir:
        csv_dir = os.path.join(save_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)
        df_detailed.to_csv(os.path.join(csv_dir, "monuseg_detailed.csv"), index=False)
        df_summary.to_csv(os.path.join(csv_dir, "monuseg_summary.csv"), index=False)
        print(f"Results saved to: {csv_dir}")

    return df_detailed, df_summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on MoNuSeg test dataset without dataloaders.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt).")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to MoNuSeg test directory.")
    parser.add_argument("--patch_size", type=int, default=None, help="Square patch size.")
    parser.add_argument("--stride", type=int, default=None, help="Step stride for patches.")
    parser.add_argument("--overlap", type=float, default=None, help="Overlap fraction in [0.0, 1.0).")
    parser.add_argument("--batch_size", type=int, default=4, help="Patch inference batch size.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary classification threshold.")
    parser.add_argument("--min_instance_size", type=int, default=10, help="Minimum connected component area.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save evaluation artifacts.")
    parser.add_argument("--save_predictions", dest="save_predictions", action="store_true", default=True, help="Save prediction masks, instances, and overlays (default: True).")
    parser.add_argument("--no_save_predictions", dest="save_predictions", action="store_false", help="Disable saving prediction artifacts.")
    parser.add_argument(
        "--save_all_heads",
        dest="save_all_heads",
        action="store_true",
        default=True,
        help="Save predictions from all heads when evaluating a multi-task model (default: True).",
    )
    parser.add_argument(
        "--no_save_all_heads",
        dest="save_all_heads",
        action="store_false",
        help="Disable saving multi-task all heads predictions.",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of evaluation samples.")
    parser.add_argument("--device", type=str, default=None, help="Compute device (e.g. cuda, cpu).")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device) if args.device else get_device()

    # Resolve data directory
    data_dir = args.data_dir
    if not data_dir:
        test_cfg = config.get("test", {})
        data_dir = test_cfg.get("data_dir", test_cfg.get("image_dir", "data/MoNuSegTestData"))
    if not os.path.exists(data_dir):
        data_dir = "data/MoNuSegTestData"

    # Find test images
    samples = find_monuseg_test_images(data_dir)
    if not samples:
        raise FileNotFoundError(f"No test images found in '{data_dir}'.")

    # Load model
    model = load_model_for_evaluation(config, checkpoint_path=args.checkpoint, device=device)

    # Patch configuration
    test_cfg = config.get("test", {})
    patch_size = args.patch_size or test_cfg.get("patch_size", 256)
    stride = args.stride or test_cfg.get("stride", None)
    overlap = args.overlap if args.overlap is not None else test_cfg.get("overlap", None)

    # Save directory
    save_dir = args.save_dir
    if not save_dir:
        exp_name = config.get("experiment", {}).get("name", "monuseg_eval")
        save_dir = os.path.join("eval_results", exp_name, "monuseg")

    run_monuseg_eval(
        model=model,
        samples=samples,
        device=device,
        patch_size=patch_size,
        stride=stride,
        overlap=overlap,
        patch_batch_size=args.batch_size,
        threshold=args.threshold,
        min_instance_size=args.min_instance_size,
        save_dir=save_dir,
        save_predictions=args.save_predictions,
        save_all_heads=args.save_all_heads,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
