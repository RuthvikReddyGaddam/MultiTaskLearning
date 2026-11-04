# scripts/test_glas.py
"""
Evaluation script for the GlaS (Gland Segmentation) dataset.
Directly reads full-size images and ground truth masks (no DataLoader).
Patchifies images based on stride, overlap, or evenly-spaced minimal uniform overlap.
Reconstructs full-size predictions, and computes binary Dice, IoU, AJI, and PQ.
Reports separate evaluation scores for TestA, TestB, and Combined.
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
    colorize_multiclass_mask,
    create_tp_fp_fn_overlay,
    create_comparison_panel,
    create_dataset_panel,
    create_all_heads_panel,
    predict_whole_image_multitask,
    save_image_safe,
)


def find_glas_test_images(
    data_dir: Union[str, Path],
    split: str = "test",
) -> List[Tuple[str, str, str, str]]:
    """
    Scans data_dir for GlaS test images and pairs them with ground truth annotations.
    Returns list of tuples: (image_path, mask_path, sample_name, group).
    Group is 'testA' or 'testB' (or 'train' if split='train').
    By default, only test images ('testA' and 'testB', 80 images total) are returned.
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        # Check subdirectories (e.g. Warwick_QU_Dataset)
        sub = data_path / "Warwick_QU_Dataset"
        if sub.is_dir():
            data_path = sub

    all_files = sorted(list(data_path.glob("*.bmp")) + list(data_path.glob("*.png")) + list(data_path.glob("*.tif")))
    samples = []
    split_lower = (split or "test").lower()

    for f in all_files:
        stem = f.stem
        # Skip annotation files
        if stem.endswith("_anno"):
            continue

        stem_lower = stem.lower()
        if stem_lower.startswith("testa"):
            group = "testA"
        elif stem_lower.startswith("testb"):
            group = "testB"
        elif stem_lower.startswith("train"):
            group = "train"
        else:
            continue

        # Filter by split: standard evaluation uses testA and testB
        if split_lower in ("test", "all", "test_all", "both"):
            if group not in ("testA", "testB"):
                continue
        elif split_lower == "testa":
            if group != "testA":
                continue
        elif split_lower == "testb":
            if group != "testB":
                continue
        elif split_lower == "train":
            if group != "train":
                continue

        # Look for corresponding annotation file
        anno_candidates = [
            f.parent / f"{stem}_anno{f.suffix}",
            f.parent / f"{stem}_anno.bmp",
            f.parent / f"{stem}_anno.png",
        ]
        anno_path = None
        for cand in anno_candidates:
            if cand.exists():
                anno_path = cand
                break

        if anno_path is None:
            continue

        samples.append((str(f), str(anno_path), stem, group))

    return samples


def run_glas_eval(
    model: torch.nn.Module,
    samples: List[Tuple[str, str, str, str]],
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
    Evaluates GlaS on full-size images:
      - Patchifies with specified stride, overlap, or evenly-spaced starts.
      - Batched patch inference.
      - Whole-image stitching.
      - Computes Dice, IoU, AJI, and PQ.
      - Aggregates by TestA, TestB, and Combined.
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
    print("GLAS EVALUATION: FULL-IMAGE PATCH INFERENCE WITHOUT DATALOADERS")
    print(f"Total test samples found: {len(samples)}")
    print(f"Patch configuration: size={patch_size}, stride={stride}, overlap={overlap}, batch_size={patch_batch_size}")
    if stride is None and overlap is None:
        print("Patch spacing: Evenly-spaced minimal uniform overlap enabled.")
    print("=" * 90)

    results = []

    print(f"\n{'Sample':<16} {'Group':<8} {'Dice':>8} {'IoU':>8} {'AJI':>8} {'PQ':>8}")
    print("-" * 65)

    is_multitask = getattr(model, "multitask", False)

    with torch.inference_mode():
        for idx, (img_path, anno_path, name, group) in enumerate(tqdm(samples, desc="GlaS Testing")):
            if max_samples is not None and idx >= max_samples:
                break

            # 1. Read full-size image and GT mask
            img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            gt_anno = cv2.imread(anno_path, cv2.IMREAD_UNCHANGED)
            if gt_anno is None:
                continue

            gt_bin = (gt_anno > 0).astype(np.uint8)
            gt_inst = gt_anno.astype(np.int32)

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
                gland_out = multi_out.get("gland", {})
                full_pred_bin = gland_out.get("binary_mask", np.zeros((h, w), dtype=np.uint8))
                full_pred_inst = gland_out.get("instances", np.zeros((h, w), dtype=np.int32))
                full_prob = gland_out.get("probability", np.zeros((h, w), dtype=np.float32))
                colorized_4class = gland_out.get("colorized_4class", None)
            else:
                # 2. Patchify image
                patches, coords, orig_shape, padded_shape = patchify_image(
                    image=img_rgb,
                    patch_size=patch_size,
                    stride=stride,
                    overlap=overlap,
                )

                # 3. Model inference on patches
                patch_probs = []
                patch_flows = []
                has_flow = False
                is_multiclass = False
                num_classes = 1
                for i in range(0, len(patches), patch_batch_size):
                    chunk = patches[i : i + patch_batch_size].to(device, non_blocking=True)
                    raw_out = model(chunk)
                    seg_logits, _, flow_out = extract_model_outputs(raw_out, task="glas", return_flow=True)
                    if seg_logits.ndim == 3:
                        seg_logits = seg_logits.unsqueeze(1)

                    is_multiclass = (seg_logits.shape[1] > 1)
                    num_classes = seg_logits.shape[1]

                    # Match patch resolution if needed
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

                # 4. Stitch back full probability and flow maps
                full_flow = None
                if has_flow and len(patch_flows) == len(coords):
                    full_flow = stitch_flow_map(
                        patch_flows=patch_flows,
                        coordinates=coords,
                        orig_shape=orig_shape,
                        padded_shape=padded_shape,
                    )

                # 5. Instance labeling (Multi-class J-loss vs Flow integration)
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

            # 6. Calculate full image metrics
            scores = compute_binary_metrics(
                pred_binary=full_pred_bin,
                gt_binary=gt_bin,
                pred_instance=full_pred_inst,
                gt_instance=gt_inst,
                min_instance_size=min_instance_size,
            )

            row = {
                "Sample": name,
                "Group": group,
                "Dice": scores["Dice"],
                "IoU": scores["IoU"],
                "AJI": scores["AJI"],
                "PQ": scores["PQ"],
            }
            results.append(row)

            print(
                f"{name:<16} {group:<8} "
                f"{scores['Dice']:>8.4f} {scores['IoU']:>8.4f} "
                f"{scores['AJI']:>8.4f} {scores['PQ']:>8.4f}"
            )

            # 7. Save visual artifacts (both individual PNGs and combined multi-panel PNGs)
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
                if is_multitask and save_all_heads and "cell" in multi_out and "phenotype" in multi_out:
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

    # 8. Compute TestA, TestB, and Combined summaries
    summary_rows = []
    for grp in ["testA", "testB", "Combined"]:
        if grp == "Combined":
            sub_df = df_detailed
        else:
            sub_df = df_detailed[df_detailed["Group"].str.lower() == grp.lower()]

        if len(sub_df) > 0:
            summary_rows.append({
                "Group": grp,
                "Count": len(sub_df),
                "Dice_Mean": float(sub_df["Dice"].mean()),
                "Dice_Std": float(sub_df["Dice"].std()) if len(sub_df) > 1 else 0.0,
                "IoU_Mean": float(sub_df["IoU"].mean()),
                "IoU_Std": float(sub_df["IoU"].std()) if len(sub_df) > 1 else 0.0,
                "AJI_Mean": float(sub_df["AJI"].mean()),
                "AJI_Std": float(sub_df["AJI"].std()) if len(sub_df) > 1 else 0.0,
                "PQ_Mean": float(sub_df["PQ"].mean()),
                "PQ_Std": float(sub_df["PQ"].std()) if len(sub_df) > 1 else 0.0,
            })

    df_summary = pd.DataFrame(summary_rows)

    print("\n" + "=" * 90)
    print("GLAS EVALUATION SUMMARY RESULTS")
    print("=" * 90)
    print(df_summary.to_string(index=False))
    print("=" * 90 + "\n")

    if save_dir:
        csv_dir = os.path.join(save_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)
        df_detailed.to_csv(os.path.join(csv_dir, "glas_detailed.csv"), index=False)
        df_summary.to_csv(os.path.join(csv_dir, "glas_summary.csv"), index=False)
        print(f"Results saved to: {csv_dir}")

    return df_detailed, df_summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on GlaS test dataset without dataloaders.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt).")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to GlaS test directory.")
    parser.add_argument("--patch_size", type=int, default=None, help="Square patch size.")
    parser.add_argument("--stride", type=int, default=None, help="Step stride for patches.")
    parser.add_argument("--overlap", type=float, default=None, help="Overlap fraction in [0.0, 1.0).")
    parser.add_argument("--batch_size", type=int, default=4, help="Patch inference batch size.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary classification threshold.")
    parser.add_argument("--min_instance_size", type=int, default=10, help="Minimum connected component area.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save evaluation artifacts.")
    parser.add_argument(
        "--save_predictions",
        dest="save_predictions",
        action="store_true",
        default=True,
        help="Save prediction masks, ground truths, overlays, and panels (default: True).",
    )
    parser.add_argument(
        "--no_save_predictions",
        dest="save_predictions",
        action="store_false",
        help="Disable saving prediction and ground truth visual artifacts.",
    )
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
    parser.add_argument("--split", type=str, default=None, help="Evaluation split: 'test' (testA + testB), 'testA', 'testB', or 'train' (default: 'test').")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of evaluation samples.")
    parser.add_argument("--device", type=str, default=None, help="Compute device (e.g. cuda, cpu).")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device) if args.device else get_device()

    # Resolve data directory
    test_cfg = config.get("test", {})
    data_dir = args.data_dir
    if not data_dir:
        data_dir = test_cfg.get("data_dir", test_cfg.get("image_dir", "data/glas/Warwick_QU_Dataset"))
    if not os.path.exists(data_dir):
        data_dir = "data/glas/Warwick_QU_Dataset"

    # Resolve split
    split = args.split or test_cfg.get("split", "test")

    # Find test images
    samples = find_glas_test_images(data_dir, split=split)
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
        exp_name = config.get("experiment", {}).get("name", "glas_eval")
        save_dir = os.path.join("eval_results", exp_name, "glas")

    run_glas_eval(
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
