# scripts/test_monusac.py
"""
Evaluation script for the MoNuSAC testing dataset.
Directly reads full-size images and ground truth annotations from patient directories (no DataLoader).
Patchifies images based on stride, overlap, or evenly-spaced minimal uniform overlap.
Reconstructs full-size dual-head predictions (binary segmentation + cell classification).
Calculates full image metrics:
  1. Full dataset metrics: Dice, IoU, AJI, PQ (both binary and per-class).
  2. Per-patient averaged metrics: Dice, IoU, AJI, PQ (both binary and per-class).
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

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
    stitch_flow_map,
    stitch_multiclass_probability_map,
    binary_to_instances,
    flow_to_instances,
    class3_to_instances,
    class4_to_instances,
    compute_binary_metrics,
    compute_monusac_cell_metrics,
    extract_model_outputs,
    load_model_for_evaluation,
    colorize_instance_mask,
    colorize_multiclass_mask,
    colorize_4class_map,
    create_tp_fp_fn_overlay,
    create_comparison_panel,
    create_dataset_panel,
    create_monusac_dataset_panel,
    create_all_heads_panel,
    predict_whole_image_multitask,
    save_image_safe,
    MONUSAC_CELL_NAMES,
)

Image.MAX_IMAGE_PIXELS = None
CLASS_TO_IDX = {name.lower(): idx for idx, name in enumerate(MONUSAC_CELL_NAMES)}


def parse_monusac_xml(xml_path: Union[str, Path], height: int, width: int) -> np.ndarray:
    """
    Parses MoNuSAC Aperio XML annotations into a 4-channel instance mask [4, H, W].
    Channels: 0: Epithelial, 1: Lymphocyte, 2: Neutrophil, 3: Macrophage.
    """
    mask_4ch = np.zeros((4, height, width), dtype=np.int32)
    if not os.path.exists(str(xml_path)):
        return mask_4ch

    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except Exception as e:
        print(f"[Warning] Could not parse XML {xml_path}: {e}")
        return mask_4ch

    for annotation in root.findall(".//Annotation"):
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
                pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
                cv2.fillPoly(mask_4ch[class_idx], [pts], color=int(instance_id))
                instance_id += 1

    return mask_4ch


def find_monusac_test_samples(data_dir: Union[str, Path]) -> List[Tuple[str, str, str, str]]:
    """
    Scans MoNuSAC testing directory structured by patient folders.
    Returns list of tuples: (image_path, xml_path, patient_id, subimage_name).
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        cand = Path("data/MoNuSAC Testing Data and Annotations")
        if cand.is_dir():
            data_path = cand

    patient_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    samples = []

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        xml_files = sorted(patient_dir.glob("*.xml"))

        for xml_path in xml_files:
            sub_name = xml_path.stem
            # Find matching image
            img_cand_tif = patient_dir / f"{sub_name}.tif"
            img_cand_svs = patient_dir / f"{sub_name}.svs"
            img_cand_png = patient_dir / f"{sub_name}.png"

            if img_cand_tif.exists():
                img_path = img_cand_tif
            elif img_cand_svs.exists():
                img_path = img_cand_svs
            elif img_cand_png.exists():
                img_path = img_cand_png
            else:
                continue

            samples.append((str(img_path), str(xml_path), patient_id, sub_name))

    return samples


def run_monusac_eval(
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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluates MoNuSAC full-size test images:
      - Patchifies with specified stride, overlap, or evenly-spaced minimal uniform overlap.
      - Dual-head model inference on patches.
      - Whole-image stitching for segmentation and cell classification heads.
      - Computes:
        * Full dataset metrics (binary and per-class Dice, IoU, AJI, PQ).
        * Per-patient averaged metrics (binary and per-class Dice, IoU, AJI, PQ).
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

    print("\n" + "=" * 105)
    print("MONUSAC EVALUATION: FULL-IMAGE PATCH INFERENCE WITHOUT DATALOADERS")
    print(f"Total test sub-images found: {len(samples)} across {len(set(s[2] for s in samples))} patients")
    print(f"Patch configuration: size={patch_size}, stride={stride}, overlap={overlap}, batch_size={patch_batch_size}")
    if stride is None and overlap is None:
        print("Patch spacing: Evenly-spaced minimal uniform overlap enabled.")
    print("=" * 105)

    results = []

    print(f"\n{'Patient':<24} {'Subimage':<26} {'BinDice':>8} {'BinIoU':>8} {'BinAJI':>8} {'BinPQ':>8} {'ClassMeanDice':>14}")
    print("-" * 105)

    with torch.inference_mode():
        for idx, (img_path, xml_path, patient_id, sub_name) in enumerate(tqdm(samples, desc="MoNuSAC Testing")):
            if max_samples is not None and idx >= max_samples:
                break

            # 1. Read full-size image
            with Image.open(img_path) as pil_img:
                img_rgb = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
            h, w = img_rgb.shape[:2]

            # 2. Parse 4-channel ground truth XML annotations
            gt_inst_4ch = parse_monusac_xml(xml_path, h, w)
            gt_class_masks = (gt_inst_4ch > 0).astype(np.uint8)
            gt_bin = (gt_inst_4ch > 0).any(axis=0).astype(np.uint8)
            gt_bin_inst = binary_to_instances(gt_bin, min_size=min_instance_size)

            is_multitask = getattr(model, "multitask", False)
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
                full_pred_bin_inst = cell_out.get("instances", np.zeros((h, w), dtype=np.int32))
                full_prob = cell_out.get("probability", np.zeros((h, w), dtype=np.float32))
                colorized_4class = cell_out.get("colorized_4class", None)

                pheno_out = multi_out.get("phenotype", {})
                pred_class_masks = pheno_out.get("class_masks", np.zeros((4, h, w), dtype=np.uint8))
                pred_classes_color = pheno_out.get("colorized_classes", np.zeros((h, w, 3), dtype=np.uint8))
            else:
                # 3. Patchify image
                patches, coords, orig_shape, padded_shape = patchify_image(
                    image=img_rgb,
                    patch_size=patch_size,
                    stride=stride,
                    overlap=overlap,
                )

                # 4. Model inference on patches
                patch_seg_probs = []
                patch_cls_probs = []
                patch_flows = []
                has_cls_head = False
                has_flow = False
                is_multiclass = False
                num_seg_classes = 1

                for i in range(0, len(patches), patch_batch_size):
                    chunk = patches[i : i + patch_batch_size].to(device, non_blocking=True)
                    raw_out = model(chunk)
                    seg_logits, cls_logits, flow_out = extract_model_outputs(raw_out, task="monusac", return_flow=True)

                    if seg_logits.ndim == 3:
                        seg_logits = seg_logits.unsqueeze(1)

                    is_multiclass = (seg_logits.shape[1] > 1)
                    num_seg_classes = seg_logits.shape[1]

                    p_h, p_w = chunk.shape[-2:]
                    if seg_logits.shape[-2:] != (p_h, p_w):
                        seg_logits = F.interpolate(seg_logits, size=(p_h, p_w), mode="bilinear", align_corners=False)

                    if is_multiclass:
                        seg_probs = F.softmax(seg_logits, dim=1).detach().cpu().numpy()
                        patch_seg_probs.extend(seg_probs)
                    else:
                        seg_probs = torch.sigmoid(seg_logits)[:, 0].detach().cpu().numpy()
                        patch_seg_probs.extend(seg_probs)

                    if flow_out is not None:
                        has_flow = True
                        if flow_out.shape[-2:] != (p_h, p_w):
                            flow_out = F.interpolate(flow_out, size=(p_h, p_w), mode="bilinear", align_corners=False)
                        patch_flows.extend(flow_out.detach().cpu().numpy())

                    if cls_logits is not None:
                        has_cls_head = True
                        if cls_logits.shape[-2:] != (p_h, p_w):
                            cls_logits = F.interpolate(cls_logits, size=(p_h, p_w), mode="bilinear", align_corners=False)
                        cls_probs = torch.softmax(cls_logits, dim=1).detach().cpu().numpy()
                        patch_cls_probs.extend(cls_probs)

                # 5. Stitch back whole probability maps
                full_flow = None
                if has_flow and len(patch_flows) == len(coords):
                    full_flow = stitch_flow_map(
                        patch_flows=patch_flows,
                        coordinates=coords,
                        orig_shape=orig_shape,
                        padded_shape=padded_shape,
                    )

                if is_multiclass:
                    full_seg_prob = stitch_multiclass_probability_map(
                        patch_probs=patch_seg_probs,
                        coordinates=coords,
                        orig_shape=orig_shape,
                        padded_shape=padded_shape,
                    )
                    if num_seg_classes == 4:
                        full_pred_bin_inst = class4_to_instances(full_seg_prob, min_size=min_instance_size)
                        full_4class = np.argmax(full_seg_prob, axis=0)
                        colorized_4class = colorize_4class_map(full_4class)
                        full_prob = full_seg_prob[1] + full_seg_prob[2]
                    elif num_seg_classes == 3:
                        full_pred_bin_inst = class3_to_instances(full_seg_prob, min_size=min_instance_size)
                        full_4class = np.argmax(full_seg_prob, axis=0)
                        colorized_4class = colorize_4class_map(full_4class)
                        full_prob = full_seg_prob[1] + full_seg_prob[2]
                    else:
                        pred_labels = np.argmax(full_seg_prob, axis=0)
                        full_pred_bin_inst = binary_to_instances((pred_labels > 0).astype(np.uint8), min_size=min_instance_size)
                        full_prob = 1.0 - full_seg_prob[0]
                    full_pred_bin = (full_pred_bin_inst > 0).astype(np.uint8)
                else:
                    full_prob = stitch_probability_map(
                        patch_probs=patch_seg_probs,
                        coordinates=coords,
                        orig_shape=orig_shape,
                        padded_shape=padded_shape,
                    )
                    full_pred_bin = (full_prob >= threshold).astype(np.uint8)
                    full_pred_bin_inst = flow_to_instances(full_pred_bin, flow_vector=full_flow, min_size=min_instance_size)

                # 6. Reconstruct class prediction masks
                pred_class_masks = np.zeros((4, h, w), dtype=np.uint8)
                if has_cls_head and len(patch_cls_probs) > 0:
                    full_cls_prob = stitch_multiclass_probability_map(
                        patch_probs=patch_cls_probs,
                        coordinates=coords,
                        orig_shape=orig_shape,
                        padded_shape=padded_shape,
                    )
                    if full_cls_prob.shape[0] >= 5:
                        fg_argmax = np.argmax(full_cls_prob[1:5], axis=0)  # 0..3
                    else:
                        fg_argmax = np.argmax(full_cls_prob, axis=0)

                    for c in range(4):
                        pred_class_masks[c] = ((full_pred_bin > 0) & (fg_argmax == c)).astype(np.uint8)

                pred_classes_color = colorize_multiclass_mask(pred_class_masks)

            # 7. Compute binary metrics
            bin_scores = compute_binary_metrics(
                pred_binary=full_pred_bin,
                gt_binary=gt_bin,
                pred_instance=full_pred_bin_inst,
                gt_instance=gt_bin_inst,
                min_instance_size=min_instance_size,
            )

            # 8. Compute per-class metrics
            cell_scores = compute_monusac_cell_metrics(
                pred_class_masks=pred_class_masks,
                gt_class_masks=gt_class_masks,
                gt_class_instances=gt_inst_4ch,
                min_instance_size=min_instance_size,
            )

            row = {
                "Patient": patient_id,
                "Subimage": sub_name,
                "bin_dice": bin_scores["Dice"],
                "bin_iou": bin_scores["IoU"],
                "bin_aji": bin_scores["AJI"],
                "bin_pq": bin_scores["PQ"],
                **cell_scores,
            }
            results.append(row)

            print(
                f"{patient_id:<24} {sub_name:<26} "
                f"{bin_scores['Dice']:>8.4f} {bin_scores['IoU']:>8.4f} "
                f"{bin_scores['AJI']:>8.4f} {bin_scores['PQ']:>8.4f} "
                f"{cell_scores['mean_class_dice']:>14.4f}"
            )

            # 9. Save visualizations (GT mask, GT instance, pred binary mask, per class pred mask, instance mask pred, overlays, panels)
            if save_dir and save_predictions:
                sample_key = f"{patient_id}_{sub_name}"
                gt_classes_color = colorize_multiclass_mask(gt_class_masks)

                # Traditional outputs
                save_image_safe(os.path.join(save_dir, "ground_truth", f"{sample_key}_gt_mask.png"), gt_bin * 255)
                save_image_safe(os.path.join(save_dir, "ground_truth", f"{sample_key}_gt_binary_mask.png"), gt_bin * 255)
                save_image_safe(os.path.join(save_dir, "ground_truth", f"{sample_key}_gt_instance.png"), colorize_instance_mask(gt_bin_inst))
                save_image_safe(os.path.join(save_dir, "ground_truth", f"{sample_key}_gt_instance_ids.png"), gt_bin_inst.astype(np.uint16))
                for c_idx, c_name in enumerate(MONUSAC_CELL_NAMES):
                    save_image_safe(os.path.join(save_dir, "ground_truth", f"{sample_key}_gt_{c_name.lower()}.png"), gt_class_masks[c_idx] * 255)
                save_image_safe(os.path.join(save_dir, "ground_truth", f"{sample_key}_gt_classes_color.png"), gt_classes_color)

                save_image_safe(os.path.join(save_dir, "predictions", f"{sample_key}_pred_binary_mask.png"), full_pred_bin * 255)
                save_image_safe(os.path.join(save_dir, "predictions", f"{sample_key}_pred_mask.png"), full_pred_bin * 255)
                save_image_safe(os.path.join(save_dir, "predictions", f"{sample_key}_pred_instance.png"), colorize_instance_mask(full_pred_bin_inst))
                save_image_safe(os.path.join(save_dir, "predictions", f"{sample_key}_pred_instance_ids.png"), full_pred_bin_inst.astype(np.uint16))
                for c_idx, c_name in enumerate(MONUSAC_CELL_NAMES):
                    save_image_safe(os.path.join(save_dir, "predictions", f"{sample_key}_pred_{c_name.lower()}.png"), pred_class_masks[c_idx] * 255)
                save_image_safe(os.path.join(save_dir, "predictions", f"{sample_key}_pred_classes_color.png"), pred_classes_color)

                prob_heatmap = (np.clip(full_prob, 0.0, 1.0) * 255).astype(np.uint8)
                prob_color = cv2.applyColorMap(prob_heatmap, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(save_dir, "probabilities", f"{sample_key}_prob.png"), prob_color)

                overlay = create_tp_fp_fn_overlay(img_rgb, gt_bin, full_pred_bin)
                save_image_safe(os.path.join(save_dir, "overlays", f"{sample_key}_overlay.png"), overlay)

                panel = create_comparison_panel(img_rgb, gt_bin, full_pred_bin, overlay)
                save_image_safe(os.path.join(save_dir, "panels", f"{sample_key}_panel.png"), panel)

                # --- 1. INDIVIDUAL PNGs ---
                indiv_dir = os.path.join(save_dir, "individual", sample_key)
                save_image_safe(os.path.join(indiv_dir, "original_image.png"), img_rgb)
                save_image_safe(os.path.join(indiv_dir, "gt_binary_mask.png"), gt_bin * 255)
                save_image_safe(os.path.join(indiv_dir, "gt_instance.png"), colorize_instance_mask(gt_bin_inst))
                save_image_safe(os.path.join(indiv_dir, "gt_phenotypes.png"), gt_classes_color)
                for c_idx, c_name in enumerate(MONUSAC_CELL_NAMES):
                    save_image_safe(os.path.join(indiv_dir, f"gt_{c_name.lower()}.png"), gt_class_masks[c_idx] * 255)
                    save_image_safe(os.path.join(indiv_dir, f"pred_{c_name.lower()}.png"), pred_class_masks[c_idx] * 255)

                save_image_safe(os.path.join(indiv_dir, "pred_binary_mask.png"), full_pred_bin * 255)
                save_image_safe(os.path.join(indiv_dir, "pred_instance.png"), colorize_instance_mask(full_pred_bin_inst))
                save_image_safe(os.path.join(indiv_dir, "pred_probability.png"), prob_color)
                save_image_safe(os.path.join(indiv_dir, "error_overlay.png"), overlay)
                save_image_safe(os.path.join(indiv_dir, "pred_phenotypes.png"), pred_classes_color)
                if colorized_4class is not None:
                    save_image_safe(os.path.join(indiv_dir, "pred_4class.png"), colorized_4class)

                # --- 2. COMBINED PNGs ---
                comb_dir = os.path.join(save_dir, "combined")
                ds_panel = create_monusac_dataset_panel(
                    image=img_rgb,
                    gt_inst=gt_bin_inst,
                    pred_inst=full_pred_bin_inst,
                    overlay=overlay,
                    gt_pheno_rgb=gt_classes_color,
                    pred_pheno_rgb=pred_classes_color,
                )
                save_image_safe(os.path.join(comb_dir, f"{sample_key}_dataset_panel.png"), ds_panel)

                # --- 3. ALL HEADS OUTPUT (Individual + Combined) ---
                if is_multitask and save_all_heads and "gland" in multi_out:
                    all_heads_dir = os.path.join(indiv_dir, "all_heads")
                    save_image_safe(os.path.join(all_heads_dir, "gland_instances.png"), colorize_instance_mask(multi_out["gland"]["instances"]))
                    save_image_safe(os.path.join(all_heads_dir, "gland_4class.png"), multi_out["gland"]["colorized_4class"])
                    save_image_safe(os.path.join(all_heads_dir, "cell_instances.png"), colorize_instance_mask(multi_out["cell"]["instances"]))
                    save_image_safe(os.path.join(all_heads_dir, "cell_4class.png"), multi_out["cell"]["colorized_4class"])
                    save_image_safe(os.path.join(all_heads_dir, "phenotypes.png"), pred_classes_color)

                    all_heads_panel = create_all_heads_panel(
                        image=img_rgb,
                        gland_inst=multi_out["gland"]["instances"],
                        cell_inst=multi_out["cell"]["instances"],
                        pheno_rgb=pred_classes_color,
                    )
                    save_image_safe(os.path.join(comb_dir, f"{sample_key}_all_heads_panel.png"), all_heads_panel)
                panel = create_comparison_panel(img_rgb, gt_bin, full_pred_bin, overlay)
                save_image_safe(os.path.join(save_dir, "panels", f"{sample_key}_panel.png"), panel)

                # Also save inside patient folder structure for nested navigation
                sub_save_dir = os.path.join(save_dir, "predictions", patient_id, sub_name)
                save_image_safe(os.path.join(sub_save_dir, "pred_binary_mask.png"), full_pred_bin * 255)
                save_image_safe(os.path.join(sub_save_dir, "pred_instance.png"), colorize_instance_mask(full_pred_bin_inst))
                save_image_safe(os.path.join(sub_save_dir, "pred_classes_color.png"), pred_classes_color)
                save_image_safe(os.path.join(sub_save_dir, "gt_binary_mask.png"), gt_bin * 255)
                save_image_safe(os.path.join(sub_save_dir, "gt_instance.png"), colorize_instance_mask(gt_bin_inst))
                save_image_safe(os.path.join(sub_save_dir, "gt_classes_color.png"), gt_classes_color)
                for c_idx, c_name in enumerate(MONUSAC_CELL_NAMES):
                    save_image_safe(os.path.join(sub_save_dir, f"pred_{c_name.lower()}.png"), pred_class_masks[c_idx] * 255)
                    save_image_safe(os.path.join(sub_save_dir, f"gt_{c_name.lower()}.png"), gt_class_masks[c_idx] * 255)
                save_image_safe(os.path.join(sub_save_dir, "overlay_tp_fp_fn.png"), overlay)
                save_image_safe(os.path.join(sub_save_dir, "summary_panel.png"), panel)

    df_detailed = pd.DataFrame(results)

    # 10. Compute Full Dataset Summary Table
    def build_metric_table(df_source: pd.DataFrame, title: str) -> pd.DataFrame:
        metrics_list = ["Dice", "IoU", "AJI", "PQ"]
        table_rows = []
        for m in metrics_list:
            m_lower = m.lower()
            table_rows.append({
                "Metric": m,
                "Binary": float(df_source[f"bin_{m_lower}"].mean()),
                "Epithelial": float(df_source[f"{m_lower}_epithelial"].mean()),
                "Lymphocyte": float(df_source[f"{m_lower}_lymphocyte"].mean()),
                "Neutrophil": float(df_source[f"{m_lower}_neutrophil"].mean()),
                "Macrophage": float(df_source[f"{m_lower}_macrophage"].mean()),
                "Class_Mean": float(df_source[f"mean_class_{m_lower}"].mean()),
            })
        return pd.DataFrame(table_rows)

    # A: Full dataset metrics (averaged across all sub-images)
    df_full_dataset_summary = build_metric_table(df_detailed, "FULL DATASET SUMMARY (ALL SUB-IMAGES)")

    # B: Per-patient averaged metrics
    # Step 1: Average sub-images per patient
    non_numeric = {"Patient", "Subimage"}
    metric_cols = [c for c in df_detailed.columns if c not in non_numeric]
    df_per_patient = df_detailed.groupby("Patient")[metric_cols].mean().reset_index()

    # Step 2: Average across all patients (macro-patient average)
    df_per_patient_summary = build_metric_table(df_per_patient, "PER-PATIENT AVERAGED SUMMARY")

    print("\n" + "=" * 90)
    print("1. MONUSAC FULL DATASET METRICS (Macro-Average Across All Sub-Images)")
    print("=" * 90)
    print(df_full_dataset_summary.to_string(index=False))
    print("=" * 90)

    print("\n" + "=" * 90)
    print("2. MONUSAC PER-PATIENT AVERAGED METRICS (Patient-Level Macro-Average)")
    print("=" * 90)
    print(df_per_patient_summary.to_string(index=False))
    print("=" * 90 + "\n")

    if save_dir:
        csv_dir = os.path.join(save_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)
        df_detailed.to_csv(os.path.join(csv_dir, "monusac_subimages_detailed.csv"), index=False)
        df_per_patient.to_csv(os.path.join(csv_dir, "monusac_per_patient_breakdown.csv"), index=False)
        df_full_dataset_summary.to_csv(os.path.join(csv_dir, "monusac_full_dataset_summary.csv"), index=False)
        df_per_patient_summary.to_csv(os.path.join(csv_dir, "monusac_per_patient_summary.csv"), index=False)
        print(f"Results saved to: {csv_dir}")

    return df_detailed, df_full_dataset_summary, df_per_patient_summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on MoNuSAC test dataset without dataloaders.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt).")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to MoNuSAC testing directory.")
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
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of evaluation samples.")
    parser.add_argument("--device", type=str, default=None, help="Compute device (e.g. cuda, cpu).")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device) if args.device else get_device()

    # Resolve data directory
    data_dir = args.data_dir
    if not data_dir:
        test_cfg = config.get("test", {})
        data_dir = test_cfg.get("data_dir", test_cfg.get("image_dir", "data/MoNuSAC Testing Data and Annotations"))
    if not os.path.exists(data_dir):
        data_dir = "data/MoNuSAC Testing Data and Annotations"

    # Find test images
    samples = find_monusac_test_samples(data_dir)
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
        exp_name = config.get("experiment", {}).get("name", "monusac_eval")
        save_dir = os.path.join("eval_results", exp_name, "monusac")

    run_monusac_eval(
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
