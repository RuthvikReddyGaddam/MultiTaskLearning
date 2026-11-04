# scripts/evaluate_multitask.py
"""
Unified Multi-Task Evaluation Script.
Evaluates a single multi-task trained checkpoint across all 3 official test benchmarks:
  1. GLaS (80 test images: 60 TestA benign/malignant + 20 TestB challenging)
  2. MoNuSeg (14 multi-organ WSIs / test images)
  3. MoNuSAC (Multi-organ cell nuclei + 4-class cell phenotype evaluation)

Features:
  - Reads full-size images without dataloaders.
  - Runs batched patch inference on GPU with sliding window and seamless stitching.
  - Saves dataset-specific outputs and all-heads outputs.
  - Saves both INDIVIDUAL PNGs and COMBINED multi-panel comparison PNGs.
  - Outputs per-dataset quantitative evaluation metrics and unified benchmark summaries.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from utils.config import load_config
from utils.device import get_device
from utils.test_utils import load_model_for_evaluation
from scripts.test_glas import run_glas_eval, find_glas_test_images
from scripts.test_monuseg import run_monuseg_eval, find_monuseg_test_images
from scripts.test_monusac import run_monusac_eval, find_monusac_test_samples


def main():
    parser = argparse.ArgumentParser(description="Unified Multi-Task Model Evaluation on GLaS, MoNuSeg, and MoNuSAC.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multitask/multitask_sdenetpointgradexp_512.yaml",
        help="Path to multi-task YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained checkpoint (.pt). If omitted, resolves best.pt from config.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results/multitask_evaluation",
        help="Directory to save evaluation reports and visualizations.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "glas", "monuseg", "monusac"],
        help="Dataset benchmark to evaluate (default: 'all').",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Patch batch size during inference.")
    parser.add_argument("--patch_size", type=int, default=512, help="Square patch size.")
    parser.add_argument("--stride", type=int, default=None, help="Patch stride.")
    parser.add_argument("--overlap", type=float, default=None, help="Patch overlap fraction.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample limit for quick testing.")
    parser.add_argument(
        "--save_predictions",
        dest="save_predictions",
        action="store_true",
        default=True,
        help="Save visual predictions and ground truths (default: True).",
    )
    parser.add_argument(
        "--no_save_predictions",
        dest="save_predictions",
        action="store_false",
        help="Disable saving visual prediction artifacts.",
    )
    parser.add_argument(
        "--save_all_heads",
        dest="save_all_heads",
        action="store_true",
        default=True,
        help="Save predictions from all heads (gland, cell, phenotype) for each sample (default: True).",
    )
    parser.add_argument(
        "--no_save_all_heads",
        dest="save_all_heads",
        action="store_false",
        help="Disable saving predictions from non-target heads.",
    )
    args = parser.parse_args()

    device = get_device()
    config = load_config(args.config)

    print("=" * 90)
    print("UNIFIED MULTI-TASK EVALUATION SUITE")
    print("=" * 90)
    print(f"Device:           {device}")
    print(f"Config:           {args.config}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Target Dataset:   {args.dataset}")
    print(f"Save All Heads:   {args.save_all_heads}")
    print("=" * 90)

    # 1. Load multi-task model
    model = load_model_for_evaluation(config, checkpoint_path=args.checkpoint, device=device)
    model.eval()

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    summary_results: Dict[str, float] = {}

    # -------------------------------------------------------------
    # 1. GLaS Evaluation
    # -------------------------------------------------------------
    if args.dataset in ("all", "glas"):
        glas_data_dir = PROJECT_ROOT / "data" / "glas" / "Warwick_QU_Dataset"
        if glas_data_dir.exists():
            print("\n" + "#" * 90)
            print("EVALUATING TASK 1: GLaS (Gland Segmentation)")
            print("#" * 90)
            glas_samples = find_glas_test_images(glas_data_dir, split="test")
            df_detailed, df_summary = run_glas_eval(
                model=model,
                samples=glas_samples,
                device=device,
                patch_size=args.patch_size,
                stride=args.stride,
                overlap=args.overlap,
                patch_batch_size=args.batch_size,
                save_dir=str(output_base / "glas"),
                save_predictions=args.save_predictions,
                save_all_heads=args.save_all_heads,
                max_samples=args.max_samples,
            )
            if len(df_detailed) > 0:
                summary_results["GLaS_Dice"] = float(df_detailed["Dice"].mean())
                summary_results["GLaS_IoU"] = float(df_detailed["IoU"].mean())
                summary_results["GLaS_AJI"] = float(df_detailed["AJI"].mean())
                summary_results["GLaS_PQ"] = float(df_detailed["PQ"].mean())
        else:
            print(f"[Warning] GLaS data directory not found at {glas_data_dir}")

    # -------------------------------------------------------------
    # 2. MoNuSeg Evaluation
    # -------------------------------------------------------------
    if args.dataset in ("all", "monuseg"):
        monuseg_data_dir = PROJECT_ROOT / "data" / "MoNuSegTestData"
        if monuseg_data_dir.exists():
            print("\n" + "#" * 90)
            print("EVALUATING TASK 2: MoNuSeg (Multi-Organ Nuclei Segmentation)")
            print("#" * 90)
            monuseg_samples = find_monuseg_test_images(monuseg_data_dir)
            monuseg_df, _ = run_monuseg_eval(
                model=model,
                samples=monuseg_samples,
                device=device,
                patch_size=args.patch_size,
                stride=args.stride,
                overlap=args.overlap,
                patch_batch_size=args.batch_size,
                save_dir=str(output_base / "monuseg"),
                save_predictions=args.save_predictions,
                save_all_heads=args.save_all_heads,
                max_samples=args.max_samples,
            )
            if len(monuseg_df) > 0:
                summary_results["MoNuSeg_Dice"] = float(monuseg_df["Dice"].mean())
                summary_results["MoNuSeg_IoU"] = float(monuseg_df["IoU"].mean())
                summary_results["MoNuSeg_AJI"] = float(monuseg_df["AJI"].mean())
                summary_results["MoNuSeg_PQ"] = float(monuseg_df["PQ"].mean())
        else:
            print(f"[Warning] MoNuSeg data directory not found at {monuseg_data_dir}")

    # -------------------------------------------------------------
    # 3. MoNuSAC Evaluation
    # -------------------------------------------------------------
    if args.dataset in ("all", "monusac"):
        monusac_data_dir = PROJECT_ROOT / "data" / "MoNuSAC Testing Data and Annotations"
        if monusac_data_dir.exists():
            print("\n" + "#" * 90)
            print("EVALUATING TASK 3: MoNuSAC (Nuclei Segmentation & 4-Class Classification)")
            print("#" * 90)
            monusac_samples = find_monusac_test_samples(monusac_data_dir)
            monusac_df, df_full_summary, df_patient_summary = run_monusac_eval(
                model=model,
                samples=monusac_samples,
                device=device,
                patch_size=args.patch_size,
                stride=args.stride,
                overlap=args.overlap,
                patch_batch_size=args.batch_size,
                save_dir=str(output_base / "monusac"),
                save_predictions=args.save_predictions,
                save_all_heads=args.save_all_heads,
                max_samples=args.max_samples,
            )
            if len(monusac_df) > 0:
                summary_results["MoNuSAC_BinDice"] = float(monusac_df["bin_dice"].mean())
                summary_results["MoNuSAC_BinIoU"] = float(monusac_df["bin_iou"].mean()) if "bin_iou" in monusac_df else 0.0
                summary_results["MoNuSAC_BinAJI"] = float(monusac_df["bin_aji"].mean())
                summary_results["MoNuSAC_BinPQ"] = float(monusac_df["bin_pq"].mean())
                if "mean_class_dice" in monusac_df:
                    summary_results["MoNuSAC_MeanClassDice"] = float(monusac_df["mean_class_dice"].mean())
                if "dice_epithelial" in monusac_df:
                    summary_results["MoNuSAC_Dice_Epithelial"] = float(monusac_df["dice_epithelial"].mean())
                if "dice_lymphocyte" in monusac_df:
                    summary_results["MoNuSAC_Dice_Lymphocyte"] = float(monusac_df["dice_lymphocyte"].mean())
                if "dice_neutrophil" in monusac_df:
                    summary_results["MoNuSAC_Dice_Neutrophil"] = float(monusac_df["dice_neutrophil"].mean())
                if "dice_macrophage" in monusac_df:
                    summary_results["MoNuSAC_Dice_Macrophage"] = float(monusac_df["dice_macrophage"].mean())
        else:
            print(f"[Warning] MoNuSAC data directory not found at {monusac_data_dir}")

    # -------------------------------------------------------------
    # Summary Table and CSV Export
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("FINAL MULTI-TASK BENCHMARK SUMMARY")
    print("=" * 90)
    for k, v in summary_results.items():
        print(f"  {k:<28}: {v:.4f}")
    print("=" * 90)

    # Save summary table CSV
    if summary_results:
        summary_rows = [{"Metric": k, "Value": v} for k, v in summary_results.items()]
        df_bench = pd.DataFrame(summary_rows)
        summary_csv_path = output_base / "multitask_benchmark_summary.csv"
        df_bench.to_csv(summary_csv_path, index=False)
        print(f"\nUnified benchmark summary saved to: {summary_csv_path}")


if __name__ == "__main__":
    main()
