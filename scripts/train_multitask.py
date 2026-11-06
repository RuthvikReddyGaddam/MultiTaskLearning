# scripts/train_multitask.py

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import (
    load_yaml_config,
    setup_experiment,
    cleanup_experiment,
    build_optimizer,
    build_scheduler,
    print_model_summary,
)
from scripts.build_models import build_model
from datasets import build_multitask_dataloader
from scripts.build_losses import build_loss
from training.trainer_multitask import MultiTaskTrainer


def main():
    parser = argparse.ArgumentParser(description="Multi-Task Training for GLaS, MoNuSeg, and MoNuSAC (512x512).")
    parser.add_argument("--config", type=str, required=True, help="Path to multi-task YAML configuration file.")
    args = parser.parse_args()

    # 1. Load configuration and initialize experiment environment
    config = load_yaml_config(args.config)
    device, exp_dir, log_file = setup_experiment(config, dataset_name="multitask")

    try:
        # 2. Build Multi-Task Model (all 7 prediction heads)
        model = build_model(config)
        model.to(device)

        print_model_summary(model, config=config, device=device)

        # 3. Optimizer & Learning Rate Scheduler
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)

        # 4. Multi-Task Balanced DataLoader (512x512 with EqualTaskBatchSampler)
        dataset_cfg = config.get("dataset", {})
        training_cfg = config.get("training", {})
        batch_size = int(training_cfg.get("batch_size", 8))
        patch_size = int(dataset_cfg.get("patch_size", 512))
        num_workers = int(dataset_cfg.get("num_workers", 4))
        pin_memory = bool(dataset_cfg.get("pin_memory", True))

        glas_dir = dataset_cfg.get("glas_dir")
        monuseg_dir = dataset_cfg.get("monuseg_dir")
        monusac_dir = dataset_cfg.get("monusac_dir")
        class_balance_monusac = bool(dataset_cfg.get("class_balance_monusac", True))
        patches_per_class = int(dataset_cfg.get("patches_per_class", 80))
        class_weights = dataset_cfg.get("class_weights")
        steps_per_epoch = training_cfg.get("steps_per_epoch", 160)
        if steps_per_epoch is not None:
            steps_per_epoch = int(steps_per_epoch)

        train_loader = build_multitask_dataloader(
            glas_dir=glas_dir,
            monuseg_dir=monuseg_dir,
            monusac_dir=monusac_dir,
            patch_size=patch_size,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            steps_per_epoch=steps_per_epoch,
            class_balance_monusac=class_balance_monusac,
            patches_per_class=patches_per_class,
            class_weights=class_weights,
            augment=True,
        )

        dataset = train_loader.dataset
        sac_ds = dataset.monusac_dataset
        if hasattr(sac_ds, "patches_per_class"):
            sac_info = f"{len(sac_ds)} pseudo-patches ({sac_ds.patches_per_class}/class: 80 Epi, 80 Lym, 80 Neu, 80 Mac - 1:1:1:1)"
        else:
            sac_info = f"{len(sac_ds)} patches"

        num_batches = len(train_loader)
        total_patches_epoch = num_batches * batch_size
        k_glas = batch_size // 2
        k_monuseg = batch_size // 4
        k_monusac = batch_size // 4

        print("\n" + "=" * 80)
        print("                  MULTI-TASK DATALOADER STATISTICS")
        print("=" * 80)
        print("Raw Dataset Patch Pools:")
        print(f"  - GLaS (Gland Segmentation):           {len(dataset.glas_dataset):>6d} patches")
        print(f"  - MoNuSeg (Nuclei Segmentation):       {len(dataset.monuseg_dataset):>6d} patches")
        print(f"  - MoNuSAC (Nuclei + Phenotype):        {len(sac_ds):>6d} patches ({sac_info})")
        print(f"  - Total Combined Unique Pool:          {len(dataset):>6d} patches")
        print("-" * 80)
        print("Batch Sampling (Strict 2:1:1 Balanced Ratio, Shuffled):")
        print(f"  - Batch Size:                          {batch_size:>6d} patches/batch")
        print(f"    * GLaS (task 0):                     {k_glas:>6d} patches/batch ({k_glas/batch_size*100:.0f}%)")
        print(f"    * MoNuSeg (task 1):                  {k_monuseg:>6d} patches/batch ({k_monuseg/batch_size*100:.0f}%)")
        print(f"    * MoNuSAC (task 2):                  {k_monusac:>6d} patches/batch ({k_monusac/batch_size*100:.0f}%)")
        print("-" * 80)
        print("Epoch Throughput Statistics:")
        print(f"  - Total Batches per Epoch:             {num_batches:>6d} batches")
        print(f"  - Total Patches processed per Epoch:   {total_patches_epoch:>6d} patches")
        print(f"    * GLaS patches per epoch:            {num_batches * k_glas:>6d} patches/epoch")
        print(f"    * MoNuSeg patches per epoch:         {num_batches * k_monuseg:>6d} patches/epoch")
        print(f"    * MoNuSAC patches per epoch:         {num_batches * k_monusac:>6d} patches/epoch")
        print("=" * 80 + "\n")

        # 5. Multi-Task Loss Function
        criterion = build_loss(config)

        # 6. Run MultiTaskTrainer
        trainer = MultiTaskTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            config=config,
            scheduler=scheduler,
            exp_dir=exp_dir,
        )

        trainer.fit()

    finally:
        cleanup_experiment(log_file)


if __name__ == "__main__":
    main()

