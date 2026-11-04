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
        num_workers = int(dataset_cfg.get("num_workers", 2))
        pin_memory = bool(dataset_cfg.get("pin_memory", True))

        glas_dir = dataset_cfg.get("glas_dir")
        monuseg_dir = dataset_cfg.get("monuseg_dir")
        monusac_dir = dataset_cfg.get("monusac_dir")
        monusac_resample_size = int(dataset_cfg.get("monusac_resample_size", 600))
        class_weights = dataset_cfg.get("class_weights")

        train_loader = build_multitask_dataloader(
            glas_dir=glas_dir,
            monuseg_dir=monuseg_dir,
            monusac_dir=monusac_dir,
            patch_size=patch_size,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            monusac_resample_size=monusac_resample_size,
            monusac_class_weights=class_weights,
            augment=True,
        )

        dataset = train_loader.dataset
        print(f"Initialized MultiTaskDataset with {len(dataset)} total samples:")
        print(f"  - GLaS:    {len(dataset.glas_dataset)} patches")
        print(f"  - MoNuSeg: {len(dataset.monuseg_dataset)} patches")
        print(f"  - MoNuSAC: {len(dataset.monusac_dataset)} virtual patches (minority resampled)")
        print(f"Batch size: {batch_size} (exact {batch_size//2}:{batch_size//4}:{batch_size//4} 2:1:1 split per batch)")
        print(f"Total batches per epoch: {len(train_loader)}\n")

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

