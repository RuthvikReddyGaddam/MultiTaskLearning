# training/trainer_multitask.py

import os
import csv
import time
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F

from metrics.monusac_training_metrics import compute_dice_iou_binary, compute_multiclass_metrics


class MultiTaskTrainer:
    """
    Dedicated Multi-Task Trainer for:
      - GLaS (Gland segmentation + 2D flow field)
      - MoNuSeg (Nuclei segmentation + 2D flow field)
      - MoNuSAC (Nuclei segmentation + 2D flow field + 4-class classification)

    Features:
      - Sliced forward pass with task_ids
      - Strict zero-gradient preservation for inactive heads
      - Epoch-boundary resampling for minority cell types in MoNuSAC
      - Comprehensive per-task and aggregate metric tracking
      - Checkpoint saving (best.pt, latest.pt) and history.csv logging
    """
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader,
        device: torch.device,
        config: Dict[str, Any],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        exp_dir: Optional[str] = None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = device
        self.config = config
        self.scheduler = scheduler

        # Training epochs
        training_cfg = config.get("training", {})
        self.epochs = int(training_cfg.get("epochs", config.get("epochs", 50)))

        # Experiment output directory
        if exp_dir is not None:
            self.exp_dir = exp_dir
        else:
            chk_path = config.get("checkpoint", {}).get("path", "checkpoints/multitask")
            exp_name = config.get("experiment", {}).get("name", "multitask_512_run")
            self.exp_dir = os.path.join(chk_path, exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        self.best_checkpoint_path = os.path.join(self.exp_dir, "best.pt")
        self.last_checkpoint_path = os.path.join(self.exp_dir, "latest.pt")
        self.history_path = os.path.join(self.exp_dir, "history.csv")

        # Resume state
        self.resume = config.get("checkpoint", {}).get("resume", True)
        self.start_epoch = 1
        self.best_score = float("-inf")
        self.best_epoch = 0

        self._resume_if_needed()

    def _resume_if_needed(self) -> None:
        """Loads last checkpoint if resume is True and file exists, else starts from scratch."""
        if not self.resume:
            print("[MultiTaskTrainer] Starting training from scratch at epoch 1.")
            return

        if not os.path.exists(self.last_checkpoint_path):
            print(f"[MultiTaskTrainer] No checkpoint found at {self.last_checkpoint_path}. Starting from scratch.")
            return

        print(f"[MultiTaskTrainer] Resuming from checkpoint: {self.last_checkpoint_path}")
        checkpoint = torch.load(self.last_checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.start_epoch = checkpoint.get("epoch", 0) + 1
        self.best_score = checkpoint.get("best_score", float("-inf"))
        self.best_epoch = checkpoint.get("best_epoch", 0)
        print(f"[MultiTaskTrainer] Resumed from epoch {self.start_epoch - 1} (best score: {self.best_score:.4f} at epoch {self.best_epoch})")

    def _save_checkpoint(self, epoch: int, is_best: bool, metrics: Dict[str, float]) -> None:
        """Saves latest and best checkpoints."""
        checkpoint = {
            "epoch": epoch,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "metrics": metrics,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, self.last_checkpoint_path)
        if is_best:
            torch.save(checkpoint, self.best_checkpoint_path)
            print(f"[*] Best checkpoint updated at epoch {epoch} (score: {self.best_score:.4f})")

    def _log_history(self, epoch: int, metrics: Dict[str, float], run_time: float) -> None:
        """Appends one epoch metrics row to CSV file."""
        file_exists = os.path.exists(self.history_path)
        fieldnames = ["epoch", "time_sec", "lr"] + list(metrics.keys())

        with open(self.history_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            lr = self.optimizer.param_groups[0]["lr"]
            row = {"epoch": epoch, "time_sec": f"{run_time:.2f}", "lr": f"{lr:.6e}"}
            for k, v in metrics.items():
                row[k] = f"{v:.4f}" if isinstance(v, float) else v
            writer.writerow(row)

    def train_one_epoch(self) -> Dict[str, float]:
        """Runs one full training epoch across the multi-task balanced dataloader."""
        self.model.train()

        # Resample minority MoNuSAC classes at epoch boundary
        dataset = getattr(self.train_loader, "dataset", None)
        if dataset is not None and hasattr(dataset, "monusac_dataset") and hasattr(dataset.monusac_dataset, "resample"):
            dataset.monusac_dataset.resample()

        total_loss = 0.0
        loss_components_accum: Dict[str, float] = {}
        num_batches = len(self.train_loader)

        # Accuracy metric accumulators
        glas_dice_sum, glas_iou_sum, n_glas = 0.0, 0.0, 0
        monuseg_dice_sum, monuseg_iou_sum, n_monuseg = 0.0, 0.0, 0
        monusac_dice_sum, monusac_iou_sum, n_monusac = 0.0, 0.0, 0
        monusac_cls_dice_sum, monusac_cls_iou_sum = 0.0, 0.0
        glas_neg_bg_sum, n_glas_neg = 0.0, 0

        for batch in self.train_loader:
            # Transfer strictly required tensors with non_blocking=True
            images = batch["image"].to(self.device, non_blocking=True)
            task_ids = batch["task_id"].to(self.device, non_blocking=True)
            target_4class = batch["target_4class"].to(self.device, non_blocking=True)
            binary_masks = batch["binary_mask"].to(self.device, non_blocking=True)
            class_maps = batch["class_map"].to(self.device, non_blocking=True)

            batch_gpu = {
                "image": images,
                "task_id": task_ids,
                "target_4class": target_4class,
                "target_3class": target_4class,
                "binary_mask": binary_masks,
                "class_map": class_maps,
                "is_gland": batch.get("is_gland"),
                "is_monuseg": batch.get("is_monuseg"),
                "is_monusac": batch.get("is_monusac"),
            }
            if "class_mask" in batch and batch["class_mask"] is not None:
                batch_gpu["class_mask"] = batch["class_mask"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Pure FP32 forward pass with sliced multi-task prediction head
            preds = self.model(images, task_ids=task_ids)

            # Compute decoupled multi-task loss
            loss, batch_logs = self.criterion(preds, batch_gpu)

            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.detach())
            for k, v in batch_logs.items():
                loss_components_accum[k] = loss_components_accum.get(k, 0.0) + float(v.detach() if isinstance(v, torch.Tensor) else v)

            # Compute batch evaluation metrics
            with torch.no_grad():
                B = images.shape[0]
                half = B // 2
                q3 = half + B // 4
                if B % 8 == 0 and (task_ids[:half] == 0).all() and (task_ids[half:q3] == 1).all() and (task_ids[q3:] == 2).all():
                    # 1. GLaS (slice 0 : half)
                    p_argmax = preds["gland_seg"].argmax(dim=1)
                    p_bin_g = ((p_argmax == 1) | (p_argmax == 2)).float().unsqueeze(1)
                    d_g, i_g = compute_dice_iou_binary(p_bin_g, binary_masks[:half])
                    glas_dice_sum += d_g * half
                    glas_iou_sum += i_g * half
                    n_glas += half

                    # 2. MoNuSeg (slice half : q3)
                    p_argmax_m = preds["cell_seg"][: B // 4].argmax(dim=1)
                    p_bin_m = ((p_argmax_m == 1) | (p_argmax_m == 2)).float().unsqueeze(1)
                    d_m, i_m = compute_dice_iou_binary(p_bin_m, binary_masks[half:q3])
                    monuseg_dice_sum += d_m * (B // 4)
                    monuseg_iou_sum += i_m * (B // 4)
                    n_monuseg += (B // 4)

                    # 3. MoNuSAC Seg (slice q3 : B)
                    p_argmax_s = preds["cell_seg"][B // 4 :].argmax(dim=1)
                    p_bin_s = ((p_argmax_s == 1) | (p_argmax_s == 2)).float().unsqueeze(1)
                    d_s, i_s = compute_dice_iou_binary(p_bin_s, binary_masks[q3:])
                    monusac_dice_sum += d_s * (B // 4)
                    monusac_iou_sum += i_s * (B // 4)
                    n_monusac += (B // 4)

                    # 4. MoNuSAC Phenotype Classification (slice q3 : B)
                    cls_metrics = compute_multiclass_metrics(preds["cell_class"], class_maps[q3:])
                    monusac_cls_dice_sum += cls_metrics["cell_dice"] * (B // 4)
                    monusac_cls_iou_sum += cls_metrics["cell_iou"] * (B // 4)

                    # 5. Negative supervision suppression tracking
                    if "gland_neg" in preds:
                        neg_pred_bg = (preds["gland_neg"].argmax(dim=1) == 0).float().mean()
                        k_neg = preds["gland_neg"].shape[0]
                        glas_neg_bg_sum += float(neg_pred_bg.item()) * k_neg
                        n_glas_neg += k_neg
                else:
                    # Dynamic fallback for non-standard batches
                    is_gland = batch_gpu.get("is_gland", task_ids == 0)
                    p_gland = preds.get("gland_seg", preds.get("glas_seg"))
                    if is_gland.any() and p_gland is not None:
                        p_argmax = p_gland.argmax(dim=1) if p_gland.shape[1] > 1 else (torch.sigmoid(p_gland) > 0.5).long().squeeze(1)
                        p_bin = ((p_argmax == 1) | (p_argmax == 2)).float().unsqueeze(1) if p_gland.shape[1] > 1 else p_argmax.float().unsqueeze(1)
                        gt_glas = binary_masks[is_gland]
                        k_g = int(is_gland.sum().item())
                        d, i = compute_dice_iou_binary(p_bin, gt_glas)
                        glas_dice_sum += d * k_g
                        glas_iou_sum += i * k_g
                        n_glas += k_g

                    is_monuseg = batch_gpu.get("is_monuseg", task_ids == 1)
                    p_cell_monuseg = preds.get("monuseg_seg", preds.get("cell_seg"))
                    if is_monuseg.any() and p_cell_monuseg is not None:
                        p_argmax = p_cell_monuseg.argmax(dim=1) if p_cell_monuseg.shape[1] > 1 else (torch.sigmoid(p_cell_monuseg) > 0.5).long().squeeze(1)
                        p_bin = ((p_argmax == 1) | (p_argmax == 2)).float().unsqueeze(1) if p_cell_monuseg.shape[1] > 1 else p_argmax.float().unsqueeze(1)
                        gt_monuseg = binary_masks[is_monuseg]
                        k_m = int(is_monuseg.sum().item())
                        d, i = compute_dice_iou_binary(p_bin, gt_monuseg)
                        monuseg_dice_sum += d * k_m
                        monuseg_iou_sum += i * k_m
                        n_monuseg += k_m

                    is_monusac = batch_gpu.get("is_monusac", task_ids == 2)
                    p_cell_monusac = preds.get("monusac_seg", preds.get("cell_seg"))
                    if is_monusac.any():
                        k_s = int(is_monusac.sum().item())
                        if p_cell_monusac is not None:
                            p_argmax = p_cell_monusac.argmax(dim=1) if p_cell_monusac.shape[1] > 1 else (torch.sigmoid(p_cell_monusac) > 0.5).long().squeeze(1)
                            p_bin = ((p_argmax == 1) | (p_argmax == 2)).float().unsqueeze(1) if p_cell_monusac.shape[1] > 1 else p_argmax.float().unsqueeze(1)
                            gt_sac = binary_masks[is_monusac]
                            d, i = compute_dice_iou_binary(p_bin, gt_sac)
                            monusac_dice_sum += d * k_s
                            monusac_iou_sum += i * k_s
                            n_monusac += k_s

                        p_cls = preds.get("cell_class", preds.get("monusac_class"))
                        if p_cls is not None:
                            gt_cls = class_maps[is_monusac]
                            cls_metrics = compute_multiclass_metrics(p_cls, gt_cls)
                            monusac_cls_dice_sum += cls_metrics["cell_dice"] * k_s
                            monusac_cls_iou_sum += cls_metrics["cell_iou"] * k_s

                    if "gland_neg" in preds:
                        neg_pred_bg = (preds["gland_neg"].argmax(dim=1) == 0).float().mean()
                        k_neg = preds["gland_neg"].shape[0]
                        glas_neg_bg_sum += float(neg_pred_bg.item()) * k_neg
                        n_glas_neg += k_neg

        total_cell_n = n_monuseg + n_monusac
        overall_cell_dice = (monuseg_dice_sum + monusac_dice_sum) / max(1, total_cell_n)
        overall_cell_iou = (monuseg_iou_sum + monusac_iou_sum) / max(1, total_cell_n)

        metrics = {
            "loss": total_loss / max(1, num_batches),
            "glas_bin_dice": glas_dice_sum / max(1, n_glas),
            "glas_bin_iou": glas_iou_sum / max(1, n_glas),
            "monuseg_bin_dice": monuseg_dice_sum / max(1, n_monuseg),
            "monuseg_bin_iou": monuseg_iou_sum / max(1, n_monuseg),
            "monusac_bin_dice": monusac_dice_sum / max(1, n_monusac),
            "monusac_bin_iou": monusac_iou_sum / max(1, n_monusac),
            "cell_bin_dice": overall_cell_dice,
            "cell_bin_iou": overall_cell_iou,
            "monusac_cell_dice": monusac_cls_dice_sum / max(1, n_monusac),
            "monusac_cell_iou": monusac_cls_iou_sum / max(1, n_monusac),
            "glas_neg_bg_ratio": glas_neg_bg_sum / max(1, n_glas_neg),
        }

        # Average loss components
        for k, v in loss_components_accum.items():
            metrics[k] = v / max(1, num_batches)

        return metrics

    def fit(self) -> None:
        """Runs the multi-task training loop across all epochs."""
        if self.start_epoch > self.epochs:
            print(f"[MultiTaskTrainer] Training already completed ({self.start_epoch - 1}/{self.epochs} epochs).")
            return

        print(f"\n[MultiTaskTrainer] Starting multi-task training: epochs {self.start_epoch} to {self.epochs}")
        print(f"Batch composition: strict 1:1:1 equal task sampling per batch.")
        print("-" * 80)

        for epoch in range(self.start_epoch, self.epochs + 1):
            t0 = time.time()
            metrics = self.train_one_epoch()

            if self.scheduler is not None:
                self.scheduler.step()

            run_time = time.time() - t0
            # Composite score across tasks: gland dice + nuclei dice + cell dice
            score = (
                metrics["glas_bin_dice"]
                + metrics["monuseg_bin_dice"]
                + metrics["monusac_bin_dice"]
                + metrics["monusac_cell_dice"]
            )
            is_best = score > self.best_score
            if is_best:
                self.best_score = score
                self.best_epoch = epoch

            self._save_checkpoint(epoch, is_best, metrics)
            self._log_history(epoch, metrics, run_time)

            print(
                f"[Epoch {epoch:03d}/{self.epochs:03d}] "
                f"Loss: {metrics['loss']:.4f} | "
                f"GLaS: {metrics['glas_bin_dice']:.4f} | "
                f"Shared Cell: {metrics['cell_bin_dice']:.4f} "
                f"(Seg: {metrics['monuseg_bin_dice']:.4f}, SAC: {metrics['monusac_bin_dice']:.4f}) | "
                f"MoNuSAC Pheno: {metrics['monusac_cell_dice']:.4f} | "
                f"Score: {score:.4f} "
                f"({'BEST' if is_best else ''}) | "
                f"Time: {run_time:.1f}s"
            )

        print("-" * 80)
        print(f"[MultiTaskTrainer] Multi-task training completed. Best composite score: {self.best_score:.4f} at epoch {self.best_epoch}")

