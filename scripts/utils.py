# scripts/utils.py

import os
import sys
import pprint
import random
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import yaml
import numpy as np
import torch


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a YAML configuration file from disk."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class TeeLogger:
    """Writes output simultaneously to both terminal and log file."""
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        if self.log_file and not self.log_file.closed:
            self.log_file.write(message)
            self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        if self.log_file and not self.log_file.closed:
            self.log_file.flush()

    def close(self):
        if self.log_file and not self.log_file.closed:
            self.log_file.flush()
            self.log_file.close()
        if sys.stdout is self:
            sys.stdout = self.terminal



def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seed across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Returns CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_experiment(config: Dict[str, Any], dataset_name: str) -> Tuple[torch.device, str, Any]:
    """
    Initializes seed, resolves device, creates output/checkpoint directories,
    and sets up TeeLogger to stream stdout to both console and train.log.
    """
    seed = config.get("experiment", {}).get("seed", config.get("seed", 42))
    set_seed(seed)
    device = get_device()

    exp_name = config.get("experiment", {}).get("name", f"{dataset_name}_run")
    chk_path = config.get("checkpoint", {}).get("path", f"checkpoints/{dataset_name}")
    exp_dir = os.path.join(chk_path, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Resolve log directory from config (falls back to exp_dir if not specified)
    log_dir = config.get("experiment", {}).get("log_dir") or config.get("log_dir") or exp_dir
    os.makedirs(log_dir, exist_ok=True)

    log_file_path = os.path.join(log_dir, f"{exp_name}.log")
    log_file = open(log_file_path, "a")
    sys.stdout = TeeLogger(sys.stdout, log_file)

    print("=" * 70)
    print(f"{dataset_name.upper()} Training Starting")
    print(f"Experiment Name: {exp_name}")
    print(f"Checkpoint Dir:  {exp_dir}")
    print(f"Log File:        {log_file_path}")
    print(f"Device:          {device}")
    print("=" * 70)
    pprint.pprint(config)

    return device, exp_dir, log_file


def cleanup_experiment(log_file: Optional[Any] = None) -> None:
    """Restores standard stdout and cleanly flushes and closes log file."""
    if isinstance(sys.stdout, TeeLogger):
        sys.stdout.close()
    if log_file is not None and hasattr(log_file, "closed") and not log_file.closed:
        log_file.flush()
        log_file.close()


def build_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Builds optimizer dynamically according to config parameters.
    Checks config['optimizer'] and falls back to config['training'].
    """
    opt_cfg = config.get("optimizer", {})
    train_cfg = config.get("training", {})

    opt_name = str(opt_cfg.get("name", train_cfg.get("optimizer", "adamw"))).lower().strip()
    lr = float(opt_cfg.get("learning_rate", opt_cfg.get("lr", train_cfg.get("learning_rate", 1e-4))))
    weight_decay = float(opt_cfg.get("weight_decay", train_cfg.get("weight_decay", 1e-5)))

    if opt_name in ("adamw", "adam_w"):
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        momentum = float(opt_cfg.get("momentum", train_cfg.get("momentum", 0.9)))
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer '{opt_name}'. Choose 'adamw', 'adam', or 'sgd'.")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Builds learning rate scheduler dynamically according to config parameters.
    Returns None if no scheduler is configured.
    """
    sched_cfg = config.get("scheduler", {})
    train_cfg = config.get("training", {})
    if not sched_cfg and "scheduler" not in train_cfg:
        return None

    sched_name = str(sched_cfg.get("name", train_cfg.get("scheduler", ""))).lower().strip()
    if not sched_name or sched_name in ("none", "null"):
        return None

    epochs = int(train_cfg.get("epochs", config.get("epochs", 50)))

    if sched_name in ("cosine", "cosineannealinglr"):
        min_lr = float(sched_cfg.get("eta_min", sched_cfg.get("min_lr", train_cfg.get("min_lr", 1e-6))))
        t_max = int(sched_cfg.get("t_max", epochs))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)
    elif sched_name in ("step", "steplr"):
        step_size = int(sched_cfg.get("step_size", train_cfg.get("step_size", 30)))
        gamma = float(sched_cfg.get("gamma", train_cfg.get("gamma", 0.1)))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_name in ("plateau", "reducelronplateau"):
        patience = int(sched_cfg.get("patience", train_cfg.get("patience", 10)))
        factor = float(sched_cfg.get("factor", train_cfg.get("factor", 0.5)))
        mode = str(sched_cfg.get("mode", "min"))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
    else:
        raise ValueError(f"Unsupported scheduler '{sched_name}'.")


def print_model_summary(
    model: torch.nn.Module,
    config: Optional[Union[Dict[str, Any], int]] = None,
    patch_size: int = 256,
    device: Optional[torch.device] = None,
) -> None:
    """
    Prints a detailed model summary whenever a training script runs, including:
      1. Model Architecture & parameter counts (Total, Trainable, Non-trainable)
      2. Top-level module parameter breakdown
      3. Layer-by-layer shapes, parameters, and FLOPs via torchinfo (or fallback to PyTorch structure)
    Ensures model is restored to training mode afterwards.
    """
    if isinstance(config, int):
        patch_size = config
        config_dict = None
    elif isinstance(config, dict):
        config_dict = config
    else:
        config_dict = None

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    model_name = (
        config_dict.get("model", {}).get("name", model.__class__.__name__)
        if config_dict
        else model.__class__.__name__
    )

    print("\n" + "=" * 80)
    print("MODEL SUMMARY & PARAMETER COUNTS")
    print("=" * 80)
    print(f"Model Name / Type:        {model_name}")
    print(f"Total Parameters:         {total_params:,}")
    print(f"Trainable Parameters:     {trainable_params:,}")
    print(f"Non-trainable Parameters: {non_trainable_params:,}")
    print("-" * 80)
    print("Top-Level Module Breakdown:")
    for child_name, child in model.named_children():
        child_total = sum(p.numel() for p in child.parameters())
        child_train = sum(p.numel() for p in child.parameters() if p.requires_grad)
        print(f"  - {child_name:<30} Total: {child_total:>11,} | Trainable: {child_train:>11,}")
    print("=" * 80)

    # Determine patch size
    patch_sz = patch_size
    if config_dict is not None:
        patch_sz = (
            config_dict.get("dataset", {}).get("patch_size")
            or config_dict.get("training", {}).get("patch_size")
            or config_dict.get("model", {}).get("patch_size")
            or config_dict.get("test", {}).get("patch_size")
            or patch_size
        )

    if isinstance(patch_sz, (list, tuple)):
        patch_sz = int(patch_sz[0])
    else:
        try:
            patch_sz = int(patch_sz)
        except Exception:
            patch_sz = 256

    try:
        import torchinfo
    except ImportError:
        torchinfo = None

    input_size = (1, 3, patch_sz, patch_sz)
    summary_printed = False

    if torchinfo is not None:
        try:
            stats = torchinfo.summary(model, input_size=input_size, device=device, verbose=0)
            print(stats)
            summary_printed = True
        except Exception:
            try:
                stats = torchinfo.summary(model, input_size=input_size, verbose=0)
                print(stats)
                summary_printed = True
            except Exception as e:
                print(f"[Note] Detailed torchinfo summary skipped ({e}). Displaying PyTorch model hierarchy:")

    if not summary_printed:
        print(model)

    print("=" * 80 + "\n")
    model.train()


__all__ = [
    "load_yaml_config",
    "TeeLogger",
    "set_seed",
    "get_device",
    "setup_experiment",
    "cleanup_experiment",
    "build_optimizer",
    "build_scheduler",
    "print_model_summary",
]
