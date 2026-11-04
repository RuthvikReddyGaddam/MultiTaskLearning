# utils/training_utils.py

from typing import Optional, Dict, Any, Union
import torch

try:
    import torchinfo
except ImportError:
    torchinfo = None


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    Builds the optimizer from the training configuration dictionary.
    Supports AdamW, Adam, and SGD with momentum.
    """
    training_config = config.get("training", {})
    optimizer_config = config.get("optimizer", {})

    optimizer_name = training_config.get(
        "optimizer", optimizer_config.get("name", "adamw")
    ).lower()

    lr = float(training_config.get(
        "learning_rate", optimizer_config.get("learning_rate", 1e-4)
    ))
    weight_decay = float(training_config.get(
        "weight_decay", optimizer_config.get("weight_decay", 1e-2)
    ))

    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        momentum = float(training_config.get("momentum", 0.9))
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: '{optimizer_name}'. Supported: 'adamw', 'adam', 'sgd'.")


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Builds learning rate scheduler from configuration dictionary.
    Supports CosineAnnealingLR and StepLR.
    """
    training_config = config.get("training", {})
    scheduler_config = config.get("scheduler", {})

    scheduler_name = training_config.get(
        "scheduler", scheduler_config.get("name", "cosine")
    ).lower().strip()

    epochs = int(training_config.get("epochs", 50))

    if scheduler_name in ("cosine", "cosineannealinglr"):
        min_lr = float(training_config.get(
            "min_lr", scheduler_config.get("eta_min", 1e-6)
        ))
        t_max = int(scheduler_config.get("t_max", epochs))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)
    elif scheduler_name in ("step", "steplr"):
        step_size = int(training_config.get(
            "step_size", scheduler_config.get("step_size", 30)
        ))
        gamma = float(training_config.get(
            "gamma", scheduler_config.get("gamma", 0.1)
        ))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return None


def print_model_summary(
    model: torch.nn.Module,
    config: Optional[Union[Dict[str, Any], int]] = None,
    patch_size: int = 256,
    device: Optional[torch.device] = None,
) -> None:
    """Print model parameter counts and architecture summary before training or testing."""
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
