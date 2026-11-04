# datasets/monusac_dataloader.py

from typing import Dict, Any, Union, Optional
from pathlib import Path
from torch.utils.data import DataLoader

from datasets.monusac_dataset import MoNuSACDataset
from datasets.transforms import get_train_transforms
from datasets.utils import load_config, resolve_patch_dir, build_dataloader


def create_monusac_dataloaders(
    config: Union[Dict[str, Any], str, Path],
    transform: Optional[Any] = None,
    class_weights: Optional[Union[Dict[str, float], list]] = None,
    **kwargs,
) -> DataLoader:
    """
    Creates MoNuSAC training DataLoader using parameters from YAML config or dict.
    Resolves data_dir, applies transforms, and supports runtime class-weighted sampling
    (e.g. oversampling macrophage patches based on class_weights).
    """
    cfg = load_config(config)
    data_dir = resolve_patch_dir(cfg, dataset_name="monusac", patch_size=256)
    transforms = transform if transform is not None else get_train_transforms()

    weights = class_weights if class_weights is not None else cfg.get("dataset", {}).get("class_weights", None)

    dataset = MoNuSACDataset(data_dir=data_dir, transform=transforms, class_weights=weights)
    return build_dataloader(dataset, cfg, class_weights=weights, **kwargs)


get_monusac_dataloader = create_monusac_dataloaders

__all__ = ["create_monusac_dataloaders", "get_monusac_dataloader"]
