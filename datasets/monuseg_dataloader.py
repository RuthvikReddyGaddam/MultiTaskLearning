# datasets/monuseg_dataloader.py

from typing import Dict, Any, Union, Optional
from pathlib import Path
from torch.utils.data import DataLoader

from datasets.monuseg_dataset import MoNuSegDataset
from datasets.transforms import get_train_transforms
from datasets.utils import load_config, resolve_patch_dir, build_dataloader


def create_monuseg_dataloaders(
    config: Union[Dict[str, Any], str, Path],
    transform: Optional[Any] = None,
    **kwargs,
) -> DataLoader:
    """
    Creates MoNuSeg training DataLoader using parameters from YAML config or dict.
    Resolves data_dir, applies transforms, and configures batch_size, num_workers, pin_memory.
    """
    cfg = load_config(config)
    data_dir = resolve_patch_dir(cfg, dataset_name="monuseg", patch_size=256)
    transforms = transform if transform is not None else get_train_transforms()

    dataset = MoNuSegDataset(data_dir=data_dir, transform=transforms)
    return build_dataloader(dataset, cfg, shuffle=True, **kwargs)


get_monuseg_dataloader = create_monuseg_dataloaders

__all__ = ["create_monuseg_dataloaders", "get_monuseg_dataloader"]
