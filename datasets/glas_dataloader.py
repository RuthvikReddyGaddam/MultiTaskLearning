# datasets/glas_dataloader.py

from typing import Dict, Any, Union, Optional
from pathlib import Path
from torch.utils.data import DataLoader

from datasets.glas_dataset import GlasDataset
from datasets.transforms import get_train_transforms
from datasets.utils import load_config, resolve_patch_dir, build_dataloader


def create_glas_dataloaders(
    config: Union[Dict[str, Any], str, Path],
    transform: Optional[Any] = None,
    **kwargs,
) -> DataLoader:
    """
    Creates GLaS training DataLoader using parameters from YAML config or dict.
    Resolves data_dir, applies transforms, and configures batch_size, num_workers, pin_memory.
    """
    cfg = load_config(config)
    data_dir = resolve_patch_dir(cfg, dataset_name="glas", patch_size=256)
    transforms = transform if transform is not None else get_train_transforms()

    dataset = GlasDataset(data_dir=data_dir, transform=transforms)
    return build_dataloader(dataset, cfg, shuffle=True, **kwargs)


get_glas_dataloader = create_glas_dataloaders

__all__ = ["create_glas_dataloaders", "get_glas_dataloader"]
