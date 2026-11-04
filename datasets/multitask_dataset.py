# datasets/multitask_dataset.py

from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import torch
from torch.utils.data import Dataset, DataLoader

from datasets.glas_dataset import GlasDataset
from datasets.monuseg_dataset import MoNuSegDataset
from datasets.monusac_dataset import MoNuSACDataset, ResampledMoNuSACDataset
from datasets.multitask_sampler import EqualTaskBatchSampler
from datasets.multitask_collate import MultiTaskCollateFn
from datasets.multitask_augment import MultiTaskSpatialAugmentor
from datasets.utils import resolve_patch_dir, load_config, PROJECT_ROOT


class MultiTaskDataset(Dataset):
    """
    Unified multi-task dataset combining:
      - Dataset 0: GLaS (Gland instance/semantic segmentation)
      - Dataset 1: MoNuSeg (Multi-organ cell nuclei segmentation)
      - Dataset 2: MoNuSAC (Cell nuclei segmentation + 4-class cell type classification)

    Ensures consistent key schemas and tags each sample with its integer `task_id`.
    Applies joint spatial/vector-field augmentations synchronously across all maps.
    """
    def __init__(
        self,
        glas_dataset: GlasDataset,
        monuseg_dataset: MoNuSegDataset,
        monusac_dataset: Union[MoNuSACDataset, ResampledMoNuSACDataset],
        spatial_augmentor: Optional[MultiTaskSpatialAugmentor] = None,
    ):
        self.glas_dataset = glas_dataset
        self.monuseg_dataset = monuseg_dataset
        self.monusac_dataset = monusac_dataset
        self.spatial_augmentor = spatial_augmentor

        self.lengths = [
            len(self.glas_dataset),
            len(self.monuseg_dataset),
            len(self.monusac_dataset),
        ]
        self.offsets = [
            0,
            self.lengths[0],
            self.lengths[0] + self.lengths[1],
        ]
        self.total_len = sum(self.lengths)

    def __len__(self) -> int:
        return self.total_len

    def get_dataset_lengths(self) -> List[int]:
        """Returns [N_glas, N_monuseg, N_monusac]"""
        return list(self.lengths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < self.offsets[1]:
            # GLaS sample
            sample = self.glas_dataset[idx]
            sample["task_id"] = 0
            sample["dataset_name"] = "glas"
        elif idx < self.offsets[2]:
            # MoNuSeg sample
            local_idx = idx - self.offsets[1]
            sample = self.monuseg_dataset[local_idx]
            sample["task_id"] = 1
            sample["dataset_name"] = "monuseg"
        elif idx < self.total_len:
            # MoNuSAC sample
            local_idx = idx - self.offsets[2]
            sample = self.monusac_dataset[local_idx]
            sample["task_id"] = 2
            sample["dataset_name"] = "monusac"
        else:
            raise IndexError(f"Index {idx} out of bounds for MultiTaskDataset with length {self.total_len}")

        # Apply joint spatial & vector field rotation augmentations if configured
        if self.spatial_augmentor is not None:
            sample = self.spatial_augmentor(sample)

        return sample


def build_multitask_dataloader(
    glas_dir: Optional[Union[str, Path]] = None,
    monuseg_dir: Optional[Union[str, Path]] = None,
    monusac_dir: Optional[Union[str, Path]] = None,
    patch_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    augment: bool = True,
    steps_per_epoch: Optional[int] = None,
    ignore_index: int = -1,
    pin_memory: bool = True,
    monusac_resample_size: Optional[int] = None,
    monusac_class_weights: Optional[Dict[str, float]] = None,
    class_weights: Optional[Dict[str, float]] = None,
) -> DataLoader:
    """
    Factory function to construct a balanced Multi-Task DataLoader.
    
    Guarantees:
      - Exactly batch_size // 3 samples from GLaS, MoNuSeg, and MoNuSAC in EVERY batch.
      - Approach A: MoNuSAC is wrapped in ResampledMoNuSACDataset with minority-class oversampling.
      - Orientation-preserving vector field rotations on 2-channel flow vectors.
      - Padded class maps (-1) and boolean task indicator masks for decoupled heads.
    """
    if monusac_class_weights is None and class_weights is not None:
        monusac_class_weights = class_weights
    # 1. Resolve directories (defaults to data/<name>_train_patch_<patch_size>)
    if glas_dir is None:
        glas_dir = PROJECT_ROOT / "data" / f"glas_train_patch_{patch_size}"
    else:
        glas_dir = Path(glas_dir)

    if monuseg_dir is None:
        monuseg_dir = PROJECT_ROOT / "data" / f"monuseg_train_patch_{patch_size}"
    else:
        monuseg_dir = Path(monuseg_dir)

    if monusac_dir is None:
        monusac_dir = PROJECT_ROOT / "data" / f"monusac_train_patch_{patch_size}"
    else:
        monusac_dir = Path(monusac_dir)

    # 2. Instantiate individual datasets (transforms handled by MultiTaskSpatialAugmentor)
    glas_ds = GlasDataset(data_dir=glas_dir, transform=None)
    monuseg_ds = MoNuSegDataset(data_dir=monuseg_dir, transform=None)


    if monusac_class_weights is None:
        monusac_class_weights = {
            "background": 0.05,
            "epithelial": 1.0,
            "lymphocyte": 1.0,
            "neutrophil": 5.0,
            "macrophage": 10.0,
        }

    raw_monusac_ds = MoNuSACDataset(data_dir=monusac_dir, transform=None, class_weights=monusac_class_weights)

    # 3. Approach A: Wrap MoNuSAC in ResampledMoNuSACDataset
    target_resample = monusac_resample_size
    if target_resample is None:
        target_resample = 600 if patch_size == 512 else 4000

    balanced_monusac_ds = ResampledMoNuSACDataset(
        base_dataset=raw_monusac_ds,
        num_samples=target_resample,
        class_weights=monusac_class_weights,
    )

    # 4. Setup spatial augmentor
    augmentor = MultiTaskSpatialAugmentor(
        p_hflip=0.5,
        p_vflip=0.5,
        p_rot90=0.5,
        color_jitter=True,
    ) if augment else None

    # 5. Wrap into unified MultiTaskDataset
    multitask_ds = MultiTaskDataset(
        glas_dataset=glas_ds,
        monuseg_dataset=monuseg_ds,
        monusac_dataset=balanced_monusac_ds,
        spatial_augmentor=augmentor,
    )

    # 6. Build EqualTaskBatchSampler
    batch_sampler = EqualTaskBatchSampler(
        dataset_lengths=multitask_ds.get_dataset_lengths(),
        batch_size=batch_size,
        shuffle=shuffle,
        steps_per_epoch=steps_per_epoch,
    )

    # 7. Build MultiTaskCollateFn
    collate_fn = MultiTaskCollateFn(ignore_index=ignore_index, num_classes=5, check_equal_split=True)

    # 8. Construct and return DataLoader
    loader = DataLoader(
        dataset=multitask_ds,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return loader

