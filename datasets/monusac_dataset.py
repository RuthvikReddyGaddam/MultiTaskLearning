# datasets/monusac_dataset.py

from pathlib import Path
from typing import Optional, Union, Any, List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from datasets.utils import (
    load_patch_paths,
    create_monusac_class_maps,
    apply_patch_transforms,
    compute_or_load_patch_class_stats,
    parse_class_weights,
    compute_patch_sampling_weights,
    PatchSample,
    CLASS_NAMES,
    ALL_CLASS_NAMES,
)
from datasets.multitask_collate import compute_4class_target


class MoNuSACDataset(Dataset):
    """
    Dataset for MoNuSAC train patches.
    Keeps track of all patches along with their percent of all classes, and supports
    runtime class-weighted sampling (e.g. oversampling macrophage-containing patches).
    
    Yields PatchSample containing:
      - "image":          [3, H, W] float32 image patch in [0, 1]
      - "mask":           [1, H, W] float32 binary nucleus mask (0.0 or 1.0)
      - "sdf":            [1, H, W] float32 signed distance function patch
      - "class_mask":     [5, H, W] float32 5-channel mask (ch 0: background, ch 1-4: 4 cell classes)
      - "class_map":      [H, W] int64 class map with integer values in {0, 1, 2, 3, 4}
      - "class_percents": [5] float32 tensor of patch percentage for each class [BG, Epi, Lym, Neu, Mac]
      - "patch_name":     str identifier of the patch
      - "patch_idx":      int dataset index
    """
    def __init__(
        self,
        data_dir: Union[str, Path],
        transform: Optional[Any] = None,
        class_weights: Optional[Union[List[float], Dict[str, float]]] = None,
        force_recompute_stats: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples: List[tuple] = load_patch_paths(self.data_dir)
        if len(self.samples) == 0:
            raise RuntimeError(f"No patch files found in {self.data_dir}")

        # Track per-patch class statistics and percentages
        self.patch_stats: Dict[str, Dict[str, Any]] = compute_or_load_patch_class_stats(
            self.data_dir,
            force_recompute=force_recompute_stats,
        )
        # Precompute parallel list of records for O(1) index access
        self.patch_records: List[Dict[str, Any]] = [
            self.patch_stats.get(s[0].stem, {}) for s in self.samples
        ]
        # Precompute [N, 5] tensor of class percentages across all patches:
        # Columns: [Background %, Epithelial %, Lymphocyte %, Neutrophil %, Macrophage %]
        percents_list = [
            r.get("class_percents_5ch", [100.0, 0.0, 0.0, 0.0, 0.0])
            for r in self.patch_records
        ]
        self.class_percents = torch.tensor(percents_list, dtype=torch.float32)

        # Runtime sampling weights
        self.cell_weights, self.bg_weight = parse_class_weights(class_weights)
        self.class_weights = self.cell_weights

    def __len__(self) -> int:
        return len(self.samples)

    def set_class_weights(self, class_weights: Union[List[float], Dict[str, float]]) -> None:
        """Updates class sampling weights dynamically at runtime."""
        self.cell_weights, self.bg_weight = parse_class_weights(class_weights)
        self.class_weights = self.cell_weights

    def get_patch_stats(self, idx: int) -> Dict[str, Any]:
        """Returns the complete class percentage statistics dict for a specific patch index."""
        if 0 <= idx < len(self.patch_records):
            return self.patch_records[idx]
        name = self.samples[idx][0].stem
        return self.patch_stats.get(name, {})

    def get_class_percents(self, idx: Optional[int] = None) -> torch.Tensor:
        """
        Returns class percentages:
          - If idx is None: [N, 5] tensor of [BG%, Epi%, Lym%, Neu%, Mac%] for all patches
          - If idx is int:  [5] tensor of percentages for patch at idx
        """
        if idx is None:
            return self.class_percents
        return self.class_percents[idx]

    def get_sampling_weights(
        self,
        class_weights: Optional[Union[List[float], Dict[str, float]]] = None,
        empty_weight: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Computes 1D sampling weights for all patches based on their class percentages
        and the specified (or currently configured) class weights.
        """
        weights = class_weights if class_weights is not None else self.class_weights
        emp_w = empty_weight if empty_weight is not None else self.bg_weight
        return compute_patch_sampling_weights(
            samples=self.samples,
            patch_stats=self.patch_stats,
            class_weights=weights,
            empty_weight=emp_w,
        )

    def get_sampler(
        self,
        class_weights: Optional[Union[List[float], Dict[str, float]]] = None,
        num_samples: Optional[int] = None,
        replacement: bool = True,
    ) -> WeightedRandomSampler:
        """
        Creates a PyTorch WeightedRandomSampler that samples patches with probability
        proportional to their class percentages and runtime class weights.
        """
        weights = self.get_sampling_weights(class_weights)
        n_samples = num_samples if num_samples is not None else len(self.samples)
        return WeightedRandomSampler(
            weights=weights,
            num_samples=n_samples,
            replacement=replacement,
        )

    def get_class_distribution(self) -> Dict[str, Any]:
        """Summarizes dataset-wide class presence, counts, and average area percentages."""
        class_counts = [0, 0, 0, 0]
        class_area_sums = [0.0, 0.0, 0.0, 0.0]
        empty_count = 0
        total = len(self.samples)

        for record in self.patch_records:
            present = record.get("classes_present", [])
            if not present or not record.get("has_cells", False):
                empty_count += 1
            for c in present:
                if c < 4:
                    class_counts[c] += 1
            cell_pcts = record.get("class_percents", [0.0, 0.0, 0.0, 0.0])
            for i, p in enumerate(cell_pcts[:4]):
                class_area_sums[i] += p

        return {
            "total_patches": total,
            "empty_patches": empty_count,
            "empty_percent": (empty_count / total * 100.0) if total > 0 else 0.0,
            "class_patch_counts": {name: class_counts[i] for i, name in enumerate(CLASS_NAMES)},
            "class_patch_percents": {
                name: (class_counts[i] / total * 100.0 if total > 0 else 0.0)
                for i, name in enumerate(CLASS_NAMES)
            },
            "avg_patch_class_area_percent": {
                name: (class_area_sums[i] / total if total > 0 else 0.0)
                for i, name in enumerate(CLASS_NAMES)
            },
        }

    def print_class_distribution(self) -> None:
        """Prints a formatted summary table of dataset class distribution and runtime weights."""
        dist = self.get_class_distribution()
        total = dist["total_patches"]
        print("\n" + "=" * 70)
        print(f"MoNuSAC Dataset Class Distribution ({total:,} total patches in {self.data_dir.name}):")
        print(f"{'Class':<14} | {'Patches Present':<16} | {'% of Dataset':<14} | {'Avg Area %':<12} | {'Weight'}")
        print("-" * 70)
        for i, name in enumerate(CLASS_NAMES):
            count = dist["class_patch_counts"][name]
            pct = dist["class_patch_percents"][name]
            avg_area = dist["avg_patch_class_area_percent"][name]
            w = self.cell_weights[i] if i < len(self.cell_weights) else 1.0
            print(f"{name:<14} | {count:6d} patches   | {pct:6.2f}%       | {avg_area:6.2f}%     | {w:.2f}")
        emp = dist["empty_patches"]
        emp_pct = dist["empty_percent"]
        print(f"{'Background/Empty':<14} | {emp:6d} patches   | {emp_pct:6.2f}%       | {'-':<12} | {self.bg_weight:.2f}")
        print("=" * 70 + "\n")

    def __getitem__(self, idx: int) -> PatchSample:
        img_path, mask_path, _ = self.samples[idx]

        # 1. Load image [H, W, 3] -> [3, H, W] float32 in [0, 1]
        image = np.load(img_path)
        img_t = torch.from_numpy(image).permute(2, 0, 1).float()
        if img_t.max() > 1.0:
            img_t = img_t / 255.0

        H, W = img_t.shape[-2:]

        # 2. Load precomputed 4-class target [H, W] int64
        t4_path = self.data_dir / "targets_4class" / img_path.name
        if t4_path.exists():
            target_4class = torch.from_numpy(np.load(t4_path)).long()
        else:
            mask_4ch = np.load(mask_path)
            unified_inst = np.zeros(mask_4ch.shape[1:], dtype=np.int32)
            for c in range(4):
                c_mask = mask_4ch[c]
                unified_inst[c_mask > 0] = c_mask[c_mask > 0]
            target_4class = compute_4class_target(torch.from_numpy(unified_inst).unsqueeze(0)).squeeze(0)

        # 3. Load precomputed binary mask [1, H, W] float32
        bin_path = self.data_dir / "binary_masks" / img_path.name
        if bin_path.exists():
            binary_mask = torch.from_numpy(np.load(bin_path)).float().unsqueeze(0)
        else:
            binary_mask = (target_4class > 0).float().unsqueeze(0)

        # 4. Standard 2D integer class map [H, W] in {0: bg, 1: epi, 2: lym, 3: neu, 4: mac}
        mask_4ch = np.load(mask_path)
        class_map_np = np.zeros((H, W), dtype=np.int64)
        for c in range(4):
            class_map_np[mask_4ch[c] > 0] = c + 1
        class_map = torch.from_numpy(class_map_np).long()

        # 5. Legacy 5-channel class mask (only for evaluation/tests if needed)
        class_mask_5ch = torch.zeros((5, H, W), dtype=torch.float32)
        class_mask_5ch[0] = 1.0 - binary_mask.squeeze(0)
        for c in range(4):
            class_mask_5ch[c + 1] = torch.from_numpy((mask_4ch[c] > 0).astype(np.float32))

        sample = PatchSample(
            image=img_t,
            target_4class=target_4class,
            target_3class=target_4class,
            binary_mask=binary_mask,
            mask=binary_mask,
            class_map=class_map,
            class_mask=class_mask_5ch,
            class_percents=self.class_percents[idx],
            task_id=2,
            patch_name=self.samples[idx][0].stem,
            patch_idx=idx,
        )
        return sample


class ResampledMoNuSACDataset(Dataset):
    """
    Virtual oversampled MoNuSAC dataset for multi-task class balancing (Approach A).
    Presents an expanded virtual length M where patches containing minority classes
    (macrophages, neutrophils) are sampled with replacement proportional to their class weights.
    """
    def __init__(
        self,
        base_dataset: MoNuSACDataset,
        num_samples: Optional[int] = None,
        class_weights: Optional[Union[Dict[str, float], List[float]]] = None,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
    ):
        self.base_dataset = base_dataset
        # Default to 600 for 512 patches, 4000 for 256 patches, or max(len(base_dataset), 600)
        if num_samples is not None:
            self.num_samples = int(num_samples)
        else:
            self.num_samples = 600 if len(base_dataset) < 1000 else 4000

        self.class_weights = class_weights
        self.generator = generator
        if seed is not None:
            self.generator = torch.Generator().manual_seed(seed)

        self.sampling_weights = base_dataset.get_sampling_weights(class_weights)
        self.resample()

    def resample(self) -> None:
        """Draws a new multinomial sample of indices for the epoch."""
        self.sampled_indices = torch.multinomial(
            self.sampling_weights,
            num_samples=self.num_samples,
            replacement=True,
            generator=self.generator,
        ).tolist()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> PatchSample:
        base_idx = self.sampled_indices[idx]
        sample = self.base_dataset[base_idx]
        if isinstance(sample, dict):
            sample["resampled_idx"] = idx
            sample["base_idx"] = base_idx
        return sample

