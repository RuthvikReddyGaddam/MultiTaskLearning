# tests/test_class_balanced_monusac.py

import unittest
from unittest.mock import MagicMock
import torch
from torch.utils.data import Dataset, DataLoader

from datasets.monusac_dataset import ClassBalancedMoNuSACDataset
from datasets.multitask_sampler import EqualTaskBatchSampler
from datasets.multitask_dataset import MultiTaskDataset
from datasets.multitask_collate import MultiTaskCollateFn


class MockMoNuSACDataset(Dataset):
    """Mock MoNuSAC dataset with known imbalanced class distributions."""
    def __init__(self, n_epi=150, n_lym=100, n_neu=20, n_mac=10):
        super().__init__()
        self.patch_records = []
        self.samples = []

        idx = 0
        # Epithelial (class 0)
        for _ in range(n_epi):
            self.patch_records.append({
                "classes_present": [0],
                "class_pixels": [500, 0, 0, 0],
                "has_cells": True,
            })
            self.samples.append((f"img_epi_{idx}", f"mask_epi_{idx}", None))
            idx += 1

        # Lymphocyte (class 1)
        for _ in range(n_lym):
            self.patch_records.append({
                "classes_present": [1],
                "class_pixels": [0, 300, 0, 0],
                "has_cells": True,
            })
            self.samples.append((f"img_lym_{idx}", f"mask_lym_{idx}", None))
            idx += 1

        # Neutrophil (class 2)
        for _ in range(n_neu):
            self.patch_records.append({
                "classes_present": [2],
                "class_pixels": [0, 0, 50, 0],
                "has_cells": True,
            })
            self.samples.append((f"img_neu_{idx}", f"mask_neu_{idx}", None))
            idx += 1

        # Macrophage (class 3)
        for _ in range(n_mac):
            self.patch_records.append({
                "classes_present": [3],
                "class_pixels": [0, 0, 0, 30],
                "has_cells": True,
            })
            self.samples.append((f"img_mac_{idx}", f"mask_mac_{idx}", None))
            idx += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            "image": torch.zeros((3, 32, 32), dtype=torch.float32),
            "binary_mask": torch.zeros((1, 32, 32), dtype=torch.float32),
            "class_map": torch.zeros((32, 32), dtype=torch.long),
            "target_4class": torch.zeros((32, 32), dtype=torch.long),
            "task_id": 2,
            "patch_name": f"mock_patch_{idx}",
        }


class MockDataset(Dataset):
    def __init__(self, length, task_id):
        self.length = length
        self.task_id = task_id

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "image": torch.zeros((3, 32, 32), dtype=torch.float32),
            "binary_mask": torch.zeros((1, 32, 32), dtype=torch.float32),
            "class_map": torch.zeros((32, 32), dtype=torch.long),
            "target_4class": torch.zeros((32, 32), dtype=torch.long),
            "task_id": self.task_id,
            "patch_name": f"task_{self.task_id}_patch_{idx}",
        }


class TestClassBalancedMoNuSAC(unittest.TestCase):

    def test_equal_class_counts(self):
        """Verify that ClassBalancedMoNuSACDataset outputs exactly patches_per_class for each class."""
        mock_ds = MockMoNuSACDataset(n_epi=150, n_lym=100, n_neu=20, n_mac=10)
        patches_per_class = 80
        balanced_ds = ClassBalancedMoNuSACDataset(
            base_dataset=mock_ds,
            patches_per_class=patches_per_class,
            seed=42,
        )

        self.assertEqual(len(balanced_ds), 4 * patches_per_class)
        self.assertEqual(len(balanced_ds), 320)

        # Count drawn classes in the resampled list
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for c in balanced_ds.sampled_classes:
            class_counts[c] += 1

        self.assertEqual(class_counts[0], 80, "Epithelial must have exactly 80 patches")
        self.assertEqual(class_counts[1], 80, "Lymphocyte must have exactly 80 patches")
        self.assertEqual(class_counts[2], 80, "Neutrophil must have exactly 80 patches")
        self.assertEqual(class_counts[3], 80, "Macrophage must have exactly 80 patches")

    def test_major_class_without_replacement(self):
        """Verify that a major class (150 patches) does not repeat any patch in a single 80-patch epoch."""
        mock_ds = MockMoNuSACDataset(n_epi=150, n_lym=100, n_neu=20, n_mac=10)
        balanced_ds = ClassBalancedMoNuSACDataset(
            base_dataset=mock_ds,
            patches_per_class=80,
            seed=123,
        )

        # Find all drawn indices corresponding to class 0 (Epi)
        epi_indices = [
            balanced_ds.sampled_indices[i]
            for i, c in enumerate(balanced_ds.sampled_classes)
            if c == 0
        ]
        self.assertEqual(len(epi_indices), 80)
        self.assertEqual(len(set(epi_indices)), 80, "Major class must be drawn without replacement in an 80-patch draw")

    def test_minor_class_cyclic_exhaustion_and_reintroduction(self):
        """Verify that a minor class (10 patches) exhausts and draws all 10 patches before any repeats."""
        mock_ds = MockMoNuSACDataset(n_epi=150, n_lym=100, n_neu=20, n_mac=10)
        balanced_ds = ClassBalancedMoNuSACDataset(
            base_dataset=mock_ds,
            patches_per_class=80,
            seed=999,
        )

        mac_indices = [
            balanced_ds.sampled_indices[i]
            for i, c in enumerate(balanced_ds.sampled_classes)
            if c == 3
        ]
        self.assertEqual(len(mac_indices), 80)
        # All 10 unique macrophage patches must be present in the 80 drawn patches
        self.assertEqual(len(set(mac_indices)), 10, "All 10 unique macrophage patches must be represented")

        # In 80 draws from 10 patches, each patch must be drawn exactly 8 times (80 / 10 = 8)
        counts = {}
        for idx in mac_indices:
            counts[idx] = counts.get(idx, 0) + 1
        for patch_id, cnt in counts.items():
            self.assertEqual(cnt, 8, f"Patch {patch_id} should appear exactly 8 times in 80 cyclic draws")

    def test_dataloader_160_steps(self):
        """Verify that DataLoader with steps_per_epoch=160 and batch_size=8 yields 160 batches with 2:1:1 split."""
        glas_ds = MockDataset(length=328, task_id=0)
        monuseg_ds = MockDataset(length=120, task_id=1)
        mock_sac = MockMoNuSACDataset(n_epi=150, n_lym=100, n_neu=20, n_mac=10)
        sac_ds = ClassBalancedMoNuSACDataset(base_dataset=mock_sac, patches_per_class=80, seed=42)

        multitask_ds = MultiTaskDataset(
            glas_dataset=glas_ds,
            monuseg_dataset=monuseg_ds,
            monusac_dataset=sac_ds,
        )

        sampler = EqualTaskBatchSampler(
            dataset_lengths=multitask_ds.get_dataset_lengths(),
            batch_size=8,
            steps_per_epoch=160,
        )

        self.assertEqual(len(sampler), 160)

        collate_fn = MultiTaskCollateFn(ignore_index=-1, num_classes=5, check_equal_split=True)
        loader = DataLoader(
            dataset=multitask_ds,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=0,
        )

        self.assertEqual(len(loader), 160)

        # Inspect first 3 batches to verify strict 2:1:1 split and shuffling
        for i, batch in enumerate(loader):
            if i >= 3:
                break
            task_ids = batch["task_id"]
            self.assertEqual(task_ids.shape[0], 8)
            self.assertEqual((task_ids == 0).sum().item(), 4, "Must have exactly 4 GLaS patches")
            self.assertEqual((task_ids == 1).sum().item(), 2, "Must have exactly 2 MoNuSeg patches")
            self.assertEqual((task_ids == 2).sum().item(), 2, "Must have exactly 2 MoNuSAC patches")


if __name__ == "__main__":
    unittest.main()

