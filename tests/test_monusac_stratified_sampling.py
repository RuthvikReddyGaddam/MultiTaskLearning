# tests/test_monusac_stratified_sampling.py

import sys
from pathlib import Path
from unittest.mock import MagicMock
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.monusac_dataset import ResampledMoNuSACDataset
from datasets.utils import assign_patch_hierarchical_class


def test_assign_patch_hierarchical_class():
    # Priority 1: Macrophage presence
    stat_mac = {
        "has_cells": True,
        "class_pixels": [500, 200, 100, 5],  # Epi, Lym, Neu, Mac
    }
    assert assign_patch_hierarchical_class(stat_mac) == 3

    # Priority 2: Neutrophil presence (no Mac)
    stat_neu = {
        "has_cells": True,
        "class_pixels": [500, 200, 10, 0],  # Epi, Lym, Neu, Mac
    }
    assert assign_patch_hierarchical_class(stat_neu) == 2

    # Majority 1: Epithelial > Lymphocyte
    stat_epi = {
        "has_cells": True,
        "class_pixels": [300, 100, 0, 0],
    }
    assert assign_patch_hierarchical_class(stat_epi) == 0

    # Majority 2: Lymphocyte > Epithelial
    stat_lym = {
        "has_cells": True,
        "class_pixels": [50, 400, 0, 0],
    }
    assert assign_patch_hierarchical_class(stat_lym) == 1

    # Empty patch
    stat_empty = {
        "has_cells": False,
        "class_pixels": [0, 0, 0, 0],
    }
    assert assign_patch_hierarchical_class(stat_empty) == -1


def test_resampled_monusac_equal_quotas():
    # Mock base MoNuSAC dataset with skewed class buckets:
    # Epithelial: 100 patches, Lymphocyte: 60 patches, Neutrophil: 10 patches, Macrophage: 4 patches
    mock_base = MagicMock()
    mock_base.__len__.return_value = 174
    mock_base.class_indices = {
        0: list(range(0, 100)),       # 100
        1: list(range(100, 160)),     # 60
        2: list(range(160, 170)),     # 10
        3: list(range(170, 174)),     # 4
    }
    mock_base.__getitem__.side_effect = lambda idx: {"image": torch.zeros(3, 32, 32), "idx": idx}

    # Initialize resampled dataset
    # Max class size N_max = 100.
    # Total virtual pool = 4 * 100 = 400.
    resampled_ds = ResampledMoNuSACDataset(
        base_dataset=mock_base,
        num_samples=400,
        seed=42,
    )

    assert len(resampled_ds) == 400
    assert resampled_ds.n_max == 100
    assert resampled_ds.samples_per_class == 100

    # Check that in the 400 drawn samples, every class is sampled exactly 100 times
    # Map index to class
    idx_to_class = {}
    for c, idxs in mock_base.class_indices.items():
        for i in idxs:
            idx_to_class[i] = c

    drawn_classes = [idx_to_class[idx] for idx in resampled_ds.sampled_indices]
    counts = {c: drawn_classes.count(c) for c in range(4)}

    print("Drawn class counts:", counts)
    assert counts[0] == 100, f"Expected 100 Epithelial, got {counts[0]}"
    assert counts[1] == 100, f"Expected 100 Lymphocyte, got {counts[1]}"
    assert counts[2] == 100, f"Expected 100 Neutrophil, got {counts[2]}"
    assert counts[3] == 100, f"Expected 100 Macrophage, got {counts[3]}"

    # Verify interleaving: every group of 4 consecutive samples has 1 of each class
    for i in range(0, len(drawn_classes), 4):
        slice_classes = set(drawn_classes[i:i+4])
        assert slice_classes == {0, 1, 2, 3}, f"Slice at {i} not fully balanced: {drawn_classes[i:i+4]}"


if __name__ == "__main__":
    test_assign_patch_hierarchical_class()
    print("test_assign_patch_hierarchical_class PASSED!")
    test_resampled_monusac_equal_quotas()
    print("test_resampled_monusac_equal_quotas PASSED!")
