# tests/test_dataset_creation_ground_truths.py

import unittest
import sys
import shutil
import tempfile
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import skimage
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

if HAS_SKIMAGE:
    from create_datasets.monuseg_utils import (
        compute_instance_sdf as compute_monuseg_sdf,
        compute_3class_target as compute_monuseg_3class,
        compute_4class_target_with_bottom_hat as compute_monuseg_4class,
        save_patch_pentaplet as save_monuseg_pentaplet,
        save_patch_heptaplet as save_monuseg_heptaplet,
    )
    from create_datasets.glas_utils import (
        compute_instance_sdf as compute_glas_sdf,
        compute_3class_target as compute_glas_3class,
        compute_4class_target_with_bottom_hat as compute_glas_4class,
        save_patch_pentaplet as save_glas_pentaplet,
        save_patch_heptaplet as save_glas_heptaplet,
    )
    from create_datasets.monusac_utils import (
        compute_instance_sdf as compute_monusac_sdf,
        compute_3class_target as compute_monusac_3class,
        compute_4class_target_with_bottom_hat as compute_monusac_4class,
        save_patch_pentaplet as save_monusac_pentaplet,
        save_patch_heptaplet as save_monusac_heptaplet,
    )


@unittest.skipUnless(HAS_SKIMAGE, "scikit-image not installed in environment")
class TestDatasetCreationGroundTruths(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.H, self.W = 100, 100

        # Create two touching circular nuclei:
        # Cell 1: center (40, 40), radius 12
        # Cell 2: center (40, 62), radius 12 (touching at x=51)
        # Plus a tiny nucleus (radius 1) at (80, 80) to test small-cell degeneracy
        y, x = np.mgrid[:self.H, :self.W]
        d1 = np.sqrt((y - 40)**2 + (x - 40)**2)
        d2 = np.sqrt((y - 40)**2 + (x - 62)**2)
        d3 = np.sqrt((y - 80)**2 + (x - 80)**2)

        self.instance_mask = np.zeros((self.H, self.W), dtype=np.int32)
        self.instance_mask[d1 <= 12] = 1
        self.instance_mask[d2 <= 12] = 2
        self.instance_mask[d3 <= 1.5] = 3

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_per_instance_sdf_properties(self):
        """Verify SDF calculates per-instance inside negative distance and 0.0 boundary."""
        sdf = compute_monuseg_sdf(self.instance_mask)

        # Background must be strictly positive
        self.assertTrue(np.all(sdf[self.instance_mask == 0] > 0))

        # Deep inside Cell 1 center (40, 40), distance must be negative
        self.assertLess(sdf[40, 40], -5.0)

        # Deep inside Cell 2 center (40, 62), distance must be negative
        self.assertLess(sdf[40, 62], -5.0)

        # Boundary pixels must be 0.0
        self.assertTrue(np.any(sdf == 0.0))
        self.assertTrue(np.all(sdf[self.instance_mask > 0] <= 0.0))

    def test_per_instance_4class_target_with_bottom_hat(self):
        """Verify 4-class target generates interior (1), boundary (2), and bottom-hat gap (3)."""
        t4 = compute_monuseg_4class(self.instance_mask, kernel_size=5, boundary_width=1)

        # Check classes present: {0, 1, 2, 3}
        unique_classes = set(np.unique(t4))
        self.assertEqual(unique_classes, {0, 1, 2, 3})

        # Cell centers must be class 1 (interior core)
        self.assertEqual(t4[40, 40], 1)
        self.assertEqual(t4[40, 62], 1)

        # Touching seam at x=51 between Cell 1 and Cell 2 must be class 3 (gap)
        self.assertEqual(t4[40, 51], 3)

        # Tiny cell at (80, 80) must still retain an interior seed (class 1)
        self.assertTrue(np.any(t4[(self.instance_mask == 3)] == 1), "Tiny instance lost interior seed!")

    def test_save_patch_pentaplet_creates_all_5_gt_files(self):
        """Verify save_patch_pentaplet creates all 5 expected directories and files."""
        dirs = {
            "images": self.temp_dir / "images",
            "binary_masks": self.temp_dir / "binary_masks",
            "instance_masks": self.temp_dir / "instance_masks",
            "sdfs": self.temp_dir / "sdfs",
            "targets_4class": self.temp_dir / "targets_4class",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        patch_name = "test_sample_0001"
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        bin_mask = (self.instance_mask > 0).astype(np.float32)
        inst_mask = self.instance_mask
        sdf = compute_monuseg_sdf(inst_mask)
        t4 = compute_monuseg_4class(inst_mask)

        save_monuseg_pentaplet(dirs, patch_name, img, bin_mask, inst_mask, sdf, t4)

        for key, d in dirs.items():
            fpath = d / f"{patch_name}.npy"
            self.assertTrue(fpath.exists(), f"Missing file: {fpath} for {key}")
            loaded = np.load(fpath)
            self.assertEqual(loaded.shape[:2], (self.H, self.W))


    def test_per_instance_3class_target(self):
        """Verify 3-class target generates interior (1) and boundary (2), with no gap class."""
        t3 = compute_monuseg_3class(self.instance_mask, boundary_width=1)

        # Check classes present: strictly {0, 1, 2}
        unique_classes = set(np.unique(t3))
        self.assertEqual(unique_classes, {0, 1, 2})

        # Cell centers must be class 1 (interior core)
        self.assertEqual(t3[40, 40], 1)
        self.assertEqual(t3[40, 62], 1)

        # Tiny cell at (80, 80) must still retain an interior seed (class 1)
        self.assertTrue(np.any(t3[(self.instance_mask == 3)] == 1), "Tiny instance lost interior seed in 3-class target!")

    def test_save_patch_heptaplet_creates_all_7_gt_files(self):
        """Verify save_patch_heptaplet creates all 7 expected directories and files."""
        dirs = {
            "images": self.temp_dir / "images",
            "binary_masks": self.temp_dir / "binary_masks",
            "instance_masks": self.temp_dir / "instance_masks",
            "sdfs": self.temp_dir / "sdfs",
            "targets_3class": self.temp_dir / "targets_3class",
            "targets_4class": self.temp_dir / "targets_4class",
            "hv_maps": self.temp_dir / "hv_maps",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        patch_name = "test_heptaplet_0001"
        img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        bin_mask = (self.instance_mask > 0).astype(np.float32)
        inst_mask = self.instance_mask
        sdf = compute_monuseg_sdf(inst_mask)
        t3 = compute_monuseg_3class(inst_mask)
        t4 = compute_monuseg_4class(inst_mask)
        hv = np.zeros((2, self.H, self.W), dtype=np.float32)

        save_monuseg_heptaplet(dirs, patch_name, img, bin_mask, inst_mask, sdf, t3, t4, hv)

        for key, d in dirs.items():
            fpath = d / f"{patch_name}.npy"
            self.assertTrue(fpath.exists(), f"Missing file: {fpath} for {key}")
            loaded = np.load(fpath)
            if key == "hv_maps":
                self.assertEqual(loaded.shape, (2, self.H, self.W))
            else:
                self.assertEqual(loaded.shape[:2], (self.H, self.W))


if __name__ == "__main__":
    unittest.main()

