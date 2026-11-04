import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2
from utils.test_utils import class4_to_instances


class TestGlandLumenReclamation(unittest.TestCase):
    def test_hollow_gland_lumen_is_reclaimed(self):
        # Create a single gland with hollow lumen (class 0: background)
        H, W = 100, 100
        seg = np.zeros((H, W), dtype=np.int64)
        yy, xx = np.ogrid[:H, :W]
        dist_from_center = np.sqrt((yy - 50)**2 + (xx - 50)**2)

        # Epithelial ring between r=20 and r=35
        seg[(dist_from_center >= 20) & (dist_from_center <= 35)] = 2
        # Small core interior at r=25..30
        seg[(dist_from_center >= 25) & (dist_from_center <= 30)] = 1
        # Inside r < 20 is hollow white lumen (class 0)

        # Create optical white image (V=250, S=10)
        img_rgb = np.full((H, W, 3), 245, dtype=np.uint8)

        # With gland lumen filling enabled:
        inst_filled = class4_to_instances(seg, is_gland=True, fill_lumen=True, close_kernel=15, original_rgb=img_rgb)
        self.assertGreater(inst_filled[50, 50], 0, "Enclosed optical white lumen must be reclaimed as gland interior")

    def test_pink_stroma_is_strictly_rejected_not_filled(self):
        # Create a single gland enclosing a PINK stroma pocket (eosin stain: high saturation S=90, V=190)
        H, W = 100, 100
        seg = np.zeros((H, W), dtype=np.int64)
        yy, xx = np.ogrid[:H, :W]
        dist_from_center = np.sqrt((yy - 50)**2 + (xx - 50)**2)

        seg[(dist_from_center >= 20) & (dist_from_center <= 35)] = 2
        seg[(dist_from_center >= 25) & (dist_from_center <= 30)] = 1

        # Create pink stroma image (R=220, G=120, B=170) -> high saturation
        img_pink = np.zeros((H, W, 3), dtype=np.uint8)
        img_pink[:, :, 0] = 220 # R
        img_pink[:, :, 1] = 120 # G
        img_pink[:, :, 2] = 170 # B

        inst_rejected = class4_to_instances(seg, is_gland=True, fill_lumen=True, close_kernel=15, original_rgb=img_pink)
        # Center pink stroma must NOT be filled!
        self.assertEqual(inst_rejected[50, 50], 0, "Pink stroma must be strictly rejected by saturation gate!")

    def test_inter_gland_stroma_between_multiple_glands_not_filled(self):
        # Create TWO separate glands with background between them
        H, W = 100, 100
        seg = np.zeros((H, W), dtype=np.int64)
        yy, xx = np.ogrid[:H, :W]
        
        # Gland 1 on left
        d1 = np.sqrt((yy - 50)**2 + (xx - 30)**2)
        seg[d1 <= 15] = 1
        
        # Gland 2 on right
        d2 = np.sqrt((yy - 50)**2 + (xx - 70)**2)
        seg[d2 <= 15] = 1
        
        # Space between them at (50, 50) is inter-gland background
        inst = class4_to_instances(seg, is_gland=True, fill_lumen=True, close_kernel=5)
        self.assertEqual(inst[50, 50], 0, "Inter-gland background between separate glands must NOT be filled")


if __name__ == "__main__":
    unittest.main()

