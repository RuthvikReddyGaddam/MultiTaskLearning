# tests/test_j4_formulation.py

import unittest
import numpy as np
import torch

from datasets.multitask_collate import compute_4class_target
from utils.test_utils import j4_reconstruct_instances, assign_instance_classes_by_majority_vote


class TestJ4Formulation(unittest.TestCase):
    def test_compute_4class_target_touching_and_gap(self):
        # Create synthetic image with 2 touching circular cells
        # Instance 1: circle at (32, 22) with radius 10
        # Instance 2: circle at (32, 40) with radius 10 (touching at x around 31)
        inst = np.zeros((64, 64), dtype=np.int32)
        y, x = np.ogrid[:64, :64]
        c1 = (y - 32)**2 + (x - 22)**2 <= 10**2
        c2 = (y - 32)**2 + (x - 40)**2 <= 10**2
        inst[c1] = 1
        inst[c2] = 2

        t_inst = torch.from_numpy(inst).unsqueeze(0)  # [1, 64, 64]
        target_4class = compute_4class_target(t_inst, kernel_size=5).squeeze(0).numpy()

        # Class 0: Background
        # Class 1: Cell Interior
        # Class 2: Touching Region
        # Class 3: Gap (cleft between the two circles)
        self.assertTrue((target_4class == 0).any(), "Background missing")
        self.assertTrue((target_4class == 1).any(), "Cell interior missing")
        self.assertTrue((target_4class == 2).any(), "Touching class missing on adjacent cells")
        self.assertTrue((target_4class == 3).any(), "Gap class missing on touching cleft")

        # Isolated cell test: should have ZERO touching class
        isolated_inst = np.zeros((64, 64), dtype=np.int32)
        isolated_inst[c1] = 1
        t_iso = torch.from_numpy(isolated_inst).unsqueeze(0)
        target_iso = compute_4class_target(t_iso, kernel_size=5).squeeze(0).numpy()
        self.assertEqual((target_iso == 2).sum(), 0, "Isolated cell should have 0 touching pixels")

    def test_j4_reconstruct_instances(self):
        # 4-channel probability map: [4, 64, 64]
        # Simulating 2 cells:
        # Cell 1 interior at [20:30, 10:20]
        # Cell 2 interior at [20:30, 30:40]
        # Touching region at [20:30, 21:29]
        probs = np.zeros((4, 64, 64), dtype=np.float32)
        probs[0] = 0.9  # default background

        # Cell 1
        probs[1, 20:30, 10:21] = 0.95
        probs[0, 20:30, 10:21] = 0.05

        # Touching region between the two
        probs[2, 20:30, 21:29] = 0.92
        probs[0, 20:30, 21:29] = 0.08

        # Cell 2
        probs[1, 20:30, 29:40] = 0.95
        probs[0, 20:30, 29:40] = 0.05

        # Normalize
        probs = probs / probs.sum(axis=0, keepdims=True)

        recon_instances = j4_reconstruct_instances(probs, min_size=5)

        # There must be exactly 2 distinct cell instances
        unique_ids = np.unique(recon_instances)
        unique_cells = unique_ids[unique_ids > 0]
        self.assertEqual(len(unique_cells), 2, f"Expected 2 instances, found {len(unique_cells)}")

        # Touching region must have been assigned to cell 1 or cell 2
        touching_slice = recon_instances[20:30, 21:29]
        self.assertTrue(np.all(touching_slice > 0), "Touching pixels must be assigned to cells")
        self.assertTrue(1 in touching_slice and 2 in touching_slice, "Touching pixels should be partitioned between cell 1 and cell 2")

    def test_assign_instance_classes_majority_voting(self):
        # 2 detected instances
        inst = np.zeros((32, 32), dtype=np.int32)
        inst[5:15, 5:15] = 1   # instance 1
        inst[18:28, 18:28] = 2 # instance 2

        # 5-channel classification probability map:
        # [0: BG, 1: Epithelial, 2: Lymphocyte, 3: Neutrophil, 4: Macrophage]
        cls_prob = np.zeros((5, 32, 32), dtype=np.float32)
        cls_prob[0] = 0.9  # background everywhere

        # Instance 1: Epithelial (channel 1)
        cls_prob[1, 5:15, 5:15] = 0.8
        cls_prob[0, 5:15, 5:15] = 0.1

        # Instance 2: Lymphocyte (channel 2)
        cls_prob[2, 18:28, 18:28] = 0.85
        cls_prob[0, 18:28, 18:28] = 0.1

        pred_class_masks, colorized, inst_to_cls = assign_instance_classes_by_majority_vote(
            instances=inst,
            cls_prob=cls_prob,
            num_classes=4,
        )

        # Instance 1 -> Epithelial (class 0 in 0-indexed 4-class foreground)
        # Instance 2 -> Lymphocyte (class 1 in 0-indexed 4-class foreground)
        self.assertEqual(inst_to_cls[1], 0)
        self.assertEqual(inst_to_cls[2], 1)
        self.assertEqual(pred_class_masks[0].sum(), 100)
        self.assertEqual(pred_class_masks[1].sum(), 100)
        self.assertEqual(pred_class_masks[2].sum(), 0)  # No neutrophils
        self.assertEqual(pred_class_masks[3].sum(), 0)  # No macrophages


if __name__ == "__main__":
    unittest.main()

