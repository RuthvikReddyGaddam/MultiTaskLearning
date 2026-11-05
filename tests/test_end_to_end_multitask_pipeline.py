# tests/test_end_to_end_multitask_pipeline.py

import unittest
import sys
from pathlib import Path
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import load_yaml_config
from scripts.build_models import build_model
from scripts.build_losses import build_loss
from datasets import build_multitask_dataloader
from utils.test_utils import extract_model_outputs, flow_to_instances, class3_to_instances, binary_to_instances, class4_to_instances


class TestEndToEndMultiTaskPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = PROJECT_ROOT / "configs" / "multitask_sdenetpointgradexp_512.yaml"
        cls.config = load_yaml_config(str(cls.config_path))
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_1_dataset_and_dataloader_generation(self):
        """Step 1: Check that DataLoader yields 512x512 batches with 4:2:2 equality and 4-class target."""
        glas_dir = Path(self.config["dataset"]["glas_dir"])
        if not glas_dir.exists():
            self.skipTest(f"Data directory does not exist: {glas_dir}")
        loader = build_multitask_dataloader(
            glas_dir=self.config["dataset"]["glas_dir"],
            monuseg_dir=self.config["dataset"]["monuseg_dir"],
            monusac_dir=self.config["dataset"]["monusac_dir"],
            patch_size=512,
            batch_size=8,
            num_workers=0,
            monusac_resample_size=600,
            augment=True,
        )
        self.assertGreater(len(loader), 0)

        batch = next(iter(loader))
        self.assertEqual(batch["image"].shape, (8, 3, 512, 512))
        self.assertEqual(batch["target_4class"].shape, (8, 512, 512))
        self.assertEqual(batch["target_3class"].shape, (8, 512, 512))
        self.assertEqual(batch["binary_mask"].shape, (8, 1, 512, 512))
        self.assertEqual(batch["class_map"].shape, (8, 512, 512))
        self.assertEqual(batch["class_mask"].shape, (8, 5, 512, 512))

        # Check exact 4:2:2 (2:1:1) task allocation
        task_counts = torch.bincount(batch["task_id"], minlength=3)
        self.assertEqual(task_counts[0].item(), 4, "GLaS count must be 4 (B//2)")
        self.assertEqual(task_counts[1].item(), 2, "MoNuSeg count must be 2 (B//4)")
        self.assertEqual(task_counts[2].item(), 2, "MoNuSAC count must be 2 (B//4)")

    def test_2_model_forward_backward_training_step(self):
        """Step 2: Check model forward pass, J-regularization loss, and parameter gradient update."""
        model = build_model(self.config).to(self.device)
        model.train()
        criterion = build_loss(self.config).to(self.device)

        # Create simulated batch (B=8: 4 GLaS, 2 MoNuSeg, 2 MoNuSAC)
        B, H, W = 8, 256, 256
        task_ids = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2], device=self.device)
        images = torch.randn(B, 3, H, W, device=self.device)
        target_4class = torch.randint(0, 4, (B, H, W), device=self.device)
        class_map = torch.randint(-1, 5, (B, H, W), device=self.device)

        batch = {
            "image": images,
            "target_4class": target_4class,
            "target_3class": target_4class,
            "binary_mask": (target_4class > 0).float().unsqueeze(1),
            "class_map": class_map,
            "class_mask": torch.randn(B, 5, H, W, device=self.device),
            "task_id": task_ids,
            "is_gland": (task_ids == 0),
            "is_monuseg": (task_ids == 1),
            "is_monusac": (task_ids == 2),
        }

        preds = model(images, task_ids=task_ids)
        loss, logs = criterion(preds, batch)
        self.assertFalse(torch.isnan(loss), "Training loss is NaN!")
        self.assertTrue(loss.requires_grad)
        self.assertIn("loss_glas_neg", logs, "loss_glas_neg must be logged")

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        loss.backward()

        # Check gradients on shared cell head and gland head
        shared_cell_head = model.prediction_head.cell_seg
        self.assertIsNotNone(shared_cell_head.weight.grad, "No grad on cell_seg!")
        glas_head = model.prediction_head.gland_seg
        self.assertIsNotNone(glas_head.weight.grad, "No grad on gland_seg!")
        optimizer.step()

    def test_3_inference_pipeline_and_instance_splitting(self):
        """Step 3: Test inference extraction for all 3 tasks and 4-class instance segmentation."""
        model = build_model(self.config).to(self.device)
        model.eval()

        dummy_patch = torch.randn(1, 3, 256, 256, device=self.device)

        with torch.no_grad():
            # 1. GLaS inference
            out_glas = model(dummy_patch, task="glas")
            seg_glas, cls_glas = extract_model_outputs(out_glas, task="glas")
            self.assertEqual(seg_glas.shape, (1, 4, 256, 256))
            self.assertIsNone(cls_glas)

            # 2. MoNuSeg inference
            out_monuseg = model(dummy_patch, task="monuseg")
            seg_monuseg, cls_monuseg = extract_model_outputs(out_monuseg, task="monuseg")
            self.assertEqual(seg_monuseg.shape, (1, 4, 256, 256))
            self.assertIsNone(cls_monuseg)

            # 3. MoNuSAC inference
            out_monusac = model(dummy_patch, task="monusac")
            seg_monusac, cls_monusac = extract_model_outputs(out_monusac, task="monusac")
            self.assertEqual(seg_monusac.shape, (1, 4, 256, 256))
            self.assertEqual(cls_monusac.shape, (1, 5, 256, 256))

        # 4. Test 4-class instance splitting on touching synthetic cells
        H, W = 100, 100
        y, x = np.mgrid[:H, :W]
        # Circle 1 at (40, 40) r=14
        d1 = np.sqrt((y - 40)**2 + (x - 40)**2)
        # Circle 2 at (40, 65) r=14 (touching between x=50 and x=55)
        d2 = np.sqrt((y - 40)**2 + (x - 65)**2)

        pred_4class = np.zeros((H, W), dtype=np.int64)
        # Assign boundary (class 2) to outer shells
        pred_4class[d1 <= 14] = 2
        pred_4class[d2 <= 14] = 2
        # Assign interior (class 1) to inner cores (r=10)
        pred_4class[d1 <= 10] = 1
        pred_4class[d2 <= 10] = 1
        # Assign gap (class 3) to the cleft between them (where they touch)
        pred_4class[(d1 <= 15) & (d2 <= 15)] = 3

        # Check standard binary segmentation merges them
        binary_mask = ((d1 <= 14) | (d2 <= 14)).astype(np.uint8)
        merged = binary_to_instances(binary_mask)
        self.assertEqual(merged.max(), 1, "Connected components on binary mask merges touching cells into 1!")

        # 4-class instance extraction splits them cleanly
        instances = class4_to_instances(pred_4class, min_size=10, expand_boundary=True)
        self.assertEqual(instances.max(), 2, "4-class segmentation must split touching cells into 2 instances!")
        self.assertTrue(np.all((instances > 0) == ((pred_4class == 1) | (pred_4class == 2))), "Interior and boundary pixels must be retained as instances, gap preserved as barrier.")


if __name__ == "__main__":
    unittest.main()
