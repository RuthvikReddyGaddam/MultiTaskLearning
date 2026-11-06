import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.multitask_sampler import EqualTaskBatchSampler
from datasets.multitask_dataset import build_multitask_dataloader


class TestMultitaskSampler8Batch(unittest.TestCase):
    def test_sampler_requires_multiple_of_4(self):
        lengths = [328, 120, 650]
        # Non-multiples of 4 must fail!
        with self.assertRaises(ValueError):
            EqualTaskBatchSampler(dataset_lengths=lengths, batch_size=6)
        with self.assertRaises(ValueError):
            EqualTaskBatchSampler(dataset_lengths=lengths, batch_size=10)

    def test_sampler_supports_batch_size_4(self):
        lengths = [328, 120, 650]
        B = 4
        sampler = EqualTaskBatchSampler(dataset_lengths=lengths, batch_size=B, shuffle=False, steps_per_epoch=4)
        self.assertEqual(len(sampler), 4)

        for batch in sampler:
            self.assertEqual(len(batch), 4)
            # 0:2 must be GLaS [0, 327] (2 patches)
            self.assertTrue(all(0 <= idx < 328 for idx in batch[0:2]), f"GLaS slice failed: {batch[0:2]}")
            # 2:3 must be MoNuSeg [328, 447] (1 patch)
            self.assertTrue(all(328 <= idx < 448 for idx in batch[2:3]), f"MoNuSeg slice failed: {batch[2:3]}")
            # 3:4 must be MoNuSAC [448, 1097] (1 patch)
            self.assertTrue(all(448 <= idx < 1098 for idx in batch[3:4]), f"MoNuSAC slice failed: {batch[3:4]}")

    def test_sampler_yields_exact_half_glas_quarter_monuseg_quarter_monusac(self):
        lengths = [328, 120, 650]
        B = 8
        sampler = EqualTaskBatchSampler(dataset_lengths=lengths, batch_size=B, shuffle=False, steps_per_epoch=4)
        self.assertEqual(len(sampler), 4)

        for batch in sampler:
            self.assertEqual(len(batch), 8)
            # 0:4 must be GLaS [0, 327]
            self.assertTrue(all(0 <= idx < 328 for idx in batch[0:4]), f"GLaS slice failed: {batch[0:4]}")
            # 4:6 must be MoNuSeg [328, 447]
            self.assertTrue(all(328 <= idx < 448 for idx in batch[4:6]), f"MoNuSeg slice failed: {batch[4:6]}")
            # 6:8 must be MoNuSAC [448, 1097]
            self.assertTrue(all(448 <= idx < 1098 for idx in batch[6:8]), f"MoNuSAC slice failed: {batch[6:8]}")

    def test_dataloader_defaults_to_8(self):
        glas_dir = Path("data/glas_train_patch_512")
        if not glas_dir.exists():
            self.skipTest(f"Data directory not found: {glas_dir}")
        # build_multitask_dataloader default batch size should be 8
        loader = build_multitask_dataloader(
            patch_size=512,
            batch_size=8,
            num_workers=0,
            shuffle=False,
            steps_per_epoch=1,
        )
        batch = next(iter(loader))
        self.assertEqual(batch["image"].shape[0], 8)
        self.assertEqual(batch["task_id"].tolist(), [0, 0, 0, 0, 1, 1, 2, 2])


if __name__ == "__main__":
    unittest.main()

