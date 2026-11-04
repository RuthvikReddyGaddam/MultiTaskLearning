# datasets/glas_dataset.py

from pathlib import Path
from typing import Optional, Union, Any, List
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.utils import (
    load_patch_paths,
    apply_patch_transforms,
    PatchSample,
)
from datasets.multitask_collate import compute_4class_target


class GlasDataset(Dataset):
    """
    Dataset for GLaS train patches.
    Yields:
      - "image":         [3, H, W] float32 image patch in [0, 1]
      - "mask":          [1, H, W] float32 binary gland mask (0.0 or 1.0)
      - "binary_mask":   [1, H, W] float32 binary gland mask (alias)
      - "instance_mask": [1, H, W] int64 individual gland instance identifiers
      - "sdf":           [1, H, W] float32 signed distance function patch
      - "flow_vector":   [2, H, W] float32 normalized 2D flow vector (Fx, Fy) in [-1.0, 1.0]
    """
    def __init__(
        self,
        data_dir: Union[str, Path],
        transform: Optional[Any] = None,
        target_size: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_size = target_size
        if self.target_size is None:
            if "512" in self.data_dir.name:
                self.target_size = 512
            elif "256" in self.data_dir.name:
                self.target_size = 256
        self.samples: List[tuple] = load_patch_paths(self.data_dir)
        if len(self.samples) == 0:
            raise RuntimeError(f"No patch files found in {self.data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

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
            inst_mask = np.load(mask_path)
            target_4class = compute_4class_target(torch.from_numpy(inst_mask).unsqueeze(0)).squeeze(0)

        # 3. Load precomputed binary mask [1, H, W] float32
        bin_path = self.data_dir / "binary_masks" / img_path.name
        if bin_path.exists():
            binary_mask = torch.from_numpy(np.load(bin_path)).float().unsqueeze(0)
        else:
            binary_mask = (target_4class > 0).float().unsqueeze(0)

        # 4. Standard semantic class map (-1 for GLaS)
        class_map = torch.full((H, W), -1, dtype=torch.long)

        # 5. Optional padding if target_size is set
        if self.target_size is not None and (H < self.target_size or W < self.target_size):
            pad_bottom = max(0, self.target_size - H)
            pad_right = max(0, self.target_size - W)
            pad_tuple = (0, pad_right, 0, pad_bottom)
            img_t = torch.nn.functional.pad(img_t, pad_tuple, value=1.0)
            target_4class = torch.nn.functional.pad(target_4class.unsqueeze(0), pad_tuple, value=0).squeeze(0)
            binary_mask = torch.nn.functional.pad(binary_mask, pad_tuple, value=0.0)
            class_map = torch.nn.functional.pad(class_map.unsqueeze(0), pad_tuple, value=-1).squeeze(0)

        sample = PatchSample(
            image=img_t,
            target_4class=target_4class,
            target_3class=target_4class,
            binary_mask=binary_mask,
            mask=binary_mask,
            class_map=class_map,
            task_id=0,
            patch_name=img_path.stem,
        )
        return sample
