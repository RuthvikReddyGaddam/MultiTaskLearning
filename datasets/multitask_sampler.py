# datasets/multitask_sampler.py

import math
from typing import List, Iterator, Optional
import torch
from torch.utils.data import Sampler


class EqualTaskBatchSampler(Sampler[List[int]]):
    """
    BatchSampler for multi-task learning across 3 datasets (GLaS, MoNuSeg, MoNuSAC).
    
    Guarantees strict 2:1:1 task sampling per batch:
      - Exactly half (batch_size // 2) samples from Dataset 0 (GLaS)
      - Exactly one-quarter (batch_size // 4) samples from Dataset 1 (MoNuSeg)
      - Exactly one-quarter (batch_size // 4) samples from Dataset 2 (MoNuSAC)

    Batch size must be a multiple of 8 (e.g. 8, 16, 24, 32).
    
    Handles dataset size disparities via cyclic independent shuffling:
    When a smaller dataset exhausts its patches, its index pool is reshuffled
    with a new random permutation and wrapped around, preventing starvation of
    larger datasets without repetitive correlation patterns.
    """
    def __init__(
        self,
        dataset_lengths: List[int],
        batch_size: int = 8,
        shuffle: bool = True,
        drop_last: bool = False,
        steps_per_epoch: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        shuffle_within_batch: Optional[bool] = None,
    ):
        if len(dataset_lengths) != 3:
            raise ValueError(f"EqualTaskBatchSampler requires exactly 3 dataset lengths, got {len(dataset_lengths)}")
        for i, l in enumerate(dataset_lengths):
            if l <= 0:
                raise ValueError(f"Dataset length for dataset {i} must be > 0, got {l}")

        if batch_size % 4 != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be a multiple of 4 (e.g. 4, 8, 16) to ensure "
                f"2:1:1 task sampling (half GLaS, quarter MoNuSeg, quarter MoNuSAC)."
            )

        self.dataset_lengths = dataset_lengths
        self.batch_size = batch_size
        self.k_glas = batch_size // 2
        self.k_monuseg = batch_size // 4
        self.k_monusac = batch_size // 4
        self.task_k = [self.k_glas, self.k_monuseg, self.k_monusac]
        self.k = self.k_glas
        self.shuffle = shuffle
        self.shuffle_within_batch = shuffle if shuffle_within_batch is None else shuffle_within_batch
        self.drop_last = drop_last
        self.generator = generator

        # Cumulative offsets to map local dataset index to global concatenated index
        # Dataset 0 (GLaS):    [0, N0 - 1]
        # Dataset 1 (MoNuSeg): [N0, N0 + N1 - 1]
        # Dataset 2 (MoNuSAC): [N0 + N1, N0 + N1 + N2 - 1]
        self.offsets = [
            0,
            dataset_lengths[0],
            dataset_lengths[0] + dataset_lengths[1],
        ]
        self.total_size = sum(dataset_lengths)

        # Epoch length calculation
        if steps_per_epoch is not None:
            self.num_batches = steps_per_epoch
        else:
            max_len = max(dataset_lengths)
            eff_k = min(self.task_k)
            if drop_last:
                self.num_batches = max_len // eff_k
            else:
                self.num_batches = math.ceil(max_len / eff_k)

    def _get_permutation(self, length: int) -> torch.Tensor:
        if self.shuffle:
            return torch.randperm(length, generator=self.generator)
        return torch.arange(length)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        # Initialize index pools and pointers for each dataset
        pools = [self._get_permutation(l).tolist() for l in self.dataset_lengths]
        ptrs = [0, 0, 0]

        for _ in range(self.num_batches):
            batch = []
            for d_idx in range(3):
                d_len = self.dataset_lengths[d_idx]
                offset = self.offsets[d_idx]
                k_task = self.task_k[d_idx]
                d_samples = []

                while len(d_samples) < k_task:
                    remaining_needed = k_task - len(d_samples)
                    available = d_len - ptrs[d_idx]

                    if available >= remaining_needed:
                        # Draw from current pool
                        slice_end = ptrs[d_idx] + remaining_needed
                        d_samples.extend(pools[d_idx][ptrs[d_idx]:slice_end])
                        ptrs[d_idx] = slice_end
                        if ptrs[d_idx] == d_len:
                            # Reshuffle on pool exhaustion
                            pools[d_idx] = self._get_permutation(d_len).tolist()
                            ptrs[d_idx] = 0
                    else:
                        # Take all remaining, then reshuffle and take the rest
                        d_samples.extend(pools[d_idx][ptrs[d_idx]:])
                        pools[d_idx] = self._get_permutation(d_len).tolist()
                        ptrs[d_idx] = 0

                # Convert local indices to global concatenated indices
                batch.extend([idx + offset for idx in d_samples])

            if self.shuffle_within_batch:
                perm = torch.randperm(len(batch), generator=self.generator).tolist()
                batch = [batch[p] for p in perm]

            yield batch

