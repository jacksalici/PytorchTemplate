"""
This module provides a synthetic dataset and dataloader utilities for
training causal (autoregressive) transformer models on the "Sort" problem.

Based on: https://github.com/karpathy/minGPT/blob/master/demo.ipynb
"""

from typing import List, Set, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

SortDatasetSample = Tuple[torch.Tensor, torch.Tensor]


class SortDataset(Dataset):
    """
    Dataset for the "Sort" problem.

    This is designed for the training of a causal (autoregressive) transformer,
    whose goal is to take an input sequence of digits and autoregressively
    generate its sorted version.

    Example (when length = 6):
        - Input sequence:     [0, 0, 2, 1, 0, 1]
        - Sorted sequence:    [0, 0, 0, 1, 1, 2]

        - Model input (x):    [0, 0, 2, 1, 0, 1, 0, 0, 0, 1, 1]
            - the model input consists of the input sequence followed by
              the sorted sequence (excluding the last element)

        - Target output (y):  [-1, -1, -1, -1, -1, 0, 0, 0, 1, 1, 2]
            - the target output consists of the input sequence (excluding the
              first element) followed by the sorted sequence
            - note that positions corresponding to the input sequence are masked
              with -1 and do not contribute to the loss during training
    """

    def __init__(
        self,
        samples: List[SortDatasetSample],
        length: int,
        num_digits: int,
    ) -> None:
        """
        Args:
            length: Length of input sequence to sort
            num_digits: Number of possible digit values (0 to num_digits-1)
            samples: List of samples to include in the dataset
        """
        self.length = length
        self.num_digits = num_digits
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def get_vocab_size(self) -> int:
        """
        Returns:
            vocabulary size for the model.
        """
        return self.num_digits

    @property
    def get_block_size(self) -> int:
        """
        Returns:
            Length of the input sequence for the model, which is the
            concatenation of the input sequence and the sorted sequence
            (excluding the last element, hence the -1).
        """
        return self.length * 2 - 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a training example.

        Returns:
            x: Model input sequence
            y: Target output sequence
        """
        return self.samples[idx]


def _generate_example(
    length: int, num_digits: int, boost_repeats_prob: float = 0.4
) -> SortDatasetSample:
    """
    Randomly generates a single example for the sort task.

    Args:
        length: Length of input sequence to sort
        num_digits: Number of possible digit values (0 to num_digits-1)
        boost_repeats_prob: Probability to boost examples with repeated digits

    Returns:
        Tuple of (x, y) where:
            - x: Model input sequence
            - y: Target output sequence
    """
    # Generate random sequence
    inp = torch.randint(num_digits, size=(length,), dtype=torch.long)

    # Boost examples with many repeated digits
    # (harder for model, according to the original demo)
    if torch.rand(1).item() < boost_repeats_prob:
        unique_count = inp.unique().numel()
        if unique_count > length // 2:
            num_unique = int(torch.randint(1, length // 2 + 1, (1,)).item())
            unique_vals = torch.randperm(num_digits)[:num_unique]
            inp = unique_vals[torch.randint(num_unique, (length,))]

    sol = torch.sort(inp)[0]
    sequence = torch.cat([inp, sol])

    x = sequence[:-1].clone()
    y = sequence[1:].clone()
    y[: length - 1] = -1

    return x, y


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    length: int = 6,
    num_digits: int = 5,
    train_len: int = 12000,
    test_len: int = 640,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for the sort task.

    Both training and testing datasets are randomly generated, ensuring
    that all sequences are unique within and across the sets.

    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        length: Length of sequences to sort
        num_digits: Number of possible digit values
        train_len: Number of training samples
        test_len: Number of testing samples

    Returns:
        Tuple of (train_loader, test_loader)
    """

    if train_len + test_len > num_digits**length:
        raise ValueError(
            f"Cannot create dataset of size {train_len} + {test_len} = "
            f"{train_len + test_len}, given num_digits={num_digits} and "
            f"length={length}. You can have at most {num_digits**length} "
            f"unique sequences."
        )

    # generate train and test samples, ensuring uniqueness
    _support_set: Set[bytes] = set([])
    all_samples: List[SortDatasetSample] = []
    while len(all_samples) < train_len + test_len:
        s = _generate_example(length, num_digits)
        hashable_s = s[0].numpy().tobytes()
        if hashable_s not in _support_set:
            _support_set.add(hashable_s)
            all_samples.append(s)

    # split into train and test
    train_samples = all_samples[:train_len]
    test_samples = all_samples[train_len:]

    train_ds = SortDataset(samples=train_samples, length=length, num_digits=num_digits)
    test_ds = SortDataset(samples=test_samples, length=length, num_digits=num_digits)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, test_loader
