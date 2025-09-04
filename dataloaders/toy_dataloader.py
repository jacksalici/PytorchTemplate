from typing import List, Set, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

SortDatasetSample = Tuple[torch.Tensor, torch.Tensor]


class SortDataset(Dataset):
    """
    Dataset for the Sort problem. E.g. for problem length 6:
    Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2

    The transformer receives concatenated input-output as:
    input:  0 0 2 1 0 1 0 0 0 1 1
    output: I I I I I 0 0 0 1 1 2
    where I is "ignore" (masked with -1 during training)

    Based on: https://github.com/karpathy/minGPT/blob/master/demo.ipynb
    """

    def __init__(
        self,
        samples: List[SortDatasetSample],
        length: int,
        num_digits: int,
    ) -> None:
        """ "
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

    def get_vocab_size(self) -> int:
        """Returns vocabulary size for the model."""
        return self.num_digits

    def get_block_size(self) -> int:
        """
        Returns sequence length for transformer input.
        Concatenated input + output - 1 (since transformer predicts next token).
        """
        return self.length * 2 - 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a training example.

        Returns:
            x: Input sequence (concatenated input + partial output)
            y: Target sequence (shifted by 1, with input positions masked as -1)
        """
        return self.samples[idx]


def _generate_example(
    length: int, num_digits: int, boost_repeats_prob: float = 0.4
) -> SortDatasetSample:
    """
    Generate a single input sequence for the sort task.

    Args:
        length: Length of input sequence to sort
        num_digits: Number of possible digit values (0 to num_digits-1)
        boost_repeats_prob: Probability to boost examples with many repeated digits

    Returns:
        Tuple of (x, y) where:
        - x: Input sequence (concatenated input + partial output)
        - y: Target sequence (shifted by 1, with input positions masked as -1)
    """
    # Generate random sequence
    inp = torch.randint(num_digits, size=(length,), dtype=torch.long)

    # Boost examples with many repeated digits
    # (harder for model, according to the original demo)
    if torch.rand(1).item() < boost_repeats_prob:
        unique_count = inp.unique().numel()
        if unique_count > length // 2:
            num_unique = torch.randint(1, length // 2 + 1, (1,)).item()
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
