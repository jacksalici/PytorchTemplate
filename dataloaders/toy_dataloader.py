from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

SortDatasetSample = Tuple[torch.Tensor, torch.Tensor]


def is_in_list(query: SortDatasetSample, sample_list: List[SortDatasetSample]) -> bool:
    """
    Check if a dataset sample is contained in a list of samples.

    WARNING: this function is not optimized for performance,
        but is sufficient for utility purposes within this toy dataset.

    Args:
        query: Sample to check
        sample_list: List of samples to check against

    Returns:
        True if the query sample is in sample_list, False otherwise
    """
    for s in sample_list:
        # to tell if two samples are equal, we can just check if their inputs
        # are equal, since outputs are deterministic given inputs
        if torch.equal(query[0], s[0]):
            return True
    return False


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

    BOOST_REPEATS_PROB = 0.4  # Probability to boost examples with repeats

    def __init__(
        self,
        length: int = 6,
        num_digits: int = 3,
        size: int = 10000,
        avoid_samples: List[Tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> None:
        """ "
        Args:
            length: Length of input sequence to sort
            num_digits: Number of possible digit values (0 to num_digits-1)
            size: Dataset size (for consistent iteration)
            avoid_samples: List of samples to avoid;
                - useful to avoid overlap between train and test sets; to
                  do so, init the train set first with this argument as None,
                  then init the test set with this argument as the list of
                  samples from the train set.
        """
        self.length = length
        self.num_digits = num_digits
        self.size = size

        if avoid_samples is None:
            avoid_samples = []

        if size + len(avoid_samples) > num_digits**length:
            raise ValueError(
                f"Cannot create dataset of size {size} with {len(avoid_samples)} "
                f"avoided samples, given num_digits={num_digits} and length={length}. "
                f"You can have at most {num_digits**length} unique sequences."
            )

        self.data: List[SortDatasetSample] = []
        while len(self.data) < size:
            s = self._generate_example()
            # avoid duplicates and overlap with avoid_samples
            if not is_in_list(s, self.data) and not is_in_list(s, avoid_samples):
                self.data.append(s)

    def __len__(self) -> int:
        return self.size

    def get_vocab_size(self) -> int:
        """Returns vocabulary size for the model."""
        return self.num_digits

    def get_block_size(self) -> int:
        """
        Returns sequence length for transformer input.
        Concatenated input + output - 1 (since transformer predicts next token).
        """
        return self.length * 2 - 1

    def _generate_example(self) -> SortDatasetSample:
        """Generate a single input sequence."""
        # Generate random sequence
        inp = torch.randint(self.num_digits, size=(self.length,), dtype=torch.long)

        # Boost examples with many repeated digits (harder for model, according to the original demo)
        if torch.rand(1).item() < self.BOOST_REPEATS_PROB:
            unique_count = inp.unique().numel()
            if unique_count > self.length // 2:
                num_unique = torch.randint(1, self.length // 2 + 1, (1,)).item()
                unique_vals = torch.randperm(self.num_digits)[:num_unique]
                inp = unique_vals[torch.randint(num_unique, (self.length,))]

        sol = torch.sort(inp)[0]
        sequence = torch.cat([inp, sol])

        x = sequence[:-1].clone()
        y = sequence[1:].clone()
        y[: self.length - 1] = -1

        return x, y

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a training example.

        Returns:
            x: Input sequence (concatenated input + partial output)
            y: Target sequence (shifted by 1, with input positions masked as -1)
        """
        return self.data[idx]


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    length: int = 6,
    num_digits: int = 5,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for the sort task.

    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        length: Length of sequences to sort
        num_digits: Number of possible digit values

    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_dataset = SortDataset(length, num_digits, 64**2)
    test_dataset = SortDataset(length, num_digits, 64 * 10)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, test_loader
