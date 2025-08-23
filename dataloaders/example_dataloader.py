import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class TestDataset(Dataset):
    def __init__(self, **kwargs):
        """
        Args:
            ...
        """
        pass
    
    def __len__(self) -> int:
        pass
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a training example.
        
        Returns:
            x: Input sequence (concatenated input + partial output)
            y: Target sequence (shifted by 1, with input positions masked as -1)
        """
        pass

def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for the sort task.
    
    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_dataset = TestDataset(**kwargs)
    test_dataset = TestDataset(**kwargs)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True 
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, test_loader


