import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from typing import Dict, List, Optional
import torch.nn.functional as F


class SortDataset(Dataset):
    """ 
    # from https://github.com/karpathy/minGPT/blob/master/demo.ipynb
    Dataset for the Sort problem. E.g. for problem length 6:
    Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2
    Which will feed into the transformer concatenated as:
    input:  0 0 2 1 0 1 0 0 0 1 1
    output: I I I I I 0 0 0 1 1 2
    where I is "ignore", as the transformer is reading the input sequence
    """

    def __init__(self, split, length=6, num_digits=3):
        assert split in {'train', 'test'}
        self.split = split
        self.length = length
        self.num_digits = num_digits
    
    def __len__(self):
        return 10000 # This should be large enough to avoid repetition issues
    
    def get_vocab_size(self):
        return self.num_digits
    
    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.length * 2 - 1

    def __getitem__(self, idx):
        # use rejection sampling to generate an input example from the desired split
        while True:
            # generate some random integers
            inp = torch.randint(self.num_digits, size=(self.length,), dtype=torch.long)
            # half of the time let's try to boost the number of examples that 
            # have a large number of repeats, as this is what the model seems to struggle
            # with later in training, and they are kind of rare
            if torch.rand(1).item() < 0.5:
                if inp.unique().nelement() > self.length // 2:
                    # too many unique digits, re-sample
                    continue
            # figure out if this generated example is train or test based on its hash
            h = hash(pickle.dumps(inp.tolist()))
            inp_split = 'test' if h % 4 == 0 else 'train' # designate 25% of examples as test
            if inp_split == self.split:
                break # ok
        
        # solve the task: i.e. sort
        sol = torch.sort(inp)[0]

        # concatenate the problem specification and the solution
        cat = torch.cat((inp, sol), dim=0)

        # the inputs to the transformer will be the offset sequence
        x = cat[:-1].clone()
        y = cat[1:].clone()
        # we only want to predict at output locations, mask out the loss at the input locations
        y[:self.length-1] = -1
        return x, y


def get_dataloaders(batch_size=32, num_workers=0, pin_memory=True, length=6, num_digits=3) -> tuple[DataLoader, DataLoader]:
    train_dataset = SortDataset(split='train', length=length, num_digits=num_digits)
    test_dataset = SortDataset(split='test', length=length, num_digits=num_digits)
    
    # Use the same approach as the working demo notebook
    # The key is using replacement=True for training with a large number of samples
    train_loader = DataLoader(
        train_dataset,
        sampler=torch.utils.data.RandomSampler(
            train_dataset, 
            replacement=True, 
            num_samples=int(2000)  # Very large number like in demo
        ),
        shuffle=False,
        pin_memory=pin_memory,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    # For test, use a reasonable finite number
    test_loader = DataLoader(
        test_dataset,
        sampler=torch.utils.data.RandomSampler(
            test_dataset, 
            replacement=False, 
            num_samples=int(1000)  # Increased from 1000
        ),
        shuffle=False,
        pin_memory=pin_memory,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    return train_loader, test_loader


def test_dataloader():
    """Test function to verify the dataloader works correctly"""
    print("Testing SortDataset and DataLoader...")
    
    # Test dataset directly
    train_dataset = SortDataset('train', length=6, num_digits=3)
    test_dataset = SortDataset('test', length=6, num_digits=3)
    
    print(f"Train dataset vocab size: {train_dataset.get_vocab_size()}")
    print(f"Train dataset block size: {train_dataset.get_block_size()}")
    
    # Test a few samples
    for i in range(3):
        x, y = train_dataset[i]
        print(f"Sample {i}: x={x.tolist()}, y={y.tolist()}")
        
        # Verify the structure
        assert len(x) == train_dataset.get_block_size()
        assert len(y) == train_dataset.get_block_size()
        assert (y[:train_dataset.length-1] == -1).all(), "Input positions should be masked"
    
    # Test dataloader
    train_loader, test_loader = get_dataloaders(batch_size=4, length=6, num_digits=3)
    
    # Get one batch
    batch_x, batch_y = next(iter(train_loader))
    print(f"Batch shape - x: {batch_x.shape}, y: {batch_y.shape}")
    print(f"First sample in batch - x: {batch_x[0].tolist()}, y: {batch_y[0].tolist()}")
    
    print("Dataloader test passed!")


if __name__ == "__main__":
    test_dataloader()