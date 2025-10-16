

from models.transformer import Transformer
from experiments.exp_toysort import ToySortExperiment, CustomCrossEntropyLoss

import torch
import torch.nn as nn
import torch.optim as optim
from utils.logger import Logger
from utils.reproducibility import set_seeds
from configs.config import Config

from pathlib import Path


models = {
    "transformer": {"model": Transformer},
}

experiments = {
    "toysort": {"experiment": ToySortExperiment, "criterion": CustomCrossEntropyLoss},
}

optimizers = {
   "sgd": optim.SGD,
   "adam": optim.Adam,
   "adamw": optim.AdamW,
}

def main():
    config = Config.from_args()
    
    device = config.get_device()
    
    set_seeds(config.seed, force=config.force_reproducibility)
        
    logger = Logger(
        project_name = config.project_name,
        avoid_wandb = config.avoid_wandb)
    
    
    if config.task == "training":
    
        logger.print_config(config.to_dict())
        
        # Load data
        if config.dataloader == "example_ds":
            import dataloaders.example_dataloader as get_dataloaders
            raise NotImplementedError("Please implement the example dataloader.")
        elif config.dataloader == "toysort_ds":
            from dataloaders.toy_dataloader import get_dataloaders
            train_loader, test_loader = get_dataloaders(
                batch_size=config.batch_size, 
                num_workers=4,
                pin_memory=(device == "cuda"),  # Only pin memory for CUDA
                length=config.seq_len,
                num_digits=config.num_digits,
            )
        else:
            raise ValueError(f"Unknown dataloader: {config.dataloader}")
        
        if config.experiment in experiments:
            criterion = experiments[config.experiment]["criterion"]()
        else:
            raise ValueError(f"Unknown experiment: {config.experiment}")
        
        if config.model_name in models:
            model = models[config.model_name]["model"](**config.to_dict()).to(device)
        else:
            raise ValueError(f"Unknown model: {config.model_name}")
        
        logger(f"Model: {config.model_name}, Parameters: {model.n_param:,}")
        
        if config.optim in optimizers:
            optimizer = optimizers[config.optim](model.parameters(), lr=config.lr)
        else:
            raise ValueError(f"Unknown optimizer: {config.optim}")
        
        if config.experiment in experiments:
            experiment = experiments[config.experiment]["experiment"](model, criterion, optimizer, device, logger, config)
            experiment.run(train_loader, test_loader, num_epochs=config.num_epochs)
        else:
            raise ValueError(f"experiment {config.experiment} not available - please implement the experiment classes")

    elif config.task == "inference":
        logger(f"Running inference for {config.experiment} on {config.model_name} model...")
        
        # Try to load as new checkpoint format first, then fall back to old format
        checkpoint_path = config.get_checkpoint_path()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if config.model_name in models:
            model = models[config.model_name]["model"](**config.to_dict()).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger(f"Loaded model state from new checkpoint format: {checkpoint_path}")

        if config.experiment in experiments:
            exp = experiments[config.experiment]["experiment"]
            exp.inference(model, config)
        else:
            raise ValueError(f"Unknown experiment: {config.experiment}")

    else:
        raise ValueError(f"Unknown task: {config.task}")
        
        
if __name__ == "__main__":
    main()
