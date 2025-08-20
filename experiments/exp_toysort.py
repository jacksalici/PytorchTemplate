import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from utils.logger import Logger
import torch.nn.functional as F
from argparse import ArgumentParser
from experiments.exp_base import BaseExperiment
from configs.config import Config

class CustomCrossEntropyLoss(nn.Module):
    """
    Custom cross entropy loss for transformer model with support for ignore_index.
    """
    def __init__(self, ):
        super().__init__()
    
    def forward(self, predictions, targets, ignore_index=-1):
              
        return F.cross_entropy(
            predictions, 
            targets, 
            ignore_index=ignore_index,
        )

class ToySortExperiment(BaseExperiment):
    """
    Experiment class for the toy sorting task.
    Handles sequence-to-sequence learning where the model learns to sort sequences.
    """
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        logger: Logger,
        conf: Config,
    ):
        super().__init__(model, criterion, optimizer, device, logger, conf)
        self.sequence_length = conf.get("sequence_length", 6)
        self.num_digits = conf.get("num_digits", 1)

    def train(self, train_loader: DataLoader) -> float:

        self.model.train()
        self.model.requires_grad_(True)
        total_loss = 0.0
        

        for batch_idx, (x, y) in enumerate(train_loader):
            print(f"Train Batch {batch_idx + 1}/{len(train_loader)}", end="\r")
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(x)
            
            loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
           
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> tuple[float, dict]:

        self.model.eval()
        total_correct = 0
        total_sequences = 0
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                print(f"Eval Batch {batch_idx + 1}/{len(val_loader)}", end="\r")

                x, y = x.to(self.device), y.to(self.device)

                inp = x[:, : self.sequence_length]
                target_sol = y[:, -self.sequence_length :]

                generated_full = self.model.generate(inp, max_length=self.sequence_length)
                generated_sol = generated_full[:, self.sequence_length:]
                
                
                correct = (generated_sol == target_sol).all(dim=-1)
                total_correct += correct.sum().item()
                total_sequences += generated_sol.size(0)
                
                        
        return 0, {
            "Accuracy": total_correct / total_sequences,
            "Correct Sequences": total_correct,
            "Total Sequences": total_sequences,
        }

    @staticmethod
    def inference(model: nn.Module, config: Config):
        input = torch.randint(0, model.num_digits, (1, 6), dtype=torch.long)
        model.eval()
        with torch.no_grad():
            
            generated = model.generate(input, max_length=6)
            
        print("Input Sequence:", input.tolist()[0])
        print("Generated Sequence:", generated.tolist()[0][6:])    
            