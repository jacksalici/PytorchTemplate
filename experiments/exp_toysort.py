"""
This module defines the ToySortExperiment class and its custom loss function.
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from configs.config import Config
from experiments.exp_base import BaseExperiment
from utils.logger import Logger


class CustomCrossEntropyLoss(nn.Module):
    """
    Cross entropy loss for transformer model with support for ignore_index.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1
    ) -> torch.Tensor:
        """
        Computes the cross-entropy loss.

        Args:
            predictions: Predicted logits from the model
            targets: Ground truth target indices
            ignore_index: Specifies a target value that is ignored
                and does not contribute to the input gradient.
        """
        return torch.nn.functional.cross_entropy(
            predictions,
            targets,
            ignore_index=ignore_index,
        )


class ToySortExperiment(BaseExperiment):
    """
    Experiment class for the toy sorting task.

    Handles sequence-to-sequence learning where the model learns to sort
    digit sequences in ascending order.
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
        """
        Train the model for one epoch using the provided DataLoader.

        Args:
            train_loader: DataLoader for the training set.

        Returns:
            Average batch training loss.
        """
        self.model.train()
        self.model.requires_grad_(True)
        total_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            print(f"Train Batch {batch_idx + 1}/{len(train_loader)}", end="\r")
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(x)

            loss = self.criterion(
                logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader) -> tuple[float, dict]:
        """
        Validate the model using the provided DataLoader.

        Args:
            val_loader: DataLoader for the validation set.

        Returns:
            A tuple containing a placeholder value and a dictionary containing
            validation metrics (accuracy, number of correctly predicted
            sequences, and total number of validated sequences).
        """
        self.model.eval()
        total_correct = 0
        total_sequences = 0

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                print(f"Eval Batch {batch_idx + 1}/{len(val_loader)}", end="\r")

                x, y = x.to(self.device), y.to(self.device)

                inp = x[:, : self.sequence_length]
                target_sol = y[:, -self.sequence_length :]

                generated_full = self.model.generate(
                    inp, max_length=self.sequence_length
                )
                generated_sol = generated_full[:, self.sequence_length :]

                correct = (generated_sol == target_sol).all(dim=-1)
                total_correct += correct.sum().item()
                total_sequences += generated_sol.size(0)

        return 0, {
            "Accuracy": total_correct / total_sequences,
            "Correct Sequences": total_correct,
            "Total Sequences": total_sequences,
        }

    @staticmethod
    def inference(model: nn.Module, config: Config, logger: Logger | None = None):
        """
        Run inference on a random input sequence and log the results.

        Args:
            model: The model to use for inference.
            config: Configuration object.
            logger: Logger instance for logging results.
                If None, prints to console.
        """
        input = torch.randint(0, model.num_digits, (1, 6), dtype=torch.long)
        model.eval()
        with torch.no_grad():
            generated = model.generate(input, max_length=6)

        if logger is None:
            logger = print  # very pythonic :)

        logger("Input Sequence:", input.tolist()[0])
        logger("Generated Sequence:", generated.tolist()[0][6:])
