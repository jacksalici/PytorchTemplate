"""
Module containing the BaseExperiment class for managing AI/ML experiments.
"""

from abc import ABC, abstractmethod

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from configs.config import Config
from utils.logger import Logger


class BaseExperiment(ABC):
    """
    Abstract base class that defines the structure of a basic ML/AI experiment.

    This class provides a template for training, validation, and inference
    procedures while also handling logging and model checkpointing.

    Subclasses are expected to implement the `train`, `validate`, and
    `inference` methods according to their specific use case.
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
        """
        Initializes the BaseExperiment class with the provided model, criterion,
        optimizer, device, logger, and arguments.

        Args:
            model: The model to be used in the experiment.
            criterion: The loss function to be used.
            optimizer: The optimizer to be used for training.
            device: The device (CPU or GPU) on which the model will be trained.
            logger: Logger instance for logging experiment results.
            conf: Configuration class
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.conf = conf
        self.min_metric_model = float("inf")  # minimum metric value for model saving

    @abstractmethod
    def train(self, train_loader: DataLoader) -> float:
        """
        Train the model for one epoch using the provided DataLoader.

        Args:
            train_loader: DataLoader for the training set.

        Returns:
            Average batch training loss.
        """
        pass

    @abstractmethod
    def validate(self, val_loader: DataLoader) -> tuple[float, dict]:
        """
        Validate the model using the provided DataLoader.

        Args:
            val_loader: DataLoader for the testing dataset.

        Returns:
            A tuple containing the average test loss and a Metrics object.
        """
        pass

    def run(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        model_saving_metric: str = "Test Loss",
    ):
        """
        Runs the training and testing loop for the specified number of epochs.

        Args:
            train_loader: DataLoader for the training dataset.
            val_loader: DataLoader for the testing dataset.
            num_epochs: Number of epochs to run the training and testing loop.
            model_saving_metric: The metric used to determine if the model
                should be saved
        """

        for epoch in range(num_epochs):
            train_loss = self.train(train_loader)
            test_loss, metric = self.validate(val_loader)

            log_dict = {
                "Epoch": epoch,
                "Train Loss": train_loss,
                "Test Loss": test_loss,
                **metric,
            }
            self.logger(log_dict)

            self._save_model(epoch, log_dict, model_saving_metric)

    def _save_model(self, epoch: int, log_dict: dict, model_saving_metric: str):
        """
        Saves the model state to a file.

        Args:
            epoch: The current epoch number.
            log_dict: Dictionary containing the metrics for the current epoch.
            model_saving_metric: The metric used to determine if the model
                should be saved
        """
        if model_saving_metric not in log_dict:
            raise Warning(
                f"Model saving metric '{model_saving_metric}' not found in log_dict"
            )
        metric_value = log_dict[model_saving_metric]

        if metric_value <= self.min_metric_model:
            self.min_metric_model = metric_value
            if self.conf.save_model:
                self.logger(
                    f"Saving model at epoch {epoch} with "
                    f"{model_saving_metric} {metric_value}"
                )
                torch.save(self.model, self.conf.get_checkpoint_path())

    @staticmethod
    @abstractmethod
    def inference(model: nn.Module, config: Config, logger: Logger | None = None):
        """
        Perform inference using the provided model. Static method since it does
        not require an instance of the class but it's useful to have it in the
        same class.

        Args:
            model: The model to be used for inference.
            config: Configuration object containing inference settings.
            logger: Logger instance for logging inference results.

        Returns:
            Any: The result of the inference process.
        """
        pass
