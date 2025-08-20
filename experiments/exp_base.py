import torch
from torch import nn, optim, device
from torch.utils.data import DataLoader
from utils.logger import Logger
from utils.plotter import plot
from utils.metrics import Metrics
import torch.nn.functional as F
from argparse import ArgumentParser
import os

class BaseExperiment():
    @staticmethod
    def add_args(parser: ArgumentParser):
        """
        Adds command-line arguments to the given ArgumentParser instance.
        This function is intended to be used for extending the parser with experiment-specific arguments.
        Args:
            parser (ArgumentParser): The argument parser to which new arguments will be added.
        """
        
        return parser

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        logger: Logger,
        conf: dict
    ):
        """
        Initializes the BaseExperiment class with the provided model, criterion, optimizer, device, logger, and arguments.
        Args:
            model (nn.Module): The model to be used in the experiment.
            criterion (nn.Module): The loss function to be used.
            optimizer (optim.Optimizer): The optimizer to be used for training.
            device (torch.device): The device (CPU or GPU) on which the model will be trained.
            logger (Logger): Logger instance for logging experiment results.
            conf (dict): Dictionary of arguments for the experiment.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.conf = conf
        
    def train(self, train_loader: DataLoader) -> float:
        """
        Train the model for one epoch using the provided DataLoader.
        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
        
        Returns:
            float: The average training loss for the epoch.
        """
        
        return 0
    
    
    def validate(self, val_loader: DataLoader) -> tuple[float, dict]:
        """
        Validate the model using the provided DataLoader.
        Args:
            val_loader (DataLoader): DataLoader for the testing dataset.
        
        Returns:
            tuple: A tuple containing the average test loss and a Metrics object.
        """
        
        return 0, {}
    
    def run(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        """
        Runs the training and testing loop for the specified number of epochs.
        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the testing dataset.
            num_epochs (int): Number of epochs to run the training and testing loop.
        """
        min_test_loss = float("inf")
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            train_loss = self.train(train_loader)
            test_loss, metric = self.validate(val_loader)
            
            #self.logger.log({"Epoch": epoch, "Train Loss": train_loss, "Test Loss": test_loss,
            #               **metric.get_dict()})
                     
            #if test_loss < min_test_loss:
            #    min_test_loss = test_loss
            #    self.save_model(epoch, test_loss)
    
    
    def save_model(self, epoch: int, test_loss: float):
        """
        Saves the model state to a file.
        Args:
            epoch (int): The current epoch number.
            test_loss (float): The test loss at the current epoch.
        """
        if self.conf.save_model:
            print(f"Saving model at epoch {epoch} with test loss {test_loss}")
            if not os.path.exists(self.conf["checkpoint_path"]):
                os.makedirs(self.conf["checkpoint_path"])
            
            if "checkpoint_name" in self.conf:
                path = os.path.join(self.conf["checkpoint_path"], self.conf["checkpoint_name"])
            else:
                path = f"{str(os.path.join(self.conf["checkpoint_path"], self.conf["model_name"]))}.pth"
            torch.save(self.model, path)


