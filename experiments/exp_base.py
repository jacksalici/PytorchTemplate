import torch
from torch import nn, optim, device
from torch.utils.data import DataLoader
from utils.logger import Logger
from utils.scheduler import LRSchedulerWrapper, create_scheduler
from utils.early_stopping import EarlyStopping
import torch.nn.functional as F
from argparse import ArgumentParser
from configs.config import Config
import os
from typing import Optional, Dict, Any

class BaseExperiment():
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        logger: Logger,
        conf: Config
    ):
        """
        Initializes the BaseExperiment class with the provided model, criterion, optimizer, device, logger, and arguments.
        Args:
            model (nn.Module): The model to be used in the experiment.
            criterion (nn.Module): The loss function to be used.
            optimizer (optim.Optimizer): The optimizer to be used for training.
            device (torch.device): The device (CPU or GPU) on which the model will be trained.
            logger (Logger): Logger instance for logging experiment results.
            conf (Config): Configuration class
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.conf = conf
        
        # Initialize tracking variables based on mode
        if conf.model_saving_mode == "min":
            self.best_metric_value = float("inf")
        else:
            self.best_metric_value = float("-inf")
        
        # Initialize scheduler if enabled
        self.scheduler = None
        if conf.use_scheduler:
            scheduler_config = {
                "type": conf.scheduler_type,
                "params": conf.scheduler_params or {},
                "metric_mode": conf.scheduler_mode,
                "step_on_epoch": True
            }
            self.scheduler = create_scheduler(optimizer, scheduler_config)
        
        # Initialize early stopping if enabled
        self.early_stopping = None
        if conf.use_early_stopping:
            self.early_stopping = EarlyStopping(
                patience=conf.early_stopping_patience,
                min_delta=conf.early_stopping_min_delta,
                mode=conf.early_stopping_mode,
                restore_best_weights=conf.early_stopping_restore_best_weights,
                verbose=True
            )
        
        # For resuming training
        self.start_epoch = 0
        
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
        
    def run(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, model_saving_metric: str = None):
        """
        Runs the training and testing loop for the specified number of epochs.
        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the testing dataset.
            num_epochs (int): Number of epochs to run the training and testing loop.
            model_saving_metric (str): Metric to use for model saving (overrides config default)
        """
        
        # Use provided metric or fall back to config default
        saving_metric = model_saving_metric or self.conf.model_saving_metric
        
        # Resume training if requested
        if self.conf.resume_training:
            self._resume_training()
        
        for epoch in range(self.start_epoch, num_epochs):
            train_loss = self.train(train_loader)
            test_loss, metric = self.validate(val_loader)
            
            log_dict = {"Epoch": epoch, "Train Loss": train_loss, "Test Loss": test_loss,
                           **metric}
            
            # Add learning rate to log
            if self.scheduler:
                log_dict["Learning Rate"] = self.scheduler.get_last_lr()[0]
            
            self.logger(log_dict)
                     
            # Save model based on specified metric
            self._save_model(epoch, log_dict, saving_metric)
            
            # Step scheduler if enabled
            if self.scheduler:
                if self.scheduler.is_metric_based:
                    # Use the metric specified in config for scheduler
                    scheduler_metric_value = log_dict.get(self.conf.scheduler_metric)
                    if scheduler_metric_value is not None:
                        self.scheduler.step(scheduler_metric_value, epoch)
                else:
                    self.scheduler.step()
            
            # Check early stopping if enabled
            if self.early_stopping:
                early_stopping_metric_value = log_dict.get(self.conf.early_stopping_metric)
                if early_stopping_metric_value is not None:
                    if self.early_stopping(early_stopping_metric_value, self.model, epoch):
                        self.logger(f"Early stopping triggered at epoch {epoch}")
                        if self.conf.early_stopping_restore_best_weights:
                            self.early_stopping.restore_weights(self.model)
                        break
    
    
    def _save_model(self, epoch: int, log_dict: dict, model_saving_metric: str):
        """
        Saves the model state to a file based on the specified metric and mode.
        Args:
            epoch (int): The current epoch number.
            log_dict (dict): Dictionary containing metrics for this epoch.
            model_saving_metric (str): The metric to use for determining when to save.
        """
        if model_saving_metric not in log_dict:
            self.logger(f"Warning: Model saving metric '{model_saving_metric}' not found in log_dict")
            return
            
        metric_value = log_dict[model_saving_metric]
        
        # Check if this is the best model based on the specified mode
        is_best = False
        if self.conf.model_saving_mode == "min":
            if metric_value <= self.best_metric_value:
                self.best_metric_value = metric_value
                is_best = True
        else:  # mode == "max"
            if metric_value >= self.best_metric_value:
                self.best_metric_value = metric_value
                is_best = True
        
        if is_best and self.conf.save_model:
            self.logger(f"Saving model at epoch {epoch} with {model_saving_metric} {metric_value:.6f}")
            
            # Save both model and training state for resuming
            checkpoint = {
                'epoch': epoch + 1,  # Next epoch to start from
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_metric_value': self.best_metric_value,
                'config': self.conf.to_dict()
            }
            
            # Add scheduler state if present
            if self.scheduler:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
            # Add early stopping state if present
            if self.early_stopping:
                checkpoint['early_stopping_state_dict'] = self.early_stopping.state_dict()
            
            torch.save(checkpoint, self.conf.get_checkpoint_path())


    def _resume_training(self):
        """
        Resume training from a checkpoint.
        """
        checkpoint_path = self.conf.resume_checkpoint_path or self.conf.get_checkpoint_path()
        
        if not os.path.exists(checkpoint_path):
            self.logger(f"Warning: Checkpoint file not found at {checkpoint_path}. Starting from scratch.")
            return
        
        try:
            self.logger(f"Resuming training from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load training progress
            self.start_epoch = checkpoint.get('epoch', 0)
            self.best_metric_value = checkpoint.get('best_metric_value', 
                                                   float("inf") if self.conf.model_saving_mode == "min" else float("-inf"))
            
            # Load scheduler state if present
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load early stopping state if present
            if self.early_stopping and 'early_stopping_state_dict' in checkpoint:
                self.early_stopping.load_state_dict(checkpoint['early_stopping_state_dict'])
            
            self.logger(f"Resumed training from epoch {self.start_epoch}")
            
        except Exception as e:
            self.logger(f"Error loading checkpoint: {e}. Starting from scratch.")
            self.start_epoch = 0


    @staticmethod
    def inference(model: nn.Module, config: Config, logger: Logger = None):
        """
        Perform inference using the provided model. Static method since it does not require an instance of the class but it's useful to have it in the same class.
        Args:
            model (nn.Module): The model to be used for inference.
            config (Config): Configuration object containing inference settings.
            logger (Logger, optional): Logger instance for logging inference results. Defaults to None.
        
        Returns:
            Any: The result of the inference process.
        """
        pass