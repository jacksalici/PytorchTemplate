"""
Early Stopping mechanism for PyTorch training.

This module provides an early stopping utility that monitors a specified metric
and stops training when the metric stops improving for a given patience period.
"""

import numpy as np
from typing import Optional, Literal, Union
import torch


class EarlyStopping:
    """
    Early stopping utility to stop training when a monitored metric stops improving.
    
    This class tracks the best value of a specified metric and stops training
    if the metric doesn't improve for a specified number of epochs (patience).
    """
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: Literal["min", "max"] = "min",
        baseline: Optional[float] = None,
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the EarlyStopping callback.
        
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change in the monitored quantity to qualify as an improvement
            mode: One of {"min", "max"}. In "min" mode, training will stop when the quantity
                 monitored has stopped decreasing; in "max" mode it will stop when the
                 quantity monitored has stopped increasing
            baseline: Baseline value for the monitored quantity. Training will stop if the
                     model doesn't show improvement over the baseline
            restore_best_weights: Whether to restore model weights from the epoch with the
                                 best value of the monitored quantity
            verbose: Whether to print messages when improvement is detected
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_epoch = 0
        
        if mode == "min":
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
        
        self._reset()
    
    def _reset(self):
        """Reset the early stopping state."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_epoch = 0
        
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.inf if self.mode == "min" else -np.inf
    
    def __call__(
        self, 
        current_value: float, 
        model: torch.nn.Module,
        epoch: int
    ) -> bool:
        """
        Check if training should be stopped based on the current metric value.
        
        Args:
            current_value: Current value of the monitored metric
            model: PyTorch model whose weights should be saved if this is the best epoch
            epoch: Current epoch number
        
        Returns:
            bool: True if training should be stopped, False otherwise
        """
        if self.monitor_op(current_value - self.min_delta, self.best):
            self.best = current_value
            self.wait = 0
            self.best_epoch = epoch
            
            if self.restore_best_weights:
                # Save the best model weights
                self.best_weights = {
                    key: value.clone().detach() 
                    for key, value in model.state_dict().items()
                }
            
            if self.verbose:
                print(f"Epoch {epoch}: Metric improved to {current_value:.6f}")
                
        else:
            self.wait += 1
            
            if self.verbose and self.wait == 1:
                print(f"Epoch {epoch}: Metric did not improve from {self.best:.6f}")
        
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            if self.verbose:
                print(f"Early stopping triggered after {self.patience} epochs without improvement")
                print(f"Best metric: {self.best:.6f} at epoch {self.best_epoch}")
            return True
        
        return False
    
    def restore_weights(self, model: torch.nn.Module) -> bool:
        """
        Restore the best weights to the model.
        
        Args:
            model: PyTorch model to restore weights to
            
        Returns:
            bool: True if weights were restored, False if no best weights were saved
        """
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                print(f"Restored best weights from epoch {self.best_epoch}")
            return True
        else:
            if self.verbose:
                print("No best weights to restore")
            return False
    
    def get_best_value(self) -> float:
        """Get the best metric value encountered so far."""
        return self.best
    
    def get_best_epoch(self) -> int:
        """Get the epoch number where the best metric value was achieved."""
        return self.best_epoch
    
    def get_patience_left(self) -> int:
        """Get the number of epochs remaining before early stopping triggers."""
        return max(0, self.patience - self.wait)
    
    def state_dict(self) -> dict:
        """Get the early stopping state for saving/resuming."""
        return {
            'wait': self.wait,
            'stopped_epoch': self.stopped_epoch,
            'best': self.best,
            'best_epoch': self.best_epoch,
            'best_weights': self.best_weights
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load the early stopping state for resuming."""
        self.wait = state_dict.get('wait', 0)
        self.stopped_epoch = state_dict.get('stopped_epoch', 0)
        self.best = state_dict.get('best', np.inf if self.mode == "min" else -np.inf)
        self.best_epoch = state_dict.get('best_epoch', 0)
        self.best_weights = state_dict.get('best_weights', None)


class EarlyStoppingManager:
    """
    Manager class for multiple early stopping criteria.
    
    This allows monitoring multiple metrics simultaneously and stopping
    training when any of them triggers early stopping.
    """
    
    def __init__(self, early_stopping_configs: dict):
        """
        Initialize the early stopping manager.
        
        Args:
            early_stopping_configs: Dictionary mapping metric names to early stopping configs
                                   Each config should contain parameters for EarlyStopping
        """
        self.early_stoppers = {}
        
        for metric_name, config in early_stopping_configs.items():
            self.early_stoppers[metric_name] = EarlyStopping(**config)
    
    def __call__(
        self, 
        metrics: dict, 
        model: torch.nn.Module,
        epoch: int
    ) -> tuple[bool, list]:
        """
        Check all early stopping criteria.
        
        Args:
            metrics: Dictionary of metric values
            model: PyTorch model
            epoch: Current epoch number
        
        Returns:
            tuple: (should_stop, list_of_triggered_metrics)
        """
        should_stop = False
        triggered_metrics = []
        
        for metric_name, early_stopper in self.early_stoppers.items():
            if metric_name in metrics:
                if early_stopper(metrics[metric_name], model, epoch):
                    should_stop = True
                    triggered_metrics.append(metric_name)
        
        return should_stop, triggered_metrics
    
    def restore_best_weights(self, model: torch.nn.Module, metric_name: Optional[str] = None):
        """
        Restore the best weights for a specific metric or the first available.
        
        Args:
            model: PyTorch model
            metric_name: Specific metric to restore weights for, or None for first available
        """
        if metric_name and metric_name in self.early_stoppers:
            self.early_stoppers[metric_name].restore_weights(model)
        else:
            # Restore weights from the first early stopper that has best weights
            for early_stopper in self.early_stoppers.values():
                if early_stopper.restore_weights(model):
                    break
    
    def get_status(self) -> dict:
        """Get status information for all early stoppers."""
        status = {}
        for metric_name, early_stopper in self.early_stoppers.items():
            status[metric_name] = {
                'best_value': early_stopper.get_best_value(),
                'best_epoch': early_stopper.get_best_epoch(),
                'patience_left': early_stopper.get_patience_left()
            }
        return status
    
    def state_dict(self) -> dict:
        """Get state dict for all early stoppers."""
        return {
            metric_name: early_stopper.state_dict()
            for metric_name, early_stopper in self.early_stoppers.items()
        }
    
    def load_state_dict(self, state_dict: dict):
        """Load state dict for all early stoppers."""
        for metric_name, early_stopper in self.early_stoppers.items():
            if metric_name in state_dict:
                early_stopper.load_state_dict(state_dict[metric_name])
