"""
Learning Rate Scheduler utilities for PyTorch training.

This module provides a unified interface for various PyTorch learning rate schedulers
with support for both step-based and metric-based scheduling.
"""

import torch
from torch.optim import lr_scheduler
from typing import Dict, Any, Optional, Union, Literal
import warnings


class LRSchedulerWrapper:
    """
    Wrapper class for PyTorch learning rate schedulers with unified interface.
    
    Supports both step-based schedulers (called every epoch) and metric-based
    schedulers (called with validation metrics).
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_type: str,
        scheduler_params: Optional[Dict[str, Any]] = None,
        metric_mode: Literal["min", "max"] = "min",
        step_on_epoch: bool = True
    ):
        """
        Initialize the learning rate scheduler wrapper.
        
        Args:
            optimizer: PyTorch optimizer instance
            scheduler_type: Type of scheduler ("step", "cosine", "reduce_on_plateau", 
                          "exponential", "cosine_annealing_warm_restarts", "linear_warm_up")
            scheduler_params: Dictionary of parameters for the scheduler
            metric_mode: Whether to minimize or maximize the metric (for ReduceLROnPlateau)
            step_on_epoch: Whether to step the scheduler every epoch (True) or every batch (False)
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type.lower()
        self.scheduler_params = scheduler_params or {}
        self.metric_mode = metric_mode
        self.step_on_epoch = step_on_epoch
        self.scheduler = self._create_scheduler()
        
        # Track if this is a metric-based scheduler
        self.is_metric_based = isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau)
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create the appropriate PyTorch scheduler based on type."""
        
        if self.scheduler_type == "step":
            return lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.scheduler_params.get("step_size", 30),
                gamma=self.scheduler_params.get("gamma", 0.1),
                last_epoch=self.scheduler_params.get("last_epoch", -1)
            )
        
        elif self.scheduler_type == "cosine":
            return lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.scheduler_params.get("T_max", 50),
                eta_min=self.scheduler_params.get("eta_min", 0),
                last_epoch=self.scheduler_params.get("last_epoch", -1)
            )
        
        elif self.scheduler_type == "reduce_on_plateau":
            return lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.metric_mode,
                factor=self.scheduler_params.get("factor", 0.1),
                patience=self.scheduler_params.get("patience", 10),
                threshold=self.scheduler_params.get("threshold", 1e-4),
                threshold_mode=self.scheduler_params.get("threshold_mode", "rel"),
                cooldown=self.scheduler_params.get("cooldown", 0),
                min_lr=self.scheduler_params.get("min_lr", 0),
                eps=self.scheduler_params.get("eps", 1e-8),
                verbose=self.scheduler_params.get("verbose", False)
            )
        
        elif self.scheduler_type == "exponential":
            return lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.scheduler_params.get("gamma", 0.95),
                last_epoch=self.scheduler_params.get("last_epoch", -1)
            )
        
        elif self.scheduler_type == "cosine_annealing_warm_restarts":
            return lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.scheduler_params.get("T_0", 10),
                T_mult=self.scheduler_params.get("T_mult", 1),
                eta_min=self.scheduler_params.get("eta_min", 0),
                last_epoch=self.scheduler_params.get("last_epoch", -1)
            )
        
        elif self.scheduler_type == "linear_warm_up":
            return lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=self.scheduler_params.get("start_factor", 1.0/3),
                end_factor=self.scheduler_params.get("end_factor", 1.0),
                total_iters=self.scheduler_params.get("total_iters", 5),
                last_epoch=self.scheduler_params.get("last_epoch", -1)
            )
        
        elif self.scheduler_type == "multiplicative":
            if "lr_lambda" not in self.scheduler_params:
                # Default lambda function that multiplies by 0.95 every epoch
                self.scheduler_params["lr_lambda"] = lambda epoch: 0.95
            return lr_scheduler.MultiplicativeLR(
                self.optimizer,
                lr_lambda=self.scheduler_params["lr_lambda"],
                last_epoch=self.scheduler_params.get("last_epoch", -1)
            )
        
        elif self.scheduler_type == "none" or self.scheduler_type == "disabled":
            # Return a dummy scheduler that does nothing
            return DummyScheduler(self.optimizer)
        
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
    
    def step(self, metric: Optional[float] = None, epoch: Optional[int] = None):
        """
        Step the learning rate scheduler.
        
        Args:
            metric: Validation metric (required for ReduceLROnPlateau)
            epoch: Current epoch (optional, for logging purposes)
        """
        if self.is_metric_based:
            if metric is None:
                warnings.warn(
                    "Metric-based scheduler requires a metric value but None was provided. "
                    "Skipping scheduler step."
                )
                return
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
    
    def get_last_lr(self) -> list:
        """Get the last learning rate(s)."""
        if hasattr(self.scheduler, 'get_last_lr'):
            return self.scheduler.get_last_lr()
        else:
            # For ReduceLROnPlateau which doesn't have get_last_lr
            return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self) -> Dict[str, Any]:
        """Get the scheduler state dict for saving."""
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load the scheduler state dict for resuming."""
        self.scheduler.load_state_dict(state_dict)


class DummyScheduler:
    """Dummy scheduler that does nothing - used when no scheduling is desired."""
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    def step(self, metric=None):
        """Do nothing."""
        pass
    
    def get_last_lr(self):
        """Return current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """Return empty state dict."""
        return {}
    
    def load_state_dict(self, state_dict):
        """Do nothing."""
        pass


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: Dict[str, Any]
) -> Optional[LRSchedulerWrapper]:
    """
    Factory function to create a learning rate scheduler from configuration.
    
    Args:
        optimizer: PyTorch optimizer instance
        scheduler_config: Configuration dictionary containing scheduler settings
    
    Returns:
        LRSchedulerWrapper instance or None if no scheduler specified
    """
    if not scheduler_config or scheduler_config.get("type", "none").lower() in ["none", "disabled"]:
        return None
    
    scheduler_type = scheduler_config.get("type", "step")
    scheduler_params = scheduler_config.get("params", {})
    metric_mode = scheduler_config.get("metric_mode", "min")
    step_on_epoch = scheduler_config.get("step_on_epoch", True)
    
    return LRSchedulerWrapper(
        optimizer=optimizer,
        scheduler_type=scheduler_type,
        scheduler_params=scheduler_params,
        metric_mode=metric_mode,
        step_on_epoch=step_on_epoch
    )
