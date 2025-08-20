from torch import nn
import argparse


class BaseModel(nn.Module): 
    @property
    def n_param(self) -> int:
        """
        Returns the number of (trainable) parameters in the model.

        Returns:
            Number of parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def __init__(self, **kwargs):
        
        super().__init__()

        
    def requires_grad(self, flag: bool) -> None:
        """
        Sets the `requires_grad` attribute of all model parameters to `flag`.

        Args:
            flag: True if the model requires gradient, False otherwise.
        """
        for p in self.parameters():
            p.requires_grad = flag