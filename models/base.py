"""
Base model utilities built on top of `torch.nn.Module`.

This module defines `BaseModel`, a lightweight extension of PyTorch's base
module that centralizes a few recurring conveniences.

Usage example:
    ```
    class MyModel(BaseModel):
        def __init__(self):
            super().__init__()
            ...

    model = MyModel()
    print(model.n_param)
    model.requires_grad(False)  # freeze all parameters
    ```
"""

from torch import nn


class BaseModel(nn.Module):
    """
    Base class for all torch models.

    This is a simple extension of `torch.nn.Module` that adds some
    utility functions, such as counting the number of trainable parameters
    and setting the `requires_grad` attribute of all parameters.
    """

    @property
    def n_param(self) -> int:
        """
        Returns the number of (trainable) parameters in the model.

        Returns:
            Number of parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __init__(self) -> None:
        super().__init__()

    def requires_grad(self, flag: bool) -> None:
        """
        Sets the `requires_grad` attribute of all model parameters to `flag`.

        Args:
            flag: True if the model requires gradient, False otherwise.
        """
        for p in self.parameters():
            p.requires_grad = flag
