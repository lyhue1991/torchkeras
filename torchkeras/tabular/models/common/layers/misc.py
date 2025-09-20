# W605
from typing import Callable, Union

import torch
from torch import nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Lambda(nn.Module):
    """A wrapper for a lambda function as a pytorch module."""

    def __init__(self, func: Callable):
        """Initialize lambda module
        Args:
            func: any function/callable
        """
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class ModuleWithInit(nn.Module):
    """Base class for pytorch module with data-aware initializer on first batch."""

    def __init__(self):
        super().__init__()
        self._is_initialized_tensor = nn.Parameter(torch.tensor(0, dtype=torch.uint8), requires_grad=False)
        self._is_initialized_bool = None
        # Note: this module uses a separate flag self._is_initialized so as to achieve both
        # * persistence: is_initialized is saved alongside model in state_dict
        # * speed: model doesn't need to cache
        # please DO NOT use these flags in child modules

    def initialize(self, *args, **kwargs):
        """Initialize module tensors using first batch of data."""
        raise NotImplementedError("Please implement ")

    def __call__(self, *args, **kwargs):
        if self._is_initialized_bool is None:
            self._is_initialized_bool = bool(self._is_initialized_tensor.item())
        if not self._is_initialized_bool:
            self.initialize(*args, **kwargs)
            self._is_initialized_tensor.data[...] = 1
            self._is_initialized_bool = True
        return super().__call__(*args, **kwargs)


class Add(nn.Module):
    """A module that adds a constant/parameter value to the input."""

    def __init__(self, add_value: Union[float, torch.Tensor]):
        """Initialize the module.

        Args:
            add_value: The value to add to the input

        """
        super().__init__()
        self.add_value = add_value

    def forward(self, x):
        return x + self.add_value
