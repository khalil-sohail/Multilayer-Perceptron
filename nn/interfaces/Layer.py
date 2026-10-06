from numpy.typing import NDArray
from typing import Protocol

import numpy as np

Array = NDArray[np.float64]

class Layer(Protocol):
    """A protocol for layers in a neural network."""
    trainable: bool

    def forward(self, inputs: Array, training: bool = False) -> Array:
        ...

    def backward(self, grad_output: Array) -> Array:
        ...