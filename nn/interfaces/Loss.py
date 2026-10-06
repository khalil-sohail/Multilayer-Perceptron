from numpy.typing import NDArray
from typing import Protocol

import numpy as np

Array = NDArray[np.float64]

class Loss(Protocol):
    """A protocol for loss functions."""

    def forward(self, y_pred: Array, y_true: Array, training: bool = False) -> float:
        ...

    def backward(self) -> Array:
        ...