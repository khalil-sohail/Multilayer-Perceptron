from numpy.typing import NDArray
from typing import Protocol

import numpy as np

Array = NDArray[np.float64]

class TrainableLayer(Protocol):
    """
    A protocol for defining trainable layers.
    """
    weights: Array
    biases: Array
    grad_weights: Array
    grad_biases: Array

class Optimizer(Protocol):
    """
    A protocol for defining optimizers.
    """
    def step(self, layer: TrainableLayer) -> None:
        ...