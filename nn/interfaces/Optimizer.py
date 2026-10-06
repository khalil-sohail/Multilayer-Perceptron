from numpy.typing import NDArray
from typing import Protocol

import numpy as np

Array = NDArray[np.float64]

class TrainableLayer(Protocol):
    weights: Array
    biases: Array
    grad_weights: Array
    grad_biases: Array


class Optimizer(Protocol):
    def step(self, layer: TrainableLayer) -> None:
        ...