from collections.abc import Callable
from numpy.typing import NDArray

import numpy as np

Array = NDArray[np.float64]

Metric = Callable[
    [Array, Array],
    float,
]
