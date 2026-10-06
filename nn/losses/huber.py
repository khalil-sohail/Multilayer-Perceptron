from numpy.typing import NDArray

import numpy as np

Array = NDArray[np.float64]

class Huber:
    """
    This class implements the Huber loss function.
    It behaves like MSE when the error is small, but more like absolute error when the error gets large.
    Making it is less sensitive to outliers than the mean squared error loss.

    Equation:
        L  = ( |P - T| <= delta ) ? ( 1/2 * (P - T)^2 ) : ( delta * (|P - T| - 1/2 * delta) )
        L' = ( |P - T| <= delta ) ?      ( P - T )      :       ( delta * sign(P - T) )
          
    Attributes:
        delta (float): The threshold at which to switch between quadratic and linear loss.
    """
    def __init__(self, delta: float = 1.0):
        """
        Initializes the Huber loss function with a specified delta.
        
        Args:
            delta (float): The threshold at which to switch between quadratic and linear loss.
        """
        self.delta = delta

    def forward(self, y_pred: Array, y_true: Array, training: bool=True) -> float:
        """
        Computes the Huber loss.
        """
        if training:
            self.y_pred = y_pred
            self.y_true = y_true

        error = y_pred - y_true
        absolute_error = np.abs(error)

        losses = np.where(
            absolute_error <= self.delta,
            0.5 * error**2,
            self.delta * (
                absolute_error - 0.5 * self.delta
            ),
        )

        return float(np.mean(losses))

    def backward(self) -> Array:
        """
        Computes the gradient of the Huber loss with respect to the predictions.
        """
        error = self.y_pred - self.y_true
        absolute_error = np.abs(error)

        grad = np.where(
            absolute_error <= self.delta,
            error,
            self.delta * np.sign(error),
        )

        return grad / self.y_pred.size
