import numpy as np

class Huber:
    """
    This class implements the Huber loss function.
    It behaves like MSE when the error is small, but more like absolute error when the error gets large.
    Making it is less sensitive to outliers than the mean squared error loss.

    Equation:
        L = ( |P - T| <= delta ) ? ( 1/2 * (P - T)^2 ) : ( delta * (|P - T| - 1/2 * delta) )
          
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

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes the Huber loss.
        """
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
