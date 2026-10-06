from numpy.typing import NDArray

import numpy as np

Array = NDArray[np.float64]

class MSE:
    """
    Mean Squared Error Loss Function.
    This class implements the mean squared error loss function, which is commonly used for regression problems.
    It measures the average squared difference between the predicted values and the true values.
    
    Equation:
        MSE = (1/n) * Σ_i(P_i - T_i)^2
        MSE' = (2/n) * [(P_0 - T_0), (P_1 - T_1), ..., (P_n-1 - T_n-1)]
    
    Attributes:
        None
    """

    def __init__(self):
        """
        Initializes the Mean Squared Error loss function.
        """
        pass

    def forward(self, y_pred: Array, y_true: Array, training: bool=True) -> float:
        """
        Computes the mean squared error loss.
        """
        if training:
            self.y_pred = y_pred
            self.y_true = y_true

        return float(np.mean((y_pred - y_true) ** 2))

    def backward(self) -> Array:
        """
        Computes the gradient of the mean squared error loss with respect to the predictions.
        """
        return (2.0 / self.y_pred.size) * (self.y_pred - self.y_true)
