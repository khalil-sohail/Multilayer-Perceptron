from numpy.typing import NDArray

import numpy as np

Array = NDArray[np.float64]

class BinaryCrossEntropy:
    """
    Binary Cross-Entropy Loss Function.
    This class implements the binary cross-entropy loss function, which is commonly used for binary classification problems.
    It measures the dissimilarity between the predicted probability distribution and the true distribution (binary labels).
    
    Equation:
        L = -1/N Σ_i[Y_i * log(P_i) + (1 - Y_i) * log(1 - P_i)]
        L' = -1/N [Y/P - (1 - Y)/(1 - P)]

    Attributes:
        None
    """

    def __init__(self):
        """
        Initializes the Binary Cross-Entropy loss function.
        """
        pass

    def forward(self, y_pred: Array, y_true: Array, training: bool=True) -> float:
        """
        Computes the binary cross-entropy loss.
        """
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        y_true = y_true

        if training:
            self.y_pred = y_pred
            self.y_true = y_true
        
        losses = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
        return float(np.mean(losses))

    def backward(self) -> Array:
        """
        Computes the gradient of the binary cross-entropy loss with respect to the predictions.
        """
        
        grad = -(self.y_true / self.y_pred) + (1.0 - self.y_true) / (1.0 - self.y_pred)
        return grad / self.y_pred.size
