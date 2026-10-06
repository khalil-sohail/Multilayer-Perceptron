from numpy.typing import NDArray

import numpy as np

Array = NDArray[np.float64]

class CategoricalCrossEntropy:
    """
    Categorical Cross-Entropy Loss Function.
    This class implements the categorical cross-entropy loss function, which is commonly used for multi-class classification problems.
    It measures the dissimilarity between the predicted probability distribution and the true distribution (one-hot encoded labels).
    
    Equation:
        L  = -Σ_i [ Y_i * log(P_i) ]
        L'_i = -Y_i / P_i
        L' = -Σ_i [ Y_i / P_i ]

    Attributes:
        None
    """

    def __init__(self):
        """
        Initializes the Categorical Cross-Entropy loss function.
        """
        pass
    
    def forward(self, y_pred: Array, y_true: Array, training: bool=True) -> float:
        """
        Computes the categorical cross-entropy loss.
        """
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        y_true = y_true
        
        if training:
            self.y_pred = y_pred
            self.y_true = y_true
        
        sample_losses = -np.sum(y_true * np.log(y_pred), axis=1)
        return float(np.mean(sample_losses))

    def backward(self) -> Array:
        """
        Computes the gradient of the categorical cross-entropy loss with respect to the predictions.
        """
        
        grad = -( self.y_true / self.y_pred )
        return grad / self.y_pred.shape[0]
