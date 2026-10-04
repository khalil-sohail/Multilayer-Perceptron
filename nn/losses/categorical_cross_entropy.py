import numpy as np

class CategoricalCrossEntropy:
    """
    Categorical Cross-Entropy Loss Function.
    This class implements the categorical cross-entropy loss function, which is commonly used for multi-class classification problems.
    It measures the dissimilarity between the predicted probability distribution and the true distribution (one-hot encoded labels).
    
    Equation:
        L = -Σ_i[ Y_i * log(P_i) ]

    Attributes:
        None
    """
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes the categorical cross-entropy loss.
        """
        
        epsilon = 1e-7

        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        sample_losses = -np.sum(y_true * np.log(y_pred), axis=1)

        return float(np.mean(sample_losses))
