import numpy as np

class BinaryCrossEntropy:
    """
    Binary Cross-Entropy Loss Function.
    This class implements the binary cross-entropy loss function, which is commonly used for binary classification problems.
    It measures the dissimilarity between the predicted probability distribution and the true distribution (binary labels).
    
    Equation:
        L = -[Y * log(P) + (1 - Y) * log(1 - P)]

    Attributes:
        None
    """
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes the binary cross-entropy loss.
        """
        
        epsilon = 1e-7

        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        losses = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))

        return float(np.mean(losses))
