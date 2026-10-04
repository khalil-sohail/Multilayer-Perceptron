import numpy as np

class MSE:
    """
    Mean Squared Error Loss Function.
    This class implements the mean squared error loss function, which is commonly used for regression problems.
    It measures the average squared difference between the predicted values and the true values.
    
    Equation:
        MSE = (1/n) * Σ_i(P_i - T_i)^2
    
    Attributes:
        None
    """
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Computes the mean squared error loss.
        """
        
        return float(np.mean((y_pred - y_true) ** 2))
