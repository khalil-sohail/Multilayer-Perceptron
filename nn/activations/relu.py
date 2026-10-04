import numpy as np

class ReLU:
    """
    A ReLU (Rectified Linear Unit) activation function for the MLP.
    This activation function applies the ReLU operation to the input data, which sets all negative values to zero and keeps positive values unchanged.

    Equation:
        relu(x) = max(0, x)

    Attributes:
        inputs (numpy.ndarray): The input data to the activation function.
    """

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the ReLU activation function.
        Args:
            inputs (numpy.ndarray): The input data to the activation function.
        Returns:
            numpy.ndarray: The output of the activation function, where all negative values are set to zero and positive values remain unchanged.
        """
        self.inputs = inputs
        return np.maximum(0, inputs)
