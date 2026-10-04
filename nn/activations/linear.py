import numpy as np

class Linear:
    """
    A linear activation function for neural networks.
    This activation function applies a linear transformation to the input data.

    Equation:
        f(x) = x

    Attributes:
        None
    """
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the Linear activation function.
        Args:
            inputs (numpy.ndarray): The input data to the activation function.
        Returns:
            numpy.ndarray: The output of the activation function, which is the same as the input.
        """
        return inputs
