from numpy.typing import NDArray

import numpy as np

Array = NDArray[np.float64]

class ReLU:
    """
    A ReLU (Rectified Linear Unit) activation function for the MLP.
    This activation function applies the ReLU operation to the input data, which sets all negative values to zero and keeps positive values unchanged.

    Equation:
        relu(x) = max(0, x)
        relu'(x) = 1 if x > 0 else 0

    Attributes:
        inputs (numpy.ndarray): The input data to the activation function.
    """

    def __init__(self):
        """
        Initializes the ReLU activation function.
        """
        self.trainable = False

    def forward(self, inputs: Array, training: bool=True) -> Array:
        """
        Performs the forward pass of the ReLU activation function.

        Args:
            inputs (numpy.ndarray): The input data to the activation function.

        Returns:
            numpy.ndarray: The output of the activation function, where all negative values are set to zero and positive values remain unchanged.
        """
        if training:
            self.inputs = inputs
            
        return np.maximum(0, inputs)

    def backward(self, grad_output: Array) -> Array:
        """
        Performs the backward pass of the ReLU activation function.

        Args:
            grad_output (numpy.ndarray): The gradient of the loss with respect to the output of the activation function.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input of the activation function,
                where gradients for negative input values are set to zero and gradients for positive input values remain unchanged.
        """
        dinputs = grad_output.copy()
        dinputs[self.inputs <= 0] = 0

        return dinputs
