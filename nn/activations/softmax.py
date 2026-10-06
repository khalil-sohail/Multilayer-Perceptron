from numpy.typing import NDArray

import numpy as np

Array = NDArray[np.float64]

class Softmax:
    """
    A Softmax activation function for the MLP.
    This activation function applies the Softmax operation to the input data, which converts the input values into probabilities that sum to 1.

    Equation:
        softmax(x) = Σ_i_j (exp(x_i) / sum(exp(x_j)))
        softmax'(x) = softmax(x) * (1 - softmax(x))

    Attributes:
        inputs (numpy.ndarray): The input data to the activation function.
        output (numpy.ndarray): The output of the activation function after applying the Softmax operation.
    """

    def __init__(self):
        """
        Initializes the Softmax activation function.
        """
        self.trainable = False

    def forward(self, inputs: Array, training: bool=True) -> Array:
        """
        Performs the forward pass of the Softmax activation function.

        Args:
            inputs (numpy.ndarray): The input data to the activation function.
            training (bool): Whether the layer is in training mode.

        Returns:
            numpy.ndarray: The output of the activation function, where the input values are converted into probabilities that sum to 1.
        """
        shifted = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        if training:
            self.inputs = inputs
            self.output = output

        return output

    def backward(self, grad_output: Array) -> Array:
        """
        Performs the backward pass of the Softmax activation function.

        Args:
            grad_output (numpy.ndarray): The gradient of the loss with respect to the output of the activation function.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input of the activation function.
        """
        
        dot = np.sum(grad_output * self.output, axis=1, keepdims=True)
        grad_input = self.output * (grad_output - dot)

        return grad_input
