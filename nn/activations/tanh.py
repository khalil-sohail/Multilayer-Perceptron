from numpy.typing import NDArray

import numpy as np

Array = NDArray[np.float64]

class Tanh:
    """
    A Tanh (Hyperbolic Tangent) activation function for the MLP.
    This activation function applies the Tanh operation to the input data, which maps the input values to a range between -1 and 1.

    Equation:
        tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        tanh'(x) = 1 - tanh(x)^2
    
    Attributes:
        inputs (numpy.ndarray): The input data to the activation function.
        output (numpy.ndarray): The output of the activation function after applying the Tanh operation.
    """

    def __init__(self):
        """
        Initializes the Tanh activation function.
        """
        self.trainable = False

    def forward(self, inputs: Array, training: bool=True) -> Array:
        """
        Performs the forward pass of the Tanh activation function.

        Args:
            inputs (numpy.ndarray): The input data to the activation function.

        Returns:
            numpy.ndarray: The output of the activation function, where the input values are mapped to a range between -1 and 1.
        """
        output = np.tanh(inputs)
        
        if training:
            self.inputs = inputs
            self.output = output

        return output

    def backward(self, grad_output: Array) -> Array:
        """
        Performs the backward pass of the Tanh activation function.

        Args:
            grad_output (numpy.ndarray): The gradient of the loss with respect to the output of the activation function.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input of the activation function,
                calculated using the derivative of the Tanh function.
        """
        dinputs = grad_output * (1 - self.output ** 2)
        return dinputs
    