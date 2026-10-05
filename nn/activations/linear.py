import numpy as np

class Linear:
    """
    A linear activation function for neural networks.
    This activation function applies a linear transformation to the input data.

    Equation:
        f(x) = x
        f'(x) = 1

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

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass of the Linear activation function.

        Args:
            grad_output (numpy.ndarray): The gradient of the loss with respect to the output of the activation function.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input of the activation function,
                which is the same as grad_output since the derivative of a linear function is 1.
        """
        return grad_output
