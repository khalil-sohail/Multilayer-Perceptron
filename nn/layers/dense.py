from ..initializers.he import he_normal
from numpy.typing import NDArray

import numpy as np

Array = NDArray[np.float64]

class Dense:
    """
    A fully connected layer in a neural network.
    This layer performs a linear transformation on the input data, followed by an optional activation function.

    Attributes:
        weights (numpy.ndarray): The weight matrix of the layer.
        biases (numpy.ndarray): The bias vector of the layer.
    """
    def __init__(self, input_size: int, output_size: int, initializer=he_normal, rng=None,):
        """
        Initializes the Dense layer with random weights and zero biases.
        
        Args:
            input_size (int): The number of input units.
            output_size (int): The number of output units.
            initializer: The function to use for initializing the weights.
            rng: The random number generator to use.
        """
        if initializer.__name__ == "random_normal":
            self.weights = initializer(input_size, output_size, 0.2, rng)
        else:
            self.weights = initializer(input_size, output_size, rng)
        self.biases = np.zeros((1, output_size))
        self.trainable = True

    def forward(self, inputs: Array, training: bool=True) -> Array:
        """
        Performs the forward pass of the Dense layer.

        Args:
            inputs (numpy.ndarray): The input data to the layer.
            training (bool): Whether the layer is in training mode.

        Returns:
            numpy.ndarray: The output of the layer after applying the linear transformation.
        """
        if training:
            self.inputs = inputs

        return inputs @ self.weights + self.biases

    def backward(self, grad_output: Array) -> Array:
        """
        Performs the backward pass of the Dense layer.


        Args:
            grad_output (numpy.ndarray): The gradient of the loss with respect to the output of the layer.
            training (bool): Whether the layer is in training mode.
            inputs (numpy.ndarray): The input data to the layer.

        Equation:
            chain rule: dL/dX = dL/dY * dY/dX
            where:
            - dL/dX is the gradient of the loss with respect to the input of the layer.
            - dL/dY is the gradient of the loss with respect to the output of the layer.
            - dY/dX is the gradient of the output of the layer with respect to the input of the layer.

        Returns:
            tuple: A tuple containing:
                - dinputs (numpy.ndarray): The gradient of the loss with respect to the input of the layer.
                - dweights (numpy.ndarray): The gradient of the loss with respect to the weights of the layer.
                - dbiases (numpy.ndarray): The gradient of the loss with respect to the biases of the layer.
        """
        self.grad_weights = self.inputs.T @ grad_output
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        
        dinputs = grad_output @ self.weights.T
        return dinputs
