import numpy as np

from ..initializers.he import he_normal

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
        self.weights = initializer(input_size, output_size, rng)
        self.biases = np.zeros((1, output_size))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the Dense layer.

        Args:
            inputs (numpy.ndarray): The input data to the layer.

        Returns:
            numpy.ndarray: The output of the layer after applying the linear transformation.
        """
        self.inputs = inputs
        return inputs @ self.weights + self.biases

    def backward(self, grad_output: np.ndarray):
        """
        Performs the backward pass of the Dense layer.


        Args:
            grad_output (numpy.ndarray): The gradient of the loss with respect to the output of the layer.
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
        dweights = self.inputs.T @ grad_output
        dbiases = np.sum(grad_output, axis=0, keepdims=True)
        dinputs = grad_output @ self.weights.T
        
        return dinputs, dweights, dbiases
