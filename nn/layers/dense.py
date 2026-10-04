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
        return inputs @ self.weights + self.biases
