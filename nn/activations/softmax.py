import numpy as np

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

    def forward(self, inputs):
        """
        Performs the forward pass of the Softmax activation function.
        Args:
            inputs (numpy.ndarray): The input data to the activation function.
        Returns:
            numpy.ndarray: The output of the activation function, where the input values are converted into probabilities that sum to 1.
        """
        self.inputs = inputs

        shifted = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        return self.output
