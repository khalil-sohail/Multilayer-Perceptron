import numpy as np

class Sigmoid:
    """
    A Sigmoid activation function for the MLP.
    This activation function applies the Sigmoid operation to the input data, which maps the input values to a range between 0 and 1.

    Equation:
        sigmoid(x) = 1 / (1 + exp(-x))
        sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))

    Attributes:
        inputs (numpy.ndarray): The input data to the activation function.
        output (numpy.ndarray): The output of the activation function after applying the Sigmoid operation.
    """

    def forward(self, inputs):
        """
        Performs the forward pass of the Sigmoid activation function.

        Args:
            inputs (numpy.ndarray): The input data to the activation function.

        Returns:
            numpy.ndarray: The output of the activation function, where the input values are mapped to a range between 0 and 1.
        """
        self.inputs = inputs
        self.output = 1.0 / (1.0 + np.exp(-inputs))
        return self.output

    def backward(self, grad_output):
        """
        Performs the backward pass of the Sigmoid activation function.

        Args:
            grad_output (numpy.ndarray): The gradient of the loss with respect to the output of the activation function.

        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input of the activation function,
                calculated using the derivative of the Sigmoid function.
        """
        dinputs = grad_output * (self.output * (1 - self.output))
        return dinputs
    