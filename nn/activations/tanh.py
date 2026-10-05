import numpy as np

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

    def forward(self, inputs):
        """
        Performs the forward pass of the Tanh activation function.

        Args:
            inputs (numpy.ndarray): The input data to the activation function.

        Returns:
            numpy.ndarray: The output of the activation function, where the input values are mapped to a range between -1 and 1.
        """
        self.inputs = inputs
        self.output = np.tanh(inputs)
        return self.output

    def backward(self, grad_output):
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
    