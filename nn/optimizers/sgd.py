from nn.interfaces.Optimizer import TrainableLayer

class SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer.
    This class implements the stochastic gradient descent optimization algorithm.
    It updates the parameters of a layer by moving in the direction of the negative gradient.
    
    Equation:
        W = W - learning_rate * dW
        b = b - learning_rate * db
        Where:
            W: Weights of the layer
            b: Biases of the layer
            dW: Gradient of the loss with respect to weights
            db: Gradient of the loss with respect to biases
            learning_rate: Step size for each iteration of the optimization

    Attributes:
        learning_rate (float): The step size for each iteration of the optimization.
    """

    def __init__(self, learning_rate: float=0.01):
        """
        Initializes the SGD optimizer with a specified learning rate.
        
        Args:
            learning_rate (float): The step size for each iteration of the optimization.
        """
        self.learning_rate = learning_rate

    def step(self, layer: TrainableLayer):
        """
        Updates the parameters of the given layer using the gradients computed during backpropagation.
        """

        layer.weights -= (self.learning_rate * layer.grad_weights)
        layer.biases -= (self.learning_rate * layer.grad_biases)

