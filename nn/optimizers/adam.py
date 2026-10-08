import numpy as np

from nn.interfaces.Optimizer import TrainableLayer

class Adam:
    """
    Adam optimizer.
    This class implements the Adam optimization algorithm.
    It combines the advantages of two other extensions of stochastic gradient descent,
        namely Adaptive Gradient Algorithm (AdaGrad) and Root Mean Square Propagation (RMSProp).
    
    Equation:
        m = beta1 * m + (1 - beta1) * dW
        v = beta2 * v + (1 - beta2) * (dW ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        W = W - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
        b = b - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
        Where:
            W: Weights of the layer
            b: Biases of the layer
            dW: Gradient of the loss with respect to weights
            db: Gradient of the loss with respect to biases
            learning_rate: Step size for each iteration of the optimization
            beta1: Exponential decay rate for the first moment estimates
            beta2: Exponential decay rate for the second moment estimates
            epsilon: Small constant for numerical stability
    
    Attributes:
        learning_rate (float): The step size for each iteration of the optimization.
        beta1 (float): Exponential decay rate for the first moment estimates.
        beta2 (float): Exponential decay rate for the second moment estimates.
        epsilon (float): Small constant for numerical stability.
    """
    def __init__(
        self,
        learning_rate: float = 0.0005,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ) -> None:
        """
        Initializes the Adam optimizer with specified hyperparameters.
        
        Args:
            learning_rate: The step size for each iteration of the optimization.
            beta1: Exponential decay rate for the first moment estimates.
            beta2: Exponential decay rate for the second moment estimates.
            epsilon: Small constant for numerical stability.
        
        Raises:
            ValueError: If any of the hyperparameters are out of their valid ranges.
        """
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

        if not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1)")
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self._states = {}

    def step(self, layer: TrainableLayer) -> None:
        """
        Updates the parameters of the given layer using the gradients computed during backpropagation.
        """
        state = self._states.get(layer)

        if state is None:
            state = {
                "t": 0,
                "m_weights": np.zeros_like(layer.weights),
                "v_weights": np.zeros_like(layer.weights),
                "m_biases": np.zeros_like(layer.biases),
                "v_biases": np.zeros_like(layer.biases)
            }

            self._states[layer] = state

        state["t"] += 1
        t = state["t"]

        state["m_weights"] = (
            self.beta1
            * state["m_weights"]
            + (1.0 - self.beta1)
            * layer.grad_weights
        )

        state["m_biases"] = (
            self.beta1
            * state["m_biases"]
            + (1.0 - self.beta1)
            * layer.grad_biases
        )

        state["v_weights"] = (
            self.beta2
            * state["v_weights"]
            + (1.0 - self.beta2)
            * np.square(
                layer.grad_weights
            )
        )

        state["v_biases"] = (
            self.beta2
            * state["v_biases"]
            + (1.0 - self.beta2)
            * np.square(
                layer.grad_biases
            )
        )

        m_weights_hat = (
            state["m_weights"] / (
                1.0 - self.beta1 ** t
            )
        )

        m_biases_hat = (
            state["m_biases"] / (
                1.0 - self.beta1 ** t
            )
        )

        v_weights_hat = (
            state["v_weights"] / (
                1.0 - self.beta2 ** t
            )
        )

        v_biases_hat = (
            state["v_biases"] / (
                1.0 - self.beta2 ** t
            )
        )

        layer.weights -= (
            self.learning_rate
            * m_weights_hat / (
                np.sqrt(v_weights_hat)
                + self.epsilon
            )
        )

        layer.biases -= (
            self.learning_rate
            * m_biases_hat / (
                np.sqrt(v_biases_hat)
                + self.epsilon
            )
        )