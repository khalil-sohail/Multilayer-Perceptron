from nn.interfaces.Optimizer import TrainableLayer

class Adam:
    """
    Adam optimizer.
    """

    def __init__(self, learning_rate: float=0.001, beta1: float=0.9, beta2: float=0.999, epsilon: float=1e-8):
        """
        """
        ...

    def step(self, layer: TrainableLayer):
        """
        """
        ...