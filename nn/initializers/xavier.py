import numpy as np


def xavier_normal(input_size: int, output_size: int, rng=None):
    """
    Generates a weight matrix with values drawn from a normal distribution. (Best with Sigmoid or Tanh activation functions)
    Var(W) = sqrt(2 / (input_size + output_size))
    
    Args:
        input_size (int): The number of input units.
        output_size (int): The number of output units.
        rng (numpy.random.Generator): The random number generator to use.

    Returns:
        numpy.ndarray: A weight matrix with values drawn from a normal distribution.
    """
    if rng is None:
        rng = np.random.default_rng()

    var = np.sqrt(2.0 / (input_size + output_size))
    return (rng.standard_normal((input_size, output_size)) * var)