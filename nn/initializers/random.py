import numpy as np


def random_normal(input_size: int, output_size: int, scale: float = 0.2, rng=None):
    """
    Generates a weight matrix with random values drawn from a normal distribution.
    
    Equation:
        weights ~ N(0, scale^2)
    
    Args:
        input_size (int): The number of input units.
        output_size (int): The number of output units.
        scale (float): The standard deviation of the normal distribution.
        rng (numpy.random.Generator): The random number generator to use.

    Returns:
        numpy.ndarray: A weight matrix with random values drawn from a normal distribution.
    """
    if rng is None:
        rng = np.random.default_rng()

    return rng.standard_normal((input_size, output_size)) * scale