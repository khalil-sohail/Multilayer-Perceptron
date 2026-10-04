import numpy as np

def he_normal(input_size: int, output_size: int, rng=None):
    """
    Generates a weight matrix with values drawn from a normal distribution using He initialization. (Best with ReLU-family activation functions)

    Equation:
        sigma = sqrt(2 / input_size)
        weights ~ N(0, sigma^2)
    
    Args:
        input_size (int): The number of input units.
        output_size (int): The number of output units.
        rng (numpy.random.Generator, optional): A random number generator. If None, a new default generator is used.
    Returns:
        numpy.ndarray: A weight matrix with values drawn from a normal distribution using He initialization.
    """
    if rng is None:
        rng = np.random.default_rng()

    sigma = np.sqrt(2.0 / input_size)
    return (rng.standard_normal((input_size, output_size)) * sigma)
