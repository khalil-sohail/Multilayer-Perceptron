import unittest

import numpy as np

from nn.activations.linear import Linear


class TestLinear(unittest.TestCase):
    def test_linear_returns_input_unchanged(self):
        activation = Linear()

        inputs = np.array([
            [-2.0, 0.0, 3.0],
            [5.0, -1.0, 8.0],
        ])

        output = activation.forward(inputs)

        np.testing.assert_allclose(output, inputs)


if __name__ == "__main__":
    unittest.main()
