import unittest

import numpy as np

from nn.activations.tanh import Tanh


class TestTanh(unittest.TestCase):
    def test_zero_returns_zero(self):
        activation = Tanh()

        output = activation.forward(
            np.array([[0.0]])
        )

        np.testing.assert_allclose(
            output,
            [[0.0]],
        )

    def test_matches_numpy_tanh(self):
        activation = Tanh()

        inputs = np.array([
            [-2.0, -1.0, 0.0, 1.0, 2.0],
        ])

        output = activation.forward(inputs)

        np.testing.assert_allclose(
            output,
            np.tanh(inputs),
        )

    def test_output_range(self):
        activation = Tanh()

        inputs = np.array([
            [-100.0, -1.0, 0.0, 1.0, 100.0],
        ])

        output = activation.forward(inputs)

        self.assertTrue(np.all(output >= -1.0))
        self.assertTrue(np.all(output <= 1.0))


if __name__ == "__main__":
    unittest.main()
