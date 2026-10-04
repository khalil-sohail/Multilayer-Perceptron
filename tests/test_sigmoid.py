import unittest

import numpy as np

from nn.activations.sigmoid import Sigmoid


class TestSigmoid(unittest.TestCase):
    def test_zero_returns_half(self):
        activation = Sigmoid()

        output = activation.forward(
            np.array([[0.0]])
        )

        np.testing.assert_allclose(
            output,
            [[0.5]],
        )

    def test_output_is_between_zero_and_one(self):
        activation = Sigmoid()

        inputs = np.array([
            [-10.0, -1.0, 0.0, 1.0, 10.0],
        ])

        output = activation.forward(inputs)

        self.assertTrue(np.all(output > 0.0))
        self.assertTrue(np.all(output < 1.0))

    def test_known_values(self):
        activation = Sigmoid()

        inputs = np.array([
            [-1.0, 0.0, 1.0],
        ])

        expected = np.array([
            [
                1.0 / (1.0 + np.exp(1.0)),
                0.5,
                1.0 / (1.0 + np.exp(-1.0)),
            ]
        ])

        output = activation.forward(inputs)

        np.testing.assert_allclose(output, expected)


if __name__ == "__main__":
    unittest.main()
