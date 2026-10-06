import unittest

import numpy as np

from nn.layers.dense import Dense
from nn.optimizers.sgd import SGD


class TestSGD(unittest.TestCase):
    def test_parameter_update(self):
        dense = Dense(2, 2)

        dense.weights = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])

        dense.biases = np.array([
            [0.5, -0.5],
        ])

        dense.grad_weights = np.array([
            [0.1, 0.2],
            [0.3, 0.4],
        ])

        dense.grad_biases = np.array([
            [0.5, -0.25],
        ])

        optimizer = SGD(
            learning_rate=0.1
        )

        optimizer.step(dense)

        expected_weights = np.array([
            [0.99, 1.98],
            [2.97, 3.96],
        ])

        expected_biases = np.array([
            [0.45, -0.475],
        ])

        np.testing.assert_allclose(
            dense.weights,
            expected_weights,
        )

        np.testing.assert_allclose(
            dense.biases,
            expected_biases,
        )


if __name__ == "__main__":
    unittest.main()