import unittest

import numpy as np

from nn.layers.dense import Dense


class TestDense(unittest.TestCase):
    def test_output_shape(self):
        dense = Dense(input_size=3, output_size=4)

        inputs = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])

        output = dense.forward(inputs)

        self.assertEqual(output.shape, (2, 4))

    def test_weights_shape(self):
        dense = Dense(input_size=3, output_size=4)

        self.assertEqual(dense.weights.shape, (3, 4))

    def test_biases_shape(self):
        dense = Dense(input_size=3, output_size=4)

        self.assertEqual(dense.biases.shape, (1, 4))

    def test_forward_known_values(self):
        dense = Dense(input_size=2, output_size=2)

        dense.weights = np.array([
            [0.5, 1.0],
            [2.0, 3.0],
        ])

        dense.biases = np.array([
            [0.1, 0.2],
        ])

        inputs = np.array([
            [1.0, 2.0],
        ])

        expected = np.array([
            [4.6, 7.2],
        ])

        output = dense.forward(inputs)

        np.testing.assert_allclose(output, expected)


if __name__ == "__main__":
    unittest.main()
