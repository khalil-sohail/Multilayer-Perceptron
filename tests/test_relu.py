import unittest

import numpy as np

from nn.activations.relu import ReLU


class TestReLU(unittest.TestCase):
    def test_negative_values_become_zero(self):
        activation = ReLU()

        inputs = np.array([
            [-3.0, -1.0, 0.0, 2.0, 5.0],
        ])

        expected = np.array([
            [0.0, 0.0, 0.0, 2.0, 5.0],
        ])

        output = activation.forward(inputs)

        np.testing.assert_allclose(output, expected)

    def test_output_shape_is_preserved(self):
        activation = ReLU()

        inputs = np.random.default_rng(42).standard_normal((8, 5))

        output = activation.forward(inputs)

        self.assertEqual(output.shape, inputs.shape)

    def test_output_is_never_negative(self):
        activation = ReLU()

        inputs = np.array([
            [-100.0, -1.0, 0.0, 1.0, 100.0],
        ])

        output = activation.forward(inputs)

        self.assertTrue(np.all(output >= 0.0))


if __name__ == "__main__":
    unittest.main()
