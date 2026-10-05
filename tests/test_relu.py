import unittest

import numpy as np

from nn.activations.relu import ReLU
from tests.numerical_gradient import numerical_gradient

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


class TestReLUBackward(unittest.TestCase):
    def test_negative_inputs_block_gradient(self):
        activation = ReLU()

        inputs = np.array([
            [-2.0, 0.0, 3.0],
        ])

        grad_output = np.array([
            [10.0, 4.0, -2.0],
        ])

        activation.forward(inputs)
        grad_input = activation.backward(grad_output)

        expected = np.array([
            [0.0, 0.0, -2.0],
        ])

        np.testing.assert_allclose(
            grad_input,
            expected,
        )

    def test_positive_inputs_pass_gradient(self):
        activation = ReLU()

        inputs = np.array([
            [1.0, 2.0, 3.0],
        ])

        grad_output = np.array([
            [5.0, -3.0, 7.0],
        ])

        activation.forward(inputs)
        grad_input = activation.backward(grad_output)

        np.testing.assert_allclose(
            grad_input,
            grad_output,
        )

    def test_numerical_gradient_away_from_zero(self):
        activation = ReLU()

        inputs = np.array([
            [-2.0, -0.5, 1.5, 3.0],
        ])

        grad_output = np.array([
            [1.0, -2.0, 0.5, 3.0],
        ])

        activation.forward(inputs)
        analytical = activation.backward(grad_output)

        numerical = numerical_gradient(
            ReLU(),
            inputs,
            grad_output,
        )

        np.testing.assert_allclose(
            analytical,
            numerical,
            rtol=1e-5,
            atol=1e-7,
        )

if __name__ == "__main__":
    unittest.main()
