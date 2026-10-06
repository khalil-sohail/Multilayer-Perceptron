import unittest

import numpy as np

from tests.utils.gradients import numerical_gradient
from nn import Linear


class TestLinear(unittest.TestCase):
    def test_linear_returns_input_unchanged(self):
        activation = Linear()

        inputs = np.array([
            [-2.0, 0.0, 3.0],
            [5.0, -1.0, 8.0],
        ])

        output = activation.forward(inputs)

        np.testing.assert_allclose(output, inputs)


class TestLinearBackward(unittest.TestCase):
    def test_gradient_passes_through_unchanged(self):
        activation = Linear()

        inputs = np.array([
            [1.0, -2.0, 3.0],
        ])

        grad_output = np.array([
            [4.0, 5.0, -6.0],
        ])

        activation.forward(inputs)
        grad_input = activation.backward(grad_output)

        expected = grad_output

        np.testing.assert_allclose(
            grad_input,
            expected,
        )

    def test_numerical_gradient(self):
        activation = Linear()

        inputs = np.array([
            [1.2, -0.7, 2.5],
        ])

        grad_output = np.array([
            [0.3, -1.2, 2.0],
        ])

        activation.forward(inputs)
        analytical = activation.backward(grad_output)

        numerical = numerical_gradient(
            Linear(),
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
