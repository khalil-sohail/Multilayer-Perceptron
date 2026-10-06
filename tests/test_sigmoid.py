import unittest

import numpy as np

from nn import Sigmoid
from tests.utils.gradients import numerical_gradient

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


class TestSigmoidBackward(unittest.TestCase):
    def test_known_derivative_at_zero(self):
        activation = Sigmoid()

        inputs = np.array([
            [0.0],
        ])

        grad_output = np.array([
            [1.0],
        ])

        activation.forward(inputs)
        grad_input = activation.backward(grad_output)

        # sigmoid(0) = 0.5
        #
        # sigmoid'(0)
        # = 0.5 * (1 - 0.5)
        # = 0.25
        expected = np.array([
            [0.25],
        ])

        np.testing.assert_allclose(
            grad_input,
            expected,
        )

    def test_incoming_gradient_is_applied(self):
        activation = Sigmoid()

        inputs = np.array([
            [0.0],
        ])

        grad_output = np.array([
            [4.0],
        ])

        activation.forward(inputs)
        grad_input = activation.backward(grad_output)

        # 4 * 0.25
        expected = np.array([
            [1.0],
        ])

        np.testing.assert_allclose(
            grad_input,
            expected,
        )

    def test_numerical_gradient(self):
        activation = Sigmoid()

        inputs = np.array([
            [-1.0, 0.2, 1.5],
            [2.0, -0.7, 0.4],
        ])

        grad_output = np.array([
            [0.5, -1.0, 2.0],
            [1.5, 0.3, -0.8],
        ])

        activation.forward(inputs)
        analytical = activation.backward(grad_output)

        numerical = numerical_gradient(
            Sigmoid(),
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
