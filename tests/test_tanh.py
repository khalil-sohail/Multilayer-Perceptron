import unittest

import numpy as np

from nn.activations.tanh import Tanh
from tests.utils.gradients import numerical_gradient

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

class TestTanhBackward(unittest.TestCase):
    def test_known_derivative_at_zero(self):
        activation = Tanh()

        inputs = np.array([
            [0.0],
        ])

        grad_output = np.array([
            [1.0],
        ])

        activation.forward(inputs)
        grad_input = activation.backward(grad_output)

        # tanh(0) = 0
        #
        # tanh'(0)
        # = 1 - 0²
        # = 1
        expected = np.array([
            [1.0],
        ])

        np.testing.assert_allclose(
            grad_input,
            expected,
        )

    def test_incoming_gradient_is_applied(self):
        activation = Tanh()

        inputs = np.array([
            [0.0],
        ])

        grad_output = np.array([
            [3.5],
        ])

        activation.forward(inputs)
        grad_input = activation.backward(grad_output)

        expected = np.array([
            [3.5],
        ])

        np.testing.assert_allclose(
            grad_input,
            expected,
        )

    def test_numerical_gradient(self):
        activation = Tanh()

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
            Tanh(),
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
