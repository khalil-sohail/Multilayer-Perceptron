import unittest

import numpy as np

from nn import Linear
from nn import ReLU
from nn import Sigmoid
from nn import Softmax
from nn import Tanh
from nn import he_normal
from nn import random_normal
from nn import xavier_normal
from nn import Dense


def numerical_activation_gradient(
    activation,
    inputs,
    grad_output,
    epsilon=1e-5,
):
    gradient = np.zeros_like(inputs, dtype=float)

    for index in np.ndindex(inputs.shape):
        plus = inputs.copy()
        minus = inputs.copy()

        plus[index] += epsilon
        minus[index] -= epsilon

        out_plus = activation.forward(
            plus,
            training=False,
        )

        out_minus = activation.forward(
            minus,
            training=False,
        )

        loss_plus = np.sum(
            out_plus * grad_output
        )

        loss_minus = np.sum(
            out_minus * grad_output
        )

        gradient[index] = (
            loss_plus - loss_minus
        ) / (2.0 * epsilon)

    return gradient


class TestInitializers(unittest.TestCase):
    def test_random_shape(self):
        rng = np.random.default_rng(42)

        weights = random_normal(
            3,
            4,
            rng=rng,
        )

        self.assertEqual(
            weights.shape,
            (3, 4),
        )

    def test_random_reproducible(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        a = random_normal(
            3,
            4,
            rng=rng1,
        )

        b = random_normal(
            3,
            4,
            rng=rng2,
        )

        np.testing.assert_allclose(a, b)

    def test_xavier_formula(self):
        input_size = 3
        output_size = 4

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        actual = xavier_normal(
            input_size,
            output_size,
            rng=rng1,
        )

        std = np.sqrt(
            2.0 / (
                input_size + output_size
            )
        )

        expected = (
            rng2.standard_normal(
                (input_size, output_size)
            )
            * std
        )

        np.testing.assert_allclose(
            actual,
            expected,
        )

    def test_he_formula(self):
        input_size = 3
        output_size = 4

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        actual = he_normal(
            input_size,
            output_size,
            rng=rng1,
        )

        std = np.sqrt(
            2.0 / input_size
        )

        expected = (
            rng2.standard_normal(
                (input_size, output_size)
            )
            * std
        )

        np.testing.assert_allclose(
            actual,
            expected,
        )


class TestDense(unittest.TestCase):
    def test_shapes(self):
        dense = Dense(3, 4)

        self.assertEqual(
            dense.weights.shape,
            (3, 4),
        )

        self.assertEqual(
            dense.biases.shape,
            (1, 4),
        )

    def test_known_forward(self):
        dense = Dense(2, 2)

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

        actual = dense.forward(
            inputs,
            training=True,
        )

        np.testing.assert_allclose(
            actual,
            expected,
        )

    def test_known_backward(self):
        dense = Dense(2, 2)

        dense.weights = np.array([
            [0.5, -1.0],
            [2.0, 3.0],
        ])

        dense.biases = np.zeros((1, 2))

        inputs = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])

        grad_output = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])

        dense.forward(
            inputs,
            training=True,
        )

        grad_input = dense.backward(
            grad_output
        )

        expected_dw = np.array([
            [10.0, 14.0],
            [14.0, 20.0],
        ])

        expected_db = np.array([
            [4.0, 6.0],
        ])

        expected_dx = np.array([
            [-1.5, 8.0],
            [-2.5, 18.0],
        ])

        np.testing.assert_allclose(
            dense.grad_weights,
            expected_dw,
        )

        np.testing.assert_allclose(
            dense.grad_biases,
            expected_db,
        )

        np.testing.assert_allclose(
            grad_input,
            expected_dx,
        )


class TestLinear(unittest.TestCase):
    def test_forward(self):
        activation = Linear()

        x = np.array([
            [-2.0, 0.0, 3.0],
        ])

        np.testing.assert_allclose(
            activation.forward(x),
            x,
        )

    def test_backward(self):
        activation = Linear()

        x = np.array([[1.0, 2.0]])
        grad = np.array([[3.0, -4.0]])

        activation.forward(
            x,
            training=True,
        )

        np.testing.assert_allclose(
            activation.backward(grad),
            grad,
        )


class TestReLU(unittest.TestCase):
    def test_forward(self):
        activation = ReLU()

        x = np.array([
            [-2.0, 0.0, 3.0],
        ])

        expected = np.array([
            [0.0, 0.0, 3.0],
        ])

        np.testing.assert_allclose(
            activation.forward(x),
            expected,
        )

    def test_backward(self):
        activation = ReLU()

        x = np.array([
            [-2.0, 0.0, 3.0],
        ])

        grad = np.array([
            [10.0, 4.0, -2.0],
        ])

        activation.forward(
            x,
            training=True,
        )

        expected = np.array([
            [0.0, 0.0, -2.0],
        ])

        np.testing.assert_allclose(
            activation.backward(grad),
            expected,
        )

    def test_numerical_gradient(self):
        x = np.array([
            [-2.0, -0.5, 1.5, 3.0],
        ])

        grad = np.array([
            [1.0, -2.0, 0.5, 3.0],
        ])

        activation = ReLU()

        activation.forward(
            x,
            training=True,
        )

        analytical = activation.backward(
            grad
        )

        numerical = numerical_activation_gradient(
            ReLU(),
            x,
            grad,
        )

        np.testing.assert_allclose(
            analytical,
            numerical,
            rtol=1e-5,
            atol=1e-7,
        )


class TestSigmoid(unittest.TestCase):
    def test_forward_zero(self):
        activation = Sigmoid()

        result = activation.forward(
            np.array([[0.0]])
        )

        np.testing.assert_allclose(
            result,
            [[0.5]],
        )

    def test_backward_zero(self):
        activation = Sigmoid()

        x = np.array([[0.0]])

        activation.forward(
            x,
            training=True,
        )

        result = activation.backward(
            np.array([[1.0]])
        )

        np.testing.assert_allclose(
            result,
            [[0.25]],
        )

    def test_numerical_gradient(self):
        x = np.array([
            [-1.0, 0.2, 1.5],
        ])

        grad = np.array([
            [0.5, -1.0, 2.0],
        ])

        activation = Sigmoid()

        activation.forward(
            x,
            training=True,
        )

        analytical = activation.backward(
            grad
        )

        numerical = numerical_activation_gradient(
            Sigmoid(),
            x,
            grad,
        )

        np.testing.assert_allclose(
            analytical,
            numerical,
            rtol=1e-5,
            atol=1e-7,
        )


class TestTanh(unittest.TestCase):
    def test_forward_matches_numpy(self):
        activation = Tanh()

        x = np.array([
            [-2.0, 0.0, 2.0],
        ])

        np.testing.assert_allclose(
            activation.forward(x),
            np.tanh(x),
        )

    def test_backward_zero(self):
        activation = Tanh()

        x = np.array([[0.0]])

        activation.forward(
            x,
            training=True,
        )

        result = activation.backward(
            np.array([[1.0]])
        )

        np.testing.assert_allclose(
            result,
            [[1.0]],
        )


class TestSoftmax(unittest.TestCase):
    def test_rows_sum_to_one(self):
        activation = Softmax()

        x = np.array([
            [1.0, 2.0, 3.0],
            [5.0, 2.0, 1.0],
        ])

        output = activation.forward(x)

        np.testing.assert_allclose(
            output.sum(axis=1),
            np.ones(2),
        )

    def test_equal_logits(self):
        activation = Softmax()

        output = activation.forward(
            np.array([
                [1.0, 1.0, 1.0, 1.0],
            ])
        )

        np.testing.assert_allclose(
            output,
            [[0.25, 0.25, 0.25, 0.25]],
        )

    def test_large_values_are_stable(self):
        activation = Softmax()

        output = activation.forward(
            np.array([
                [1000.0, 1001.0, 1002.0],
            ])
        )

        self.assertTrue(
            np.all(np.isfinite(output))
        )

    def test_backward_numerically(self):
        x = np.array([
            [1.0, 2.0, 0.5],
            [-0.5, 1.5, 2.0],
        ])

        grad = np.array([
            [0.3, -1.2, 2.0],
            [1.0, 0.4, -0.7],
        ])

        activation = Softmax()

        activation.forward(
            x,
            training=True,
        )

        analytical = activation.backward(
            grad
        )

        numerical = numerical_activation_gradient(
            Softmax(),
            x,
            grad,
        )

        np.testing.assert_allclose(
            analytical,
            numerical,
            rtol=1e-5,
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
