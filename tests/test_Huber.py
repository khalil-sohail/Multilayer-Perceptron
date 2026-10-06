import unittest
import numpy as np

from nn.losses.huber import Huber
from tests.utils.gradients import numerical_gradient_loss

class TestHuber(unittest.TestCase):
    def test_zero_error_is_zero(self):
        loss = Huber()

        values = np.array([[1.0, 2.0]])

        self.assertEqual(
            loss.forward(values, values),
            0.0,
        )

    def test_small_error_uses_quadratic_region(self):
        loss = Huber(delta=1.0)

        y_true = np.array([[0.0]])
        y_pred = np.array([[0.5]])

        self.assertAlmostEqual(
            loss.forward(y_pred, y_true),
            0.125,
        )

class TestHuberBackward(unittest.TestCase):
    def test_small_error_uses_linear_gradient(self):
        loss = Huber(delta=1.0)

        y_pred = np.array([
            [0.5],
        ])

        y_true = np.array([
            [0.0],
        ])

        loss.forward(y_pred, y_true)
        gradient = loss.backward()

        # |error| <= delta
        #
        # gradient = error = 0.5

        expected = np.array([
            [0.5],
        ])

        np.testing.assert_allclose(
            gradient,
            expected,
        )

    def test_large_positive_error_is_clipped_to_delta(self):
        loss = Huber(delta=1.0)

        y_pred = np.array([
            [5.0],
        ])

        y_true = np.array([
            [0.0],
        ])

        loss.forward(y_pred, y_true)
        gradient = loss.backward()

        expected = np.array([
            [1.0],
        ])

        np.testing.assert_allclose(
            gradient,
            expected,
        )

    def test_large_negative_error_is_negative_delta(self):
        loss = Huber(delta=1.0)

        y_pred = np.array([
            [-5.0],
        ])

        y_true = np.array([
            [0.0],
        ])

        loss.forward(y_pred, y_true)
        gradient = loss.backward()

        expected = np.array([
            [-1.0],
        ])

        np.testing.assert_allclose(
            gradient,
            expected,
        )

    def test_numerical_gradient(self):
        # Don't use error exactly equal to delta because
        # that's the boundary between Huber's two regions.
        y_pred = np.array([
            [0.2, 2.0],
            [-0.5, -3.0],
        ])

        y_true = np.zeros_like(y_pred)

        loss = Huber(delta=1.0)

        loss.forward(y_pred, y_true)
        analytical = loss.backward()

        numerical = numerical_gradient_loss(
            lambda: Huber(delta=1.0),
            y_pred,
            y_true,
        )

        np.testing.assert_allclose(
            analytical,
            numerical,
            rtol=1e-5,
            atol=1e-7,
        )

if __name__ == "__main__":
    unittest.main()
