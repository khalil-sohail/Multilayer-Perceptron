import unittest
import numpy as np

from nn.losses.mse import MSE
from tests.utils.gradients import numerical_gradient_loss

class TestMSE(unittest.TestCase):
    def test_perfect_prediction_is_zero(self):
        loss = MSE()

        values = np.array([
            [1.0, 2.0, 3.0],
        ])

        self.assertEqual(
            loss.forward(values, values),
            0.0,
        )

    def test_known_value(self):
        loss = MSE()

        y_true = np.array([[1.0, 2.0]])
        y_pred = np.array([[2.0, 4.0]])

        self.assertAlmostEqual(
            loss.forward(y_pred, y_true),
            2.5,
        )

class TestMSEBackward(unittest.TestCase):
    def test_known_gradient(self):
        loss = MSE()

        y_pred = np.array([
            [2.0, 4.0],
        ])

        y_true = np.array([
            [1.0, 2.0],
        ])

        loss.forward(y_pred, y_true)
        gradient = loss.backward()

        # MSE:
        #
        # L = mean((pred - true)^2)
        #
        # dL/dpred =
        # 2 * (pred - true) / number_of_elements
        #
        # errors = [1, 2]
        # N = 2
        #
        # gradient = [1, 2]

        expected = np.array([
            [1.0, 2.0],
        ])

        np.testing.assert_allclose(
            gradient,
            expected,
        )

    def test_perfect_prediction_has_zero_gradient(self):
        loss = MSE()

        y = np.array([
            [1.0, 2.0, 3.0],
        ])

        loss.forward(y, y)
        gradient = loss.backward()

        np.testing.assert_allclose(
            gradient,
            np.zeros_like(y),
        )

    def test_numerical_gradient(self):
        y_pred = np.array([
            [0.2, 1.3, -0.7],
            [2.0, -1.0, 0.4],
        ])

        y_true = np.array([
            [0.0, 1.0, -1.0],
            [1.5, -0.5, 0.0],
        ])

        loss = MSE()

        loss.forward(y_pred, y_true)
        analytical = loss.backward()

        numerical = numerical_gradient_loss(
            MSE,
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
