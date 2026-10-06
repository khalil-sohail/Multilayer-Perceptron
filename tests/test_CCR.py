import unittest
import numpy as np

from nn import CategoricalCrossEntropy
from tests.utils.gradients import numerical_gradient_loss

class TestCategoricalCrossEntropy(unittest.TestCase):
    def test_good_prediction_has_low_loss(self):
        loss = CategoricalCrossEntropy()

        y_true = np.array([
            [1.0, 0.0],
        ])

        good = np.array([
            [0.99, 0.01],
        ])

        bad = np.array([
            [0.01, 0.99],
        ])

        self.assertLess(
            loss.forward(good, y_true),
            loss.forward(bad, y_true),
        )

    def test_perfect_prediction_is_near_zero(self):
        loss = CategoricalCrossEntropy()

        y_true = np.array([
            [1.0, 0.0],
        ])

        y_pred = np.array([
            [1.0, 0.0],
        ])

        self.assertAlmostEqual(
            loss.forward(y_pred, y_true),
            0.0,
            places=6,
        )

class TestCategoricalCrossEntropyBackward(unittest.TestCase):
    def test_known_gradient(self):
        loss = CategoricalCrossEntropy()

        y_pred = np.array([
            [0.7, 0.2, 0.1],
        ])

        y_true = np.array([
            [1.0, 0.0, 0.0],
        ])

        loss.forward(y_pred, y_true)
        gradient = loss.backward()

        # -y_true / y_pred
        # batch size = 1

        expected = np.array([
            [
                -1.0 / 0.7,
                0.0,
                0.0,
            ]
        ])

        np.testing.assert_allclose(
            gradient,
            expected,
        )

    def test_batch_normalization(self):
        loss = CategoricalCrossEntropy()

        y_pred = np.array([
            [0.8, 0.2],
            [0.3, 0.7],
        ])

        y_true = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        loss.forward(y_pred, y_true)
        gradient = loss.backward()

        expected = np.array([
            [
                -1.0 / 0.8 / 2.0,
                0.0,
            ],
            [
                0.0,
                -1.0 / 0.7 / 2.0,
            ],
        ])

        np.testing.assert_allclose(
            gradient,
            expected,
        )

    def test_gradient_is_finite_at_zero_probability(self):
        loss = CategoricalCrossEntropy()

        y_pred = np.array([
            [0.0, 1.0],
        ])

        y_true = np.array([
            [1.0, 0.0],
        ])

        loss.forward(y_pred, y_true)
        gradient = loss.backward()

        self.assertTrue(
            np.all(np.isfinite(gradient))
        )

    def test_numerical_gradient(self):
        y_pred = np.array([
            [0.6, 0.3, 0.1],
            [0.2, 0.5, 0.3],
        ])

        y_true = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        loss = CategoricalCrossEntropy()

        loss.forward(y_pred, y_true)
        analytical = loss.backward()

        numerical = numerical_gradient_loss(
            CategoricalCrossEntropy,
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
