import unittest

import numpy as np

from nn import BinaryCrossEntropy
from tests.utils.gradients import numerical_gradient_loss

class TestBinaryCrossEntropy(unittest.TestCase):
    def test_good_prediction_has_low_loss(self):
        loss = BinaryCrossEntropy()

        y_true = np.array([[1.0]])

        good = np.array([[0.99]])
        bad = np.array([[0.01]])

        self.assertLess(
            loss.forward(good, y_true),
            loss.forward(bad, y_true),
        )

    def test_no_inf_for_zero_probability(self):
        loss = BinaryCrossEntropy()

        result = loss.forward(
            np.array([[0.0]]),
            np.array([[1.0]]),
        )

        self.assertTrue(np.isfinite(result))

class TestBinaryCrossEntropyBackward(unittest.TestCase):
    def test_known_gradient(self):
        loss = BinaryCrossEntropy()

        y_pred = np.array([
            [0.8],
            [0.3],
        ])

        y_true = np.array([
            [1.0],
            [0.0],
        ])

        loss.forward(y_pred, y_true)
        gradient = loss.backward()

        # BCE derivative:
        #
        # -(y/p - (1-y)/(1-p)) / N
        #
        # N = 2

        expected = np.array([
            [
                -(1.0 / 0.8) / 2.0
            ],
            [
                (1.0 / (1.0 - 0.3)) / 2.0
            ],
        ])

        np.testing.assert_allclose(
            gradient,
            expected,
        )

    def test_gradient_is_finite_near_boundaries(self):
        loss = BinaryCrossEntropy()

        y_pred = np.array([
            [0.0],
            [1.0],
        ])

        y_true = np.array([
            [1.0],
            [0.0],
        ])

        loss.forward(y_pred, y_true)
        gradient = loss.backward()

        self.assertTrue(
            np.all(np.isfinite(gradient))
        )

    def test_numerical_gradient(self):
        # Stay away from exactly 0 and 1 because clipping
        # introduces boundaries there.
        y_pred = np.array([
            [0.2],
            [0.7],
            [0.4],
        ])

        y_true = np.array([
            [0.0],
            [1.0],
            [1.0],
        ])

        loss = BinaryCrossEntropy()

        loss.forward(y_pred, y_true)
        analytical = loss.backward()

        numerical = numerical_gradient_loss(
            BinaryCrossEntropy,
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