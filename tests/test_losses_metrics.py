import unittest

import numpy as np

from nn import BinaryCrossEntropy
from nn import CategoricalCrossEntropy
from nn import Huber
from nn import MSE

from nn import accuracy
from nn import f1_score
from nn import precision
from nn import recall


def numerical_loss_gradient(
    factory,
    y_pred,
    y_true,
    epsilon=1e-5,
):
    gradient = np.zeros_like(
        y_pred,
        dtype=float,
    )

    for index in np.ndindex(y_pred.shape):
        plus = y_pred.copy()
        minus = y_pred.copy()

        plus[index] += epsilon
        minus[index] -= epsilon

        loss_plus = factory().forward(
            plus,
            y_true,
            training=False,
        )

        loss_minus = factory().forward(
            minus,
            y_true,
            training=False,
        )

        gradient[index] = (
            loss_plus - loss_minus
        ) / (2.0 * epsilon)

    return gradient


class TestMSE(unittest.TestCase):
    def test_forward_known_value(self):
        loss = MSE()

        y_pred = np.array([
            [2.0, 4.0],
        ])

        y_true = np.array([
            [1.0, 2.0],
        ])

        self.assertAlmostEqual(
            loss.forward(
                y_pred,
                y_true,
            ),
            2.5,
        )

    def test_backward_numerically(self):
        y_pred = np.array([
            [0.2, 1.3, -0.7],
            [2.0, -1.0, 0.4],
        ])

        y_true = np.array([
            [0.0, 1.0, -1.0],
            [1.5, -0.5, 0.0],
        ])

        loss = MSE()

        loss.forward(
            y_pred,
            y_true,
            training=True,
        )

        analytical = loss.backward()

        numerical = numerical_loss_gradient(
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


class TestBinaryCrossEntropy(unittest.TestCase):
    def test_good_prediction_lower_loss(self):
        loss = BinaryCrossEntropy()

        y_true = np.array([[1.0]])

        good = loss.forward(
            np.array([[0.99]]),
            y_true,
        )

        bad = loss.forward(
            np.array([[0.01]]),
            y_true,
        )

        self.assertLess(good, bad)

    def test_boundary_is_finite(self):
        loss = BinaryCrossEntropy()

        result = loss.forward(
            np.array([
                [0.0],
                [1.0],
            ]),
            np.array([
                [1.0],
                [0.0],
            ]),
        )

        self.assertTrue(
            np.isfinite(result)
        )

    def test_backward_numerically(self):
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

        loss.forward(
            y_pred,
            y_true,
            training=True,
        )

        analytical = loss.backward()

        numerical = numerical_loss_gradient(
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


class TestCategoricalCrossEntropy(unittest.TestCase):
    def test_good_prediction_lower_loss(self):
        loss = CategoricalCrossEntropy()

        y_true = np.array([
            [1.0, 0.0],
        ])

        good = loss.forward(
            np.array([
                [0.99, 0.01],
            ]),
            y_true,
        )

        bad = loss.forward(
            np.array([
                [0.01, 0.99],
            ]),
            y_true,
        )

        self.assertLess(good, bad)

    def test_backward_numerically(self):
        y_pred = np.array([
            [0.6, 0.3, 0.1],
            [0.2, 0.5, 0.3],
        ])

        y_true = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        loss = CategoricalCrossEntropy()

        loss.forward(
            y_pred,
            y_true,
            training=True,
        )

        analytical = loss.backward()

        numerical = numerical_loss_gradient(
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


class TestHuber(unittest.TestCase):
    def test_zero_error(self):
        loss = Huber()

        y = np.array([
            [1.0, 2.0],
        ])

        self.assertEqual(
            loss.forward(y, y),
            0.0,
        )

    def test_small_error(self):
        loss = Huber(delta=1.0)

        result = loss.forward(
            np.array([[0.5]]),
            np.array([[0.0]]),
        )

        self.assertAlmostEqual(
            result,
            0.125,
        )

    def test_backward_numerically(self):
        y_pred = np.array([
            [0.2, 2.0],
            [-0.5, -3.0],
        ])

        y_true = np.zeros_like(y_pred)

        loss = Huber(delta=1.0)

        loss.forward(
            y_pred,
            y_true,
            training=True,
        )

        analytical = loss.backward()

        numerical = numerical_loss_gradient(
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


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([
            1, 0, 1, 1
        ])

        self.y_pred = np.array([
            1, 0, 0, 1
        ])

    def test_accuracy(self):
        self.assertAlmostEqual(
            accuracy(
                self.y_true,
                self.y_pred,
            ),
            0.75,
        )

    def test_precision(self):
        self.assertAlmostEqual(
            precision(
                self.y_true,
                self.y_pred,
            ),
            1.0,
        )

    def test_recall(self):
        self.assertAlmostEqual(
            recall(
                self.y_true,
                self.y_pred,
            ),
            2.0 / 3.0,
        )

    def test_f1(self):
        self.assertAlmostEqual(
            f1_score(
                self.y_true,
                self.y_pred,
            ),
            0.8,
        )

    def test_precision_division_by_zero(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])

        self.assertEqual(
            precision(y_true, y_pred),
            0.0,
        )

    def test_recall_division_by_zero(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])

        self.assertEqual(
            recall(y_true, y_pred),
            0.0,
        )

    def test_f1_division_by_zero(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])

        self.assertEqual(
            f1_score(y_true, y_pred),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
