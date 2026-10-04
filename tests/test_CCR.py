import unittest
import numpy as np

from nn.losses.categorical_cross_entropy import CategoricalCrossEntropy

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

if __name__ == "__main__":
    unittest.main()
