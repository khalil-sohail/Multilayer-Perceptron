import unittest

import numpy as np

from nn.losses.binary_cross_entropy import BinaryCrossEntropy

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

if __name__ == "__main__":
    unittest.main()