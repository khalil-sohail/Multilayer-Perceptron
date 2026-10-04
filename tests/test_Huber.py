import unittest
import numpy as np

from nn.losses.huber import Huber

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

if __name__ == "__main__":
    unittest.main()
