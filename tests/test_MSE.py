import unittest
import numpy as np

from nn.losses.mse import MSE

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

if __name__ == "__main__":
    unittest.main()
