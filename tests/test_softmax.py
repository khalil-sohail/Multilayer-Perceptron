import unittest

import numpy as np

from nn.activations.softmax import Softmax


class TestSoftmax(unittest.TestCase):
    def test_each_row_sums_to_one(self):
        activation = Softmax()

        inputs = np.array([
            [1.0, 2.0, 3.0],
            [5.0, 2.0, 1.0],
        ])

        output = activation.forward(inputs)

        np.testing.assert_allclose(
            output.sum(axis=1),
            np.ones(2),
        )

    def test_output_is_between_zero_and_one(self):
        activation = Softmax()

        inputs = np.array([
            [1.0, 2.0, 3.0],
            [-5.0, 0.0, 5.0],
        ])

        output = activation.forward(inputs)

        self.assertTrue(np.all(output >= 0.0))
        self.assertTrue(np.all(output <= 1.0))

    def test_equal_inputs_produce_equal_probabilities(self):
        activation = Softmax()

        inputs = np.array([
            [1.0, 1.0, 1.0, 1.0],
        ])

        expected = np.array([
            [0.25, 0.25, 0.25, 0.25],
        ])

        output = activation.forward(inputs)

        np.testing.assert_allclose(output, expected)

    def test_large_values_are_numerically_stable(self):
        activation = Softmax()

        inputs = np.array([
            [1000.0, 1001.0, 1002.0],
        ])

        output = activation.forward(inputs)

        self.assertTrue(np.all(np.isfinite(output)))

        np.testing.assert_allclose(
            output.sum(axis=1),
            [1.0],
        )

    def test_adding_constant_does_not_change_result(self):
        activation = Softmax()

        inputs = np.array([
            [1.0, 2.0, 3.0],
        ])

        shifted_inputs = inputs + 1000.0

        output1 = activation.forward(inputs)
        output2 = activation.forward(shifted_inputs)

        np.testing.assert_allclose(
            output1,
            output2,
        )


if __name__ == "__main__":
    unittest.main()
