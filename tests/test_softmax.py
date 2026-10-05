import unittest

import numpy as np

from nn.activations.softmax import Softmax
from tests.numerical_gradient import numerical_gradient

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

# add later once backward is implemented
# class TestSoftmaxBackward(unittest.TestCase):
#     def test_shape_is_preserved(self):
#         activation = Softmax()

#         inputs = np.array([
#             [1.0, 2.0, 3.0],
#             [3.0, 2.0, 1.0],
#         ])

#         grad_output = np.array([
#             [1.0, 2.0, 3.0],
#             [-1.0, 0.5, 2.0],
#         ])

#         activation.forward(inputs)
#         grad_input = activation.backward(grad_output)

#         self.assertEqual(
#             grad_input.shape,
#             inputs.shape,
#         )

#     def test_gradient_of_sum_of_probabilities_is_zero(self):
#         activation = Softmax()

#         inputs = np.array([
#             [1.0, 2.0, 3.0],
#         ])

#         # If:
#         #
#         # L = s0 + s1 + s2
#         #
#         # then L is always 1 because softmax probabilities
#         # always sum to 1.
#         #
#         # Therefore dL/dx must be zero.
#         grad_output = np.ones_like(inputs)

#         activation.forward(inputs)
#         grad_input = activation.backward(grad_output)

#         expected = np.zeros_like(inputs)

#         np.testing.assert_allclose(
#             grad_input,
#             expected,
#             atol=1e-12,
#         )

#     def test_gradients_sum_to_zero_for_each_sample(self):
#         activation = Softmax()

#         inputs = np.array([
#             [1.0, 2.0, 3.0],
#             [-1.0, 4.0, 0.5],
#         ])

#         grad_output = np.array([
#             [2.0, -1.0, 0.5],
#             [-3.0, 1.0, 4.0],
#         ])

#         activation.forward(inputs)
#         grad_input = activation.backward(grad_output)

#         # Softmax is invariant to adding the same constant
#         # to every logit:
#         #
#         # softmax(x + c) = softmax(x)
#         #
#         # therefore the gradients across the logits sum to 0.
#         np.testing.assert_allclose(
#             np.sum(grad_input, axis=1),
#             np.zeros(inputs.shape[0]),
#             atol=1e-12,
#         )

#     def test_numerical_gradient(self):
#         activation = Softmax()

#         inputs = np.array([
#             [1.0, 2.0, 0.5],
#             [-0.5, 1.5, 2.0],
#         ])

#         grad_output = np.array([
#             [0.3, -1.2, 2.0],
#             [1.0, 0.4, -0.7],
#         ])

#         activation.forward(inputs)
#         analytical = activation.backward(grad_output)

#         numerical = numerical_gradient(
#             Softmax(),
#             inputs,
#             grad_output,
#         )

#         np.testing.assert_allclose(
#             analytical,
#             numerical,
#             rtol=1e-5,
#             atol=1e-7,
#         )

if __name__ == "__main__":
    unittest.main()
