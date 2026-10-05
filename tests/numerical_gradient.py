import numpy as np

def numerical_gradient(activation, inputs, grad_output, epsilon=1e-5):
    """
    Numerically approximate:

        dL / dX

    where:

        L = sum(activation(X) * grad_output)

    using the centered finite-difference formula:

        dL/dx ≈ (L(x + eps) - L(x - eps)) / (2 * eps)
    """

    gradient = np.zeros_like(inputs, dtype=float)

    for index in np.ndindex(inputs.shape):
        inputs_plus = inputs.copy()
        inputs_minus = inputs.copy()

        inputs_plus[index] += epsilon
        inputs_minus[index] -= epsilon

        output_plus = activation.forward(inputs_plus)
        loss_plus = np.sum(output_plus * grad_output)

        output_minus = activation.forward(inputs_minus)
        loss_minus = np.sum(output_minus * grad_output)

        gradient[index] = (
            loss_plus - loss_minus
        ) / (2.0 * epsilon)

    return gradient
