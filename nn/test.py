import numpy as np

from nn.activations.relu import ReLU
from nn.activations.softmax import Softmax
from nn.layers.dense import Dense
from nn.losses.categorical_cross_entropy import CategoricalCrossEntropy
from nn.metrics.accuracy import accuracy
from nn.metrics.precision import precision
from nn.metrics.recall import recall
from nn.metrics.f1_score import f1_score
from nn.model import Sequential
from nn.optimizers.sgd import SGD


def main():
    # ---------------------------------------------------------
    # Simple binary classification dataset
    #
    # Class 0:
    #   first feature is generally larger
    #
    # Class 1:
    #   second feature is generally larger
    # ---------------------------------------------------------

    X = np.array([
        # Class 0
        [2.0, 0.0],
        [3.0, 0.0],
        [2.0, 0.5],
        [4.0, 1.0],
        [3.0, 0.5],

        # Class 1
        [0.0, 2.0],
        [0.0, 3.0],
        [0.5, 2.0],
        [1.0, 4.0],
        [0.5, 3.0],
    ])

    # One-hot encoded targets:
    #
    # class 0 -> [1, 0]
    # class 1 -> [0, 1]

    y = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],

        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])

    # ---------------------------------------------------------
    # Network
    #
    # 2 inputs
    #    ↓
    # Dense(2, 4)
    #    ↓
    # ReLU
    #    ↓
    # Dense(4, 2)
    #    ↓
    # Softmax
    #    ↓
    # [P(class 0), P(class 1)]
    # ---------------------------------------------------------

    dense1 = Dense(
        input_size=2,
        output_size=4,
    )

    relu = ReLU()

    dense2 = Dense(
        input_size=4,
        output_size=2,
    )

    softmax = Softmax()

    # ---------------------------------------------------------
    # Deterministic weights
    #
    # This keeps the test reproducible.
    # ---------------------------------------------------------

    dense1.weights = np.array([
        [0.20, -0.10, 0.30, -0.20],
        [-0.10, 0.20, -0.20, 0.30],
    ])

    dense1.biases = np.array([
        [0.10, 0.10, 0.10, 0.10],
    ])

    dense2.weights = np.array([
        [0.10, -0.10],
        [-0.10, 0.10],
        [0.05, -0.05],
        [-0.05, 0.05],
    ])

    dense2.biases = np.zeros((1, 2))

    model = Sequential([
        dense1,
        relu,
        dense2,
        softmax,
    ])

    loss = CategoricalCrossEntropy()

    optimizer = SGD(
        learning_rate=0.05,
    )

    # Keep metrics out of fit() for this integration test.
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[],
    )

    # ---------------------------------------------------------
    # Initial state
    # ---------------------------------------------------------

    initial_predictions = model.predict(X)

    initial_loss = loss.forward(
        initial_predictions,
        y,
        training=False,
    )

    initial_weights_1 = dense1.weights.copy()
    initial_weights_2 = dense2.weights.copy()

    print("Initial probabilities:")
    print(initial_predictions)

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    history = model.fit(
        X,
        y,
        epochs=200,
        batch_size=2,
        shuffle=True,
    )

    # ---------------------------------------------------------
    # Final predictions
    # ---------------------------------------------------------

    final_predictions = model.predict(X)

    final_loss = model.evaluate(
        X,
        y,
    )

    print()
    print(f"Initial loss: {initial_loss:.8f}")
    print(f"Final loss:   {final_loss:.8f}")

    print()
    print("Final probabilities:")
    print(final_predictions)

    # ---------------------------------------------------------
    # Convert probabilities to class IDs
    # ---------------------------------------------------------

    predicted_classes = np.argmax(
        final_predictions,
        axis=1,
    )

    true_classes = np.argmax(
        y,
        axis=1,
    )

    print()
    print("Predictions:")

    for (
        inputs,
        probabilities,
        predicted_class,
        true_class,
    ) in zip(
        X,
        final_predictions,
        predicted_classes,
        true_classes,
    ):
        print(
            f"{inputs} -> "
            f"probs={probabilities} "
            f"pred={predicted_class} "
            f"target={true_class}"
        )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    acc = accuracy(
        true_classes,
        predicted_classes,
    )

    prec = precision(
        true_classes,
        predicted_classes,
    )

    rec = recall(
        true_classes,
        predicted_classes,
    )

    f1 = f1_score(
        true_classes,
        predicted_classes,
    )

    print()
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 score:  {f1:.4f}")

    # ---------------------------------------------------------
    # Assertions
    # ---------------------------------------------------------

    # Loss must decrease.
    assert final_loss < initial_loss, (
        f"Training failed: loss did not decrease "
        f"({initial_loss} -> {final_loss})"
    )

    # Both Dense layers must have learned.
    assert not np.allclose(
        dense1.weights,
        initial_weights_1,
    ), "dense1 weights did not change."

    assert not np.allclose(
        dense2.weights,
        initial_weights_2,
    ), "dense2 weights did not change."

    # Softmax probabilities must sum to 1 for every sample.
    np.testing.assert_allclose(
        np.sum(final_predictions, axis=1),
        np.ones(len(X)),
        rtol=1e-6,
        atol=1e-7,
    )

    # Probabilities must be valid.
    assert np.all(final_predictions >= 0.0)
    assert np.all(final_predictions <= 1.0)

    # No NaN / inf.
    assert np.all(
        np.isfinite(final_predictions)
    )

    # This dataset is intentionally very easy.
    assert acc >= 0.9, (
        f"Classification accuracy too low: {acc}"
    )

    print()
    print("Classification training test passed.")


if __name__ == "__main__":
    main()