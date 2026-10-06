import io
import unittest

from contextlib import redirect_stdout

import numpy as np

from nn.activations.linear import Linear
from nn.layers.dense import Dense
from nn.losses.mse import MSE
from nn.model import Sequential
from nn.optimizers.sgd import SGD


class TestSequential(unittest.TestCase):
    def build_model(self):
        dense = Dense(1, 1)

        dense.weights = np.array([
            [0.0],
        ])

        dense.biases = np.array([
            [0.0],
        ])

        model = Sequential([
            dense,
            Linear(),
        ])

        model.compile(
            optimizer=SGD(
                learning_rate=0.05
            ),
            loss=MSE(),
            metrics=[],
        )

        return model

    def test_predict_shape(self):
        model = self.build_model()

        X = np.array([
            [0.0],
            [1.0],
            [2.0],
        ])

        predictions = model.predict(X)

        self.assertEqual(
            predictions.shape,
            (3, 1),
        )

    def test_evaluate_returns_loss(self):
        model = self.build_model()

        X = np.array([
            [0.0],
            [1.0],
        ])

        y = np.array([
            [1.0],
            [3.0],
        ])

        value = model.evaluate(X, y)

        self.assertIsInstance(
            value,
            float,
        )

        self.assertGreaterEqual(
            value,
            0.0,
        )

    def test_fit_reduces_loss(self):
        model = self.build_model()

        X = np.array([
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ])

        # y = 2x + 1
        y = np.array([
            [1.0],
            [3.0],
            [5.0],
            [7.0],
        ])

        initial_loss = model.evaluate(
            X,
            y,
        )

        with redirect_stdout(io.StringIO()):
            history = model.fit(
                X,
                y,
                epochs=300,
                batch_size=4,
                shuffle=False,
            )

        final_loss = model.evaluate(
            X,
            y,
        )

        self.assertLess(
            final_loss,
            initial_loss,
        )

        self.assertLess(
            final_loss,
            1e-3,
        )

        self.assertEqual(
            len(history["loss"]),
            300,
        )

    def test_validation_history(self):
        model = self.build_model()

        X_train = np.array([
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ])

        y_train = np.array([
            [1.0],
            [3.0],
            [5.0],
            [7.0],
        ])

        X_valid = np.array([
            [4.0],
            [5.0],
        ])

        y_valid = np.array([
            [9.0],
            [11.0],
        ])

        with redirect_stdout(io.StringIO()):
            history = model.fit(
                X_train,
                y_train,
                epochs=50,
                batch_size=4,
                validation_data=(
                    X_valid,
                    y_valid,
                ),
                shuffle=False,
            )

        self.assertIn(
            "loss",
            history,
        )

        self.assertIn(
            "val_loss",
            history,
        )

        self.assertEqual(
            len(history["loss"]),
            50,
        )

        self.assertEqual(
            len(history["val_loss"]),
            50,
        )

    def test_weights_change_during_fit(self):
        model = self.build_model()

        dense = model.trainable_layers[0]

        before = dense.weights.copy()

        X = np.array([
            [0.0],
            [1.0],
            [2.0],
        ])

        y = np.array([
            [1.0],
            [3.0],
            [5.0],
        ])

        with redirect_stdout(io.StringIO()):
            model.fit(
                X,
                y,
                epochs=10,
                batch_size=3,
                shuffle=False,
            )

        self.assertFalse(
            np.allclose(
                before,
                dense.weights,
            )
        )


if __name__ == "__main__":
    unittest.main()