import io
import os
import tempfile
import unittest

from contextlib import redirect_stdout

import numpy as np

from nn.activations.linear import Linear
from nn.activations.relu import ReLU
from nn.activations.softmax import Softmax
from nn.layers.dense import Dense
from nn.losses.categorical_cross_entropy import CategoricalCrossEntropy
from nn.losses.mse import MSE
from nn.metrics.accuracy import accuracy
from nn.metrics.f1_score import f1_score
from nn.metrics.precision import precision
from nn.metrics.recall import recall
from nn.model import Sequential
from nn.optimizers.sgd import SGD


class TestRegressionIntegration(unittest.TestCase):
    def test_network_learns_regression(self):
        X = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ])

        y = np.array([
            [1.0],
            [4.0],
            [3.0],
            [6.0],
            [8.0],
            [9.0],
        ])

        dense1 = Dense(2, 4)
        dense2 = Dense(4, 1)

        dense1.weights = np.array([
            [0.2, 0.1, 0.3, 0.4],
            [0.3, 0.4, 0.2, 0.1],
        ])

        dense1.biases = np.zeros((1, 4))

        dense2.weights = np.array([
            [0.1],
            [-0.2],
            [0.3],
            [0.2],
        ])

        dense2.biases = np.zeros((1, 1))

        model = Sequential([
            dense1,
            ReLU(),
            dense2,
            Linear(),
        ])

        model.compile(
            optimizer=SGD(
                learning_rate=0.01
            ),
            loss=MSE(),
            metrics=[],
        )

        initial_loss = model.evaluate(
            X,
            y,
        )

        with redirect_stdout(io.StringIO()):
            model.fit(
                X,
                y,
                epochs=2000,
                batch_size=len(X),
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
            0.01,
        )


class TestClassificationIntegration(unittest.TestCase):
    def build_model_and_data(self):
        X = np.array([
            [2.0, 0.0],
            [3.0, 0.0],
            [2.0, 0.5],
            [4.0, 1.0],
            [3.0, 0.5],

            [0.0, 2.0],
            [0.0, 3.0],
            [0.5, 2.0],
            [1.0, 4.0],
            [0.5, 3.0],
        ])

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

        dense1 = Dense(2, 4)
        dense2 = Dense(4, 2)

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
            ReLU(),
            dense2,
            Softmax(),
        ])

        model.compile(
            optimizer=SGD(
                learning_rate=0.05
            ),
            loss=CategoricalCrossEntropy(),
            metrics=[],
        )

        return model, X, y

    def test_classifier_learns(self):
        model, X, y = (
            self.build_model_and_data()
        )

        initial_loss = model.evaluate(
            X,
            y,
        )

        with redirect_stdout(io.StringIO()):
            model.fit(
                X,
                y,
                epochs=200,
                batch_size=2,
                shuffle=False,
            )

        final_loss = model.evaluate(
            X,
            y,
        )

        probabilities = model.predict(X)

        predicted_classes = np.argmax(
            probabilities,
            axis=1,
        )

        true_classes = np.argmax(
            y,
            axis=1,
        )

        self.assertLess(
            final_loss,
            initial_loss,
        )

        np.testing.assert_allclose(
            probabilities.sum(axis=1),
            np.ones(len(X)),
            atol=1e-7,
        )

        self.assertTrue(
            np.all(probabilities >= 0.0)
        )

        self.assertTrue(
            np.all(probabilities <= 1.0)
        )

        self.assertGreaterEqual(
            accuracy(
                true_classes,
                predicted_classes,
            ),
            0.9,
        )

        self.assertGreaterEqual(
            precision(
                true_classes,
                predicted_classes,
            ),
            0.9,
        )

        self.assertGreaterEqual(
            recall(
                true_classes,
                predicted_classes,
            ),
            0.9,
        )

        self.assertGreaterEqual(
            f1_score(
                true_classes,
                predicted_classes,
            ),
            0.9,
        )

    def test_save_load_preserves_predictions(self):
        model, X, y = (
            self.build_model_and_data()
        )

        with redirect_stdout(io.StringIO()):
            model.fit(
                X,
                y,
                epochs=50,
                batch_size=2,
                shuffle=False,
            )

        before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(
                tmp,
                "model.pkl",
            )

            model.save(path)

            loaded = Sequential.load(path)

            after = loaded.predict(X)

        np.testing.assert_allclose(
            before,
            after,
            rtol=1e-12,
            atol=1e-12,
        )

        self.assertEqual(
            len(model.layers),
            len(loaded.layers),
        )

        for original, restored in zip(
            model.layers,
            loaded.layers,
        ):
            self.assertIs(
                type(original),
                type(restored),
            )

        for original, restored in zip(
            model.trainable_layers,
            loaded.trainable_layers,
        ):
            np.testing.assert_allclose(
                original.weights,
                restored.weights,
            )

            np.testing.assert_allclose(
                original.biases,
                restored.biases,
            )


if __name__ == "__main__":
    unittest.main()