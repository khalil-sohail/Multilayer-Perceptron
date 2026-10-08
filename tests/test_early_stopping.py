import unittest

import numpy as np

from nn import Dense, Sequential
from nn.callbacks.early_stopping import (
    EarlyStopping,
)


class TestEarlyStopping(unittest.TestCase):
    def test_stops_and_restores_best_weights(
        self,
    ):
        dense = Dense(1, 1)

        model = Sequential([
            dense,
        ])

        callback = EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )

        callback.on_train_begin(model)

        # Epoch 0: first best
        dense.weights[:] = 1.0

        stopped = callback.on_epoch_end(
            0,
            {"val_loss": 1.0},
            model,
        )

        self.assertFalse(stopped)

        # Epoch 1: improved
        dense.weights[:] = 2.0

        stopped = callback.on_epoch_end(
            1,
            {"val_loss": 0.5},
            model,
        )

        self.assertFalse(stopped)

        # Epoch 2: worse
        dense.weights[:] = 3.0

        stopped = callback.on_epoch_end(
            2,
            {"val_loss": 0.6},
            model,
        )

        self.assertFalse(stopped)

        # Epoch 3: worse again -> stop
        dense.weights[:] = 4.0

        stopped = callback.on_epoch_end(
            3,
            {"val_loss": 0.7},
            model,
        )

        self.assertTrue(stopped)

        callback.on_train_end(model)

        # Should restore epoch 1:
        # weights == 2.0
        np.testing.assert_allclose(
            dense.weights,
            np.ones_like(
                dense.weights
            ) * 2.0,
        )


if __name__ == "__main__":
    unittest.main()
