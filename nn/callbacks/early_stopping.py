from __future__ import annotations

from typing import TYPE_CHECKING

from .call_back import Callback

if TYPE_CHECKING:
    from nn.model import Sequential


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric has stopped improving.
    """
    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True
    ) -> None:
        """
        Initializes the EarlyStopping callback with specified parameters.

        Args:
            monitor: Metric to be monitored.
            patience: Number of epochs with no improvement after which training will be stopped.
            min_delta: Minimum change in the monitored metric to qualify as an improvement.
            mode: One of {"min", "max"}. In "min" mode, training will stop when the quantity monitored has stopped decreasing;
                in "max" mode it will stop when the quantity monitored has stopped increasing.
            restore_best_weights: Whether to restore model weights from the epoch with the best value of the monitored metric.
        
        Raises:
            ValueError: If patience is negative, min_delta is negative, or mode is not one of {"min", "max"}.
        """
        if patience < 0:
            raise ValueError("patience must be >= 0")

        if min_delta < 0:
            raise ValueError("min_delta must be >= 0")

        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")

        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best: float | None = None
        self.best_epoch: int | None = None
        self.best_weights = None

        self.wait = 0
        self.stopped_epoch: int | None = None

    def on_train_begin(self, model: "Sequential") -> None:
        """
        Called at the beginning of training.
        """
        self.best = None
        self.best_epoch = None
        self.best_weights = None

        self.wait = 0
        self.stopped_epoch = None

    def _is_improvement(self, current: float) -> bool:
        """
        Check if the current value is an improvement over the best value.
        """
        if self.best is None:
            return True

        if self.mode == "min":
            return (current < self.best - self.min_delta)

        return (current > self.best + self.min_delta)

    def on_epoch_end(self, epoch: int, logs: dict[str, float], model: "Sequential") -> bool:
        """
        Called at the end of an epoch.
        Returns True if training should stop.
        """
        if self.monitor not in logs:
            raise ValueError(
                f"EarlyStopping monitor "
                f"'{self.monitor}' was not found. "
                f"Available metrics: "
                f"{list(logs.keys())}"
            )

        current = logs[self.monitor]

        if self._is_improvement(current):
            self.best = current
            self.best_epoch = epoch
            self.wait = 0

            if self.restore_best_weights:
                self.best_weights = (model.get_weights())

            return False

        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            return True

        return False

    def on_train_end(self, model: "Sequential") -> None:
        """
        Called at the end of training.
        """
        if (self.restore_best_weights and self.best_weights is not None):
            model.set_weights(self.best_weights)
