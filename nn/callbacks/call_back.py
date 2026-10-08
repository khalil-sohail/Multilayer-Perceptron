from __future__ import annotations

from typing import TYPE_CHECKING
from abc import ABC

if TYPE_CHECKING:
    from nn.model import Sequential

class Callback(ABC):
    """
    Base class for all callbacks.
    """
    def on_train_begin(self, model: "Sequential") -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: dict[str, float], model: "Sequential") -> bool:
        """
        Returns True if training should stop.
        """
        return False

    def on_train_end(self, model: "Sequential") -> None:
        pass