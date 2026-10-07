import numpy as np
import pandas as pd

from .utils import *

from nn import (
    Sequential,
    Dense,
    Sigmoid,
    Softmax,
    SGD,
    RobustScaler,
    train_test_split,
    BinaryCrossEntropy,
    accuracy,
    precision,
    recall,
    f1_score,
)

class BreastCancerDataProcessor:
    def __init__(self):
        self.dataset: pd.DataFrame | None = None
        self.train_dataset: pd.DataFrame | None = None
        self.valid_dataset: pd.DataFrame | None = None

    def load_data(self, file_path: str):
        self.dataset = load_dataset(file_path)

    def split(self, test_size: float = 0.2, random_state: int = 42):
        if self.dataset is None:
            raise RuntimeError("Dataset has not been loaded.")

        train_data, valid_data = train_test_split(
            self.dataset.to_numpy(),
            test_size=test_size,
            shuffle=True,
            random_state=random_state,
        )

        self.train_dataset = pd.DataFrame(train_data, columns=COLUMNS,)
        self.valid_dataset = pd.DataFrame(valid_data, columns=COLUMNS)

    def save(self, train_path: str, valid_path: str):
        if self.train_dataset is None or self.valid_dataset is None:
            raise RuntimeError("Dataset has not been split.")

        self.train_dataset.to_csv(train_path, index=False, header=False)
        self.valid_dataset.to_csv(valid_path, index=False, header=False)

