from .utils import *
from nn import (
    Sequential,
    RobustScaler,
    BinaryCrossEntropy,
    accuracy,
    precision,
    recall,
    f1_score,
)

class BreastCancerPredictor:
    def __init__(self, model_path: str, scaler_path: str):
        self.model = Sequential.load(model_path)
        self.scaler = RobustScaler.load(scaler_path)
        self.loss = BinaryCrossEntropy()

        self.X = None
        self.y = None

    def load_data(self, file_path: str):
        dataset = load_dataset(file_path)

        self.X = dataset.drop(columns=["id", "diagnosis"]).to_numpy(dtype=np.float64)
        self.y = dataset["diagnosis"].to_numpy()

    def preprocess_data(self):
        if self.X is None or self.y is None:
            raise RuntimeError("Dataset has not been loaded.")

        self.X = self.scaler.transform(self.X)
        self.y = encode_labels(self.y)

    def predict(self):
        if self.X is None:
            raise RuntimeError("Data has not been preprocessed.")

        return self.model.predict(self.X)

    def evaluate(self):
        if self.y is None:
            raise RuntimeError("Labels are not available.")

        y_pred = self.predict()
        loss = self.loss.forward(y_pred, self.y, training=False)

        return {
            "loss": loss,
            "accuracy": accuracy(self.y, y_pred),
            "precision": precision(self.y, y_pred),
            "recall": recall(self.y, y_pred),
            "f1_score": f1_score(self.y, y_pred),
        }
