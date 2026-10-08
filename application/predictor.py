from nn.losses.categorical_cross_entropy import CategoricalCrossEntropy

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
    """
    A class for predicting breast cancer using a trained model and scaler.
    """
    def __init__(self, model_path: str, scaler_path: str):
        """
        Initializes the BreastCancerPredictor with a trained model and scaler.

        args:
            model_path (str): The path to the trained model file.
            scaler_path (str): The path to the scaler file.
        attributes:
            model (Sequential): The trained model.
            scaler (RobustScaler): The scaler used for preprocessing.
            loss (BinaryCrossEntropy): The loss function used for evaluation.
            X (np.ndarray | None): The input features for prediction.
            y (np.ndarray | None): The true labels for evaluation.
        """
        self.model = Sequential.load(model_path)
        self.scaler = RobustScaler.load(scaler_path)
        self.loss = BinaryCrossEntropy()

        self.X = None
        self.y = None

    def load_data(self, file_path: str):
        """
        Loads the dataset from a CSV file.

        Args:
            file_path (str): The path to the CSV file.
        """
        dataset = load_dataset(file_path)

        self.X = dataset.drop(columns=["id", "diagnosis"]).to_numpy(dtype=np.float64)
        self.y = dataset["diagnosis"].to_numpy()

    def preprocess_data(self):
        """
        Preprocesses the input features and labels using the scaler and one-hot encoding.
        
        Raises:
            RuntimeError: If the dataset has not been loaded before calling this method.
        """
        if self.X is None or self.y is None:
            raise RuntimeError("Dataset has not been loaded.")

        self.X = self.scaler.transform(self.X)
        self.y = encode_labels(self.y)

    def predict(self):
        """
        Makes predictions on the preprocessed input features.

        Returns:
            tuple: A tuple containing the predicted probabilities and the predicted classes.
        """
        if self.X is None:
            raise RuntimeError("Data has not been preprocessed.")

        y_pred_prob = self.model.predict(self.X)
        y_pred_class = np.argmax(y_pred_prob, axis=1)

        return y_pred_prob, y_pred_class

    def evaluate(self):
        """
        Evaluates the model's performance on the preprocessed dataset.

        Raises:
            RuntimeError: If the labels are not available for evaluation.
        Returns:
            dict: A dictionary containing the loss, accuracy, precision, recall, and F1 score.
        """
        if self.y is None:
            raise RuntimeError("Labels are not available.")

        y_pred_prob, y_pred_class = self.predict()
        y_true_binary = self.y[:, 1]

        loss = self.loss.forward(y_pred_prob, self.y, training=False)

        y_pred_labels = decode_labels(y_pred_prob)
        y_true_labels = decode_labels(self.y)

        print("Predictions:")

        for i, (pred, true) in enumerate(zip(y_pred_labels, y_true_labels)):
            print(
                f"{i:03d}: "
                f"pred=`{pred}` "
                f"true=`{true}` "
                f"{'Correct' if pred == true else 'Incorrect'}"
            )

        return {
            "loss": loss,
            "accuracy": accuracy(
                y_true_binary,
                y_pred_class,
            ),
            "precision": precision(
                y_true_binary,
                y_pred_class,
            ),
            "recall": recall(
                y_true_binary,
                y_pred_class,
            ),
            "f1_score": f1_score(
                y_true_binary,
                y_pred_class,
            ),
        }