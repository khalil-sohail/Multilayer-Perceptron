from .utils import *
from nn import (
    Sequential,
    Dense,
    Sigmoid,
    Softmax,
    SGD,
    RobustScaler,
    BinaryCrossEntropy,
    accuracy,
    precision,
    recall,
    f1_score,
    xavier_normal,
    random_normal,
    he_normal,
    ReLU,
    Tanh,
    Huber,
    MSE,
    EarlyStopping,
    Linear,
    CategoricalCrossEntropy,
    Adam
)

activation_map = {
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "linear": Linear,
}

initializer_map = {
    "random": random_normal,
    "xavier": xavier_normal,
    "he": he_normal,
}

optimizer_map = {
    "sgd": SGD,
    "adam": Adam,
}

class BreastCancerTrainer:
    """
    A class for training a neural network model on breast cancer data.
    """
    def __init__(self, args: dict[str, str], rng: np.random.Generator | None = None):
        """
        Initializes the BreastCancerTrainer using the provided training configuration.

        Args:
            args (dict[str, str]): Validated training arguments.

        Attributes:
            model (Sequential): The neural network model.
            scaler (RobustScaler): The scaler used for preprocessing the data.
            X_train (np.ndarray | None): The training input features.
            X_valid (np.ndarray | None): The validation input features.
            y_train (np.ndarray | None): The training labels.
            y_valid (np.ndarray | None): The validation labels.
        """

        hidden_layers       = [int(size) for size in args["layers"].split(",")]
        activation_class    = activation_map[args["activation"]]
        initializer         = initializer_map[args["initializer"]]
        optimizer_class     = optimizer_map[args["optimizer"]]
        learning_rate       = float(args["learning_rate"])
        layers              = []

        input_size = 30
        for output_size in hidden_layers:
            layers.append(
                Dense(
                    input_size,
                    output_size,
                    initializer=initializer,
                    rng=rng,
                )
            )

            layers.append(activation_class())
            input_size = output_size

        layers.append(
            Dense(
                input_size,
                2,
                initializer=initializer,
                rng=rng,
            )
        )
        layers.append(Softmax())
        self.model = Sequential(layers, rng=rng)

        self.model.compile(
            optimizer=optimizer_class(learning_rate=learning_rate),
            loss=CategoricalCrossEntropy(),
            metrics=[
                accuracy,
                precision,
                recall,
                f1_score,
            ]
        )

        self.scaler = RobustScaler()

        self.X_train = None
        self.X_valid = None

        self.y_train = None
        self.y_valid = None

    def load_data(self, train_path: str, valid_path: str):
        """
        Loads the training and validation datasets from CSV files.

        Args:
            train_path (str): The path to the training CSV file.
            valid_path (str): The path to the validation CSV file.
        Raises:
            RuntimeError: If the datasets cannot be loaded.
        """
        train_dataset = load_dataset(train_path)
        valid_dataset = load_dataset(valid_path)

        self.X_train = train_dataset.drop(columns=["id", "diagnosis"]).to_numpy(dtype=np.float64)
        self.X_valid = valid_dataset.drop(columns=["id", "diagnosis"]).to_numpy(dtype=np.float64)

        self.y_train = train_dataset["diagnosis"].to_numpy()
        self.y_valid = valid_dataset["diagnosis"].to_numpy()

    def preprocess_data(self):
        """
        Preprocesses the training and validation datasets using the scaler and one-hot encoding.
        
        Raises:
            RuntimeError: If the datasets have not been loaded before calling this method.
        """
        if (
            self.X_train is None
            or self.X_valid is None
            or self.y_train is None
            or self.y_valid is None
        ):
            raise RuntimeError("Training data has not been loaded.")

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_valid = self.scaler.transform(self.X_valid)
        
        self.y_train = encode_labels(self.y_train)
        self.y_valid = encode_labels(self.y_valid)

    def train(self, epochs: int = 150, batch_size: int = 16):
        """
        Trains the neural network model on the preprocessed training dataset.

        Args:
            epochs (int): The number of epochs to train for.
            batch_size (int): The batch size to use during training.

        Raises:
            RuntimeError: If the data has not been preprocessed.
        """
        if (self.X_train is None or self.y_train is None):
            raise RuntimeError("Data has not been preprocessed.")

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=5,
            min_delta=1e-4,
            mode="min",
            restore_best_weights=True,
        )

        return self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_valid, self.y_valid),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping]
        )

    def save(self, model_path: str, scaler_path: str):
        """
        Saves the trained model and scaler to specified file paths.

        Args:
            model_path (str): The path to save the trained model.
            scaler_path (str): The path to save the scaler.
        Raises:
            RuntimeError: If the model or scaler has not been trained or fitted.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained.")
        if self.scaler is None:
            raise RuntimeError("Scaler has not been fitted.")

        self.model.save(model_path)
        self.scaler.save(scaler_path)
