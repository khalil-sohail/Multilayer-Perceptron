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
)

class BreastCancerTrainer:
    def __init__(self):
        self.model = Sequential([
            Dense(30, 24, initializer=xavier_normal),
            Sigmoid(),

            Dense(24, 16, initializer=xavier_normal),
            Sigmoid(),

            Dense(16, 16, initializer=xavier_normal),
            Sigmoid(),

            Dense(16, 2, initializer=xavier_normal),
            Softmax(),
        ])

        self.model.compile(
            optimizer=SGD(learning_rate=0.01),
            loss=BinaryCrossEntropy(),
            metrics=[
                accuracy,
                precision,
                recall,
                f1_score,
            ],
        )

        self.scaler = RobustScaler()

        self.X_train = None
        self.X_valid = None

        self.y_train = None
        self.y_valid = None

    def load_data(self, train_path: str, valid_path: str):
        train_dataset = load_dataset(train_path)
        valid_dataset = load_dataset(valid_path)

        self.X_train = train_dataset.drop(columns=["id", "diagnosis"]).to_numpy(dtype=np.float64)
        self.X_valid = valid_dataset.drop(columns=["id", "diagnosis"]).to_numpy(dtype=np.float64)

        self.y_train = train_dataset["diagnosis"].to_numpy()
        self.y_valid = valid_dataset["diagnosis"].to_numpy()

    def preprocess_data(self):
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

    def train(self, epochs: int = 100, batch_size: int = 32):
        if (self.X_train is None or self.y_train is None):
            raise RuntimeError("Data has not been preprocessed.")

        return self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_valid, self.y_valid),
            epochs=epochs,
            batch_size=batch_size,
        )

    def save(self, model_path: str, scaler_path: str):
        self.model.save(model_path)
        self.scaler.save(scaler_path)
