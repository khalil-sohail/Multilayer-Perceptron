from nn.interfaces.Optimizer import Optimizer, TrainableLayer
from nn.interfaces.Metric import Metric
from nn.interfaces.Loss import Loss
from typing import List, cast

from .interfaces.Layer import Layer
from numpy.typing import NDArray

import numpy as np
import pickle

Array = NDArray[np.float64]

class Sequential:
    """
    A sequential model that consists of a list of layers.
    This model allows for the forward and backward propagation of data through the layers in a sequential manner.

    Attributes:
        layers (list): A list of layers in the model.
        trainable_layers (list): A list of layers that have trainable parameters.
        optimizer: The optimizer used for updating the model's parameters.
        loss: The loss function used for calculating the loss during training.
        metrics (list): A list of metrics to be evaluated during training.
    """
    def __init__(self, layers: List[Layer]):
        """
        Initializes the Sequential model with a list of layers.
        """
        self.layers = layers
        self.trainable_layers: List[TrainableLayer] = [
            cast(TrainableLayer, layer) for layer in layers if getattr(layer, "trainable", False)
        ]
        self.optimizer = None
        self.loss = None
        self.metrics = []

    def forward(self, inputs: Array, training: bool=True) -> Array:
        """
        Performs the forward pass of the Sequential model.

        Args:
            inputs (numpy.ndarray): The input data to the model.
        Returns:
            numpy.ndarray: The output of the model after passing through all layers.
        """
        output = inputs

        for layer in self.layers:
            output = layer.forward(output, training=training)

        return output

    def backward(self, gradient: Array) -> Array:
        """
        Performs the backward pass of the Sequential model.

        Args:
            gradient (numpy.ndarray): The gradient of the loss with respect to the output of the model.
        Returns:
            numpy.ndarray: The gradient of the loss with respect to the input of the model.
        """
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

        return gradient

    def compile(self, optimizer: Optimizer, loss: Loss, metrics: List[Metric] | None=None):
        """
        Compiles the Sequential model with an optimizer and a loss function.

        Args:
            optimizer: The optimizer to be used for updating the model's parameters.
            loss: The loss function to be used for calculating the loss during training.
            metrics (list): A list of metrics to be evaluated during training.
        """
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics or []

    def fit(self, X: Array, y: Array, epochs: int, batch_size=None, validation_data=None, shuffle=True):
        """
        Trains the Sequential model on the provided training data.

        Args:
            X (np.ndarray): Training inputs.
            y (np.ndarray): Training targets.
            epochs (int): Number of complete passes over the dataset.
            batch_size (int | None): Number of samples per optimization step, None means full-batch training.
            validation_data (tuple | None): Optional (X_valid, y_valid).
            shuffle (bool): Whether to shuffle training samples before each epoch.

        Returns:
            dict: Training history.
        """

        if self.optimizer is None or self.loss is None:
            raise ValueError("Model must be compiled with an optimizer and loss before training.")
        elif len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")
        elif epochs <= 0:
            raise ValueError("epochs must be greater than 0.")
        elif batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        n_samples = len(X)
        if batch_size is None:
            batch_size = n_samples

        history = { "loss": [] }
        for metric in self.metrics:
            history[metric.__name__] = []

        if validation_data is not None:
            history["val_loss"] = []
            for metric in self.metrics:
                history[f"val_{metric.__name__}"] = []

        for _ in range(epochs):
            if shuffle:
                indices = np.random.permutation(n_samples)
                X_epoch = X[indices]
                y_epoch = y[indices]
            else:
                X_epoch = X
                y_epoch = y

            epoch_loss = 0.0
            y_batch = None
            predictions = None
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_epoch[start:end]
                y_batch = y_epoch[start:end]

                # FORWARD
                predictions = self.forward(X_batch, training=True)
                batch_loss = self.loss.forward(predictions, y_batch, training=True)

                # BACKWARD
                gradient = self.loss.backward()
                self.backward(gradient)

                # OPTIMIZER
                for layer in self.trainable_layers:
                    self.optimizer.step(layer)

                epoch_loss += batch_loss * len(X_batch)

            epoch_loss /= n_samples
            history["loss"].append(epoch_loss)

            train_predictions = self.predict(X)
            for metric in self.metrics:
                history[metric.__name__].append(metric(y, train_predictions))

            if validation_data is not None:
                X_valid, y_valid = validation_data
                validation_predictions = self.forward(X_valid, training=False)
                validation_loss = self.loss.forward(validation_predictions, y_valid, training=False)
                history["val_loss"].append(validation_loss)

                for metric in self.metrics:
                    history[f"val_{metric.__name__}"].append(metric(y_valid, validation_predictions))

        return history

    def predict(self, X: Array) -> Array:
        """
        Makes predictions using the trained Sequential model.

        Args:
            X (numpy.ndarray): The input data for making predictions.
        Returns:
            numpy.ndarray: The predicted output of the model.
        """
        return self.forward(X, training=False)

    def evaluate(self, X: Array, y: Array) -> float:
        """
        Evaluates the model on the given data.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The true labels.

        Returns:
            float: The loss on the given data.
        """
        if self.loss is None:
            raise ValueError("Model must be compiled before evaluation.")

        predictions = self.predict(X)
        return self.loss.forward(predictions, y, training=False)

    def save(self, path: str):
        """
        Saves the model to the specified path.

        Args:
            path (str): The file path where the model will be saved.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'Sequential':
        """
        Loads the model from the specified path.

        Args:
            path (str): The file path from where the model will be loaded.

        Returns:
            Sequential: The loaded model.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)