import pandas as pd

from nn import train_test_split
from .utils import *

class BreastCancerDataProcessor:
    """
    A class to handle loading, splitting, and saving the breast cancer dataset.
    """
    def __init__(self):
        """
        Initializes the BreastCancerDataProcessor with empty datasets.
        
        attributes:
            dataset (pd.DataFrame | None): The complete dataset.
            train_dataset (pd.DataFrame | None): The training subset of the dataset.
            valid_dataset (pd.DataFrame | None): The validation subset of the dataset.
        """
        self.dataset: pd.DataFrame | None = None
        self.train_dataset: pd.DataFrame | None = None
        self.valid_dataset: pd.DataFrame | None = None

    def load_data(self, file_path: str):
        """
        Loads the dataset from a CSV file.

        Args:
            file_path (str): The path to the CSV file.
        """
        self.dataset = load_dataset(file_path)

    def split(self, test_size: float = 0.2, random_state: int = 42):
        """
        Splits the dataset into training and validation sets.

        Args:
            test_size (float): The proportion of the dataset to include in the validation set.
            random_state (int): The seed for the random number generator.
        Raises:
            RuntimeError: If the dataset has not been loaded before calling this method.
        """
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
        """
        Saves the training and validation datasets to CSV files.

        Args:
            train_path (str): The path to the CSV file for the training set.
            valid_path (str): The path to the CSV file for the validation set.
        """
        if self.train_dataset is None or self.valid_dataset is None:
            raise RuntimeError("Dataset has not been split.")

        self.train_dataset.to_csv(train_path, index=False, header=False)
        self.valid_dataset.to_csv(valid_path, index=False, header=False)

