import pandas as pd
import numpy as np

COLUMNS = [
    "id",
    "diagnosis",

    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",

    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",

    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads a dataset from a CSV file and assigns column names.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded dataset with assigned column names.
    """
    return pd.read_csv(file_path, header=None, names=COLUMNS)

def encode_labels(y: np.ndarray) -> np.ndarray:
    """
    B -> [1, 0]
    M -> [0, 1]

    args:
        y: np.ndarray of shape (n_samples,) containing labels as strings ("B" or "M")
    returns:
        np.ndarray of shape (n_samples, 2) containing one-hot encoded labels
    """
    binary = np.where(y == "M", 1, 0)
    return np.eye(2, dtype=np.float64)[binary]

def decode_labels(y: np.ndarray) -> np.ndarray:
    """
    [1, 0] -> B
    [0, 1] -> M

    args:
        y: np.ndarray of shape (n_samples, 2) containing one-hot encoded labels
    returns:
        np.ndarray of shape (n_samples,) containing labels as strings ("B" or "M")
    """
    binary = np.argmax(y, axis=1)
    return np.where(binary == 1, "M", "B")
