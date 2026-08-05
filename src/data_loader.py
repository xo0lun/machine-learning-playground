"""
data_loader.py

Utilities for loading built-in datasets from scikit-learn
and custom CSV files.
"""

from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine


def load_iris_dataset() -> pd.DataFrame:
    """
    Load the Iris dataset as a pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        Iris dataset with feature columns and target.
    """
    dataset = load_iris(as_frame=True)
    df = dataset.frame
    return df


def load_wine_dataset() -> pd.DataFrame:
    """
    Load the Wine dataset as a pandas DataFrame.
    """
    dataset = load_wine(as_frame=True)
    return dataset.frame


def load_breast_cancer_dataset() -> pd.DataFrame:
    """
    Load the Breast Cancer Wisconsin dataset.
    """
    dataset = load_breast_cancer(as_frame=True)
    return dataset.frame


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """
    Load a CSV file.

    Parameters
    ----------
    file_path : str | Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(file_path)


if __name__ == "__main__":
    iris = load_iris_dataset()

    print("Dataset loaded successfully.")
    print(f"Shape: {iris.shape}")
    print()
    print(iris.head())
