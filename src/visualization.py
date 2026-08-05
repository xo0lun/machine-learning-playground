"""
visualization.py

Visualization utilities for exploratory data analysis
and machine learning results.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_histograms(dataframe: pd.DataFrame) -> None:
    """
    Plot histograms for all numeric features.
    """

    dataframe.hist(figsize=(12, 8))
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(dataframe: pd.DataFrame) -> None:
    """
    Plot a correlation heatmap.
    """

    correlation = dataframe.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(correlation)

    ax.set_xticks(range(len(correlation.columns)))
    ax.set_xticklabels(
        correlation.columns,
        rotation=90,
        fontsize=8,
    )

    ax.set_yticks(range(len(correlation.columns)))
    ax.set_yticklabels(
        correlation.columns,
        fontsize=8,
    )

    plt.colorbar(image)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred) -> None:
    """
    Display a confusion matrix.
    """

    cm = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(confusion_matrix=cm)

    display.plot()

    plt.tight_layout()

    plt.show()
