"""
metrics.py

Evaluation metrics for classification models.
"""

from typing import Dict

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classification(y_true, y_pred) -> Dict:
    """
    Evaluate a classification model.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.

    y_pred : array-like
        Predicted labels.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }


def print_classification_report(y_true, y_pred) -> None:
    """
    Print the classification report.
    """

    print(classification_report(y_true, y_pred))


def get_confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix.
    """

    return confusion_matrix(y_true, y_pred)
