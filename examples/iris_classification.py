"""
Example: Iris Classification using Random Forest.
"""

from pathlib import Path
import sys

# Allow imports from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_iris_dataset
from src.metrics import evaluate_classification, print_classification_report
from src.models import MachineLearningModels
from src.preprocessing import split_dataset, standard_scale
from src.visualization import plot_confusion_matrix


def main():
    # Load dataset
    dataframe = load_iris_dataset()

    # Split data
    X_train, X_test, y_train, y_test = split_dataset(
        dataframe,
        target_column="target",
    )

    # Scale features
    X_train, X_test = standard_scale(X_train, X_test)

    # Create model
    model = MachineLearningModels.random_forest()

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Metrics
    metrics = evaluate_classification(y_test, predictions)

    print("\nModel Performance")
    print("-" * 40)

    for key, value in metrics.items():
        print(f"{key.capitalize():12}: {value:.4f}")

    print("\nClassification Report")
    print("-" * 40)

    print_classification_report(y_test, predictions)

    plot_confusion_matrix(y_test, predictions)


if __name__ == "__main__":
    main()
