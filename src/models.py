"""
models.py

Factory class for creating common Machine Learning models.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class MachineLearningModels:
    """
    Factory methods for creating machine learning models.
    """

    @staticmethod
    def linear_regression():
        return LinearRegression()

    @staticmethod
    def logistic_regression():
        return LogisticRegression(max_iter=1000)

    @staticmethod
    def decision_tree(random_state: int = 42):
        return DecisionTreeClassifier(random_state=random_state)

    @staticmethod
    def random_forest(
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        return RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
        )

    @staticmethod
    def svm():
        return SVC()

    @staticmethod
    def knn(n_neighbors: int = 5):
        return KNeighborsClassifier(
            n_neighbors=n_neighbors
        )

    @staticmethod
    def naive_bayes():
        return GaussianNB()
