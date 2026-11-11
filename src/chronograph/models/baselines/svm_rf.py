"""Model A: untuned SVM baseline."""

from __future__ import annotations

import numpy as np
from sklearn.svm import SVC


class SVMBaseline:
    """Classify flattened chunk connectivity as an untuned floor."""

    def __init__(self, output_classes: int = 3) -> None:
        self.output_classes = output_classes
        self.estimator = SVC(probability=True, class_weight="balanced")

    @staticmethod
    def _flatten(adjacency: np.ndarray) -> np.ndarray:
        return adjacency.reshape(adjacency.shape[0], -1)

    def fit(self, adjacency: np.ndarray, labels: np.ndarray) -> "SVMBaseline":
        self.estimator.fit(self._flatten(adjacency), labels)
        return self

    def predict(self, adjacency: np.ndarray) -> np.ndarray:
        return self.estimator.predict(self._flatten(adjacency))

    def predict_proba(self, adjacency: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(self._flatten(adjacency))
