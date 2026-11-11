"""Classification metrics used by the evaluation phase."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    classes: list[str],
    scores: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return macro-F1, per-class metrics, AUROC, and a confusion matrix.

    ``labels`` and ``predictions`` contain integer class indices. Scores, when
    supplied, must contain one column per class and are used for one-vs-rest
    AUROC. AUROC is reported as ``None`` when a class is absent from the true
    labels, rather than silently fabricating a value.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    class_indices = np.arange(len(classes))
    matrix = confusion_matrix(labels, predictions, labels=class_indices)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=class_indices,
        zero_division=0,
    )
    per_class: dict[str, dict[str, float | None]] = {}
    auc_values: list[float] = []
    for index, name in enumerate(classes):
        true_positive = matrix[index, index]
        false_positive = matrix[:, index].sum() - true_positive
        false_negative = matrix[index, :].sum() - true_positive
        true_negative = matrix.sum() - true_positive - false_positive - false_negative
        specificity = (
            float(true_negative / (true_negative + false_positive))
            if true_negative + false_positive
            else None
        )
        auc = None
        if scores is not None and np.unique(labels == index).size == 2:
            auc = float(roc_auc_score(labels == index, np.asarray(scores)[:, index]))
            auc_values.append(auc)
        per_class[name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "specificity": specificity,
            "auroc": auc,
        }
    return {
        "macro_f1": float(f1_score(labels, predictions, labels=class_indices, average="macro", zero_division=0)),
        "per_class": per_class,
        "macro_auroc": float(np.mean(auc_values)) if auc_values else None,
        "confusion_matrix": matrix.tolist(),
    }
