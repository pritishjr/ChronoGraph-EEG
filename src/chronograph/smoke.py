"""Run a one-epoch synthetic end-to-end model smoke check."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from sklearn.svm import SVC

from chronograph.training.trainer import macro_f1, train_slate


def _synthetic_data(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    samples, steps, nodes, features = 6, 12, 23, 5
    adjacency = rng.normal(0, 0.05, (samples, steps, nodes, nodes)).astype(np.float32)
    adjacency = (adjacency + adjacency.transpose(0, 1, 3, 2)) / 2
    node_features = rng.normal(0, 1, (samples, steps, nodes, features)).astype(np.float32)
    labels = np.array(["interictal", "preictal", "ictal"] * 2)
    for index, label in enumerate(labels):
        edge_strength = {"interictal": 0.1, "preictal": 0.5, "ictal": 0.9}[label]
        adjacency[index, :, 0, 1] = edge_strength
        adjacency[index, :, 1, 0] = edge_strength
        node_features[index, :, 0, 0] += edge_strength
    return adjacency[:3], node_features[:3], labels[:3], (adjacency[3:], node_features[3:], labels[3:])


def run_smoke() -> dict[str, float | int]:
    train_adjacency, train_features, train_labels, validation = _synthetic_data()
    valid_adjacency, valid_features, valid_labels = validation
    classes = ["interictal", "preictal", "ictal"]
    mapping = {name: index for index, name in enumerate(classes)}
    svm = SVC(class_weight="balanced")
    svm.fit(train_adjacency.reshape(len(train_adjacency), -1), [mapping[label] for label in train_labels])
    svm_predictions = svm.predict(valid_adjacency.reshape(len(valid_adjacency), -1))
    with tempfile.TemporaryDirectory() as directory:
        slate_metrics = train_slate(
            (train_adjacency, train_features, train_labels),
            (valid_adjacency, valid_features, valid_labels),
            Path(directory) / "slate.pt",
            classes,
            epochs=1,
        )
    return {
        "svm_validation_macro_f1": macro_f1(
            np.array([mapping[label] for label in valid_labels]), svm_predictions
        ),
        "slate_validation_macro_f1": float(slate_metrics["validation_macro_f1"]),
        "epochs": 1,
        "samples": 6,
    }


def main() -> None:
    print(json.dumps(run_smoke(), indent=2))


if __name__ == "__main__":
    main()
