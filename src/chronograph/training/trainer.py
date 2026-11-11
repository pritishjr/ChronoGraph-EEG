"""Training helpers for P7."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from chronograph.models.slate import SLATEModel
from .early_stopping import EarlyStopping
from .losses import weighted_cross_entropy


def load_chunks(root: Path, ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wanted = set(ids)
    adjacency, features, labels, records = [], [], [], []
    for path in sorted(root.glob("*/*.npz")):
        with np.load(path) as saved:
            subject = str(saved["subject"])
            source = str(saved["source_file"])
            for index, label in enumerate(saved["labels"].astype(str)):
                record_id = f"{subject}/{source}:{index}"
                if record_id in wanted:
                    adjacency.append(saved["adjacency"][index])
                    features.append(saved["node_features"][index])
                    labels.append(str(label))
                    records.append(record_id)
    if set(records) != wanted:
        raise RuntimeError(f"Missing processed chunks: {sorted(wanted - set(records))[:3]}")
    return np.stack(adjacency), np.stack(features), np.array(labels)


def macro_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(f1_score(labels, predictions, average="macro", zero_division=0))


def undersample_interictal(
    data: tuple[np.ndarray, np.ndarray, np.ndarray], seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    adjacency, features, labels = data
    interictal = np.flatnonzero(labels == "interictal")
    other = np.flatnonzero(labels != "interictal")
    if len(interictal) <= len(other) or len(other) == 0:
        return data
    selected = np.random.default_rng(seed).choice(
        interictal, size=len(other), replace=False
    )
    indices = np.concatenate((selected, other))
    return adjacency[indices], features[indices], labels[indices]


def train_slate(
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray, np.ndarray],
    output: Path,
    classes: list[str],
    epochs: int = 10,
    patience: int = 3,
    model_config: dict[str, object] | None = None,
) -> dict[str, float | int]:
    train_adjacency, train_features, train_labels = train
    valid_adjacency, valid_features, valid_labels = validation
    mapping = {name: index for index, name in enumerate(classes)}
    model_config = model_config or {}
    model = SLATEModel(
        node_features=train_features.shape[-1],
        hidden_dim=int(model_config.get("hidden_dim", 32)),
        eigenvectors=int(model_config.get("eigenvectors", 8)),
        output_classes=len(classes),
        eigenvector_cache_dir=model_config.get("eigenvector_cache_dir"),
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(model_config.get("learning_rate", 1e-3))
    )
    stopper = EarlyStopping(int(model_config.get("patience", patience)))
    best = None
    for epoch in range(1, epochs + 1):
        model.train()
        logits, _ = model(
            torch.from_numpy(train_adjacency).float(),
            torch.from_numpy(train_features).float(),
        )
        loss = weighted_cross_entropy(
            logits, torch.tensor([mapping[label] for label in train_labels])
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_logits, _ = model(
                torch.from_numpy(valid_adjacency).float(),
                torch.from_numpy(valid_features).float(),
            )
        predictions = valid_logits.argmax(dim=-1).numpy()
        score = macro_f1(
            np.array([mapping[label] for label in valid_labels]), predictions
        )
        if best is None or score > best["validation_macro_f1"]:
            best = {"epoch": epoch, "validation_macro_f1": score}
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output)
        if stopper.update(score):
            break
    return best or {"epoch": 0, "validation_macro_f1": 0.0}
