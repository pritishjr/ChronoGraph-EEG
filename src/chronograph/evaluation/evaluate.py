"""Evaluate one saved model checkpoint on one held-out fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.svm import SVC

from chronograph.models.slate import SLATEModel
from chronograph.training.trainer import load_chunks

from .metrics import classification_metrics


def evaluate_run(
    model_name: str,
    task: str,
    fold_name: str,
    run_dir: Path,
    processed_root: Path,
    folds_path: Path,
    classes: list[str],
    model_config: dict[str, object] | None = None,
) -> dict[str, object]:
    folds = json.loads(folds_path.read_text())
    fold = next(item for item in folds["folds"] if item["fold"] == fold_name)
    adjacency, features, labels = load_chunks(processed_root, fold["test"])
    mapping = {name: index for index, name in enumerate(classes)}
    true_indices = np.array([mapping[label] for label in labels])
    model_config = model_config or {}

    if model_name == "svm":
        checkpoint = run_dir / "checkpoint.pkl"
        with checkpoint.open("rb") as handle:
            import pickle
            estimator: SVC = pickle.load(handle)
        predictions = estimator.predict(adjacency.reshape(len(adjacency), -1))
        scores = estimator.predict_proba(adjacency.reshape(len(adjacency), -1))
    elif model_name == "slate":
        model = SLATEModel(
            node_features=features.shape[-1],
            hidden_dim=int(model_config.get("hidden_dim", 32)),
            eigenvectors=int(model_config.get("eigenvectors", 8)),
            output_classes=len(classes),
            eigenvector_cache_dir=model_config.get("eigenvector_cache_dir"),
        )
        model.load_state_dict(torch.load(run_dir / "checkpoint.pt", map_location="cpu"))
        model.eval()
        with torch.no_grad():
            logits, _ = model(torch.from_numpy(adjacency).float(), torch.from_numpy(features).float())
        scores = torch.softmax(logits, dim=-1).numpy()
        predictions = scores.argmax(axis=-1)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    result = classification_metrics(true_indices, predictions, classes, scores)
    result.update({"model": model_name, "task": task, "fold": fold_name})
    (run_dir / "test_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["svm", "slate"], required=True)
    parser.add_argument("--task", default="three_class")
    parser.add_argument("--fold", required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--folds", type=Path, default=Path("data/manifests/folds.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--model-config", type=Path)
    args = parser.parse_args()
    base = yaml.safe_load(args.config.read_text())
    override_path = args.model_config or args.config.parent / "models" / f"{args.model}.yaml"
    override = yaml.safe_load(override_path.read_text()) or {}
    model_config = override.get("model", {})
    classes = base["tasks"]["class_labels"][args.task]
    result = evaluate_run(
        args.model, args.task, args.fold, args.run, args.processed_root,
        args.folds, classes, model_config,
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
