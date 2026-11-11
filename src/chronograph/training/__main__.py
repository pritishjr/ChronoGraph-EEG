"""Run one requested P7 model and fold."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import yaml
from sklearn.svm import SVC

from .trainer import (
    load_chunks,
    macro_f1,
    train_slate,
    undersample_interictal,
)


def _merge_config(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(path: Path, model: str, model_config: Path | None) -> dict:
    config = yaml.safe_load(path.read_text())
    override_path = model_config or path.parent / "models" / f"{model}.yaml"
    if not override_path.exists():
        raise FileNotFoundError(f"Model config not found: {override_path}")
    override = yaml.safe_load(override_path.read_text()) or {}
    return _merge_config(config, override)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["svm", "slate"], required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--task", default="three_class")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--folds", type=Path, default=Path("data/manifests/folds.json"))
    parser.add_argument("--runs-root", type=Path, default=Path("experiments"))
    args = parser.parse_args()
    config = _load_config(args.config, args.model, args.model_config)
    folds = json.loads(args.folds.read_text())
    fold = next(item for item in folds["folds"] if item["fold"] == args.fold)
    classes = config["tasks"]["class_labels"][args.task]
    train = load_chunks(args.processed_root, fold["train"])
    validation = load_chunks(args.processed_root, fold["validation"])
    if config["training"].get("undersample_interictal", False):
        train = undersample_interictal(train, seed=config["training"]["seed"])
    run = args.runs_root / args.model / args.task / args.fold
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    if args.model == "slate":
        metrics = train_slate(
            train,
            validation,
            run / "checkpoint.pt",
            classes,
            epochs=args.epochs or config["training"].get("epochs", 10),
            model_config=config.get("model", {}),
        )
    else:
        train_adjacency, _, train_labels = train
        valid_adjacency, _, valid_labels = validation
        estimator = SVC(probability=True, class_weight="balanced")
        mapping = {name: index for index, name in enumerate(classes)}
        estimator.fit(
            train_adjacency.reshape(len(train_adjacency), -1),
            [mapping[label] for label in train_labels],
        )
        predictions = estimator.predict(
            valid_adjacency.reshape(len(valid_adjacency), -1)
        )
        metrics = {
            "validation_macro_f1": macro_f1(
                np.array([mapping[label] for label in valid_labels]), predictions
            )
        }
        with (run / "checkpoint.pkl").open("wb") as handle:
            pickle.dump(estimator, handle)
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
