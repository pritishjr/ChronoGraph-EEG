"""Create result figures from completed test-metric artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_results(runs_root: Path, output_root: Path) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    scores: dict[str, list[float]] = defaultdict(list)
    matrices: dict[str, list[np.ndarray]] = defaultdict(list)
    for path in sorted(runs_root.glob("*/*/*/test_metrics.json")):
        payload = json.loads(path.read_text())
        model = str(payload["model"])
        scores[model].append(float(payload["macro_f1"]))
        matrices[model].append(np.asarray(payload["confusion_matrix"]))
    if not scores:
        raise FileNotFoundError(f"No test_metrics.json files found under {runs_root}")

    generated: list[Path] = []
    labels = sorted(scores)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(labels, [np.mean(scores[label]) for label in labels])
    axis.set_ylabel("Macro-F1")
    axis.set_title("Held-out model comparison")
    axis.set_ylim(0, 1)
    figure.tight_layout()
    comparison = output_root / "model_comparison.png"
    figure.savefig(comparison, dpi=200)
    plt.close(figure)
    generated.append(comparison)

    for label in labels:
        matrix = np.sum(matrices[label], axis=0)
        figure, axis = plt.subplots(figsize=(5, 4))
        image = axis.imshow(matrix, cmap="Blues")
        figure.colorbar(image, ax=axis)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_title(f"{label} confusion matrix")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, int(matrix[row, column]), ha="center", va="center")
        figure.tight_layout()
        output = output_root / f"{label}_confusion_matrix.png"
        figure.savefig(output, dpi=200)
        plt.close(figure)
        generated.append(output)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("experiments"))
    parser.add_argument("--output-root", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    print("\n".join(str(path) for path in plot_results(args.runs_root, args.output_root)))


if __name__ == "__main__":
    main()
