"""Aggregate completed training runs into evaluation tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _run_rows(runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_root.glob("*/*/*/metrics.json")):
        model, task, fold = metrics_path.relative_to(runs_root).parts[:3]
        metrics_path = metrics_path.parent / ("test_metrics.json" if (metrics_path.parent / "test_metrics.json").exists() else "metrics.json")
        metrics = json.loads(metrics_path.read_text())
        row: dict[str, Any] = {
            "model": model,
            "task": task,
            "fold": fold,
            "source_run": str(metrics_path.parent),
        }
        row.update(metrics)
        rows.append(row)
    return rows


def aggregate_runs(runs_root: Path, output_root: Path) -> tuple[Path, Path]:
    """Write per-fold and model/task summary CSV files."""
    rows = _run_rows(runs_root)
    output_root.mkdir(parents=True, exist_ok=True)
    per_subject = output_root / "per_subject.csv"
    main = output_root / "main_comparison.csv"

    per_fields = ["subject", "model", "task", "metric", "value"]
    with per_subject.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "subject": row["fold"],
                    "model": row["model"],
                    "task": row["task"],
                    "metric": "macro_f1" if "macro_f1" in row else "validation_macro_f1",
                    "value": row.get("macro_f1", row.get("validation_macro_f1", "")),
                }
            )

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        score = row.get("macro_f1", row.get("validation_macro_f1"))
        if score is not None:
            grouped[(row["model"], row["task"])].append(float(score))
    main_fields = ["model", "task", "macro_f1", "sensitivity", "specificity", "auroc", "false_alarms_per_hour"]
    with main.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=main_fields, lineterminator="\n")
        writer.writeheader()
        for (model, task), values in sorted(grouped.items()):
            writer.writerow(
                {
                    "model": model,
                    "task": task,
                    "macro_f1": mean(values),
                    "sensitivity": "",
                    "specificity": "",
                    "auroc": "",
                    "false_alarms_per_hour": "",
                }
            )
    return per_subject, main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("experiments"))
    parser.add_argument("--output-root", type=Path, default=Path("results/tables"))
    args = parser.parse_args()
    per_subject, main_table = aggregate_runs(args.runs_root, args.output_root)
    print(json.dumps({"per_subject": str(per_subject), "main_comparison": str(main_table)}))


if __name__ == "__main__":
    main()
