"""Train one model and task across every manifest fold."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["svm", "slate"], required=True)
    parser.add_argument("--task", default="three_class")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--folds", type=Path, default=Path("data/manifests/folds.json"))
    parser.add_argument("--runs-root", type=Path, default=Path("experiments"))
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    manifest = json.loads(args.folds.read_text())
    for fold in manifest["folds"]:
        command = [
            sys.executable,
            "-m",
            "chronograph.training",
            "--model",
            args.model,
            "--task",
            args.task,
            "--config",
            str(args.config),
            "--processed-root",
            str(args.processed_root),
            "--folds",
            str(args.folds),
            "--runs-root",
            str(args.runs_root),
            "--fold",
            fold["fold"],
        ]
        if args.model_config:
            command.extend(["--model-config", str(args.model_config)])
        if args.epochs is not None:
            command.extend(["--epochs", str(args.epochs)])
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
