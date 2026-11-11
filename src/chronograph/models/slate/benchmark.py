"""Benchmark one Model D chunk."""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

import torch

from .model import SLATEModel


def benchmark(
    steps: int = 60,
    nodes: int = 23,
    features: int = 5,
    eigenvectors: int = 8,
) -> dict[str, float | int]:
    adjacency = torch.randn(1, steps, nodes, nodes)
    node_features = torch.randn(1, steps, nodes, features)
    model = SLATEModel(features, 32, eigenvectors)
    tracemalloc.start()
    started = time.perf_counter()
    with torch.no_grad():
        model(adjacency, node_features)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "steps": steps,
        "nodes": nodes,
        "eigenvectors": eigenvectors,
        "seconds": elapsed,
        "peak_python_bytes": peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument(
        "--output", type=Path, default=Path("results/model_d_cost.json")
    )
    args = parser.parse_args()
    result = benchmark(steps=args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
