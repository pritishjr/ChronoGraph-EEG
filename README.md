# ChronoGraph Seizure

Research scaffold for seizure analysis from dynamic EEG connectivity graphs.

The canonical Python package is `src/chronograph`. The pipeline is organized as
channel auditing and download, preprocessing, segmentation, graph construction,
leakage-safe folds, model training, evaluation, and interpretation.

## Development

This project targets Python 3.13 and supports both `uv` and a conventional
requirements file.

```bash
uv sync
uv run python -m compileall -q src scripts
uv run chronograph
```

Raw CHB-MIT recordings and generated experiment artifacts are ignored by Git.
Only manifests, configuration, summaries, and placeholder directories are kept
in the repository.

## Pipeline

Run phases individually with `make data`, `make graphs`, `make train`, `make eval`,
and `make figures`. The numbered scripts in `scripts/` define the execution order.

Configuration is split by data, graph, model, experiment, and evaluation concerns
under `configs/`. The updated package layout is documented in `file-structure.md`.
