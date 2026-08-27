# ChronoGraph-EEG

Research scaffold for seizure detection from dynamic EEG connectivity graphs.

## Repository layout

- `configs/`: versioned data, graph, model, and evaluation defaults.
- `data/`: raw, interim, and processed data locations. EEG data is not committed.
- `src/dal_518/`: importable package organized by pipeline stage.
- `notebooks/`: ordered exploration and results-analysis notebooks.
- `scripts/`: command-line entry points for the end-to-end pipeline.
- `experiments/` and `results/`: generated checkpoints, logs, tables, and figures.
- `tests/`: focused smoke tests for the scaffold and future implementations.

## Development

This project targets Python 3.13 and uses `uv` through `pyproject.toml`.

```bash
uv sync
uv run python -m compileall -q src
uv run dal-518
```

The EEG, MNE, PyTorch, and PyTorch Geometric integrations are intentionally left
as explicit implementation points. Add those dependencies when implementing the
corresponding pipeline stages and keep raw CHB-MIT files under `data/raw/`.

## Pipeline order

1. Explore raw recordings in `notebooks/01_data_exploration.ipynb`.
2. Preprocess and segment recordings with `scripts/run_preprocessing.py`.
3. Build graph sequences with `scripts/run_graph_construction.py`.
4. Train baselines and the dynamic model.
5. Evaluate held-out subjects and write artifacts under `results/`.
