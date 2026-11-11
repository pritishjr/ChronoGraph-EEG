# ChronoGraph-EEG

ChronoGraph-EEG is a reproducible research pipeline for subject-independent
seizure-state classification from the CHB-MIT scalp EEG database. It converts
short bipolar EEG windows into dynamic connectivity graphs and compares a
flattened SVM baseline with a SLATE-inspired spatio-temporal graph transformer.

The primary task has three classes:

- interictal
- preictal
- ictal

A secondary preictal-versus-interictal task is also supported.

## Method overview

The pipeline is organized as:

1. **Channel audit:** identify recordings that satisfy the fixed bipolar channel contract.
2. **Segmentation:** create seizure-aware ictal, preictal, and interictal intervals.
3. **Preprocessing:** filter, normalize per subject, flag artifacts, and save fixed windows.
4. **Graph construction:** compute Pearson connectivity, sparsify edges, and derive band-power node features.
5. **Splitting:** create grouped leave-one-subject-out folds with leakage checks.
6. **Training:** train the SVM baseline or SLATE-inspired transformer.
7. **Evaluation:** calculate macro-F1, per-class metrics, AUROC, and confusion matrices.
8. **Reporting:** aggregate fold results and generate figures.

The data contract uses 23 bipolar channel positions, 256 Hz sampling, five-second
windows with 50% overlap, and 12-window graph chunks. Raw EEG and generated
experiment artifacts are intentionally excluded from Git.

## Setup

The project targets Python 3.13 and supports `uv` or a conventional virtual
environment.

```bash
uv sync
uv run python -m compileall -q src
```

Alternatively, install the dependencies from `requirements.txt`.

The default package command only reports package status and never downloads data:

```bash
uv run chronograph
```

## Pipeline commands

Run the data stages:

```bash
make data       # channel audit, segmentation, preprocessing
make graphs     # graph construction and chunking
make p5         # generate leakage-aware LOSO folds
```

Run training and evaluation after processed data and folds exist:

```bash
make train-all MODEL=svm TASK=three_class
make train-all MODEL=slate TASK=three_class
make evaluate-all MODEL=svm TASK=three_class
make evaluate-all MODEL=slate TASK=three_class
make eval
make plots
```

Evaluate one saved fold directly:

```bash
make evaluate \
  MODEL=svm \
  TASK=three_class \
  FOLD=chb01 \
  RUN=experiments/svm/three_class/chb01
```

Run the Model D cost check and the synthetic local smoke test:

```bash
make benchmark
make smoke
```

The smoke test uses synthetic graph data, runs one training epoch, and does not
download or require CHB-MIT recordings.

## Repository layout

```text
configs/                Shared and model-specific YAML configuration
src/chronograph/data/   Download, audit, segmentation, and preprocessing
src/chronograph/graphs/ Connectivity, features, chunking, and supra-graphs
src/chronograph/models/ SVM and SLATE-inspired model implementations
src/chronograph/splits/ LOSO folds and leakage checks
src/chronograph/training/Training loops and all-fold orchestration
src/chronograph/evaluation/ Metrics, held-out evaluation, aggregation, and plots
data/manifests/        Versioned channel, segment, and fold manifests
experiments/            Ignored checkpoints, configs, and run metrics
results/                Result tables and generated figures
paper/                  LaTeX manuscript and bibliography
docs/                   Architecture, training, references, and technical reports
tests/                  Unit tests and synthetic smoke coverage
```

## Validation

Run the test suite with:

```bash
make test
```

The tests cover preprocessing boundaries, segmentation, graph construction,
model forward/backward behavior, spectral-cache reuse, split leakage, training
helpers, evaluation metrics, and the synthetic smoke run.


