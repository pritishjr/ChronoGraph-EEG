# ChronoGraph-Seizure File Structure

Each item has one purpose. The tag shows the first tier that needs the item.

```text
chronograph-seizure/
├── README.md                         # problem, method, results, and commands [T1]
├── pyproject.toml                    # package metadata and imports [T1]
├── Makefile                          # make data, graphs, train, eval, smoke [T1]
├── configs/
│   ├── base.yaml                     # shared data, graph, task, and run settings [T1]
│   └── models/                       # svm.yaml, static.yaml, gru.yaml, slate.yaml [T1/T2]
├── data/                             # generated data; ignored except manifests/ [T1]
│   ├── raw/                          # downloaded CHB-MIT files [T1]
│   ├── interim/                      # cleaned subject EEG [T1]
│   ├── processed/                    # Pearson graph chunks [T1]
│   └── manifests/                    # fixed data tables [T1]
│       ├── channel_audit.csv         # channels, subject subset, and exclusions [T1]
│       ├── segments.csv              # segment boundaries and labels [T1]
│       └── folds.json                # LOSO and seizure-based validation folds [T1]
├── src/chronograph/
│   ├── data/                         # acquisition and EEG preparation [T1]
│   │   ├── download.py               # download and verify CHB-MIT files [T1]
│   │   ├── preprocess.py             # filter, reference, reject, and normalize [T1]
│   │   └── segment.py                # create the segment manifest [T1]
│   ├── graphs/                       # build fixed graph data [T1]
│   │   ├── build.py                  # Pearson graphs and node features [T1]
│   │   ├── chunk.py                  # make valid fixed-size chunks [T1]
│   │   ├── dataset.py                # load graph chunks [T1]
│   │   └── supra.py                  # build Model D supra-graphs [T2]
│   ├── models/                       # model implementations [T1]
│   │   ├── baselines.py              # Model A and Model B [T1]
│   │   ├── gru.py                    # Model C per-window GNN plus GRU [T1]
│   │   └── slate.py                  # Model D transformer [T2]
│   ├── splits.py                     # LOSO folds and leakage checks [T1]
│   ├── train.py                      # training, loss, and checkpoints [T1]
│   ├── evaluate.py                   # metrics and result tables [T1]
│   ├── viz.py                        # connectivity plot and result figures [T1]
│   └── utils.py                      # config, seeds, paths, and run logging [T1]
├── experiments/                      # config snapshot, checkpoint, metrics.json [T1]
├── results/                          # tables and figures [T1]
├── notebooks/                        # exploration only [T1]
│   ├── 01_explore.ipynb              # inspect EEG and labels [T1]
│   └── 02_graph_sanity.ipynb         # inspect graph changes [T1]
├── tests/
│   ├── test_segments_and_chunks.py   # segment and chunk boundary tests [T1]
│   ├── test_no_leakage.py             # subject and seizure leakage test [T1]
│   ├── test_models_forward.py         # forward and backward tests for A to D [T1/T2]
│   └── test_smoke.py                  # one-epoch synthetic full-pipeline test [T1]
└── docs/
    ├── decision_log.md               # fixed choices and any remaining TBD values [T1]
    └── report.md                     # short research report and limits [T1]
```

## Shared Rules

- Use `src/chronograph/...` for all package paths.
- Use `configs/base.yaml` for shared values.
- A file in `configs/models/` overrides the same value in `base.yaml`.
- Use Pearson connectivity for T1 and T2.
- Compute z-score statistics per subject from interictal data only.
- Never share normalization statistics across subjects.
- Use a 60-window chunk. Use a 30-window training stride and a 60-window
  validation and test stride.
- Drop a chunk that crosses a class, seizure, file, or exclusion boundary.
- Use the segment label. Do not assign mixed labels near onset.
- Use LOSO folds. Split validation data by seizure ID inside training subjects.
- Use weighted cross-entropy or focal loss. Undersample interictal data in
  training folds only.
- Keep `make smoke` synthetic and one epoch. Do not download data by default.
- Keep the full CHB-MIT run separate and resumable.
- Do not add deployment, Docker, CI/CD, serving, or model registry files.

## What Was Removed

| Removed item | Reason |
|---|---|
| `scripts/` folder | Make targets call the package directly. |
| Model registry | Four models do not need a registry. |
| Extra config folders | One base file and one file per model are enough. |
| Significance tests | T3 stretch item only. |
| Ablation runner | T3 stretch item only. |
| Deployment items | The project stops at research results. |

## Stretch

T3 items are optional: PLV, ictal-versus-non-ictal classification, ablation
sweeps, false alarms per hour, lead-time analysis, exploratory attention plots,
SALoM-inspired memory, and significance tests.
