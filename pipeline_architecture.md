# ChronoGraph-Seizure: Pipeline Architecture

The pipeline has 10 phases. It uses CHB-MIT scalp EEG. The main task is
three-class classification: interictal, preictal, and ictal.

## Pipeline Overview

```
Raw EEG
  -> P1 Acquisition and channel audit
  -> P2 Preprocessing
  -> P3 Segmentation
  -> P4 Graph construction and chunking
  -> P5 LOSO splits
  -> P6 Models
  -> P7 Training
  -> P8 Evaluation
  -> P9 Results and figures
  -> P10 Packaging
```

| Phase | Name | Output |
|---|---|---|
| P0 | Data contract | Fixed data and tensor rules |
| P1 | Acquisition and channel audit | Raw files and channel report |
| P2 | Preprocessing | Clean EEG |
| P3 | Segmentation | Segment manifest |
| P4 | Graph construction and chunking | Graph chunks |
| P5 | Splits | LOSO fold manifest |
| P6 | Models | Models A to D |
| P7 | Training | Checkpoints and run metrics |
| P8 | Evaluation | Per-fold and aggregate metrics |
| P9 | Results and figures | Tables and plots |
| P10 | Packaging | Reproducible research repository |

## P0. Data Contract

**Purpose:** Fix every rule that changes the data or results.

**Input:** Project scope and CHB-MIT annotations.

**Output:** `configs/base.yaml`, model config files, and a recorded decision log.

**Must build:**

- [T1] Define a canonical channel set. Start with a subject subset that shares
  these channels. Record the subset and excluded subjects in the channel audit.
- [T1] Set the sampling rate to 256 Hz.
- [T1] Set the filter to 0.5 to 40 Hz with zero-phase filtering.
- [T1] Set the notch filter to 60 Hz.
- [T1] Set common average reference.
- [T1] Set the amplitude rejection limit to 500 µV. Do not use ICA.
- [T1] Define segment boundaries for interictal, preictal, and ictal data.
- [T1] Set the preictal horizon to 30 minutes.
- [T1] Keep an interictal buffer of at least 4 hours from any seizure.
- [T1] Exclude postictal data.
- [T1] Define the label rule once: the segment sets the label. Windows near
  onset do not receive mixed labels.
- [T1] Define the chunk rule: a chunk must lie fully inside one segment. Drop
  a chunk that crosses a class, seizure, file, or exclusion boundary. The
  chunk label is the label of its segment.
- [T1] Set the window to 5 seconds with 50% overlap.
- [T1] Set the chunk length to 60 windows.
- [T1] Set the chunk stride in `configs/base.yaml` to 30 windows for training
  and 60 windows for validation and test.
- [T1] Set Pearson connectivity and top-k sparsification per node.
- [T1] Set node features to power in five bands.
- [T1] Define fixed tensor shapes for all subjects.
- [T1] Define one fold manifest for all models.
- [T1] Compute z-score statistics per subject from interictal data only. Never
  share statistics across subjects. This subject-local normalization is a
  stated evaluation choice.
- [T1] Define LOSO evaluation. Split validation data from training subjects by
  seizure ID. Never split one seizure across train and validation.
- [T1] Set the main task to three-class classification.
- [T2] Set the second task to preictal versus interictal classification.
- [T1] Use one `configs/base.yaml`. A model file in `configs/models/` overrides
  values in `base.yaml`.

**Done when:** The data contract, config merge rule, channel audit fields, and
fold manifest fields are written. No required decision is open.

## P1. Acquisition and Channel Audit

**Purpose:** Get valid CHB-MIT files and select a common channel set.

**Input:** CHB-MIT source files and seizure annotations.

**Output:** Verified files and `data/manifests/channel_audit.csv`.

**Must build:**

- [T1] Download CHB-MIT only when the user runs the full data command.
- [T1] Check that each selected EDF file opens.
- [T1] Check that each selected file has its seizure annotations.
- [T1] List channels and sampling rates for every subject and file.
- [T1] Record the selected subject subset and every excluded subject.
- [T1] Record the reason for each exclusion.

**Done when:** The audit lists the selected subjects, canonical channels, sample
rates, files, and exclusions.

## P2. Preprocessing

**Purpose:** Create clean, subject-local EEG.

**Input:** Verified EDF files and the channel audit.

**Output:** Clean continuous EEG in `data/interim/`.

**Must build:**

- [T1] Load and normalize channel names.
- [T1] Keep only canonical channels.
- [T1] Apply the 0.5 to 40 Hz zero-phase filter.
- [T1] Apply the 60 Hz notch filter.
- [T1] Apply common average reference.
- [T1] Resample or verify data at 256 Hz.
- [T1] Mark windows above 500 µV as rejected.
- [T1] Compute each subject's z-score statistics from interictal data only.
- [T1] Record rejected samples for every file.

**Done when:** One checked plot shows no flat channels and the preprocessing log
has one entry for every selected file.

## P3. Segmentation

**Purpose:** Create non-overlapping labeled segments.

**Input:** Clean EEG and seizure annotations.

**Output:** `data/manifests/segments.csv`.

**Must build:**

- [T1] Create ictal segments from seizure onset to seizure offset.
- [T1] Create preictal segments for the 30 minutes before onset.
- [T1] Create interictal segments outside the buffer.
- [T1] Exclude postictal data.
- [T1] Exclude preictal data that overlaps a previous seizure.
- [T1] Exclude interictal data inside the buffer.
- [T1] Write subject, file, start, end, label, and seizure ID.
- [T1] Write class counts for every subject.

**Done when:** A test proves that segments from different classes do not
overlap, and the manifest has all required fields.

## P4. Graph Construction and Chunking

**Purpose:** Convert each valid segment into fixed-size graph chunks.

**Input:** Clean EEG and the segment manifest.

**Output:** Pearson graph chunks in `data/processed/`.

**Must build:**

- [T1] Cut each segment into 5-second windows with 50% overlap.
- [T1] Compute Pearson connectivity for every window.
- [T1] Keep the top-k edges for each node.
- [T1] Compute five-band power features for every node and window.
- [T1] Store one graph per window with a fixed tensor shape.
- [T1] Group 60 windows into a chunk.
- [T1] Use the configured stride for training, validation, and test.
- [T1] Drop every chunk that crosses a class, seizure, file, or exclusion
  boundary.
- [T1] Give each chunk the label of its segment.
- [T1] Add a direct chunk test for boundaries, labels, and tensor shapes.
- [T1] Add one graph sanity plot with a planted connectivity change.

**Done when:** The chunk test passes for all boundary types and all subjects
have the same tensor shapes.

## P5. LOSO Splits

**Purpose:** Make one fixed, leakage-free split for every model.

**Input:** Processed graph chunks.

**Output:** `data/manifests/folds.json` and data loaders.

**Must build:**

- [T1] Hold out one subject in each LOSO test fold.
- [T1] Split validation data from training subjects by seizure ID.
- [T1] Keep one seizure in one split only.
- [T1] Apply class weighting or focal loss during training.
- [T1] Undersample interictal data in training folds only.
- [T1] Test that no subject or seizure crosses train, validation, and test sets.
- [T1] Save the same fold manifest for every model.

**Done when:** The leakage test passes for every fold.

## P6. Models

**Purpose:** Implement the smallest model ladder that answers the research
question.

**Input:** Fixed graph chunks and data loaders.

**Output:** Models A to D with a shared input and output contract.

**Must build:**

- [T1] **Model A:** SVM or random forest on flattened connectivity features.
  Use it as an untuned classical floor.
- [T1] **Model B:** Static GCN or GAT on one aggregated graph per chunk.
  Use it as a non-temporal graph baseline.
- [T1] **Model C:** Per-window GNN, GRU, and classification head. Use it as the
  main T1 dynamic baseline.
- [T2] **Model D:** SLATE-style unified spatio-temporal transformer. Include a
  supra-graph builder, supra-Laplacian eigenvector encoder, graph transformer,
  three-class head, and attention readout.
- [T1] Test forward and backward passes for Models A to C.
- [T2] Test the forward and backward pass for Model D.
- [T2] Benchmark one Model D chunk for time and memory before full training.
- [T2] Cache the supra-Laplacian eigenvectors.
- [T2] If the cost is too high, use a shorter chunk or fewer channels. Record
  the fallback in the decision log.

**Done when:** Models A to C pass their tests. Model D passes the cost gate and
its forward and backward test.

## P7. Training

**Purpose:** Train each model on the fixed folds.

**Input:** Models, loaders, folds, and merged config.

**Output:** One run folder per model, fold, and task.

**Must build:**

- [T1] Train with validation after each epoch.
- [T1] Use weighted cross-entropy or focal loss.
- [T1] Undersample interictal data in training folds only.
- [T1] Stop early on validation macro-F1.
- [T1] Save the best checkpoint.
- [T1] Save the merged config, logs, and metrics.
- [T1] Train Models A, B, and C on all LOSO folds.
- [T2] Train Model D only after M1 and its cost gate pass.
- [T2] Run the preictal-versus-interictal task.
- [T1] Never tune on the held-out subject.

**Done when:** Every required model and task has a checkpoint and a complete run
folder for every fold.

## P8. Evaluation

**Purpose:** Measure the main and second tasks without alarm post-processing.

**Input:** Fold checkpoints and held-out predictions.

**Output:** Per-subject and aggregate result tables.

**Must build:**

- [T1] Report macro-F1.
- [T1] Report sensitivity for every class.
- [T1] Report specificity for every class.
- [T1] Report one-vs-rest AUROC for every class.
- [T1] Report each held-out subject separately.
- [T1] Report mean and standard deviation across subjects.
- [T2] Run and report preictal-versus-interictal results.
- [T1] Trace every number to a run folder.
- [T1] Keep false alarms per hour and lead time out of T1 and T2. These need
  alarm post-processing rules: threshold, minimum alarm length, and refractory
  period.

**Done when:** The result tables contain all required metrics for every model,
fold, subject, and task.

## P9. Results and Figures

**Purpose:** Create only the plots needed to inspect the data and report
results.

**Input:** Graph sanity data, predictions, and result tables.

**Output:** Tables and figures in `results/`.

**Must build:**

- [T1] Create a connectivity sanity plot with the planted change.
- [T1] Create the main model comparison table.
- [T1] Create per-subject and task tables.
- [T2] Create a cost benchmark record for Model D.

**Done when:** The required T1 and T2 tables exist, and each figure has a
caption and a source run.

## P10. Packaging

**Purpose:** Make the research pipeline repeatable without deployment work.

**Input:** Source code, configs, tests, and result artifacts.

**Output:** A clean research repository.

**Must build:**

- [T1] Keep `make smoke` as the default local check. It runs the full pipeline
  on synthetic EEG with a planted connectivity change for one epoch.
- [T1] Keep the full CHB-MIT run separate and resumable.
- [T1] Make sure the default command does not download data.
- [T1] Keep the README focused on the problem, method, results, and commands.
- [T1] Keep unit tests for segments, chunks, leakage, model passes, and smoke.
- [T1] Keep a short report with methods, results, and limits.
- [T1] Do not add deployment, Docker, CI/CD, serving, or model registry work.

**Done when:** `make smoke` completes without downloading data, and a full
CHB-MIT run can resume from saved artifacts.

## Milestones

- **M1: Data contract proven.** P0 to P5 are done. The leakage test and chunk
  test pass. The synthetic smoke test passes.
- **M2: First results.** Models A, B, and C train on all LOSO folds. The first
  results table exists.
- **M3: Main model.** Model D passes the cost benchmark and trains on all LOSO
  folds. The binary task runs. The final comparison table exists.

## Phase Dependencies

| Phase | Needs |
|---|---|
| P1 | P0 |
| P2 | P1 |
| P3 | P2 |
| P4 | P3 |
| P5 | P4 |
| P6 | P4 data shapes |
| P7 | P5 and P6 |
| P8 | P7 |
| P9 | P4, P7, and P8 |
| P10 | P8 and P9 |

Model D starts only after M1 is complete. P6 can implement Models A to C after
P4 fixes the data shapes.

## Rules for the Whole Pipeline

- Use one manifest and one fold file for every model.
- Never share normalization statistics across subjects.
- Compute subject-local statistics from interictal data only.
- Never let a chunk cross a class, seizure, file, or exclusion boundary.
- Use the segment label for every window and chunk in that segment.
- Apply class balancing to training data only.
- Never tune on the held-out subject.
- Save the merged config with every run.
- Keep the full CHB-MIT run separate from the synthetic smoke path.
- Treat attention maps as exploratory.

## Stretch

These T3 items are optional: PLV, ictal-versus-non-ictal classification,
ablation sweeps, false alarms per hour, lead-time analysis, exploratory
attention figures, SALoM-inspired memory, and significance tests.
