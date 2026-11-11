# ChronoGraph-EEG Training Pipeline

## 1. Purpose

This document defines the training workflow for the two models currently in
scope:

- **Model A:** SVM classical baseline.
- **Model D:** SLATE-style spatio-temporal graph transformer.

Both models use the same processed graph chunks and the same leakage-aware
LOSO fold manifest. This is required so that differences in performance reflect
the model rather than different data partitions.

Training is a P7 activity. It must not begin until P2, P4, and P5 have produced
final, mutually consistent artifacts.

---

## 2. Shared Training Inputs

Training consumes:

```text
data/processed/
data/manifests/folds.json
configs/base.yaml
configs/models/svm.yaml
configs/models/slate.yaml
```

### 2.1 Processed graph chunks

Each graph file contains fixed-shape chunk data:

```text
adjacency:     [chunks, 12, 23, 23]
node_features: [chunks, 12, 23, 5]
labels:        [chunks]
subject
source_file
segment_ids
```

The current chunk represents 12 overlapping five-second windows, equivalent to
32.5 seconds of signal. A chunk must remain inside one class segment and one
recording context.

### 2.2 Fold manifest

`folds.json` defines one fold per held-out subject group:

```text
fold:
  train:      processed chunk IDs
  validation: processed chunk IDs
  test:       processed chunk IDs
```

The current training entry point uses the training and validation IDs. Test
records remain untouched until the evaluation stage.

Each fold must satisfy:

- no held-out subject appears in training or validation;
- no seizure group crosses partitions;
- no interictal group crosses partitions;
- no raw time interval is reused across partitions;
- validation is drawn only from training subjects.

---

## 3. Shared P7 Workflow

For each model and each LOSO fold:

1. Load the base configuration.
2. Load the model-specific configuration.
3. Merge model values over the base values.
4. Read the selected fold from `folds.json`.
5. Load only the requested training and validation chunks.
6. Apply interictal undersampling to training data if enabled.
7. Create the model.
8. Train or fit on the training partition.
9. Select the best state using validation macro-F1.
10. Save the model checkpoint.
11. Save the merged configuration.
12. Save validation metrics.

The test partition is not used in this loop. It is reserved for P8 evaluation
after model selection is complete.

### 3.1 Class policy

The configured primary labels are:

```text
interictal, preictal, ictal
```

The secondary task uses:

```text
interictal, preictal
```

Interictal undersampling is allowed only on the training partition. Validation
and test distributions remain unchanged.

Model D uses class-weighted cross-entropy. This is a training choice and must
be recorded with every run.

### 3.2 Run directory

Runs are stored under:

```text
experiments/<model>/<task>/<fold>/
```

Expected artifacts:

```text
config.yaml
metrics.json
checkpoint.pkl   # Model A
checkpoint.pt    # Model D
```

The saved configuration is the merged configuration actually used by the run,
not only the base configuration.

---

## 4. Model A Training Pipeline

### 4.1 Role

Model A is the untuned classical floor. Its purpose is to answer whether the
graph connectivity representation contains useful class information before
introducing a larger neural architecture.

It is a reference model, not the final research model.

### 4.2 Input representation

For Model A, each chunk's adjacency tensor is flattened:

```text
[12, 23, 23] -> one feature vector
```

The current baseline uses adjacency values as the input representation. The
five-band node features are not used by this baseline.

### 4.3 Estimator

Model A uses a support vector classifier with:

- probability estimates enabled;
- balanced class weighting;
- no extensive hyperparameter search;
- one fitted estimator per LOSO fold.

The class labels are mapped to integer targets using the task's configured
class order.

### 4.4 Fold procedure

For each fold:

1. Load the training adjacency arrays.
2. Load the validation adjacency arrays.
3. Optionally undersample interictal training chunks.
4. Flatten the training adjacency tensors.
5. Fit the SVM.
6. Predict validation labels.
7. Compute validation macro-F1.
8. Serialize the fitted estimator.

### 4.5 Model A artifacts

The run writes:

```text
experiments/svm/<task>/<fold>/checkpoint.pkl
experiments/svm/<task>/<fold>/config.yaml
experiments/svm/<task>/<fold>/metrics.json
```

The minimum current metric is:

```text
validation_macro_f1
```

P8 should later add per-class recall, confusion matrices, AUROC, and held-out
test metrics.

### 4.6 Model A interpretation

Model A tests whether static flattened connectivity is already sufficient for
classification. It does not explicitly model:

- graph topology as a graph operator;
- temporal order as a learned sequence;
- node features;
- dynamic edge evolution.

Therefore, a strong Model A result would indicate that the additional
complexity of Model D must be justified by measurable improvement or useful
complementary behavior.

---

## 5. Model D Training Pipeline

### 5.1 Role

Model D is the main experimental spatio-temporal model. It tests whether a
sequence of graph snapshots can be treated as one unified dynamic graph.

### 5.2 Input representation

Model D receives both:

```text
adjacency:     [batch, 12, 23, 23]
node_features: [batch, 12, 23, 5]
```

The adjacency sequence is used to construct the supra-Laplacian spectral
encoding. The node features are combined with those structural coordinates
before transformer processing.

### 5.3 Structural encoding

The 12 graph snapshots are connected through temporal links between matching
nodes in adjacent windows.

This creates a supra-graph with:

```text
12 * 23 = 276 node-time positions
```

The smallest supra-Laplacian eigenvectors are used as positional coordinates.
The configured number of eigenvectors is eight.

Eigenvectors are cached under:

```text
data/processed/eigenvectors/
```

The cache is keyed by the graph content and encoding size. It is a derived
artifact and must be regenerated if graph construction, node order, chunk
boundaries, coupling, or encoding size changes.

### 5.4 Transformer and readout

Model D:

1. computes spectral coordinates;
2. concatenates them with the five node features;
3. treats all node-time positions as transformer tokens;
4. applies graph-transformer attention;
5. pools token representations with learned attention weights;
6. predicts the configured class set.

The default hidden width is 32. The default optimizer is Adam with a learning
rate of `0.001`. Validation macro-F1 controls early stopping with a patience
of three epochs.

### 5.5 Fold procedure

For each fold:

1. Load training adjacency, node features, and labels.
2. Load validation adjacency, node features, and labels.
3. Optionally undersample interictal training chunks.
4. Initialize Model D from the merged model configuration.
5. Compute or load cached spectral encodings.
6. Run the configured number of epochs.
7. Compute weighted cross-entropy on the training partition.
8. Compute validation macro-F1.
9. Save the best validation checkpoint.
10. Stop after the configured patience without improvement.

The current trainer processes each fold as one batch. This keeps the training
path small, but it means memory usage must be measured before full LOSO
training.

### 5.6 Model D artifacts

The run writes:

```text
experiments/slate/<task>/<fold>/checkpoint.pt
experiments/slate/<task>/<fold>/config.yaml
experiments/slate/<task>/<fold>/metrics.json
```

The minimum current metrics are:

```text
epoch
validation_macro_f1
```

The attention output is available during the forward pass for later analysis,
but it should not be treated as a clinical explanation.

---

## 6. Configuration

The base file defines shared data, task, split, and training settings:

```text
configs/base.yaml
```

Model-specific files override or extend the base:

```text
configs/models/svm.yaml
configs/models/slate.yaml
```

The merge rule is:

```text
model configuration overrides base configuration
```

Nested mappings are merged recursively so a model override does not remove
unrelated base settings.

Important Model D values are:

| Setting | Value |
|---|---:|
| Hidden width | 32 |
| Eigenvectors | 8 |
| Learning rate | 0.001 |
| Early-stopping patience | 3 |
| Eigenvector cache | Enabled |
| Output classes | 3 |

The command-line `--epochs` option overrides the configured epoch count for a
single run.

---

## 7. Commands

The training module exposes one-fold execution.

### Model A

```bash
PYTHONPATH=src .venv/bin/python -m chronograph.training \
  --model svm \
  --fold patient_01
```

### Model D

```bash
PYTHONPATH=src .venv/bin/python -m chronograph.training \
  --model slate \
  --fold patient_01
```

### Short development run

```bash
PYTHONPATH=src .venv/bin/python -m chronograph.training \
  --model slate \
  --fold patient_01 \
  --epochs 1
```

### Explicit paths

```bash
PYTHONPATH=src .venv/bin/python -m chronograph.training \
  --model slate \
  --fold patient_01 \
  --processed-root data/processed \
  --folds data/manifests/folds.json \
  --runs-root experiments
```

The commands must not be run until the corresponding fold and processed data
exist.

---

## 8. Training Gates

### Gate 1: Data readiness

- P2 materialization is complete.
- P4 was rebuilt from the same P2/P3 revision.
- Every graph file has valid shapes and labels.
- No stale processed files remain.

### Gate 2: Fold readiness

- `folds.json` was generated from final P4 data.
- Every fold has non-empty train and validation sets.
- All leakage checks pass.
- Class coverage is recorded per fold.

### Gate 3: Model A baseline

- Model A runs on every fold.
- Checkpoints and validation metrics are saved.
- Per-fold class distributions are recorded.

### Gate 4: Model D cost

Before full Model D training:

- benchmark one representative chunk;
- record runtime;
- record peak memory;
- verify eigenvector caching;
- verify forward and backward passes;
- confirm the current batch strategy fits available memory.

### Gate 5: Model D training

- train only after the cost gate passes;
- keep the fold manifest unchanged;
- save the merged configuration for each run;
- report per-subject results rather than only pooled results.

---

## 9. Evaluation Outputs Required After Training

Training metrics are not the final research result. P8 must evaluate the saved
checkpoints on untouched test partitions.

Required outputs:

- per-fold macro-F1;
- per-class recall;
- per-class precision;
- per-class specificity where defined;
- one-vs-rest AUROC;
- confusion matrices;
- per-subject results;
- mean and standard deviation across held-out subjects;
- binary preictal-versus-interictal results;
- class counts and usable-seizure counts per fold.

The artificial interictal retention cap means false alarms per hour must not be
reported from the capped training/evaluation sample without a separate
duration-preserving analysis.

---

## 10. Reproducibility Rules

Every training run must record:

- git revision;
- merged configuration;
- task;
- held-out fold;
- random seed;
- class mapping;
- interictal undersampling policy;
- processed-data manifest revision;
- model checkpoint;
- validation metrics;
- software environment;
- hardware and device information.

The same fold manifest must be used for Model A and Model D. A model comparison
is invalid if the folds, class policy, or preprocessing materialization differ.

---

## 11. Current Status

The P7 entry point and trainer are implemented for one-fold execution.
Model-specific configuration merging is wired and tested. The repository test
suite passes.

Full training is still pending because the final corrected P2/P4
materialization and final `folds.json` must be confirmed first. The recommended
order is:

1. complete and validate P2;
2. rebuild and validate P4;
3. generate and validate P5 folds;
4. benchmark Model D;
5. train Model A on all folds;
6. train Model D on all folds;
7. run held-out evaluation.

