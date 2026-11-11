# ChronoGraph-EEG Technical Report

## 1. Executive Summary

ChronoGraph-EEG is a reproducible research pipeline for classifying scalp EEG
into three seizure states:

1. **Interictal** — background activity outside the seizure-related windows.
2. **Preictal** — the configured warning period before seizure onset.
3. **Ictal** — the seizure itself.

The project uses the CHB-MIT scalp EEG dataset. The central research idea is to
convert short EEG windows into dynamic graphs, then compare a classical
connectivity baseline with a spatio-temporal graph-transformer model.

The implemented scope is intentionally limited to two models:

- **Model A:** an SVM baseline over flattened graph connectivity.
- **Model D:** a SLATE-style spatio-temporal transformer using supra-Laplacian
  spectral encodings.

The work is organized into reproducible phases P0–P10. P0–P4 are the data and
graph-construction foundation. P5 creates leakage-aware evaluation folds. P6
contains the model implementations. P7 contains training scripts, but full
training has deliberately not started yet.

At the current checkpoint:

- The selected cohort contains 10 subjects with a shared 23-position channel
  layout.
- P2/P3/P4 implementations and focused tests exist.
- The full test suite currently passes.
- The latest corrected P2 materialization was interrupted partway through and
  must finish before the generated graph data is treated as final.
- P5 folds, Model D cost benchmarking, training, and evaluation remain next.

---

## 2. Research Objective

### 2.1 Main question

Can EEG connectivity graphs distinguish interictal, preictal, and ictal states
in a subject-independent evaluation setting?

### 2.2 Secondary question

Can the same representation distinguish preictal activity from interictal
activity, which is the more clinically meaningful and difficult distinction?

### 2.3 Why a graph representation?

Each EEG channel is treated as a graph node. Statistical relationships between
channels form graph edges. The graph therefore represents spatial relationships
between EEG signals, while a sequence of graphs represents temporal evolution.

This design separates three types of information:

- **Node identity:** the fixed bipolar channel positions.
- **Edge structure:** Pearson connectivity between channels.
- **Node features:** relative spectral power in five frequency bands.

The graph sequence is then consumed by a temporal model. Model A provides a
simple floor, while Model D tests whether a unified spatio-temporal graph
representation is useful.

### 2.4 Research boundaries

The repository is a research pipeline, not a deployment system. It does not
currently claim:

- real-time prediction;
- clinical validation;
- alarm performance or false alarms per hour;
- seizure lead-time performance;
- production serving;
- a complete model-ablation study;
- generalization to every CHB-MIT subject.

Zero-phase filtering is used, which is appropriate for offline experiments but
uses future samples relative to any individual time point. Results must
therefore be described as offline classification results.

---

## 3. System Architecture

The intended pipeline is:

```text
Raw CHB-MIT EDF files
        |
        v
P1: acquisition and channel audit
        |
        v
P3: seizure-aware segment manifest
        |
        v
P2: filtered and normalized EEG windows
        |
        v
P4: graph construction and temporal chunking
        |
        v
P5: LOSO and seizure-aware validation folds
        |
        v
P6: Model A and Model D
        |
        v
P7: training and checkpoints
        |
        v
P8: evaluation
        |
        v
P9/P10: results, figures, and packaging
```

### 3.1 Why P3 runs before P2

Preprocessing is manifest-driven. The segment manifest determines which samples
belong to interictal, preictal, or ictal regions. It also determines which
samples must be excluded because they are too close to a seizure.

This order prevents P2 from creating unlabeled or ambiguously labeled windows.
The dependency is:

```text
P1 -> P3 -> P2 -> P4 -> P5 -> P6/P7
```

### 3.2 Storage strategy

The original design considered saving complete filtered EDF files. That was
rejected because the estimated intermediate storage was too large for repeated
experimentation.

The current compact strategy stores only retained five-second windows in
compressed NumPy-compatible files:

```text
data/interim/<subject>/<source_file>.npz
```

Each file contains:

- `data`
- `labels`
- `window_starts`
- `artifact_windows`
- sampling rate
- channel positions
- subject-local normalization mean
- subject-local normalization standard deviation

The graph stage then writes chunk-level files under:

```text
data/processed/<subject>/<source_file>.npz
```

---

## 4. Dataset and Cohort Decisions

### 4.1 Dataset

The project uses the CHB-MIT scalp EEG dataset distributed as EDF recordings
with subject summary files and seizure annotations.

The raw dataset contains approximately:

- 24 subject directories;
- 687 EDF files;
- multiple channel layouts;
- seizure and non-seizure recordings;
- files with different annotation formats.

### 4.2 Selected cohort

The current fixed cohort is:

```text
chb01, chb02, chb03, chb05, chb06,
chb07, chb08, chb10, chb23, chb24
```

These subjects were selected because they share the chosen 23-position layout.

The excluded subjects contain either:

- changing channel layouts across files; or
- stable layouts that do not match the selected 23-position contract.

This is a deliberate fixed-shape decision. It avoids silently inserting missing
channels, changing node order, or comparing graphs with different meanings.

### 4.3 Channel contract

The graph tensor contract is:

```text
nodes = 23
adjacency = [23, 23]
node_features = [23, 5]
sampling_rate = 256 Hz
```

The source EDF files contain a repeated `T8-P8` label. The implementation
retains the 23 source positions, which means there are 22 unique channel names
but 23 fixed channel positions.

This detail must remain documented because channel names alone are not a
complete identifier for the source layout.

### 4.4 Subject grouping

The fold logic supports aliases so recordings belonging to one physical
subject can be held out together. The current configuration maps `chb01` and
`chb21` to one patient group, although `chb21` is not in the selected cohort.

Before final P5 generation, the subject identity mapping should be verified
against the dataset documentation and applied consistently to the selected
cohort.

---

## 5. P0: Data Contract

P0 fixes choices that would otherwise change the meaning of an experiment.
Important current values are:

| Area | Current decision |
|---|---|
| Sampling rate | 256 Hz |
| Bandpass | 0.5–40 Hz |
| Filter phase | Zero phase |
| Notch | 60 Hz |
| Reference | Preserve native bipolar derivations; no common-average reference |
| ICA | Disabled |
| Window | 5 seconds |
| Window overlap | 50% |
| Chunk length | 12 windows |
| Chunk duration | 32.5 seconds |
| Connectivity | Pearson correlation |
| Edge weights | Absolute Pearson values |
| Edge sign | Stored separately |
| Sparsification | Top-k, with k = 5 |
| Node features | Log-relative power in five bands |
| Main task | Interictal / preictal / ictal |
| Secondary task | Interictal / preictal |
| Evaluation | Leave-one-subject-out |
| Validation grouping | Seizure-aware |
| Interictal retention | At most two hours per subject |

### 5.1 Why common-average reference was removed

The source channels are bipolar derivations. Applying common-average reference
to a collection of bipolar differences subtracts the mean of unlike voltage
differences and forces the channels toward a sum-zero relationship.

That can distort Pearson connectivity and create artificial negative or
dependent relationships. Native bipolar signals already suppress common-mode
activity, so the current pipeline leaves the reference unchanged.

### 5.2 Why the chunk length changed

With five-second windows and 50% overlap, a 12-window chunk spans:

```text
5 + (12 - 1) * 2.5 = 32.5 seconds
```

The seizure-duration audit found 70 seizures in the selected cohort. Approximate
coverage was:

| Chunk | Duration | Seizures long enough |
|---:|---:|---:|
| 12 windows | 32.5 seconds | 43 / 70 |
| 24 windows | 62.5 seconds | 31 / 70 |
| 60 windows | 152.5 seconds | 4 / 70 |

The original 60-window design would leave the ictal class nearly empty. The
12-window design is therefore a data-coverage compromise.

---

## 6. P1: Acquisition and Channel Audit

P1 is responsible for validating the raw input before expensive processing.

The audit:

1. Finds subject directories.
2. Reads EDF headers without loading all signal samples.
3. Records channel labels and sampling rates.
4. Counts channel layouts per subject.
5. Selects the largest stable compatible cohort.
6. Records selected and excluded subjects.
7. Checks seizure-index references against available annotation sidecars.

The primary output is:

```text
data/manifests/channel_audit.csv
```

### 6.1 P1 risks still requiring attention

The audit is functional, but the following should be completed before final
results:

- verify every parsed seizure count against `RECORDS-WITH-SEIZURES`;
- record EDF duration and inter-file timing in the audit;
- verify downloaded files using the dataset checksum information, if available;
- enforce canonical channel selection by normalized channel name and position.

The last point is especially important because selecting the first 23 channels
can preserve tensor shape while silently changing channel meaning if an EDF is
reordered.

---

## 7. P3: Segmentation

P3 converts seizure annotations into non-overlapping labeled regions.

### 7.1 Labels

For each seizure:

- **ictal:** seizure onset through seizure offset;
- **preictal:** the configured warning period before onset;
- **prediction horizon:** a five-minute gap immediately before onset;
- **interictal:** data outside seizure windows and exclusion buffers.

The five-minute horizon prevents the preictal class from becoming an almost
direct seizure-detection task.

### 7.2 Exclusion rules

The segment builder excludes:

- the four-hour neighborhood around seizures from interictal data;
- postictal regions;
- preictal regions that overlap a previous seizure;
- intervals that cannot be safely assigned to one class;
- windows that would cross an EDF or recording boundary.

### 7.3 Recording continuity

EDF files are not always safe to concatenate. P3 reads recording timestamps and
uses a configured maximum gap of 30 seconds. Large gaps are preserved in the
timeline rather than being treated as continuous EEG.

This prevents a segment from spanning a period where no EEG was recorded.

### 7.4 Manifest outputs

P3 writes:

```text
data/manifests/segments.csv
data/manifests/segment_counts.csv
```

Each segment records:

- subject;
- source EDF file;
- start time within that EDF;
- end time within that EDF;
- class label;
- seizure or interictal group identifier.

Interictal segments receive stable group identifiers so that validation cannot
split one continuous interictal region across train and validation.

### 7.5 Annotation parser

The parser accepts both:

```text
Seizure Start Time
Seizure End Time
```

and numbered forms such as:

```text
Seizure 1 Start Time
Seizure 1 End Time
```

This correction was necessary because several CHB-MIT subjects use numbered
annotation labels.

---

## 8. P2: Preprocessing

P2 turns raw EDF files into compact, labeled, normalized windows.

### 8.1 Processing sequence

For each selected subject:

1. Read EDF files.
2. Verify the sampling rate.
3. Select the configured channel positions.
4. Apply the 0.5–40 Hz zero-phase filter.
5. Apply the 60 Hz notch filter.
6. Compute subject-local mean and standard deviation.
7. Generate windows fully contained in one P3 segment.
8. Flag high-amplitude windows.
9. Reject only artifacted interictal windows.
10. Retain all preictal and ictal windows.
11. Enforce the interictal storage cap.
12. Normalize and store the retained windows.

### 8.2 Normalization

The current implementation computes statistics from all available EDF samples
for one subject, after filtering, without using seizure labels.

This avoids using held-out labels to define preprocessing statistics. Statistics
are still subject-local and are never shared across subjects.

The normalization is:

```text
normalized = (signal - subject_mean) / subject_std
```

### 8.3 Artifact handling

The amplitude threshold is 500 µV. MNE stores EEG values in volts, so the
implementation compares against:

```text
500 * 1e-6 volts
```

Artifact policy:

- artifacted interictal windows are rejected;
- artifacted preictal windows are retained;
- artifacted ictal windows are retained;
- seizure-state windows retain an artifact flag for later analysis.

This prevents high-amplitude seizure activity from being removed simply because
it exceeds a generic amplitude threshold.

### 8.4 Storage format

P2 stores:

```text
data:              [N, 23, 1280] float16
labels:            [N]
window_starts:     [N]
artifact_windows:  [N] boolean
mean_v:            [23] float32
std_v:             [23] float32
```

The five-second sample count is:

```text
256 * 5 = 1280 samples
```

### 8.5 P2 execution status

The corrected P2 rerun is resumable by skipping existing per-file outputs, but
the latest execution was interrupted after a partial run. The current generated
interim data must be completed and validated before the P4 output is considered
final.

The required validation is:

- every selected EDF is represented in the preprocessing log;
- every retained file has `[N, 23, 1280]` data;
- all stored data are float16;
- labels, starts, and artifact flags have matching lengths;
- the interictal cap is respected per subject;
- P2 and P3 use the same manifest revision.

---

## 9. P4: Graph Construction and Chunking

P4 transforms each retained EEG window into graph data.

### 9.1 Pearson connectivity

For a window with channels as rows, P4 centers each channel and computes the
channel-by-channel Pearson matrix.

The calculation is performed in float32. Final graph arrays can be stored in a
smaller representation after construction.

### 9.2 Edge weights and signs

Pearson correlation is signed, but a standard graph Laplacian requires
non-negative edge weights. P4 therefore stores:

```text
edge_weight = abs(Pearson correlation)
edge_sign   = sign(Pearson correlation)
```

The diagonal is zero.

### 9.3 Top-k sparsification

For each node, the five strongest absolute correlations are selected. The
selection is symmetrized so the final graph is compatible with undirected
Laplacian operations.

Symmetrization means a node can end up with more than five neighbors when
another node selects it. The important invariant is that the graph is symmetric
and contains the strongest directed selections.

### 9.4 Node features

For every node and window, P4 computes power in:

- delta: 0.5–4 Hz;
- theta: 4–8 Hz;
- alpha: 8–13 Hz;
- beta: 13–30 Hz;
- gamma: 30–40 Hz.

The stored feature is log-relative band power:

```text
log(band_power / total_power)
```

This reduces dependence on absolute amplitude and emphasizes spectral
composition.

### 9.5 Chunking

The current chunk contains 12 graph windows. It is valid only when:

- all windows belong to one segment;
- all windows have one class label;
- all windows have one segment/group ID;
- the sequence stays within one source file and its manifest boundaries.

The default current P4 materialization uses a 60-window stride. This produces
non-overlapping evaluation-style chunks. Training-specific overlap should be
added deliberately at the training-data construction stage rather than silently
mixing train and test stride rules in one file.

### 9.6 P4 output

Each processed graph file contains:

```text
adjacency:    [chunks, 12, 23, 23]
edge_sign:    [chunks, 12, 23, 23]
node_features:[chunks, 12, 23, 5]
labels:       [chunks]
segment_ids:  [chunks]
subject
source_file
```

P4 skips files with no valid manifest windows or no complete chunks rather than
crashing on an empty `np.stack`.

---

## 10. P5: Leakage-Safe Evaluation

P5 creates one fixed fold manifest shared by all models.

### 10.1 LOSO design

Leave-one-subject-out means one subject or patient group is held out for test
in each fold. All other subjects provide training data and a validation split.

The test subject is never used for:

- model selection;
- early stopping;
- threshold tuning;
- preprocessing statistics;
- class-balancing decisions.

### 10.2 Validation grouping

Training-subject data is split by group rather than by individual chunk. A
seizure and its preictal windows remain in one partition.

Interictal segments now also have explicit group IDs. This avoids treating every
interictal chunk as unrelated when it comes from one continuous recording.

### 10.3 Leakage checks

The intended checks are:

- no record ID appears in two partitions;
- no seizure/group ID appears in two partitions;
- no subject group appears in both test and training;
- no raw time interval is reused across partitions.

P5 is not complete until `folds.json` is generated from the final P4 outputs and
all checks pass on the generated manifest.

---

## 11. P6: Model Scope

The project intentionally implements only Model A and Model D.

### 11.1 Model A: classical baseline

Model A is an SVM baseline over flattened graph connectivity features.

Purpose:

- establish a simple performance floor;
- determine whether graph connectivity contains useful class information;
- provide a low-complexity comparison for Model D.

It is not intended to be a heavily tuned production classifier.

### 11.2 Model D: SLATE-style graph transformer

Model D represents a chunk as a temporal sequence of graph snapshots.

Its components are:

1. graph input;
2. supra-graph construction across time;
3. supra-Laplacian construction;
4. smallest-eigenvector positional encoding;
5. graph-transformer layers;
6. attention-based temporal/spatial readout;
7. three-class output head.

The eigenvectors are cached by an adjacency-derived hash to avoid repeating the
same expensive decomposition.

### 11.3 Model D research limitation

The supra-Laplacian representation is theoretically motivated, but its benefit
for EEG classification is not established by this repository alone. Attention
weights are exploratory explanations, not proof of neurophysiological
importance.

The first Model D gate must therefore be computational:

- time per chunk;
- peak memory;
- CPU/GPU behavior;
- eigenvector-cache behavior.

Only after that gate should full LOSO training begin.

---

## 12. P7: Training Design

The training scripts support:

- fold-specific data loading;
- Model A and Model D selection;
- weighted cross-entropy;
- interictal undersampling in training data only;
- validation macro-F1;
- early stopping;
- checkpoint saving;
- configuration snapshots;
- metric JSON output.

Training is intentionally not started yet because the final P2/P4/P5 chain
must be stable first.

The required training order is:

1. Model A on all LOSO folds.
2. Review Model A metrics and class coverage.
3. Run Model D cost benchmark.
4. Train Model D only if the cost is acceptable.
5. Run the secondary preictal-versus-interictal task after the primary path is
   validated.

---

## 13. Testing and Validation

The repository currently contains focused tests for:

### P2 tests

- filtering changes the signal;
- common-average reference is not applied;
- amplitude flags use voltage units;
- artifact rejection affects interictal windows only;
- windows remain inside segment boundaries;
- exact-length windows are retained.

### P3 tests

- numbered seizure annotations parse correctly;
- adjacent segments do not mix labels;
- recording gaps are preserved in the timeline.

### P4 tests

- Pearson output shape is fixed;
- sparsified graphs are non-negative and symmetric;
- band features have five values per node;
- chunks crossing a boundary are dropped;
- files with no valid manifest windows are skipped safely.

### P5 tests

- LOSO test records use the held-out subject;
- validation is grouped by seizure ID;
- patient aliases remain in one held-out fold;
- partition leakage assertions work.

### Model tests

- Model A fit/predict path;
- Model D forward pass;
- Model D backward pass;
- spectral/eigenvector components.

### Current test result

The latest complete test run passed:

```text
22 passed
```

Warnings remain for the installed scikit-learn SVM probability option and an
old NVIDIA driver. They do not currently fail the test suite.

---

## 14. Current Repository Status

### Implemented

- P0 configuration and decision records;
- P1 channel audit implementation;
- P3 segmentation implementation;
- P2 compact preprocessing implementation;
- P4 graph and chunk implementation;
- P5 fold-generation implementation;
- Model A;
- Model D;
- P7 training scaffolding;
- focused regression tests.

### Generated artifacts

The repository contains generated manifests and raw/interim data locally.
Generated graph output must be treated as valid only when it was produced from
the same P2/P3 revision.

The latest known execution state is:

- corrected P2 materialization was started;
- the run was interrupted after partial subject processing;
- P4 output must be rebuilt from the completed corrected P2 run;
- P5 folds have not yet been finalized from the final P4 output.

### Not complete

- final P2 materialization;
- final P4 materialization after the latest P2 corrections;
- generated `folds.json` from final graph chunks;
- complete P5 leakage validation on generated folds;
- Model D cost benchmark;
- full Model A training;
- Model D training;
- evaluation tables;
- results figures;
- final packaging cleanup.

---

## 15. Main Risks and Limitations

### 15.1 Cohort limitation

Only the selected 10-subject compatible cohort is currently supported. Results
must not be described as results over all CHB-MIT subjects.

### 15.2 Artificial interictal ratio

The two-hour interictal cap makes the stored dataset unsuitable for reporting
false alarms per hour. It is a storage and classification-balance decision, not
a representation of natural recording prevalence.

### 15.3 Short ictal segments

Many seizures are shorter than the 32.5-second chunk duration. The chunking
policy therefore produces fewer ictal chunks than preictal or interictal
chunks. Per-subject and per-class metrics must show this imbalance.

### 15.4 Subject-local normalization

All-data subject statistics avoid label-based normalization, but they still use
the full subject recording during offline preprocessing. This is acceptable for
the current offline research design, but it is not a causal real-time
normalization procedure.

### 15.5 Channel-layout assumptions

The fixed 23-position contract is necessary for the current graph models. It
also limits the cohort and requires strict validation of channel order.

### 15.6 Model evidence

Model D is an experimental architecture. A successful forward/backward test
does not establish that it improves seizure-state classification. Model A,
data-quality checks, and leakage-safe evaluation remain the reference points.

---

## 16. Recommended Execution Plan

### Immediate

1. Finish or resume P2 from the current `segments.csv`.
2. Validate every P2 file and the per-subject interictal cap.
3. Remove any stale P4 files.
4. Rebuild P4 using the final P2/P3 pair.
5. Validate graph tensor shapes, labels, and chunk counts.

### Before training

6. Generate `data/manifests/folds.json`.
7. Run leakage checks over every generated fold.
8. Verify that every fold contains the expected classes and non-empty
   validation data.
9. Run the Model D cost benchmark on the 12-window chunk.

### First experiments

10. Train Model A across all LOSO folds.
11. Produce per-subject macro-F1, class sensitivity, specificity, AUROC, and
    confusion matrices.
12. Train Model D only after the cost gate.
13. Run the secondary preictal-versus-interictal task.

### Final reporting

14. Record the exact merged configuration for every run.
15. Report per-subject results and the macro average across held-out subjects.
16. Explain the cohort restriction, interictal cap, chunk duration, and
    offline filtering choice.
17. Add figures only after the numerical results are traceable to run folders.

---

## 17. Presentation Summary

For a slide deck, the project can be summarized as:

1. **Problem:** classify EEG into interictal, preictal, and ictal states.
2. **Data:** CHB-MIT EDF recordings; 10 compatible subjects; 23 bipolar
   positions.
3. **Pipeline:** audit → segment → preprocess → graph → split → model →
   evaluate.
4. **Preprocessing:** 0.5–40 Hz filter, 60 Hz notch, native bipolar reference,
   subject-local normalization, artifact-aware retention.
5. **Segmentation:** seizure-centered labels, five-minute prediction horizon,
   four-hour interictal buffer, recording-gap protection.
6. **Graph:** Pearson connectivity, absolute edge weights, separate edge signs,
   five-band log-relative node features.
7. **Temporal input:** 12 five-second windows with 50% overlap; 32.5-second
   chunks.
8. **Evaluation:** LOSO with seizure/group-aware validation.
9. **Models:** Model A SVM baseline and Model D SLATE-style transformer.
10. **Current status:** implementation and tests are ready; final materialization,
    folds, benchmarking, and training remain.
11. **Main risks:** cohort restriction, short ictal segments, artificial
    interictal ratio, and the experimental status of Model D.
12. **Next milestone:** complete P2–P5 and pass the Model D cost gate before
    training.

---

## 18. Theoretical Problem Definition

The pipeline estimates a seizure state from a fixed multichannel EEG sequence:

```text
X(t) ∈ R^23
f(X[t : t + 32.5 seconds]) -> {interictal, preictal, ictal}
```

The central difficulty is that EEG is multichannel, noisy, non-stationary, and
temporally dependent. Ictal activity may be relatively easy to separate, while
preictal activity is subtler and more subject-specific. The project therefore
asks whether channel relationships and their changes over time contain
information that a simple flattened connectivity vector misses.

The main hypothesis is:

> A dynamic graph representation can capture seizure-state information through
> both spatial connectivity and temporal evolution, improving or complementing
> a classical connectivity baseline.

This is a subject-independent research question. A model must generalize to a
held-out subject, not merely memorize a patient's signal scale or recording
conditions.

---

## 19. What SLATE Means in This Project

SLATE is used here as a model pattern: represent a sequence of changing graphs
as one structured spatio-temporal object, derive graph-aware coordinates, and
let a transformer attend across channel-time positions.

A conventional baseline either flattens the graph or models spatial and
temporal structure in separate stages. Model D instead allows a token for one
channel at one time to interact with any other channel-time token. This can
represent patterns such as:

- a connectivity change that develops across several windows;
- a spectral change in one channel aligned with a change elsewhere;
- a relationship that is important only during a specific temporal phase.

We are not claiming that the original SLATE task and EEG classification are
identical. We use its unified spatio-temporal graph idea to test the hypothesis
above. The architecture must still prove its value against Model A under LOSO
evaluation.

SLATE-style encoding does not guarantee better accuracy, causal
interpretability, clinical validity, or real-time suitability.

---

## 20. Dataset and Channel Decisions

### 20.1 Selected dataset

CHB-MIT was selected because it provides scalp EEG in EDF format together with
subject summaries and seizure annotations. The files can be audited, aligned
to seizure intervals, and processed reproducibly.

### 20.2 Selected subjects

The current compatible cohort is:

```text
chb01, chb02, chb03, chb05, chb06,
chb07, chb08, chb10, chb23, chb24
```

These subjects share the fixed 23-position channel layout. Other subjects were
excluded because they either change layouts or use a stable layout with a
different number and meaning of channels.

The smaller cohort is deliberate. Silently padding or reordering channels
would produce tensors with matching dimensions but different node semantics.
The limitation is that conclusions apply to this compatible cohort, not
automatically to all CHB-MIT subjects.

### 20.3 Channel contract

Every graph uses:

```text
23 nodes
adjacency: [23, 23]
node features: [23, 5]
```

The source headers contain a repeated `T8-P8` label. The contract therefore
uses 23 fixed channel positions even though there are 22 unique names. Position
and normalized name must be audited together.

### 20.4 Sampling rate

The common sampling rate is 256 Hz. A five-second window consequently contains:

```text
256 * 5 = 1280 samples
```

This rate preserves the 0.5–40 Hz analysis band while keeping tensors
manageable and consistent across the selected subjects.

---

## 21. Channel Audit, Segmentation, and Preprocessing

### 21.1 Channel audit

The channel audit is the input-integrity gate. It records files, sampling
rates, channel labels, layout changes, selected subjects, and exclusion
reasons. It is necessary because correct tensor dimensions do not guarantee
correct channel meanings.

The audit output is:

```text
data/manifests/channel_audit.csv
```

The implementation reads EDF headers, checks layout stability, and identifies
the compatible cohort before expensive sample processing.

### 21.2 Segmentation

Segmentation converts seizure annotations into mutually exclusive time regions:

```text
interictal | preictal | prediction horizon | ictal | excluded/postictal
```

It is required because one EDF can contain background, preictal, ictal, and
postictal activity. File-level labels would mix these states.

Implemented rules include:

- numbered and unnumbered seizure-summary parsing;
- 30-minute preictal intervals;
- five-minute prediction horizon;
- four-hour interictal seizure buffer;
- postictal exclusion;
- protection against preictal overlap with a previous seizure;
- 30-second maximum recording-gap handling;
- stable interictal group IDs.

Outputs are `segments.csv` and `segment_counts.csv`. P2 uses this manifest so
every retained window receives one explicit segment label.

### 21.3 Preprocessing

P2 performs channel selection, filtering, normalization, fixed-window
extraction, artifact marking, label-aware retention, and compact storage.
Filtering and normalization occur before connectivity because Pearson
correlation and spectral power are sensitive to drift, noise, and scale.

---

## 22. Preprocessing Rationale

### 22.1 0.5–40 Hz zero-phase bandpass

The lower cutoff removes slow drift and baseline movement. The upper cutoff
reduces high-frequency noise while retaining the frequency range used by the
graph node features.

Zero-phase filtering avoids phase shifts in this offline study. It uses future
samples relative to a time point, so the result must not be described as causal
real-time preprocessing.

### 22.2 60 Hz notch

The notch suppresses power-line interference. It is retained as a defensive
step even though the 40 Hz upper cutoff already attenuates much of 60 Hz.

### 22.3 Five-second windows and 50% overlap

Five seconds is long enough for more stable correlation and band-power
estimates, but short enough to preserve changes near seizure transitions.
Fifty-percent overlap starts windows every 2.5 seconds:

```text
0, 2.5, 5.0, 7.5, ...
```

Overlap reduces the chance that an informative event falls between windows.
It also creates correlated examples, which is why chunks and folds must obey
strict group and boundary rules.

### 22.4 Twelve-window chunks

A 12-window chunk spans:

```text
5 + (12 - 1) * 2.5 = 32.5 seconds
```

The duration audit found 70 seizures in the selected cohort. Only 4 supported
the previous 152.5-second design, while 43 supported a 32.5-second chunk.
Twelve windows therefore preserve substantially more ictal coverage.

### 22.5 Normalization

For each subject:

```text
z = (x - mean_subject) / std_subject
```

Statistics are computed without seizure labels and are not shared across
subjects. This prevents label-dependent preprocessing and cross-subject scale
leakage. It remains an offline subject-level choice; a deployed causal system
would need a different normalization policy.

### 22.6 Artifact policy

The 500 µV threshold marks likely artifacts. Artifacted interictal windows are
rejected, but preictal and ictal windows are retained and flagged because
seizure activity can be high amplitude. This prevents the artifact rule from
removing the target phenomenon.

---

## 23. Graph Construction

### 23.1 Why graphs

EEG channels are relational signals, not independent scalar features. A graph
makes that structure explicit:

```text
node = channel position
edge = channel relationship
node feature = local spectral activity
```

A sequence of graphs then represents how those relationships evolve.

### 23.2 Connectivity

For each five-second window, P4 computes:

```text
C_ij = corr(channel_i, channel_j)
```

The diagonal is removed. Edge weights use absolute correlation:

```text
W_ij = |C_ij|
```

The original sign is stored separately:

```text
S_ij = sign(C_ij)
```

Absolute values are used because the standard graph Laplacian requires
non-negative weights; signs remain available for diagnostics and future
signed-graph experiments.

### 23.3 Sparsification

The five strongest absolute relationships per node are selected and then
symmetrized. Sparsification reduces noisy weak edges and computational cost.
Symmetry is required for a consistent undirected spectral representation.

Adjacent bipolar channels can correlate because they share electrodes. This is a
known confounder and should be addressed only through a planned edge-type or
masking ablation, not an undocumented change.

### 23.4 Node features

For every node, P4 estimates delta, theta, alpha, beta, and gamma power. It
stores:

```text
log(band_power / total_power)
```

Relative power reduces dependence on absolute amplitude. The logarithm limits
the dominance of large ratios and gives a more stable feature scale.

---

## 24. Implemented Model Architectures

### 24.1 Model A: SVM baseline

Model A flattens the adjacency sequence:

```text
[12, 23, 23] -> [12 * 23 * 23]
```

and fits a balanced support vector classifier. It uses connectivity only; it
does not explicitly model graph neighborhoods, node features, spectral
structure, or temporal order.

Its purpose is to establish whether connectivity contains a useful signal at
all. If Model D cannot improve on or complement this floor, its added
complexity is not justified.

### 24.2 Model D: SLATE-style transformer

Model D:

1. receives the adjacency sequence and five-dimensional node features;
2. constructs a unified supra-graph;
3. computes spectral positional coordinates;
4. concatenates those coordinates with node features;
5. applies transformer attention across all channel-time tokens;
6. pools tokens using learned attention weights;
7. predicts the configured class set.

The model can learn spatial, temporal, and joint spatial-temporal patterns
without forcing the architecture to process space and time in separate stages.

---

## 25. Supra-Graph and Spectral Encoding

### 25.1 Supra-graph

A supra-graph stacks all temporal graph snapshots into one graph. Each
supra-node is:

```text
(channel_i, time_t)
```

Spatial edges connect channels within a window. Temporal edges connect the same
channel across adjacent windows:

```text
(channel_i, time_t) <-> (channel_i, time_t + 1)
```

The current contract produces:

```text
12 * 23 = 276 supra-nodes
```

### 25.2 Supra-Laplacian

For a non-negative graph `W`:

```text
D_ii = sum_j W_ij
L = D - W
```

The supra-Laplacian applies this to the complete spatial-temporal graph. Its
diagonal blocks contain spatial graph Laplacians; off-diagonal blocks encode
temporal coupling.

### 25.3 Spectral coordinates

The model solves:

```text
L v_k = lambda_k v_k
```

and uses the eight eigenvectors associated with the smallest eigenvalues as
coordinates for each supra-node. A transformer token therefore contains:

```text
five EEG node features + eight graph-structural coordinates
```

The low-frequency eigenvectors describe smooth, large-scale organization over
the dynamic graph. They provide more structure than ordinary sequence
position alone.

Eigenvector signs are arbitrary, and repeated eigenvalues can produce
equivalent bases. Spectral coordinates are therefore model inputs, not stable
biological measurements. The encoding is cached because decomposition is
expensive; the cache is valid only when graph content, node order, chunk
boundaries, coupling, and eigenvector count are unchanged.

---

## 26. Training Pipeline and Hyperparameters

For each LOSO fold:

1. load final processed chunks;
2. load training and validation IDs;
3. verify all IDs exist;
4. undersample interictal training data if enabled;
5. initialize the selected model;
6. fit or train on the training partition;
7. select the best state using validation macro-F1;
8. save checkpoint, merged configuration, and metrics.

The test partition is not used for early stopping or model selection.

### 26.1 Shared settings

| Setting | Value |
|---|---|
| Primary task | Interictal / preictal / ictal |
| Secondary task | Preictal / interictal |
| Split | Leave-one-subject-out |
| Validation | Training subjects only |
| Imbalance policy | Interictal undersampling in training only |
| Neural loss | Weighted cross-entropy |
| Seed | 42 |
| Main validation metric | Macro-F1 |

### 26.2 Model A settings

| Setting | Value |
|---|---|
| Estimator | SVC |
| Input | Flattened adjacency sequence |
| Class handling | Balanced class weights |
| Probability output | Enabled |
| Search | None; baseline |

Model A fits once per fold and reports validation macro-F1.

### 26.3 Model D settings

| Setting | Value |
|---|---:|
| Hidden dimension | 32 |
| Eigenvectors | 8 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Weighted cross-entropy |
| Early-stopping patience | 3 epochs |
| Default epoch count | 10 |
| Spectral cache | Enabled |
| Output classes | 3 |

The ten-epoch value is a development default, not an optimized claim. Final
experiments must record the exact value and merged configuration.

Runs are stored under:

```text
experiments/<model>/<task>/<fold>/
```

with `config.yaml`, `metrics.json`, and either `checkpoint.pkl` or
`checkpoint.pt`.

---

## 27. Required Reporting and Interpretation

The final study should report:

- windows and chunks per class;
- usable seizures per subject;
- artifact counts;
- per-fold class coverage;
- per-subject macro-F1;
- per-class recall and precision;
- confusion matrices;
- one-vs-rest AUROC;
- binary preictal-versus-interictal results;
- mean and standard deviation across held-out subjects;
- Model D runtime, peak memory, and cache behavior.

Attention maps, graph edges, and spectral coordinates are model diagnostics,
not proof of causal brain networks or clinical biomarkers.

The decisive comparison is whether Model D provides reliable subject-independent
gains over Model A under the same data, folds, class policy, and evaluation
metrics.

