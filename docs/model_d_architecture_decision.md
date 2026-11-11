# Model D Architecture Decision Record

## Status

**Accepted for experimental evaluation.**

Model D is the project’s unified spatio-temporal graph model. It is an
experimental research architecture, not a production or clinical system.
Training remains gated on final data materialization, leakage validation, and a
computational cost benchmark.

## Decision

Model D will represent each EEG sample as a sequence of graph snapshots and
will process the complete sequence as one unified spatio-temporal graph.

The architecture consists of:

1. fixed EEG nodes;
2. one weighted spatial graph per time window;
3. temporal links between corresponding nodes in adjacent windows;
4. a supra-Laplacian spectral positional encoding;
5. a graph-transformer encoder;
6. an attention-based global readout;
7. a three-class classification head.

The model will be evaluated against Model A, the classical SVM baseline. Models
B and C are outside the current scope.

---

## 1. Research Role

Model D answers the following question:

> Does a unified model of evolving spatial connectivity outperform a simple
> connectivity baseline for subject-independent seizure-state classification?

The model is not intended to prove that a particular electrode or connection is
the cause of a seizure. Its attention and spectral components provide useful
analysis signals, but they are not clinical explanations or causal evidence.

The primary task is:

```text
interictal vs preictal vs ictal
```

The secondary task is:

```text
preictal vs interictal
```

The secondary task is important because ictal classification is expected to be
easier than detecting a meaningful preictal state.

---

## 2. Input Contract

Model D receives one fixed-length sequence of graph snapshots.

For the current pipeline:

```text
time steps        = 12
nodes per step   = 23
node features    = 5
adjacency        = [23, 23]
chunk duration    = 32.5 seconds
```

The five node features represent log-relative power in:

- delta;
- theta;
- alpha;
- beta;
- gamma.

Each adjacency matrix contains non-negative absolute Pearson weights. The sign
of each Pearson relationship is stored separately as an edge attribute for
analysis and future extensions. The current Model D architectural decision
uses the non-negative graph for Laplacian construction.

Every sequence must remain within one segment and one class. A sequence may not
cross:

- a class boundary;
- a seizure boundary;
- an interictal group boundary;
- an EDF file boundary;
- a recording gap;
- an exclusion boundary.

This makes the model input semantically unambiguous: one sequence has one
target label.

---

## 3. Spatial Graph Semantics

Each EEG channel position is a graph node. An edge represents the strength of
the relationship between two channel signals during one five-second window.

The graph is not a physical wiring diagram. It is a statistical connectivity
graph derived from the observed EEG signals.

### 3.1 Edge weights

Pearson correlation is converted to a non-negative weight using its absolute
value. This is required because the graph Laplacian is defined from
non-negative edge weights in this design.

The original correlation sign is retained separately. This preserves the
distinction between positive and negative relationships without making signed
weights part of the spectral operator.

### 3.2 Sparsity

The graph retains the strongest five absolute connections per node and is then
made symmetric.

Sparsity is an architectural compromise:

- it reduces computation;
- it limits weak and noisy relationships;
- it makes the graph easier to inspect;
- it may remove distributed low-strength information.

The full-graph condition is a required future ablation. It answers whether
top-k sparsification helps or harms the classification task.

### 3.3 Shared node identity

The same 23 node positions are used at every time step. Node index 7, for
example, must refer to the same canonical bipolar channel position throughout a
sequence and across subjects.

The fixed node layout is essential to the temporal links and the spectral
encoding. If channel order changes, the model no longer receives the same
graph semantics.

---

## 4. Temporal Graph Semantics

The twelve spatial graphs are connected into one supra-graph.

For every node at time `t`, a temporal link connects it to the corresponding
node at time `t + 1`. These links encode continuity of channel identity across
the sequence.

The resulting graph has:

```text
12 * 23 = 276 supra-nodes
```

The supra-graph contains two kinds of relationships:

1. **spatial edges:** relationships between channel positions within one
   window;
2. **temporal edges:** relationships between the same channel position in
   adjacent windows.

The temporal coupling controls how strongly the spectral representation
preserves continuity over time. A small coupling emphasizes each individual
snapshot; a large coupling encourages smoother temporal coordinates.

The current coupling is a fixed experimental value. It must be treated as a
model hyperparameter and reported with every experiment. It should not be
tuned using the held-out test subject.

---

## 5. Supra-Laplacian Positional Encoding

### 5.1 Motivation

Transformers do not inherently know that two tokens belong to the same EEG
channel at different times, or that two nodes are strongly connected in the
same graph. A graph-derived positional encoding supplies this structural
information.

The supra-Laplacian combines the spatial and temporal structure of the whole
sequence. Its eigenvectors provide coordinates that reflect the organization
of the dynamic graph.

### 5.2 Encoding choice

The model uses eigenvectors associated with the smallest eigenvalues of the
supra-Laplacian. These low-frequency eigenvectors describe broad, smooth
patterns over the connected spatial-temporal structure.

For every supra-node, the selected eigenvectors are appended to the five EEG
node features. The transformer therefore receives both:

- measured spectral EEG information;
- graph-derived structural coordinates.

### 5.3 Why use a supra-graph rather than separate snapshots?

Separate snapshot encodings would describe each graph independently. The
supra-graph gives the encoder one coordinate system for the complete sequence.
This allows spatial and temporal structure to influence the positional
representation jointly.

The tradeoff is that the encoding is more expensive and more sensitive to:

- chunk boundaries;
- temporal coupling;
- graph sparsification;
- eigenvalue multiplicity;
- numerical eigensolver behavior.

### 5.4 Eigenvector ambiguity

An eigenvector can be multiplied by `-1` without changing its mathematical
meaning. Repeated or near-repeated eigenvalues can also produce different but
equivalent bases.

Consequently, individual eigenvector signs or columns must not be interpreted
as stable biological features. The architecture treats them as positional
coordinates for the model, not as directly reportable biomarkers.

### 5.5 Offline computation and caching

The spectral encoding is computed offline for each graph sequence and cached
using an identity derived from the graph content and encoding configuration.
Caching is an efficiency decision only; it must not change the result.

The cache is valid only when all of the following remain unchanged:

- adjacency values;
- sequence boundaries;
- temporal coupling;
- number of eigenvectors;
- node ordering;
- numerical representation rules.

Cache files must therefore be treated as derived artifacts, never as the source
of truth for graph data.

---

## 6. Transformer Representation

After spectral coordinates are added, the time and node dimensions form one
sequence of supra-node tokens.

The transformer attends over all nodes across all time steps. This allows it to
learn:

- spatial interactions within one EEG window;
- changes in connectivity across windows;
- interactions between distant time steps;
- joint spatial-temporal patterns associated with each class.

The model does not impose a hand-written order such as “first process space,
then process time.” Instead, the supra-Laplacian supplies graph structure and
the transformer learns how to combine it.

This is the defining distinction between Model D and a per-window graph model
followed by a recurrent temporal model.

---

## 7. Readout and Prediction

The model uses an attention-based global readout over all supra-node tokens.
Each token receives a learned contribution weight, and the weighted sum forms
one representation for the entire 32.5-second chunk.

The chunk representation is passed to a three-class classification head.

The readout is selected because seizure state is not expected to depend on one
fixed channel or one fixed time point in every subject. A learned global
readout can emphasize different regions of the dynamic graph for different
chunks.

The readout weights are diagnostic only. They should be reported as model
behavior, not as proof that the highlighted channels are medically causal.

---

## 8. Architectural Alternatives Rejected

### 8.1 Static graph per chunk

A single aggregated graph would discard the order and evolution of the
five-second windows. It is suitable as a baseline but does not test the central
dynamic-graph hypothesis.

### 8. Per-window GNN plus GRU

A GNN followed by a GRU is a reasonable dynamic baseline. It is excluded from
the current implementation scope to keep the project limited to Model A and
Model D. It remains a useful future ablation if Model D cannot be interpreted
against a temporal baseline.

### 8. Signed Laplacian

Signed graph operators are possible, but they introduce a separate theoretical
choice for handling positive and negative edges. The current design uses
absolute weights for the Laplacian and preserves signs as separate metadata.

### 8. Full dense connectivity

A dense graph retains every pairwise relationship but increases computation and
may amplify weak or noisy correlations. The current model uses top-k
sparsification, with a full-graph ablation planned before making a stronger
claim about sparsity.

### 8. Cross-attention edge module

The original SLATE family includes edge-focused mechanisms for dynamic link
prediction. The current task is graph-sequence classification, not link
prediction. The architecture therefore uses node/token attention and a
classification readout rather than retaining a task-mismatched edge-prediction
head.

---

## 9. Leakage and Evaluation Constraints

Model D must use the same fold manifest as Model A.

The following are prohibited:

- computing normalization statistics from the held-out subject;
- fitting the spectral encoding to labels;
- selecting the temporal coupling from test performance;
- selecting the number of eigenvectors from test performance;
- tuning the readout or threshold using test chunks;
- placing windows from one seizure or interictal group in multiple partitions.

The supra-Laplacian is unsupervised with respect to the class label, but it can
still leak information if graph sequences from the test subject are used to
choose model settings. All architectural and hyperparameter decisions must be
made using training subjects and their validation partitions.

---

## 10. Expected Benefits

The architecture is expected to provide:

1. joint modeling of spatial connectivity and temporal evolution;
2. a common structural coordinate system for the entire chunk;
3. flexible attention over all channel-time tokens;
4. a compact global representation for classification;
5. exploratory attention and spectral diagnostics.

These are hypotheses, not guaranteed outcomes. The baseline comparison is
required to determine whether the additional complexity is justified.

---

## 11. Expected Costs and Risks

### 11.1 Computational cost

The supra-Laplacian has one row and column per channel-time token. With the
current contract it has 276 dimensions per chunk. Eigen-decomposition must be
benchmarked before full LOSO training.

### 11.2 Memory cost

Attention over all channel-time tokens can grow quadratically with sequence
length. Increasing the number of windows or nodes may make the model
impractical without changing the architecture.

### 11.3 Data imbalance

Short seizures may not produce complete 32.5-second ictal chunks. Model D can
therefore appear strong on common classes while having unreliable ictal
estimates. Per-class and per-subject reporting is mandatory.

### 11.4 Numerical stability

Near-identical graphs, repeated eigenvalues, and floating-point differences can
change the exact eigenvector basis. Evaluation must focus on predictive
behavior, not exact equality of individual eigenvector values.

### 11.5 Interpretability risk

Attention weights and eigenvectors are not explanations in the causal or
clinical sense. They are internal model quantities that may help form
hypotheses, but they require independent validation.

---

## 12. Required Validation Gates

Model D is ready for training only after these gates pass:

1. **Input gate:** all graph chunks satisfy the fixed shape and single-label
   contract.
2. **Leakage gate:** generated LOSO folds pass subject, seizure, interictal
   group, and time-disjointness checks.
3. **Structural gate:** supra-graphs are symmetric and have the expected
   spatial and temporal connectivity.
4. **Spectral gate:** the encoding produces finite values with the requested
   number of coordinates.
5. **Cache gate:** cached and uncached encodings are equivalent for the same
   graph and configuration.
6. **Model gate:** forward and backward passes succeed on representative
   batches.
7. **Cost gate:** one-chunk runtime and peak memory are recorded.
8. **Baseline gate:** Model A has been evaluated before interpreting Model D.

A failed cost gate does not justify silently changing the sequence contract. It
requires an explicit architecture decision, such as reducing the number of
eigenvectors or changing the chunk length, followed by a new validation record.

---

## 13. Experiment and Ablation Plan

The minimum comparisons are:

| Variant | Architectural question |
|---|---|
| Model A SVM | Is graph connectivity useful at all? |
| Model D without spectral encoding | Does the transformer help without graph coordinates? |
| Model D with standard time/electrode encoding | Is the supra-Laplacian encoding useful? |
| Model D without top-k sparsification | Does sparsification help? |
| Full Model D | Reference architecture |

Any ablation must use the same P2/P3/P4 outputs, fold manifest, class policy, and
evaluation metrics. Otherwise differences may come from data changes rather
than architecture.

---

## 14. Decision Summary

Model D is retained because it directly tests the project’s main hypothesis:
that a sequence of EEG connectivity graphs can be represented as one structured
spatio-temporal object.

The design deliberately accepts higher computational cost and greater
mathematical complexity in exchange for a unified representation of spatial
connectivity and temporal change.

The model will not be considered successful merely because it runs. It must:

- outperform or meaningfully complement Model A under LOSO evaluation;
- preserve class and subject-level performance;
- pass leakage checks;
- fit within measured resource limits;
- show stable results across folds.

Until those conditions are met, Model D remains an architectural experiment,
not evidence of a superior EEG classifier.

