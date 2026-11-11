# Why Model D Could Outperform Model A

## 1. Conclusion

Model D can outperform Model A because it uses more of the structure already
present in the EEG representation.

Model A treats a complete graph sequence as one flattened vector. Model D
treats the same data as a dynamic spatial-temporal system. It can therefore
learn:

- which channels interact;
- how those interactions change over time;
- which spectral features occur at those channels;
- which channel-time regions matter for the final seizure-state decision.

This is a theoretical advantage, not an observed result. Model D should only be
declared better if it improves held-out subject performance under exactly the
same preprocessing, folds, class policy, and evaluation metrics.

---

## 2. What Model A Sees

Model A is an SVM over flattened adjacency values:

```text
[time, nodes, nodes] -> one long feature vector
```

Its strengths are simplicity, low computational cost, and a useful baseline.
It can learn decision boundaries over connectivity values, but it does not
explicitly understand:

- that the same node exists at every time step;
- that two adjacent time steps form a sequence;
- that graph neighborhoods have structure;
- that node spectral features belong to particular graph positions;
- that a connectivity pattern may evolve gradually.

The SVM can still discover some of these patterns indirectly through its
weights, but the representation does not make them explicit.

---

## 3. What Additional Information Model D Uses

Model D receives:

1. the per-window adjacency matrices;
2. five-band node features;
3. the fixed node identity across time;
4. temporal links between matching nodes;
5. supra-Laplacian spectral coordinates;
6. transformer attention across all channel-time tokens.

The model therefore operates on a richer representation than Model A.

The comparison is not simply:

```text
SVM versus larger neural network
```

It is primarily:

```text
flattened connectivity versus structured dynamic graph representation
```

---

## 4. Mechanisms That Could Produce Better Performance

### 4.1 Modeling temporal evolution

Seizure-state information may be expressed by a transition rather than by one
static graph. For example, a group of channels may gradually become more
strongly connected as a seizure approaches.

Model D can attend to the sequence:

```text
G1 -> G2 -> ... -> G12
```

and learn patterns of strengthening, weakening, or redistribution. Model A
receives the same values in a flattened arrangement but has no explicit
temporal mechanism.

This is most relevant to preictal classification, where the difference from
interictal activity may be a gradual change rather than a single extreme
measurement.

### 4.2 Preserving channel-time identity

In Model D, a token represents a known pair:

```text
(channel_i, time_t)
```

Temporal links connect `(channel_i, time_t)` to `(channel_i, time_t+1)`.
This preserves the idea that the same EEG channel is being followed through
time.

Model A does not have a learned notion of channel continuity. It sees a vector
position, but not an explicit spatial-temporal identity.

### 4.3 Learning long-range interactions

Transformer attention allows one channel-time token to interact with another
token anywhere in the 32.5-second chunk.

This can represent patterns such as:

- an early change in one channel followed by a later change in another;
- a spectral shift that matters only when a distant channel is also active;
- distributed connectivity changes involving several regions.

A flattened SVM can model interactions only through its decision function and
the feature representation. It does not construct an explicit attention-based
relationship between tokens.

### 4.4 Combining connectivity and local spectral information

Model A currently uses adjacency values only. Model D combines:

```text
local node features + graph structural coordinates
```

The node features describe the signal's relative energy across delta, theta,
alpha, beta, and gamma bands. The graph structure describes relationships
between channels.

This may help when two classes have similar connectivity strength but different
frequency composition.

### 4.5 Using the dynamic graph as one object

The supra-graph connects all graph snapshots. Its spectral coordinates describe
smooth organization over both space and time.

This can help Model D recognize that two observations are related because they
are:

- neighboring channels;
- the same channel at neighboring times;
- part of the same connected dynamic pattern.

Model A has no equivalent structural coordinate system.

### 4.6 Adaptive readout

Model D uses learned attention pooling rather than treating every channel and
time point as equally important.

This may allow different chunks to emphasize different evidence:

- seizure-related channels during ictal activity;
- gradually changing regions during preictal activity;
- stable background structure during interictal activity.

Model A produces one decision from the complete flattened vector and has no
explicit evidence-weighting readout.

---

## 5. Where Model D Should Have the Biggest Advantage

The strongest expected advantage is the preictal-versus-interictal task.

Ictal activity may already be sufficiently distinctive for a simple model.
Preictal activity is more likely to require:

- weak distributed changes;
- temporal context;
- interactions between local spectral power and connectivity;
- subject-independent structural patterns.

A plausible outcome is therefore:

| Class/task | Expected Model D opportunity |
|---|---|
| Ictal | Moderate; Model A may already perform well |
| Interictal | Moderate; depends on background variability |
| Preictal | Highest theoretical opportunity |
| Three-class macro-F1 | Improvement only if preictal gains do not harm other classes |
| Binary preictal/interictal | Most important test of the hypothesis |

This is a research expectation, not a result.

---

## 6. Conditions Required for Model D to Win

Model D can only be considered better if the following conditions hold:

1. Both models use the same final P2/P4 materialization.
2. Both models use the same LOSO fold manifest.
3. The held-out subjects are never used for tuning.
4. Class balancing is applied consistently and only to training data.
5. Model D passes its runtime and memory gate.
6. Improvement appears across several held-out subjects.
7. Improvement is visible in macro-F1 and per-class metrics, not only accuracy.
8. Preictal recall improves without collapsing ictal or interictal recall.
9. The result remains meaningful in the binary preictal-versus-interictal task.
10. The gain is larger than ordinary fold-to-fold variation.

The correct comparison is:

```text
same data + same folds + same evaluation protocol + different model
```

---

## 7. Metrics That Would Support the Claim

The strongest evidence would be a consistent pattern such as:

- higher mean LOSO macro-F1 for Model D;
- higher preictal recall and AUROC;
- no major reduction in ictal recall;
- similar or better interictal recall;
- improvement on most held-out subjects rather than one subject;
- lower variance or at least no extreme instability;
- Model D cost that is acceptable for the research objective.

An illustrative result pattern that would support the claim is:

| Metric | Model A | Model D | Interpretation |
|---|---:|---:|---|
| Mean macro-F1 | 0.34 | 0.42 | Overall improvement |
| Preictal recall | 0.38 | 0.52 | Main scientific gain |
| Ictal recall | 0.63 | 0.68 | No loss on obvious class |
| Interictal recall | 0.41 | 0.46 | Fewer background errors |
| Macro AUROC | 0.67 | 0.75 | Better ranking across classes |

These values are examples of an evidential pattern, not projected experimental
results.

---

## 8. Reasons Model D May Not Win

The richer architecture can also fail.

### 8.1 Too little data

The compatible cohort contains only ten subjects, and the estimated ictal
chunk count is small. Model D may have too many degrees of freedom for the
available independent examples.

### 8.2 Subject variability

The graph structure may change across subjects for reasons unrelated to seizure
state. Model D can learn subject-specific patterns rather than general
seizure-state patterns.

### 8.3 Sparse ictal coverage

Short seizures may not produce complete 32.5-second chunks. A high ictal score
could therefore be unstable.

### 8.4 Spectral encoding instability

Eigenvector signs are arbitrary, and near-repeated eigenvalues can produce
different equivalent bases. The model must learn a robust use of these
coordinates rather than relying on one exact basis.

### 8.5 Overfitting through attention

Attention can memorize particular channel-time patterns in the training
subjects. A higher training score does not establish better cross-subject
generalization.

### 8.6 Computational limitations

The current trainer processes a fold as one batch. Memory pressure may force
small hidden dimensions or fewer epochs, reducing Model D's practical
advantage.

### 8.7 Model A may already be sufficient

If seizure-state information is strongly encoded in absolute connectivity,
flattening may lose little useful information. In that case, Model D's
additional structure may not justify its complexity.

---

## 9. Required Ablation Logic

If Model D wins, ablations should identify why.

| Ablation | Question |
|---|---|
| Model A | Is connectivity alone useful? |
| Model D without node features | Do spectral graph coordinates help alone? |
| Model D without spectral encoding | Does the transformer help without graph coordinates? |
| Model D with standard time/electrode encoding | Is supra-Laplacian structure useful? |
| Model D without sparsification | Do weak edges contain useful information? |
| Model D full | Reference architecture |

Without these comparisons, an improvement cannot be attributed specifically to
the supra-graph or spectral encoding. It may come only from the extra node
features or the larger model.

---

## 10. Decision Rule

Use the following practical decision rule:

### Choose Model D as the main model if:

- it improves mean LOSO macro-F1 over Model A;
- the gain is present on most held-out subjects;
- preictal performance improves;
- class-specific failures do not become worse;
- the result survives the binary preictal-versus-interictal analysis;
- the computational cost is acceptable.

### Retain Model A as the preferred model if:

- Model D does not beat it consistently;
- Model D improves only on validation but not held-out test data;
- Model D's apparent gain comes from ictal chunks alone;
- Model D is unstable across subjects;
- the cost is disproportionate to the improvement.

A simpler model with similar performance is the better scientific result. Model
D is justified only by reliable additional information, not by architectural
complexity itself.

---

## 11. Final Position

Model D has a credible theoretical route to outperform Model A because it
preserves the dynamic graph structure, combines connectivity with node spectra,
and learns across channel-time positions.

Its most important potential gain is improved preictal recognition. Its most
important risk is overfitting the small and imbalanced subject cohort.

Therefore, the correct conclusion before training is conditional:

> Model D is the more expressive hypothesis-testing model, while Model A is the
> essential control. Model D should replace Model A as the main model only if
> its added spatio-temporal structure produces stable held-out-subject gains
> under identical evaluation conditions.

