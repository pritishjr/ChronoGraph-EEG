# Projected Training Metrics

## Status

These are **planning estimates, not measured results**. No Model A or Model D
training run has completed.

The current evidence is limited because:

- `data/processed/` contains no final graph chunks;
- `results/tables/main_comparison.csv` contains only its header;
- the current interim data is incomplete and covers five subjects;
- the final LOSO folds have not been regenerated from final P4 output.

No value below should be reported as an experimental result.

## Observed Partial-Data Baseline

The incomplete interim materialization currently contains 36,291 windows:

| Class | Windows | Share |
|---|---:|---:|
| Interictal | 14,400 | 39.68% |
| Preictal | 21,244 | 58.54% |
| Ictal | 647 | 1.78% |

This is not the final cohort distribution. It is useful only for identifying
the expected class-imbalance risk.

### Analytical baselines

| Baseline | Macro-F1 | Balanced accuracy |
|---|---:|---:|
| Always predict preictal | 0.246 | 0.333 |
| Random class prediction | approximately 0.333 | approximately 0.333 |

The majority baseline can have a deceptively reasonable accuracy while having
zero recall for interictal and ictal samples. Macro-F1 and per-class recall are
therefore more informative.

## Planning Ranges

The following ranges are conservative expectations for a leakage-aware LOSO
experiment with the current cohort, chunking policy, and class imbalance. They
are not confidence intervals and are not derived from a fitted model.

| Model | Macro-F1 | Ictal recall | Preictal recall | Interictal recall | Macro AUROC |
|---|---:|---:|---:|---:|---:|
| Model A SVM | 0.25–0.45 | 0.25–0.65 | 0.35–0.70 | 0.25–0.65 | 0.55–0.75 |
| Model D SLATE-style | 0.30–0.55 | 0.40–0.80 | 0.35–0.70 | 0.30–0.70 | 0.60–0.80 |

The most plausible outcome is not necessarily that Model D wins. With only
ten compatible subjects, very few complete ictal chunks, and a small hidden
dimension, Model D may match or underperform Model A. A useful result would be
either:

1. Model D improves macro-F1 and preictal recall consistently across held-out
   subjects; or
2. Model A performs similarly, showing that the extra spectral-transformer
   complexity is not justified for this dataset.

## Expected Per-Class Pattern

The likely qualitative pattern is:

- **Ictal:** highest recall, because seizure activity is usually the most
  separable state.
- **Interictal:** moderate recall, affected by heterogeneous background EEG.
- **Preictal:** lowest and most variable recall, because it is subtle and
  subject-specific.

The ictal estimate is especially uncertain. Short seizures may not produce a
complete 32.5-second chunk, so a high or low ictal score may reflect a small
number of examples rather than stable generalization.

## What Would Count as Encouraging

These are practical decision rules, not statistical claims:

- Model A macro-F1 clearly above 0.33 across most folds;
- Model D macro-F1 at least 0.05 above Model A on the same folds;
- preictal recall above the majority baseline without collapsing ictal recall;
- no single subject accounting for the apparent improvement;
- similar conclusions from macro-F1, balanced accuracy, and AUROC;
- confusion matrices showing that the model is not simply predicting preictal.

## What Would Count as a Negative Result

A negative result is still useful if:

- both models remain near the majority baseline;
- Model D does not beat Model A;
- preictal recall is close to zero;
- performance varies widely by held-out subject;
- ictal performance is unstable because of too few chunks.

Such a result would indicate that the current representation, cohort size, or
label definition is insufficient for a reliable claim. It would not prove that
dynamic graph models cannot work on EEG.

## Required Measurements Before Replacing Projections

The projections should be replaced by real values only after:

1. completing P2 for all selected subjects;
2. rebuilding P4 from the final P2/P3 revision;
3. regenerating and validating `folds.json`;
4. benchmarking Model D on a 12-window chunk;
5. fitting Model A on every LOSO fold;
6. fitting Model D on every LOSO fold;
7. evaluating untouched test partitions.

The final report must show per-subject metrics, class counts, confusion
matrices, AUROC, and mean plus standard deviation across folds.

