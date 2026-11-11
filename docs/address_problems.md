The plan (@pipeline_architecture.md) is sound and leakage-aware, but three items can make the results invalid or leave the ictal class almost empty: **CAR on bipolar channels**, the **152.5 s chunk length**, and **label-based normalization**. The planted-change graph test and the leakage tests are the strongest parts of the plan. Keep them.

**Priority fixes**

| # | Stage | Problem | Fix |
|---|---|---|---|
| 1 | P2 | CAR subtracts the mean of 18 unrelated voltage differences. The channels then sum to zero. The Pearson matrix becomes singular and shows false negative correlations. | Remove CAR. A bipolar channel already cancels the common-mode signal. |
| 2 | P4 | 60 windows of 5 s with 50% overlap give 5 + 59 × 2.5 = **152.5 s** per chunk. A chunk cannot cross a class boundary. An ictal chunk therefore needs 152.5 s of ictal data. Short seizures give no ictal chunk. | Compute the seizure durations from the summary files. Choose the chunk length from that distribution. |
| 3 | P2 | Interictal-only z-score statistics use the labels of the test subject. A deployed system does not have these labels. | Compute the statistics without labels (all retained data of the subject). Use log relative band power as node features. |
| 4 | P1, P5 | chb01 and chb21 are the same subject. LOSO leaks if you treat them as two subjects. | Give both one subject ID. Hold them out together. |
| 5 | P2 | The 500 µV rejection rule conflicts with "retain all ictal windows". Seizure activity can have high amplitude. | Apply the rule to interictal windows only. Flag, but keep, preictal and ictal windows. MNE returns volts, so check the unit. |

**Chunk length options**

| Windows | Duration | Supra-graph nodes (18 × T) | Attention entries per head and layer |
|---|---|---|---|
| 12 | 32.5 s | 216 | 46,656 |
| 24 | 62.5 s | 432 | 186,624 |
| 60 | 152.5 s | 1,080 | 1,166,400 |

Choose the longest chunk for which most seizures still give at least one ictal chunk. I cannot give the duration distribution from memory. Compute it in P1.

**Notes by stage**

- **P1**
  - Some summary files can use different line formats. Cross-check the parsed seizure counts against `RECORDS-WITH-SEIZURES`.
  - Add to the audit: file duration, gap to the previous file, and the channel mask from the channel-mapping policy.
  - Verify the downloads with the checksum file, if PhysioNet provides one.
- **P2**
  - The 0.5–40 Hz filter already attenuates 60 Hz. Check the attenuation at 60 Hz. Remove the notch filter if the attenuation is sufficient.
  - A zero-phase filter uses future samples. This is acceptable for offline classification. Do not claim real-time prediction.
  - Replace the visual flat-channel check with an automated test. Set the variance threshold in the config.
  - Compute connectivity in float32. Store the final arrays in float16.
  - The interictal-hours limit means the test set has an artificial class ratio. Do not report false alarms per hour from it.
- **P3**
  - Add a **prediction horizon**: a gap of a few minutes between the preictal segment and the seizure onset. Without it, the task is close to seizure detection.
  - Write the postictal exclusion time and the minimum distance from a seizure to interictal data in the config.
  - Record which seizures you keep and why. Some subjects can have very few usable seizures after filtering. Per-subject LOSO results will then be noisy.
  - Join consecutive files only if the gap is below a configured limit. Add a test that every segment lies inside one continuous recording.
- **P4**
  - Use |r| as the edge weight. The Laplacian needs non-negative weights. Store the sign as an edge feature.
  - Adjacent bipolar channels share one electrode, so they correlate by construction. These pairs fill the top-k edges in every window. Mask them or give them a separate edge type.
  - Make the top-k graph symmetric before you build the Laplacian.
  - Broadband Pearson mostly shows low-frequency activity. Compute one matrix per band, or justify one chosen band.
  - Use overlapping chunks in training only. Use stride equal to chunk length in validation and test. Each time point then counts once.
- **P5**
  - Interictal segments have no seizure ID. Give them a group ID (file or contiguous block). Keep a time gap between groups.
  - Validation on training subjects gives optimistic early stopping for a cross-subject goal. Consider one or two held-out validation subjects.
  - Add a test that no raw sample appears in chunks of two different splits.
  - Do not stack class weighting, focal loss, and undersampling without an ablation. Stacked methods distort the class probabilities.
  - Report per-subject results and the macro average.

**Model**

SLATE converts discrete time dynamic graphs into multi-layer graphs. It uses the eigenvectors of the smallest non-zero supra-Laplacian eigenvalues as positional encoding for a fully connected transformer. The authors built it for dynamic link prediction and added a cross-attention edge module. Your task is graph-sequence classification. Drop the edge module and use the attention readout.

- **Fit:** A chunk is a sequence of T snapshots on a fixed set of 18 nodes. The supra-graph is a natural representation. No evidence yet shows that this encoding helps with EEG.
- **Supra-graph:** Link each node to its copy in the next window. This keeps the graph connected. The link weight sets the balance between space and time in the eigenvectors. Tune it on the validation set.
- **Eigenvectors:** Compute them offline and cache them. Overlapping training chunks need separate computation, because the spectrum changes with the chunk boundary. Use random sign flips during training, because eigenvector signs are arbitrary.
- **Size:** You have 22 subjects and a limited number of seizures. Start with a width of 64 and two to four layers. Use dropout and weight decay.
- **Three-class head:** The classes differ in difficulty. Ictal against the rest is easy. Preictal against interictal is the real question. Report per-class recall, per-class AUROC, and the confusion matrix. Repeat the experiment with preictal lengths of 15, 30, and 60 minutes.
- **Attention readout:** It gives weights per channel and time. Use them to form hypotheses. Do not treat them as proof of what the model uses.

**Required ablations**

| Variant | Question it answers |
|---|---|
| Band-power MLP | Does the graph help at all? |
| Per-window GNN + GRU | Does the transformer beat the standard approach? |
| Transformer + sinusoidal time encoding + fixed electrode-layout encoding | Does the supra-Laplacian encoding help? |
| Full model without top-k sparsification | Do the learned edges help? |
| Full model | Reference result |

Next step: run the P1 audit and compute the seizure duration distribution. That result decides the chunk length.

Sources: [Supra-Laplacian Encoding for Transformer on Dynamic Graphs (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1ed4723f12853cbd02aecb8160f5e0c9-Abstract.html)