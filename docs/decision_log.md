# Decision Log

Record each P0 pipeline decision, its rationale, and the validation used.

## P0 Data Contract

| Decision | Value | Reason | Validation |
|---|---|---|---|
| Dataset | CHB-MIT | Required project dataset | File audit in P1 |
| Sampling rate | 256 Hz | Fixed project default | Check every selected EDF file |
| Bandpass filter | 0.5 to 40 Hz, zero phase | Remove slow drift and high-frequency noise | Inspect a cleaned EEG plot |
| Notch filter | 60 Hz | Remove mains noise | Inspect a cleaned EEG plot |
| Reference | Common average | Use a shared scalp reference | Check channel output |
| Amplitude rejection | 500 µV | Reject large artifact windows | Count rejected windows |
| ICA | Not used | Keep the first pipeline small | Confirm the config value |
| Window | 5 seconds with 50% overlap | Fixed graph input window | Check window starts and ends |
| Chunk | 60 windows | Fixed temporal input | Check tensor shape |
| Training chunk stride | 30 windows | Provide more training samples | Check generated chunk starts |
| Validation and test chunk stride | 60 windows | Reduce duplicate evaluation samples | Check generated chunk starts |
| Preictal horizon | 30 minutes | Fixed project default | Check segment boundaries |
| Interictal buffer | At least 4 hours from any seizure | Avoid nearby seizure effects | Check segment boundaries |
| Postictal data | Excluded | Avoid mixed recovery state | Check segment boundaries |
| Connectivity | Pearson for T1 and T2 | Keep the first comparison small | Check matrix values and shape |
| Sparsification | Top-k edges per node | Limit graph size | Check edge count per node |
| Node features | Power in delta, theta, alpha, beta, and gamma bands | Capture band activity | Check feature shape |
| Normalization | Per-subject z-score from interictal data only | Avoid shared subject statistics | Check statistics source and scope |
| Normalization scope | Subject-local evaluation choice | Preserve the stated evaluation design | Check no cross-subject statistics |
| Main task | Interictal, preictal, and ictal | Required research task | Check label values |
| Secondary task | Preictal versus interictal | Required T2 task | Check label values |
| Evaluation | Leave-one-subject-out | Test subject generalization | Check fold manifest |
| Validation split | By seizure ID inside training subjects | Keep one seizure in one split | Run the leakage test |
| Held-out subject tuning | Never allowed | Prevent test leakage | Check training logs |
| Class imbalance | Weighted cross-entropy or focal loss | Reduce class bias | Check merged model config |
| Interictal undersampling | Training folds only | Do not alter validation or test data | Check sampler inputs |

## Values To Set Before P1

- Canonical channels: **TBD**. Set after the channel audit.
- Selected subject subset: **TBD**. Select subjects with the same canonical channels.
- Excluded subjects and reasons: **TBD**. Record them in the channel audit.
- Top-k edges per node: **TBD**. Set after the channel count is fixed.
- Tensor node count and shapes: **TBD**. Set after the canonical channel set is fixed.
