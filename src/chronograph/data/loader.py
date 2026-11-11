"""Load selected CHB-MIT EDF recordings by channel position."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np


def load_selected_edf(
    path: Path,
    channel_count: int,
    expected_sampling_rate: float,
) -> tuple[np.ndarray, float]:
    """Load the first canonical channel positions from one EDF file."""
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    if len(raw.ch_names) < channel_count:
        raise ValueError(
            f"{path} has {len(raw.ch_names)} channels; "
            f"{channel_count} are required"
        )
    if not np.isclose(raw.info["sfreq"], expected_sampling_rate):
        raise ValueError(
            f"{path} has sampling rate {raw.info['sfreq']}, "
            f"expected {expected_sampling_rate}"
        )
    data = raw.get_data(picks=list(range(channel_count))).astype(np.float32)
    return data, float(raw.info["sfreq"])
