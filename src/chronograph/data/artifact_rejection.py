"""Flag amplitude-based EEG artifacts."""

from __future__ import annotations

import numpy as np


def artifact_windows(
    data: np.ndarray,
    sampling_rate_hz: float,
    window_seconds: float,
    amplitude_limit_uv: float,
) -> np.ndarray:
    """Return one boolean flag per complete window."""
    samples_per_window = int(round(sampling_rate_hz * window_seconds))
    window_count = data.shape[1] // samples_per_window
    if window_count == 0:
        return np.zeros(0, dtype=bool)
    windows = data[:, : window_count * samples_per_window].reshape(
        data.shape[0], window_count, samples_per_window
    )
    return np.max(np.abs(windows), axis=(0, 2)) > amplitude_limit_uv * 1e-6
