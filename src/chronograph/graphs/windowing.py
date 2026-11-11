"""Generate fixed EEG windows."""

from __future__ import annotations

import numpy as np


def starts(sample_count: int, sampling_rate_hz: float, window_seconds: float) -> np.ndarray:
    size = int(round(sampling_rate_hz * window_seconds))
    step = size // 2
    return np.arange(0, max(0, sample_count - size + 1), step, dtype=np.int64)
