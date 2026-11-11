"""Compute Pearson connectivity."""

from __future__ import annotations

import numpy as np


def pearson(window: np.ndarray) -> np.ndarray:
    """Return a channel-by-channel Pearson matrix."""
    centered = window - window.mean(axis=1, keepdims=True)
    scale = np.linalg.norm(centered, axis=1)
    denominator = scale[:, None] * scale[None, :]
    matrix = centered @ centered.T
    return np.divide(matrix, denominator, out=np.zeros_like(matrix), where=denominator > 0)
