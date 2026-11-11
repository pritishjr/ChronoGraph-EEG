"""Apply top-k graph sparsification."""

from __future__ import annotations

import numpy as np


def top_k(matrix: np.ndarray, k: int) -> np.ndarray:
    if k < 1 or k >= matrix.shape[-1]:
        raise ValueError("top_k must be between 1 and node_count - 1")
    result = np.zeros_like(matrix, dtype=np.float32)
    scores = np.abs(matrix).copy()
    np.fill_diagonal(scores, -np.inf)
    indices = np.argpartition(scores, -k, axis=-1)[..., -k:]
    rows = np.arange(matrix.shape[-2])[:, None]
    mask = np.zeros_like(matrix, dtype=bool)
    mask[rows, indices] = True
    mask |= mask.T
    return np.where(mask, np.abs(matrix), 0.0).astype(np.float32)
