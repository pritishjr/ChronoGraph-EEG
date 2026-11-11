"""Supra-Laplacian spectral encodings."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn


class SupraLaplacianEigenvectorEncoder(nn.Module):
    """Encode chunks with optional disk-cached eigenvectors."""

    def __init__(
        self, eigenvectors: int = 8, cache_dir: Path | None = None
    ) -> None:
        super().__init__()
        self.eigenvectors = eigenvectors
        self.cache_dir = cache_dir

    def _cache_path(self, sample: torch.Tensor) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(
            sample.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()[:16]
        return self.cache_dir / f"{digest}-{self.eigenvectors}.npy"

    def _encode_sample(self, sample: torch.Tensor) -> torch.Tensor:
        cache_path = self._cache_path(sample)
        if cache_path is not None and cache_path.exists():
            return torch.from_numpy(np.load(cache_path)).to(
                device=sample.device, dtype=sample.dtype
            )
        steps, nodes = sample.shape[:2]
        supra = torch.zeros(
            (steps * nodes, steps * nodes),
            device=sample.device,
            dtype=sample.dtype,
        )
        for step in range(steps):
            graph = sample[step]
            degree = torch.diag(graph.abs().sum(dim=-1))
            left = step * nodes
            supra[left:left + nodes, left:left + nodes] = degree - graph
            if step + 1 < steps:
                right = left + nodes
                coupling = -0.1 * torch.eye(
                    nodes, device=sample.device, dtype=sample.dtype
                )
                supra[left:right, right:right + nodes] = coupling
                supra[right:right + nodes, left:right] = coupling
        _, vectors = torch.linalg.eigh(supra)
        result = vectors[:, :self.eigenvectors].reshape(
            steps, nodes, self.eigenvectors
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, result.detach().cpu().numpy())
        return result

    def forward(self, adjacency: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self._encode_sample(sample) for sample in adjacency]
        )
