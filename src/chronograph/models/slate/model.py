"""Model D: SLATE-style unified spatio-temporal transformer."""

from __future__ import annotations

import torch
from torch import nn
from pathlib import Path

from .graph_transformer import GraphTransformer
from .spectral_encoding import SupraLaplacianEigenvectorEncoder


class SLATEModel(nn.Module):
    """Combine supra-graph spectral features with graph attention."""

    def __init__(
        self,
        node_features: int = 5,
        hidden_dim: int = 32,
        eigenvectors: int = 8,
        output_classes: int = 3,
        eigenvector_cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.supra_graph_builder = nn.Identity()
        self.eigenvector_encoder = SupraLaplacianEigenvectorEncoder(
            eigenvectors,
            None if eigenvector_cache_dir is None else Path(eigenvector_cache_dir),
        )
        self.graph_transformer = GraphTransformer(
            node_features + eigenvectors, hidden_dim
        )
        self.attention_readout = nn.Linear(hidden_dim, 1)
        self.head = nn.Linear(hidden_dim, output_classes)

    def forward(
        self, adjacency: torch.Tensor, node_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eigenvectors = self.eigenvector_encoder(self.supra_graph_builder(adjacency))
        combined = torch.cat((node_features, eigenvectors), dim=-1)
        batch, steps, nodes, dimensions = combined.shape
        encoded, attention = self.graph_transformer(
            combined.reshape(batch, steps * nodes, dimensions)
        )
        scores = self.attention_readout(encoded).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        pooled = (encoded * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(pooled), attention
