"""Graph transformer blocks for SLATE."""

from __future__ import annotations

import torch
from torch import nn


class GraphTransformer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, heads: int = 4) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.projection(nodes)
        attended, weights = self.attention(values, values, values)
        return self.norm(values + attended), weights
