"""Weighted cross-entropy loss."""

from __future__ import annotations

import torch


def weighted_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    weights = torch.bincount(labels, minlength=logits.shape[-1]).float()
    weights = weights.sum() / weights.clamp_min(1)
    weights = weights / weights.mean()
    return torch.nn.functional.cross_entropy(logits, labels, weight=weights.to(logits.device))
