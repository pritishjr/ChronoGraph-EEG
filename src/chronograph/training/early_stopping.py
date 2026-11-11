"""Early stopping and best-checkpoint tracking."""

from __future__ import annotations


class EarlyStopping:
    def __init__(self, patience: int = 5) -> None:
        self.patience = patience
        self.best = float("-inf")
        self.bad_epochs = 0

    def update(self, score: float) -> bool:
        if score > self.best:
            self.best = score
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience
