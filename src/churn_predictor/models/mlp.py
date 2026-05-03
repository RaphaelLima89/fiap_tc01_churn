"""MLP simples para classificação binária de churn."""
from __future__ import annotations

import torch.nn as nn


class ChurnMLP(nn.Module):
    """MLP 23 -> 64 -> 32 -> 1 (logit) com dropout 0.3 e ReLU."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))  # logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)