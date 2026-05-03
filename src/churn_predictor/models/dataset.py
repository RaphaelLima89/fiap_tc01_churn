"""Dataset PyTorch para o pipeline de churn."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class ChurnDataset(Dataset):
    """Wrapper PyTorch sobre matriz numérica + vetor de target binário."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]