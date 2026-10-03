from __future__ import annotations

import torch
from torch import nn


class PricePredictor(nn.Module):
    """Small GRU forecasting a bounded 30-minute log return."""

    def __init__(self, feature_dim: int, hidden_size: int = 96, num_layers: int = 2,
                 dropout: float = 0.15, max_abs_return: float = 0.08):
        super().__init__()
        self.max_abs_return = max_abs_return
        self.norm = nn.LayerNorm(feature_dim)
        self.gru = nn.GRU(feature_dim, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 32),
                                  nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.gru(self.norm(x))
        raw = self.head(sequence[:, -1]).squeeze(-1)
        return torch.tanh(raw) * self.max_abs_return

    def predict_price(self, x: torch.Tensor, current_price: torch.Tensor) -> torch.Tensor:
        return current_price * torch.exp(self(x))
