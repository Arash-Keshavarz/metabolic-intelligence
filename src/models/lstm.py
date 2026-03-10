from __future__ import annotations

import torch
from torch import nn


class BaselineLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=float(dropout) if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, (h, _) = self.lstm(x)
        last = h[-1]  # (B, H)
        last = self.dropout(last)
        pred = self.head(last).squeeze(-1)  # (B,)
        return pred

