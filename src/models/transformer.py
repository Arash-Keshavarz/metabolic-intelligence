from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 2048) -> None:
        super().__init__()
        self.dropout = nn.Dropout(float(dropout))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class SimpleTimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int = 2048,
        pooling: str = "last",
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooling = pooling
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        h = self.input_proj(x)
        h = self.pos(h)
        h = self.encoder(h)  # (B, T, D)
        if self.pooling == "mean":
            pooled = h.mean(dim=1)
        else:
            pooled = h[:, -1, :]
        return self.head(pooled).squeeze(-1)

