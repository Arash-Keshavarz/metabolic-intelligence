from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from src.models.lstm import BaselineLSTM
from src.models.transformer import SimpleTimeSeriesTransformer


def build_model(model_cfg: Dict[str, Any], input_dim: int) -> nn.Module:
    name = str(model_cfg["name"]).lower()
    if name == "lstm":
        return BaselineLSTM(
            input_dim=input_dim,
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_layers=int(model_cfg["num_layers"]),
            dropout=float(model_cfg["dropout"]),
        )
    if name == "transformer":
        tcfg = model_cfg.get("transformer", {})
        return SimpleTimeSeriesTransformer(
            input_dim=input_dim,
            d_model=int(tcfg["d_model"]),
            nhead=int(tcfg["nhead"]),
            num_layers=int(tcfg["num_layers"]),
            dim_feedforward=int(tcfg["dim_feedforward"]),
            dropout=float(model_cfg["dropout"]),
            pooling="last",
        )
    raise ValueError(f"Unknown model name: {name}")

