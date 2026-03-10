from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch

from src.inference.feature_builder import FeatureBuilder
from src.models.factory import build_model
from src.utils.artifacts import FeatureSpec, load_feature_spec
from src.utils.logging import setup_logging


def _apply_standardization(x: np.ndarray, spec: FeatureSpec, *, user_id: str | None) -> np.ndarray:
    stats = spec.normalization
    stype = str(stats.get("type", "standard"))
    if stype == "standard":
        means: Dict[str, float] = stats["mean"]
        stds: Dict[str, float] = stats["std"]
    elif stype == "per_user_standard":
        if user_id is None:
            raise ValueError("user_id is required for per-user normalization.")
        per_user = stats["per_user"]
        if user_id not in per_user:
            raise ValueError(f"Unknown user_id for per-user normalization: {user_id}")
        means = per_user[user_id]["mean"]
        stds = per_user[user_id]["std"]
    else:
        raise ValueError(f"Unsupported normalization type: {stype}")

    out = x.copy()
    for j, col in enumerate(spec.feature_cols):
        out[:, j] = (out[:, j] - float(means[col])) / float(stds[col])
    return out


@dataclass
class Predictor:
    model_path: Path
    feature_spec_path: Path
    device: torch.device
    mc_dropout_samples: int = 30

    def __post_init__(self) -> None:
        self.logger = setup_logging("predictor")
        self.spec = load_feature_spec(self.feature_spec_path)
        self.builder = FeatureBuilder(self.spec)

        ckpt = torch.load(self.model_path, map_location=self.device)
        self.model = build_model(ckpt["model_cfg"], input_dim=len(self.spec.feature_cols)).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])

    @torch.no_grad()
    def predict(self, raw_window: pd.DataFrame) -> Tuple[float, float]:
        user_id = None
        if self.spec.user_id_col in raw_window.columns and len(raw_window) > 0:
            user_id = str(raw_window[self.spec.user_id_col].iloc[0])
        feats = self.builder.build_window_features(raw_window)
        feats = _apply_standardization(feats, self.spec, user_id=user_id)
        x = torch.from_numpy(feats).unsqueeze(0).to(self.device)  # (1, T, F)

        # MC dropout for uncertainty: keep dropout active.
        self.model.train()
        samples = []
        for _ in range(int(self.mc_dropout_samples)):
            samples.append(float(self.model(x).squeeze(0).detach().cpu().item()))
        mean = float(np.mean(samples))
        std = float(np.std(samples, ddof=0))

        self.model.eval()
        return mean, std

