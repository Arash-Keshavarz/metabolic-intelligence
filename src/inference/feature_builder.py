from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.artifacts import FeatureSpec


_ROLL_RE = re.compile(r"^(?P<col>.+)_roll(?P<win>\d+)m_(?P<stat>mean|std)$")


def _cyclical_encode(values: np.ndarray, period: float) -> Tuple[np.ndarray, np.ndarray]:
    x = 2 * np.pi * (values.astype(float) / period)
    return np.sin(x), np.cos(x)


@dataclass(frozen=True)
class FeatureBuilder:
    spec: FeatureSpec

    def build_window_features(self, raw_window: pd.DataFrame) -> np.ndarray:
        """
        raw_window: last N rows (N >= window_steps) with at least the base sensor columns.
        Returns features shaped (window_steps, n_features) aligned to spec.feature_cols.
        """
        df = raw_window.copy()
        df[self.spec.timestamp_col] = pd.to_datetime(df[self.spec.timestamp_col])
        df = df.sort_values(self.spec.timestamp_col).reset_index(drop=True)

        # Time encoding if expected.
        needs_hour = any(c in self.spec.feature_cols for c in ("hour_float", "hour_sin", "hour_cos"))
        if needs_hour:
            hour = df[self.spec.timestamp_col].dt.hour + df[self.spec.timestamp_col].dt.minute / 60.0
            df["hour_float"] = hour.astype(float)
            sinv, cosv = _cyclical_encode(hour.to_numpy(), 24.0)
            df["hour_sin"] = sinv
            df["hour_cos"] = cosv

        needs_dow = any(c in self.spec.feature_cols for c in ("dow", "dow_sin", "dow_cos"))
        if needs_dow:
            dow = df[self.spec.timestamp_col].dt.dayofweek.astype(int)
            df["dow"] = dow
            sinv, cosv = _cyclical_encode(dow.to_numpy(), 7.0)
            df["dow_sin"] = sinv
            df["dow_cos"] = cosv

        # Rolling features: computed within the provided window.
        for f in self.spec.feature_cols:
            m = _ROLL_RE.match(f)
            if not m:
                continue
            col = m.group("col")
            win_min = int(m.group("win"))
            stat = m.group("stat")
            w = int(win_min // self.spec.freq_minutes)
            if w <= 0:
                raise ValueError(f"Invalid rolling window: {win_min}m for freq {self.spec.freq_minutes}m")
            if col not in df.columns:
                raise ValueError(f"Missing base column for rolling feature: {col}")
            # Mirror training-time preprocessing: stable rolling stats without NaNs.
            roll = df[col].astype(float).rolling(window=w, min_periods=1)
            if stat == "mean":
                df[f] = roll.mean()
            else:
                df[f] = df[col].astype(float).rolling(window=w, min_periods=2).std(ddof=0).fillna(0.0)

        # Ensure all expected features exist
        missing = [c for c in self.spec.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns at inference: {missing}")

        feats = df[self.spec.feature_cols].astype(float).to_numpy(dtype=np.float32, copy=True)
        if len(feats) < self.spec.window_steps:
            raise ValueError(f"Need at least window_steps={self.spec.window_steps} rows, got {len(feats)}")
        feats = feats[-self.spec.window_steps :]
        return feats

