from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.artifacts import FeatureSpec, save_feature_spec
from src.utils.config import load_yaml
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def _steps(minutes: int, freq_minutes: int) -> int:
    if minutes % freq_minutes != 0:
        raise ValueError(f"minutes={minutes} must be divisible by freq_minutes={freq_minutes}")
    return minutes // freq_minutes


def _cyclical_encode(series: pd.Series, period: float, prefix: str) -> pd.DataFrame:
    x = 2 * np.pi * (series.astype(float) / period)
    return pd.DataFrame({f"{prefix}_sin": np.sin(x), f"{prefix}_cos": np.cos(x)}, index=series.index)


def _add_time_features(df: pd.DataFrame, timestamp_col: str, include_dow: bool) -> pd.DataFrame:
    ts = pd.to_datetime(df[timestamp_col])
    df = df.copy()
    df["hour_float"] = ts.dt.hour + ts.dt.minute / 60.0
    df = pd.concat([df, _cyclical_encode(df["hour_float"], 24.0, "hour")], axis=1)
    if include_dow:
        df["dow"] = ts.dt.dayofweek.astype(int)
        df = pd.concat([df, _cyclical_encode(df["dow"], 7.0, "dow")], axis=1)
    return df


def _add_rolling_features(
    df: pd.DataFrame,
    user_id_col: str,
    timestamp_col: str,
    freq_minutes: int,
    windows_minutes: List[int],
    cols: List[str],
) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values([user_id_col, timestamp_col]).reset_index(drop=True)

    for win_min in windows_minutes:
        w = _steps(int(win_min), freq_minutes)
        for c in cols:
            grp = out.groupby(user_id_col, sort=False)[c]
            # For production stability we avoid NaNs: mean uses min_periods=1; std NaNs become 0.
            out[f"{c}_roll{win_min}m_mean"] = grp.rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
            out[f"{c}_roll{win_min}m_std"] = (
                grp.rolling(window=w, min_periods=2).std(ddof=0).reset_index(level=0, drop=True).fillna(0.0)
            )
    return out


def _split_users(user_ids: List[str], seed: int, train_frac: float, val_frac: float) -> Tuple[List[str], List[str], List[str]]:
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(user_ids))
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train = ids[:n_train].tolist()
    val = ids[n_train : n_train + n_val].tolist()
    test = ids[n_train + n_val :].tolist()
    return train, val, test


def _compute_standardization_stats(df_train: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    means = df_train[feature_cols].mean(axis=0, numeric_only=True).to_dict()
    stds = df_train[feature_cols].std(axis=0, ddof=0, numeric_only=True).replace(0.0, 1.0).to_dict()
    return {"type": "standard", "mean": means, "std": stds}


def _apply_standardization(df: pd.DataFrame, feature_cols: List[str], stats: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    means: Dict[str, float] = stats["mean"]
    stds: Dict[str, float] = stats["std"]
    for c in feature_cols:
        mu = float(means[c])
        sd = float(stds[c])
        df[c] = (df[c].astype(float) - mu) / sd
    return df


def preprocess(config: Dict[str, Any]) -> None:
    logger = setup_logging("preprocess")
    seed = int(config.get("seed", 42))
    set_seed(seed)

    raw_path = Path(config["raw_path"])
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    schema = config["schema"]
    timestamp_col = schema["timestamp_col"]
    user_id_col = schema["user_id_col"]
    target_col = schema["target_col"]

    fe = config["feature_engineering"]
    freq_minutes = int(fe["freq_minutes"])
    window_steps = _steps(int(fe["window_minutes"]), freq_minutes)
    horizon_steps = _steps(int(fe["horizon_minutes"]), freq_minutes)

    logger.info("Reading raw data from %s", raw_path.as_posix())
    df = pd.read_csv(raw_path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values([user_id_col, timestamp_col]).reset_index(drop=True)

    if "sleep_stage_code" not in df.columns:
        raise ValueError("Expected `sleep_stage_code` to exist in raw dataset.")

    df = _add_time_features(
        df,
        timestamp_col=timestamp_col,
        include_dow=bool(config["feature_engineering"]["cyclical_time"]["include_day_of_week"]),
    )

    if bool(config["feature_engineering"]["rolling_features"]["enabled"]):
        df = _add_rolling_features(
            df,
            user_id_col=user_id_col,
            timestamp_col=timestamp_col,
            freq_minutes=freq_minutes,
            windows_minutes=[int(x) for x in config["feature_engineering"]["rolling_features"]["windows_minutes"]],
            cols=["glucose", "heart_rate", "HRV", "activity_level"],
        )

    # Define target at +horizon
    target_name = f"{target_col}_tplus_{int(fe['horizon_minutes'])}m"
    df[target_name] = df.groupby(user_id_col, sort=False)[target_col].shift(-horizon_steps)

    # Drop rows without future target (last horizon_steps per user)
    df = df.dropna(subset=[target_name]).reset_index(drop=True)

    # Features (model inputs)
    drop_cols = {timestamp_col, user_id_col, "sleep_stage", target_name}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Global user split (for global pretraining regime)
    splits_cfg = config["splits"]["global_users"]
    user_ids = df[user_id_col].unique().tolist()
    train_users, val_users, test_users = _split_users(
        user_ids,
        seed=seed,
        train_frac=float(splits_cfg["train_frac"]),
        val_frac=float(splits_cfg["val_frac"]),
    )
    split_path = out_dir / "splits_global_users.json"
    split_path.write_text(
        json.dumps({"train_users": train_users, "val_users": val_users, "test_users": test_users}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    normalization_strategy = str(config["normalization"]["strategy"])
    if normalization_strategy not in {"global_standard", "per_user_standard"}:
        raise ValueError(f"Unsupported normalization strategy: {normalization_strategy}")

    if normalization_strategy == "global_standard":
        # Compute normalization on training users only to avoid leakage.
        df_train_for_norm = df[df[user_id_col].isin(train_users)].copy()
        stats = _compute_standardization_stats(df_train_for_norm, feature_cols)
        df = _apply_standardization(df, feature_cols, stats)
    else:
        # Per-user stats computed on each user's own "train" time slice (within-user split),
        # enabling personalization scenarios without mixing users.
        within = config["splits"]["within_user"]
        train_frac = float(within["train_frac"])
        per_user: Dict[str, Any] = {}
        for uid, udf in df.groupby(user_id_col, sort=False):
            udf = udf.reset_index(drop=True)
            n = len(udf)
            n_train = max(1, int(np.floor(n * train_frac)))
            utrain = udf.iloc[:n_train]
            ustats = _compute_standardization_stats(utrain, feature_cols)
            per_user[str(uid)] = ustats

        stats = {
            "type": "per_user_standard",
            "per_user": per_user,
        }

        # Apply per-user standardization.
        out_parts = []
        for uid, udf in df.groupby(user_id_col, sort=False):
            ustats = per_user[str(uid)]
            out_parts.append(_apply_standardization(udf, feature_cols, ustats))
        df = pd.concat(out_parts, axis=0, ignore_index=True)

    processed_path = out_dir / "features.csv.gz"
    # Safety check: NaNs in model inputs will produce NaN loss.
    numeric = df[feature_cols + [target_name]].select_dtypes(include="number")
    if numeric.isna().any().any():
        bad = numeric.isna().sum().sort_values(ascending=False)
        bad = bad[bad > 0].head(25)
        raise ValueError(f"Preprocess produced NaNs in numeric columns (top):\n{bad.to_string()}")
    df.to_csv(processed_path, index=False, compression="gzip")
    logger.info("Wrote processed feature table to %s", processed_path.as_posix())

    feature_spec = FeatureSpec(
        feature_cols=feature_cols,
        target_col=target_name,
        timestamp_col=timestamp_col,
        user_id_col=user_id_col,
        freq_minutes=freq_minutes,
        window_steps=window_steps,
        horizon_steps=horizon_steps,
        normalization=stats,
    )

    artifacts_root = Path(config.get("artifacts", {}).get("output_dir", "artifacts"))
    artifacts_dir = artifacts_root / "features"
    save_feature_spec(artifacts_dir / "feature_spec.json", feature_spec)
    logger.info("Wrote feature spec to %s", (artifacts_dir / "feature_spec.json").as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess simulated dataset into model-ready features.")
    parser.add_argument("--config", type=str, required=True, help="Path to preprocess.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    preprocess(cfg)


if __name__ == "__main__":
    main()

