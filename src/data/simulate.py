from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.config import load_yaml
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def _clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def _make_time_index(start: str, days: int, freq_minutes: int) -> pd.DatetimeIndex:
    start_dt = pd.to_datetime(start)
    n_steps = int((days * 24 * 60) / freq_minutes)
    return pd.date_range(start=start_dt, periods=n_steps, freq=f"{freq_minutes}min")


def _sample_meals_for_day(
    rng: np.random.Generator, day_start: pd.Timestamp, freq_minutes: int, meals_cfg: Dict
) -> Dict[pd.Timestamp, float]:
    mean = float(meals_cfg["meals_per_day_mean"])
    sd = float(meals_cfg["meals_per_day_sd"])
    n_meals = int(np.clip(np.round(rng.normal(mean, sd)), 2, 5))

    candidate_hours = np.array([8.0, 13.0, 19.0, 16.0, 10.5])
    rng.shuffle(candidate_hours)
    hours = np.sort(candidate_hours[:n_meals] + rng.normal(0.0, 0.6, size=n_meals))

    carbs = rng.normal(float(meals_cfg["carbs_mean_g"]), float(meals_cfg["carbs_sd_g"]), size=n_meals)
    carbs = np.clip(carbs, 10.0, 140.0)

    events: Dict[pd.Timestamp, float] = {}
    for h, c in zip(hours, carbs):
        ts = day_start + pd.Timedelta(hours=float(h))
        # snap to grid
        minutes = int((ts - day_start) / pd.Timedelta(minutes=1))
        snapped = day_start + pd.Timedelta(minutes=(minutes // freq_minutes) * freq_minutes)
        events[snapped] = float(c)
    return events


def _sleep_window_for_day(rng: np.random.Generator, day_start: pd.Timestamp, sleep_cfg: Dict) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start_hour = 23.0 + rng.normal(0.0, 0.9)
    duration = rng.normal(float(sleep_cfg["sleep_hours_mean"]), float(sleep_cfg["sleep_hours_sd"]))
    duration = float(np.clip(duration, 5.0, 9.5))

    sleep_start = day_start + pd.Timedelta(hours=start_hour)
    sleep_end = sleep_start + pd.Timedelta(hours=duration)
    return sleep_start, sleep_end


def _sleep_stage_at(mins_asleep: float, total_mins: float) -> str:
    # Roughly mimic 90-min cycles with more deep early and more REM late.
    if mins_asleep < 0 or mins_asleep > total_mins:
        return "awake"
    cycle = (mins_asleep % 90.0) / 90.0
    progress = mins_asleep / max(total_mins, 1.0)
    if cycle < 0.10:
        return "light"
    if cycle < 0.45:
        return "deep" if progress < 0.55 else "light"
    if cycle < 0.80:
        return "light"
    return "rem" if progress > 0.25 else "light"


def _encode_sleep_stage(stage: str) -> int:
    # ordinal encoding for modeling (kept stable across pipeline)
    mapping = {"awake": 0, "light": 1, "deep": 2, "rem": 3}
    return mapping.get(stage, 0)


def simulate_user(
    rng: np.random.Generator,
    user_id: str,
    index: pd.DatetimeIndex,
    cfg: Dict,
) -> pd.DataFrame:
    physiology = cfg["physiology"]
    meals_cfg = cfg["meals"]
    activity_cfg = cfg["activity"]
    sleep_cfg = cfg["sleep"]
    noise_cfg = cfg["noise"]
    freq_minutes = int(cfg["time"]["freq_minutes"])

    base_glucose = rng.normal(float(physiology["base_glucose_mean_mgdl"]), float(physiology["base_glucose_sd_mgdl"]))
    circ_amp = float(np.clip(rng.normal(float(physiology["circadian_amp_mgdl"]), 3.0), 6.0, 22.0))
    phase_shift_h = rng.normal(0.0, float(physiology["circadian_phase_shift_hours_sd"]))
    meal_gain = float(np.clip(rng.normal(float(meals_cfg["spike_per_carb_mgdl"]), 0.12), 0.35, 1.1))
    activity_mean = float(np.clip(rng.normal(float(activity_cfg["daily_activity_mean"]), float(activity_cfg["daily_activity_sd"])), 0.25, 0.85))
    insulin_sens = float(np.clip(rng.normal(1.0, 0.08), 0.80, 1.25))

    # Activity: diurnal pattern + randomness
    hours = (index.hour + index.minute / 60.0).to_numpy()
    activity = activity_mean + 0.20 * np.sin(2 * np.pi * (hours - 14.0) / 24.0) + rng.normal(0.0, 0.08, size=len(index))
    activity = _clamp(activity, 0.0, 1.0)

    # Meals: sparse carbs series
    meal_carbs = np.zeros(len(index), dtype=np.float32)
    index_to_pos = {ts: i for i, ts in enumerate(index)}
    for day in pd.date_range(index[0].normalize(), index[-1].normalize(), freq="D"):
        for ts, carbs in _sample_meals_for_day(rng, pd.Timestamp(day), freq_minutes, meals_cfg).items():
            pos = index_to_pos.get(ts)
            if pos is not None:
                meal_carbs[pos] += carbs

    # Sleep stages per timestamp
    sleep_stage = np.array(["awake"] * len(index), dtype=object)
    sleep_stage_code = np.zeros(len(index), dtype=np.int64)
    sleep_durations: List[float] = []

    for day in pd.date_range(index[0].normalize(), index[-1].normalize(), freq="D"):
        s0, s1 = _sleep_window_for_day(rng, pd.Timestamp(day), sleep_cfg)
        total_mins = (s1 - s0) / pd.Timedelta(minutes=1)
        sleep_durations.append(float(total_mins))
        # affect next-day sensitivity; computed later from distribution
        mask = (index >= s0) & (index < s1)
        if not np.any(mask):
            continue
        mins_asleep = ((index[mask] - s0) / pd.Timedelta(minutes=1)).to_numpy()
        stages = np.array([_sleep_stage_at(m, total_mins) for m in mins_asleep], dtype=object)
        sleep_stage[mask] = stages

    for i in range(len(index)):
        sleep_stage_code[i] = _encode_sleep_stage(str(sleep_stage[i]))

    # Sleep impact: derive a simple daily "sleep score"
    mean_sleep = float(sleep_cfg["sleep_hours_mean"]) * 60.0
    sleep_score = np.clip(np.array(sleep_durations).mean() / max(mean_sleep, 1.0), 0.7, 1.2)
    sleep_effect = float(sleep_cfg["sleep_sensitivity_effect"]) * (sleep_score - 1.0)
    effective_sens = insulin_sens * (1.0 + sleep_effect)

    # Circadian rhythm in glucose
    circadian = circ_amp * np.sin(2 * np.pi * ((hours + phase_shift_h) - 4.0) / 24.0)

    # Post-meal spike: exponential decay kernel applied to carbs
    decay_minutes = float(meals_cfg["spike_decay_minutes"])
    steps = int(np.ceil((decay_minutes * 3) / freq_minutes))
    t = np.arange(steps) * freq_minutes
    kernel = np.exp(-t / decay_minutes)
    kernel = kernel / (kernel.sum() + 1e-9)
    meal_effect = np.convolve(meal_carbs, kernel, mode="full")[: len(meal_carbs)]
    # meal_effect is in grams (smoothed); meal_gain converts grams -> mg/dL.
    meal_effect = meal_gain * meal_effect

    # Activity lowers glucose (improved insulin sensitivity / uptake)
    activity_effect = -float(activity_cfg["activity_effect_mgdl"]) * (activity - activity.mean())

    glucose = base_glucose + circadian + effective_sens * meal_effect + activity_effect
    glucose = glucose + rng.normal(0.0, float(noise_cfg["glucose_noise_sd_mgdl"]), size=len(index))
    glucose = _clamp(glucose, 55.0, 260.0)

    # Heart rate & HRV correlate with activity and sleep
    hr_base = float(np.clip(rng.normal(66.0, 6.0), 52.0, 82.0))
    heart_rate = hr_base + 32.0 * activity + rng.normal(0.0, float(noise_cfg["hr_noise_sd_bpm"]), size=len(index))
    heart_rate = heart_rate + np.where(sleep_stage_code > 0, -8.0, 0.0)
    heart_rate = _clamp(heart_rate, 40.0, 190.0)

    hrv_base = float(np.clip(rng.normal(58.0, 10.0), 30.0, 95.0))
    hrv = hrv_base - 14.0 * activity + rng.normal(0.0, float(noise_cfg["hrv_noise_sd_ms"]), size=len(index))
    hrv = hrv + np.where(sleep_stage == "deep", 12.0, 0.0) + np.where(sleep_stage == "rem", 6.0, 0.0)
    hrv = _clamp(hrv, 15.0, 150.0)

    df = pd.DataFrame(
        {
            "timestamp": index,
            "user_id": user_id,
            "glucose": glucose.astype(np.float32),
            "heart_rate": heart_rate.astype(np.float32),
            "HRV": hrv.astype(np.float32),
            "sleep_stage": sleep_stage.astype(str),
            "sleep_stage_code": sleep_stage_code.astype(np.int64),
            "activity_level": activity.astype(np.float32),
            "meal_carbs": meal_carbs.astype(np.float32),
        }
    )
    return df


def simulate_dataset(config: Dict) -> pd.DataFrame:
    index = _make_time_index(
        config["time"]["start_timestamp"],
        int(config["time"]["days"]),
        int(config["time"]["freq_minutes"]),
    )
    rng = np.random.default_rng(int(config["seed"]))

    n_users = int(config["users"]["n_users"])
    prefix = str(config["users"]["user_id_prefix"])
    dfs = []
    for i in range(n_users):
        user_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        dfs.append(simulate_user(user_rng, f"{prefix}{i:03d}", index, config))
    return pd.concat(dfs, axis=0, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate multi-modal wearable dataset.")
    parser.add_argument("--config", type=str, required=True, help="Path to simulate.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))
    logger = setup_logging("simulate")

    out = Path(cfg["output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Simulating dataset.")
    df = simulate_dataset(cfg)
    df.to_csv(out, index=False)
    logger.info("Wrote %s rows to %s", f"{len(df):,}", out.as_posix())


if __name__ == "__main__":
    main()

