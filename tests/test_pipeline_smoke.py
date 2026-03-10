from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.preprocess import preprocess
from src.data.simulate import simulate_dataset
from src.data.dataset import WearableSequenceDataset
from src.utils.artifacts import load_feature_spec


class PipelineSmokeTest(unittest.TestCase):
    def test_simulate_preprocess_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            os.chdir(root)
            (root / "data/raw").mkdir(parents=True, exist_ok=True)
            (root / "data/processed").mkdir(parents=True, exist_ok=True)

            sim_cfg = {
                "seed": 1,
                "output_path": "data/raw/synth.csv",
                "users": {"n_users": 3, "user_id_prefix": "u_"},
                "time": {"days": 2, "freq_minutes": 5, "start_timestamp": "2025-01-01T00:00:00"},
                "physiology": {
                    "base_glucose_mean_mgdl": 105.0,
                    "base_glucose_sd_mgdl": 10.0,
                    "circadian_amp_mgdl": 10.0,
                    "circadian_phase_shift_hours_sd": 1.0,
                },
                "meals": {
                    "meals_per_day_mean": 3.0,
                    "meals_per_day_sd": 0.5,
                    "carbs_mean_g": 55.0,
                    "carbs_sd_g": 20.0,
                    "spike_per_carb_mgdl": 0.7,
                    "spike_decay_minutes": 90,
                },
                "activity": {
                    "daily_activity_mean": 0.55,
                    "daily_activity_sd": 0.15,
                    "activity_effect_mgdl": 10.0,
                },
                "sleep": {"sleep_hours_mean": 7.2, "sleep_hours_sd": 1.0, "sleep_sensitivity_effect": 0.15},
                "noise": {"glucose_noise_sd_mgdl": 6.0, "hr_noise_sd_bpm": 2.0, "hrv_noise_sd_ms": 4.0},
            }
            df = simulate_dataset(sim_cfg)
            df.to_csv(sim_cfg["output_path"], index=False)
            self.assertTrue(Path(sim_cfg["output_path"]).exists())

            pre_cfg = {
                "seed": 1,
                "raw_path": sim_cfg["output_path"],
                "output_dir": "data/processed",
                "schema": {"timestamp_col": "timestamp", "user_id_col": "user_id", "target_col": "glucose"},
                "feature_engineering": {
                    "horizon_minutes": 120,
                    "window_minutes": 180,
                    "freq_minutes": 5,
                    "rolling_features": {"enabled": True, "windows_minutes": [30]},
                    "cyclical_time": {"enabled": True, "include_day_of_week": True},
                },
                "normalization": {"strategy": "global_standard"},
                "splits": {
                    "global_users": {"train_frac": 0.67, "val_frac": 0.0, "test_frac": 0.33},
                    "within_user": {"train_frac": 0.70, "val_frac": 0.15, "test_frac": 0.15},
                },
            }
            preprocess(pre_cfg)
            self.assertTrue(Path("data/processed/features.csv.gz").exists())
            self.assertTrue(Path("artifacts/features/feature_spec.json").exists())

            spec = load_feature_spec("artifacts/features/feature_spec.json")
            ds = WearableSequenceDataset(
                processed_dir="data/processed",
                feature_spec_path="artifacts/features/feature_spec.json",
                split="train",
                split_mode="global_users",
            )
            x, y, uid = ds[0]
            self.assertEqual(tuple(x.shape), (spec.window_steps, len(spec.feature_cols)))
            self.assertTrue(isinstance(float(y.item()), float))
            self.assertTrue(isinstance(uid, str))


if __name__ == "__main__":
    unittest.main()

