from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.artifacts import FeatureSpec, load_feature_spec


SplitName = Literal["train", "val", "test"]
SplitMode = Literal["global_users", "within_user"]


@dataclass(frozen=True)
class SampleIndex:
    user_id: str
    pos: int  # position in that user's timeline (row index within user slice)


class WearableSequenceDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, str]]):
    """
    Sequence-to-one dataset.

    X: (window_steps, n_features)
    y: scalar glucose at +horizon
    """

    def __init__(
        self,
        processed_dir: str | Path,
        feature_spec_path: str | Path,
        split: SplitName,
        *,
        split_mode: SplitMode,
        seed: int = 42,
        within_user_fracs: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        users_subset: Optional[Iterable[str]] = None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.spec: FeatureSpec = load_feature_spec(feature_spec_path)
        self.split = split
        self.split_mode = split_mode
        self.seed = int(seed)
        self.within_user_fracs = within_user_fracs

        table_path = self.processed_dir / "features.csv.gz"
        self.df = pd.read_csv(table_path)

        self.user_id_col = self.spec.user_id_col
        self.target_col = self.spec.target_col
        self.feature_cols = self.spec.feature_cols
        self.window_steps = int(self.spec.window_steps)

        if users_subset is not None:
            users_subset = set(users_subset)
            self.df = self.df[self.df[self.user_id_col].isin(users_subset)].reset_index(drop=True)

        self._user_arrays: Dict[str, Dict[str, np.ndarray]] = {}
        self._samples: List[SampleIndex] = []
        self._build_index()

    def _global_user_split(self) -> Tuple[List[str], List[str], List[str]]:
        split_path = self.processed_dir / "splits_global_users.json"
        data = json.loads(split_path.read_text(encoding="utf-8"))
        return data["train_users"], data["val_users"], data["test_users"]

    def _select_users_for_split(self, all_users: List[str]) -> List[str]:
        if self.split_mode == "global_users":
            train_users, val_users, test_users = self._global_user_split()
            mapping = {"train": train_users, "val": val_users, "test": test_users}
            return [u for u in mapping[self.split] if u in set(all_users)]
        return all_users

    def _time_mask_for_within_user(self, n: int) -> np.ndarray:
        train_frac, val_frac, test_frac = self.within_user_fracs
        if not np.isclose(train_frac + val_frac + test_frac, 1.0):
            raise ValueError("within_user_fracs must sum to 1.0")
        i0 = int(np.floor(n * train_frac))
        i1 = int(np.floor(n * (train_frac + val_frac)))
        mask = np.zeros(n, dtype=bool)
        if self.split == "train":
            mask[:i0] = True
        elif self.split == "val":
            mask[i0:i1] = True
        else:
            mask[i1:] = True
        return mask

    def _build_index(self) -> None:
        df = self.df
        user_ids = df[self.user_id_col].unique().tolist()
        selected_users = self._select_users_for_split(user_ids)

        for user_id in selected_users:
            udf = df[df[self.user_id_col] == user_id].reset_index(drop=True)
            x = udf[self.feature_cols].to_numpy(dtype=np.float32, copy=True)
            y = udf[self.target_col].to_numpy(dtype=np.float32, copy=True)

            self._user_arrays[user_id] = {"x": x, "y": y}

            # valid sample positions are those with enough history for the window
            valid = np.zeros(len(udf), dtype=bool)
            valid[self.window_steps - 1 :] = True

            if self.split_mode == "within_user":
                valid &= self._time_mask_for_within_user(len(udf))

            positions = np.where(valid)[0]
            for pos in positions:
                self._samples.append(SampleIndex(user_id=user_id, pos=int(pos)))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        s = self._samples[idx]
        arrs = self._user_arrays[s.user_id]
        x = arrs["x"][s.pos - self.window_steps + 1 : s.pos + 1]
        y = arrs["y"][s.pos]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32), s.user_id

    def user_ids(self) -> List[str]:
        return sorted(self._user_arrays.keys())

