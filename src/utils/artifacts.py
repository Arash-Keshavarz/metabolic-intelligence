from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class FeatureSpec:
    feature_cols: List[str]
    target_col: str
    timestamp_col: str
    user_id_col: str
    freq_minutes: int
    window_steps: int
    horizon_steps: int
    normalization: Dict[str, Any]


def save_feature_spec(path: str | Path, spec: FeatureSpec) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(spec), indent=2, sort_keys=True), encoding="utf-8")


def load_feature_spec(path: str | Path) -> FeatureSpec:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return FeatureSpec(**data)

