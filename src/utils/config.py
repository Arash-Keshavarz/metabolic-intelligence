from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env_vars(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    if isinstance(obj, str):
        def repl(match: re.Match[str]) -> str:
            var = match.group(1)
            return os.getenv(var, match.group(0))

        return _ENV_VAR_PATTERN.sub(repl, obj)
    return obj


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _expand_env_vars(data)


def require_keys(d: Dict[str, Any], keys: list[str], *, context: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"Missing keys in {context}: {missing}")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @staticmethod
    def from_cwd() -> "ProjectPaths":
        return ProjectPaths(root=Path.cwd())

    def resolve(self, relative: str | Path) -> Path:
        return (self.root / relative).resolve()


def get_mlflow_tracking_uri(config_uri: Optional[str]) -> Optional[str]:
    if config_uri:
        return config_uri
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    # Use file-based tracking store by default for portability and to avoid
    # SQLite file permission/locking edge cases on some filesystems.
    return "file:./mlruns"

