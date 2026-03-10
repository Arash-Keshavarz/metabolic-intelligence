from __future__ import annotations

from typing import Any, Dict


def flatten_dict(d: Dict[str, Any], *, prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, prefix=key, sep=sep))
        else:
            out[key] = v
    return out

