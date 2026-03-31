from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml  # type: ignore[import]


def load_policy(path: str | Path = "configs/policy.yaml") -> Dict[str, Any]:
    """Load policy configuration from YAML file.

    Returns an empty dict if the file is missing or invalid.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


__all__ = ["load_policy"]
