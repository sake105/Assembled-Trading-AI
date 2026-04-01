from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml  # type: ignore[import]

logger = logging.getLogger(__name__)


def load_policy(
    path: str | Path = "configs/policy.yaml",
    *,
    validate: bool = True,
) -> Dict[str, Any]:
    """Load policy configuration from YAML file.

    Returns an empty dict if the file is missing or invalid.

    Args:
        path:     Path to policy YAML file (default: configs/policy.yaml).
        validate: If True, run soft schema validation and log warnings (default: True).
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

    if validate:
        try:
            from src.assembled_core.config.policy_schema import validate_policy  # noqa: PLC0415
            data, _ = validate_policy(data)
        except Exception as e:
            logger.debug("policy schema validation skipped: %s", e)

    return data


__all__ = ["load_policy"]
