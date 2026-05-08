from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import yaml  # type: ignore[import]

logger = logging.getLogger(__name__)

_POLICY_CACHE: dict[str, tuple[Dict[str, Any], float]] = {}


def load_policy(
    path: str | Path = "configs/policy.yaml",
    *,
    validate: bool = True,
) -> Dict[str, Any]:
    """Load policy configuration from YAML file.

    Returns an empty dict only if the file is missing. Malformed YAML or a
    non-mapping top-level document are treated as hard errors — collapsing
    them to ``{}`` used to silently drop every policy-gated safeguard
    (kill-switch halt thresholds, reconciliation gates, exposure caps),
    which is a dangerous "all defaults" failure mode in this repo.

    Args:
        path:     Path to policy YAML file (default: configs/policy.yaml).
        validate: If True, run soft schema validation and log warnings (default: True).

    Raises:
        yaml.YAMLError: If the file exists but is not parseable YAML.
        ValueError:     If the top-level YAML document is not a mapping.
    """
    env_override = os.environ.get("ASSEMBLED_POLICY_PATH")
    if env_override:
        path = env_override
    p = Path(path)
    cache_key = str(p.resolve())
    if cache_key in _POLICY_CACHE:
        cached_data, cached_mtime = _POLICY_CACHE[cache_key]
        try:
            if p.stat().st_mtime == cached_mtime:
                return cached_data
        except OSError:
            return cached_data
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"policy file {p} top-level must be a YAML mapping, got {type(data).__name__}"
        )

    if validate:
        try:
            from src.assembled_core.config.policy_schema import (
                validate_policy,
            )  # noqa: PLC0415

            data, _ = validate_policy(data)
        except Exception as e:
            logger.debug("policy schema validation skipped: %s", e)

    try:
        _POLICY_CACHE[cache_key] = (data, p.stat().st_mtime)
    except OSError:
        _POLICY_CACHE[cache_key] = (data, 0.0)
    return data


__all__ = ["load_policy"]
