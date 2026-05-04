"""Hyperparameter governance utilities.

From 39_HYPERPARAMETER_GOVERNANCE.md §7 — Walk-Forward Tuning and §8 — Config Drift.

walk_forward_objective: Optuna trial wrapper using rolling time windows.
check_config_drift: Compare strategy configs across environments.
deployment_inventory: Snapshot which model/config versions are active per env.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


def walk_forward_objective(
    trial: Any,
    data: Any,
    train_fn: Callable,
    eval_fn: Callable,
    train_months: int = 9,
    test_months: int = 3,
    stride_months: int = 3,
    param_space: dict[str, tuple] | None = None,
) -> float:
    """Optuna objective using rolling walk-forward windows.

    Prevents hyperparameters that overfit to a single test window.

    Args:
        trial: Optuna Trial object.
        data: DataFrame with a DatetimeIndex (monthly granularity assumed).
        train_fn: Callable(train_data, params) → model.
        eval_fn: Callable(model, test_data) → float score.
        train_months: Number of months per training window.
        test_months: Number of months per test window.
        stride_months: Number of months to advance each fold.
        param_space: Optional dict of {param_name: (low, high, type)} for trial.
                     Types: 'int' or 'float'. If None, trial params are passed through.

    Returns:
        Mean score across all folds (maximize this).
    """
    params: dict[str, Any] = {}
    if param_space is not None:
        for name, spec in param_space.items():
            low, high, kind = spec
            if kind == "int":
                params[name] = trial.suggest_int(name, int(low), int(high))
            else:
                params[name] = trial.suggest_float(name, float(low), float(high))

    n = len(data)
    scores: list[float] = []
    fold = 0

    for start in range(0, n - train_months - test_months + 1, stride_months):
        train = data.iloc[start : start + train_months]
        test = data.iloc[start + train_months : start + train_months + test_months]
        if len(train) < train_months or len(test) < test_months:
            break
        try:
            model = train_fn(train, params)
            score = eval_fn(model, test)
            scores.append(float(score))
            fold += 1
        except Exception as exc:
            logger.warning("walk_forward_objective fold %d failed: %s", fold, exc)

    if not scores:
        return float("-inf")
    mean_score = float(np.mean(scores))
    logger.debug("walk_forward_objective: %d folds, mean=%.4f", len(scores), mean_score)
    return mean_score


def check_config_drift(
    config_paths: dict[str, Path],
) -> list[dict[str, Any]]:
    """Compare strategy configs across environments and report drift.

    Args:
        config_paths: {env_name: Path} mapping to YAML/JSON config files.

    Returns:
        List of drift records: {key, env_values} where env_values differ.
        Empty list if no drift detected.
    """
    try:
        import yaml as _yaml

        _load = lambda p: _yaml.safe_load(p.read_text(encoding="utf-8"))  # noqa: E731
    except ImportError:
        import json as _json

        _load = lambda p: _json.loads(p.read_text(encoding="utf-8"))  # noqa: E731

    configs: dict[str, dict] = {}
    for env, path in config_paths.items():
        if not Path(path).exists():
            logger.warning(
                "check_config_drift: config not found for %s at %s", env, path
            )
            configs[env] = {}
        else:
            configs[env] = _load(Path(path))

    def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
        result = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(_flatten(v, key))
            else:
                result[key] = v
        return result

    flat_configs = {env: _flatten(cfg) for env, cfg in configs.items()}
    all_keys: set[str] = set()
    for fc in flat_configs.values():
        all_keys.update(fc.keys())

    drift_records = []
    for key in sorted(all_keys):
        values = {env: flat_configs[env].get(key) for env in configs}
        unique_vals = {str(v) for v in values.values()}
        if len(unique_vals) > 1:
            drift_records.append({"key": key, "env_values": values})

    if drift_records:
        logger.warning("Config drift detected in %d keys", len(drift_records))
    return drift_records


def deployment_inventory(
    config_paths: dict[str, Path],
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Build a snapshot of which config is active per environment.

    Args:
        config_paths: {env_name: Path} mapping to active config files.
        timestamp_utc: ISO timestamp string; defaults to utcnow.

    Returns:
        Snapshot dict: {timestamp, environments: {env: {strategy_id, model_versions, ...}}}.
    """
    from datetime import datetime, timezone

    if timestamp_utc is None:
        timestamp_utc = datetime.now(timezone.utc).isoformat()

    try:
        import yaml as _yaml

        _load = lambda p: _yaml.safe_load(p.read_text(encoding="utf-8"))  # noqa: E731
    except ImportError:
        import json as _json

        _load = lambda p: _json.loads(p.read_text(encoding="utf-8"))  # noqa: E731

    snapshot: dict[str, Any] = {"timestamp": timestamp_utc, "environments": {}}
    for env, path in config_paths.items():
        if not Path(path).exists():
            snapshot["environments"][env] = {"error": f"config not found: {path}"}
        else:
            cfg = _load(Path(path))
            snapshot["environments"][env] = {
                "strategy_id": cfg.get("strategy_id", "unknown"),
                "model_versions": cfg.get("model_versions", {}),
                "config_path": str(path),
            }

    return snapshot


__all__ = ["walk_forward_objective", "check_config_drift", "deployment_inventory"]
