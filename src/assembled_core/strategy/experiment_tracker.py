"""Lightweight experiment tracker for hyperparameter governance.

From 39_HYPERPARAMETER_GOVERNANCE.md.

Uses MLflow when available; falls back to a local JSON-append store.
API surface mirrors the MLflow idioms so the call sites are identical.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")


def _mlflow_available() -> bool:
    try:
        import mlflow  # noqa: F401
        return bool(_MLFLOW_TRACKING_URI)
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Fallback local store
# ---------------------------------------------------------------------------


class _LocalRun:
    """Minimal MLflow-compatible run backed by a JSON file."""

    def __init__(self, run_name: str, store_dir: Path) -> None:
        self.run_id = str(uuid.uuid4())[:8]
        self.run_name = run_name
        self._store_dir = store_dir
        self._data: dict[str, Any] = {
            "run_id": self.run_id,
            "run_name": run_name,
            "started_at": datetime.now(tz=timezone.utc).isoformat(),
            "params": {},
            "metrics": {},
            "tags": {},
            "status": "RUNNING",
        }

    def log_param(self, key: str, value: Any) -> None:
        self._data["params"][key] = value

    def log_params(self, params: dict[str, Any]) -> None:
        self._data["params"].update(params)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self._data["metrics"][key] = value

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._data["metrics"].update(metrics)

    def set_tag(self, key: str, value: str) -> None:
        self._data["tags"][key] = value

    def _finalize(self, status: str = "FINISHED") -> None:
        self._data["status"] = status
        self._data["ended_at"] = datetime.now(tz=timezone.utc).isoformat()
        self._store_dir.mkdir(parents=True, exist_ok=True)
        out = self._store_dir / f"{self.run_id}_{self.run_name}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, default=str)
        logger.debug("Experiment run saved → %s", out)


class _LocalTracker:
    """Context-manager wrapper for _LocalRun."""

    def __init__(self, run_name: str, store_dir: Path) -> None:
        self._run = _LocalRun(run_name, store_dir)

    def __enter__(self) -> _LocalRun:
        return self._run

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        status = "FAILED" if exc_type else "FINISHED"
        self._run._finalize(status)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def start_run(
    run_name: str = "unnamed",
    local_store_dir: str = "data/experiments",
):
    """Return a context manager that logs params/metrics.

    Uses MLflow when ``MLFLOW_TRACKING_URI`` is set and mlflow is installed;
    otherwise falls back to a local JSON store.

    Usage::

        with start_run("news_classifier_v3") as run:
            run.log_params({"lr": 0.001, "n_estimators": 200})
            run.log_metrics({"accuracy": 0.87, "f1": 0.84})
    """
    if _mlflow_available():
        import mlflow
        return mlflow.start_run(run_name=run_name)
    return _LocalTracker(run_name, Path(local_store_dir))


def log_strategy_config(
    run,
    config: "StrategyConfig | dict",  # noqa: F821
) -> None:
    """Flatten a StrategyConfig into MLflow/local params."""
    data = config.to_dict() if hasattr(config, "to_dict") else config
    flat: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat[f"{k}.{sub_k}"] = sub_v
        else:
            flat[k] = v
    run.log_params(flat)


__all__ = ["start_run", "log_strategy_config"]
