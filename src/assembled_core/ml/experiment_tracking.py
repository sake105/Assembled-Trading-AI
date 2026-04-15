"""Experiment Tracking — Local MLflow-style experiment logger (M40).

Provides experiment tracking without external dependencies:
- Track hyperparameters, metrics, and artifacts per run
- Model registry with staging/production lifecycle
- JSON-based storage (SQLite optional later)
- Compatible with MLflow format for future migration

Design: no mandatory external deps. Pure Python + JSON.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExperimentRun:
    """A single experiment run."""
    run_id: str
    experiment_name: str
    timestamp: str
    params: dict[str, Any]  # hyperparameters
    metrics: dict[str, float]  # evaluation metrics
    tags: dict[str, str] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)  # file paths
    status: str = "completed"  # running, completed, failed
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentRun":
        return cls(**d)


@dataclass
class ModelVersion:
    """A registered model version."""
    model_name: str
    version: int
    run_id: str
    stage: str  # "staging", "production", "archived"
    registered_at: str
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentTracker:
    """Local experiment tracking with JSON persistence.

    Args:
        storage_path: Directory for experiment data.
    """

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self._storage_path = Path(storage_path) if storage_path else None
        self._runs: list[ExperimentRun] = []
        self._models: dict[str, list[ModelVersion]] = {}  # model_name -> versions

        if self._storage_path:
            self._storage_path.mkdir(parents=True, exist_ok=True)
            self._load()

    def log_run(
        self,
        experiment_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        tags: dict[str, str] | None = None,
        artifacts: list[str] | None = None,
        notes: str = "",
    ) -> ExperimentRun:
        """Log a new experiment run.

        Args:
            experiment_name: Name of the experiment.
            params: Hyperparameters used.
            metrics: Evaluation metrics.
            tags: Optional tags.
            artifacts: Optional artifact file paths.
            notes: Free-text notes.

        Returns:
            Created ExperimentRun.
        """
        run = ExperimentRun(
            run_id=str(uuid.uuid4())[:8],
            experiment_name=experiment_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            params=params,
            metrics=metrics,
            tags=tags or {},
            artifacts=artifacts or [],
            notes=notes,
        )
        self._runs.append(run)

        if self._storage_path:
            self._save_run(run)

        logger.info("[ExpTracker] Logged run %s: %s", run.run_id, experiment_name)
        return run

    def get_runs(
        self,
        experiment_name: str | None = None,
        min_metric: str | None = None,
        min_value: float | None = None,
        limit: int = 50,
    ) -> list[ExperimentRun]:
        """Query experiment runs.

        Args:
            experiment_name: Filter by experiment.
            min_metric: Filter by minimum metric value.
            min_value: Minimum value for the metric filter.
            limit: Max results.

        Returns:
            List of matching ExperimentRuns.
        """
        results = self._runs
        if experiment_name:
            results = [r for r in results if r.experiment_name == experiment_name]
        if min_metric and min_value is not None:
            results = [
                r for r in results
                if r.metrics.get(min_metric, float("-inf")) >= min_value
            ]
        return results[-limit:]

    def get_best_run(
        self,
        experiment_name: str,
        metric: str = "sharpe",
        higher_is_better: bool = True,
    ) -> ExperimentRun | None:
        """Find the best run by a metric.

        Args:
            experiment_name: Experiment to search.
            metric: Metric name to optimize.
            higher_is_better: If True, maximize; else minimize.

        Returns:
            Best ExperimentRun or None.
        """
        runs = [r for r in self._runs if r.experiment_name == experiment_name]
        if not runs:
            return None

        def key_fn(r):
            v = r.metrics.get(metric, float("-inf") if higher_is_better else float("inf"))
            return v if higher_is_better else -v

        return max(runs, key=key_fn)

    def register_model(
        self,
        model_name: str,
        run_id: str,
        stage: str = "staging",
    ) -> ModelVersion:
        """Register a model version from an experiment run.

        Args:
            model_name: Model name in the registry.
            run_id: Associated run ID.
            stage: Initial stage ("staging", "production", "archived").

        Returns:
            Created ModelVersion.
        """
        versions = self._models.get(model_name, [])
        version_num = len(versions) + 1

        # Get metrics from run
        run = next((r for r in self._runs if r.run_id == run_id), None)
        metrics = run.metrics if run else {}

        mv = ModelVersion(
            model_name=model_name,
            version=version_num,
            run_id=run_id,
            stage=stage,
            registered_at=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
        )

        if model_name not in self._models:
            self._models[model_name] = []
        self._models[model_name].append(mv)

        if self._storage_path:
            self._save_registry()

        logger.info("[ExpTracker] Registered %s v%d (stage=%s)", model_name, version_num, stage)
        return mv

    def promote_model(self, model_name: str, version: int, new_stage: str) -> bool:
        """Promote a model version to a new stage.

        Args:
            model_name: Model name.
            version: Version number.
            new_stage: New stage.

        Returns:
            True if successful.
        """
        versions = self._models.get(model_name, [])
        for mv in versions:
            if mv.version == version:
                old_stage = mv.stage
                mv.stage = new_stage
                logger.info("[ExpTracker] Promoted %s v%d: %s -> %s",
                            model_name, version, old_stage, new_stage)
                if self._storage_path:
                    self._save_registry()
                return True
        return False

    def get_production_model(self, model_name: str) -> ModelVersion | None:
        """Get the current production model version."""
        versions = self._models.get(model_name, [])
        prod = [v for v in versions if v.stage == "production"]
        return prod[-1] if prod else None

    @property
    def n_runs(self) -> int:
        return len(self._runs)

    @property
    def model_names(self) -> list[str]:
        return list(self._models.keys())

    # ---- Persistence ----

    def _save_run(self, run: ExperimentRun) -> None:
        if self._storage_path:
            runs_file = self._storage_path / "runs.jsonl"
            with open(runs_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(run.to_dict(), default=str) + "\n")

    def _save_registry(self) -> None:
        if self._storage_path:
            reg_file = self._storage_path / "registry.json"
            data = {
                name: [v.to_dict() for v in versions]
                for name, versions in self._models.items()
            }
            with open(reg_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

    def _load(self) -> None:
        if not self._storage_path:
            return
        runs_file = self._storage_path / "runs.jsonl"
        if runs_file.exists():
            try:
                with open(runs_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self._runs.append(ExperimentRun.from_dict(json.loads(line)))
            except Exception as e:
                logger.warning("[ExpTracker] Failed to load runs: %s", e)

        reg_file = self._storage_path / "registry.json"
        if reg_file.exists():
            try:
                with open(reg_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for name, versions in data.items():
                    self._models[name] = [ModelVersion(**v) for v in versions]
            except Exception as e:
                logger.warning("[ExpTracker] Failed to load registry: %s", e)


__all__ = [
    "ExperimentRun",
    "ModelVersion",
    "ExperimentTracker",
]
