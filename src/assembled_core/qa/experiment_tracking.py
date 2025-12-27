"""Experiment tracking module for structured logging of research experiments.

This module provides a lightweight experiment tracking system that stores
experiment runs with their configuration, metrics, and artifacts in a
structured directory format.

Each experiment run is stored in a dedicated directory with:
- run.json: Metadata (run_id, name, config, tags, status)
- metrics.csv: Time-series metrics (step, timestamp, metric_name, metric_value)
- artifacts/: Copies of important files (reports, plots, model files)

Example:
    >>> from src.assembled_core.qa.experiment_tracking import ExperimentTracker
    >>> from src.assembled_core.config.settings import get_settings
    >>>
    >>> settings = get_settings()
    >>> tracker = ExperimentTracker(settings.experiments_dir)
    >>>
    >>> # Start a run
    >>> run = tracker.start_run(
    ...     name="trend_baseline_ma20_50",
    ...     config={"ma_fast": 20, "ma_slow": 50, "freq": "1d"},
    ...     tags=["trend", "baseline"]
    ... )
    >>>
    >>> # Log metrics
    >>> tracker.log_metrics(run, {"sharpe": 1.23, "max_drawdown": -0.15})
    >>>
    >>> # Log artifacts
    >>> tracker.log_artifact(run, "output/performance_report_1d.md", "qa_report.md")
    >>>
    >>> # Finish run
    >>> tracker.finish_run(run, status="finished")
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExperimentRun:
    """Represents a single experiment run.

    Attributes:
        run_id: Unique identifier for the run (format: YYYYMMDD_HHMMSS_<uuid8>)
        name: Human-readable name for the experiment
        created_at: ISO-format timestamp when the run was created
        config: Dictionary with experiment configuration (parameters, settings)
        tags: List of tags for categorizing experiments
        status: Current status ("running", "finished", "failed")
    """

    run_id: str
    name: str
    created_at: str
    config: dict[str, Any]
    tags: list[str]
    status: str = "running"


class ExperimentTracker:
    """Lightweight experiment tracker for storing experiment runs.

    Each run is stored in a dedicated directory with:
    - run.json: Metadata
    - metrics.csv: Time-series metrics
    - artifacts/: Important files

    Args:
        base_dir: Base directory for storing experiment runs
            (default: from settings.experiments_dir)
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        """Initialize experiment tracker.

        Args:
            base_dir: Base directory for storing experiment runs.
                If None, uses settings.experiments_dir.
        """
        if base_dir is None:
            from src.assembled_core.config.settings import get_settings

            settings = get_settings()
            base_dir = settings.experiments_dir

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def start_run(
        self,
        name: str,
        config: Mapping[str, Any] | None = None,
        tags: Sequence[str] | None = None,
    ) -> ExperimentRun:
        """Start a new experiment run.

        Creates a new run directory and initializes run.json.

        Args:
            name: Human-readable name for the experiment
            config: Dictionary with experiment configuration
            tags: List of tags for categorizing experiments

        Returns:
            ExperimentRun instance with generated run_id

        Example:
            >>> tracker = ExperimentTracker()
            >>> run = tracker.start_run(
            ...     name="trend_baseline_ma20_50",
            ...     config={"ma_fast": 20, "ma_slow": 50},
            ...     tags=["trend", "baseline"]
            ... )
            >>> print(run.run_id)
            '20250115_143022_abc12345'
        """
        # Generate run_id: YYYYMMDD_HHMMSS_<uuid8>
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        uuid_short = str(uuid4()).replace("-", "")[:8]
        run_id = f"{timestamp}_{uuid_short}"

        # Create run directory
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir = run_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        # Create ExperimentRun
        run = ExperimentRun(
            run_id=run_id,
            name=name,
            created_at=datetime.now().isoformat(),
            config=dict(config) if config else {},
            tags=list(tags) if tags else [],
            status="running",
        )

        # Save run.json
        run_json_path = run_dir / "run.json"
        try:
            run_json_path.parent.mkdir(parents=True, exist_ok=True)
            with run_json_path.open("w", encoding="utf-8") as f:
                json.dump(asdict(run), f, indent=2, ensure_ascii=False)
        except (IOError, OSError) as exc:
            logger.error("Failed to write experiment run JSON to %s: %s", run_json_path, exc)
            raise RuntimeError(f"Failed to write experiment run JSON to {run_json_path}") from exc
        except (TypeError, ValueError) as exc:
            logger.error("Failed to serialize experiment run to JSON: %s", exc)
            raise ValueError(f"Failed to serialize experiment run to JSON: {run_json_path}") from exc

        return run

    def log_metrics(
        self,
        run: ExperimentRun,
        metrics: Mapping[str, float],
        step: int | None = None,
    ) -> None:
        """Log metrics for an experiment run.

        Appends metrics to metrics.csv in the run directory.
        If metrics.csv doesn't exist, it will be created.

        Args:
            run: ExperimentRun instance
            metrics: Dictionary with metric names and values
            step: Optional step number (for time-series metrics)
                If None, uses empty string in CSV

        Example:
            >>> tracker.log_metrics(run, {"sharpe": 1.23, "max_drawdown": -0.15}, step=1)
            >>> tracker.log_metrics(run, {"sharpe": 1.45, "max_drawdown": -0.12}, step=2)
        """
        run_dir = self.base_dir / run.run_id
        metrics_csv_path = run_dir / "metrics.csv"

        # Prepare data for CSV
        timestamp = datetime.now().isoformat()
        step_str = str(step) if step is not None else ""

        # Check if CSV exists
        file_exists = metrics_csv_path.exists()

        # Append metrics
        try:
            metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
            with metrics_csv_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                # Write header if file is new
                if not file_exists:
                    writer.writerow(["step", "timestamp", "metric_name", "metric_value"])

                # Write one row per metric
                for metric_name, metric_value in metrics.items():
                    writer.writerow([step_str, timestamp, metric_name, metric_value])
        except (IOError, OSError, csv.Error) as exc:
            logger.warning("Failed to append metrics CSV to %s: %s", metrics_csv_path, exc)

    def log_artifact(
        self,
        run: ExperimentRun,
        artifact_path: str | Path,
        target_name: str | None = None,
    ) -> None:
        """Log an artifact (file) for an experiment run.

        Copies the file to the run's artifacts/ directory.

        Args:
            run: ExperimentRun instance
            artifact_path: Path to the artifact file to copy
            target_name: Optional target filename in artifacts/
                If None, uses the original filename

        Raises:
            FileNotFoundError: If artifact_path doesn't exist

        Example:
            >>> tracker.log_artifact(run, "output/performance_report_1d.md", "qa_report.md")
            >>> tracker.log_artifact(run, "output/equity_curve_1d.csv")  # Uses original filename
        """
        artifact_path = Path(artifact_path)

        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        run_dir = self.base_dir / run.run_id
        artifacts_dir = run_dir / "artifacts"

        # Determine target filename
        if target_name is None:
            target_name = artifact_path.name

        target_path = artifacts_dir / target_name

        # Copy file
        shutil.copy2(artifact_path, target_path)

    def finish_run(self, run: ExperimentRun, status: str = "finished") -> None:
        """Finish an experiment run.

        Updates the status in run.json.

        Args:
            run: ExperimentRun instance
            status: Final status ("finished" or "failed")

        Example:
            >>> tracker.finish_run(run, status="finished")
            >>> # Or in case of error:
            >>> tracker.finish_run(run, status="failed")
        """
        if status not in ["finished", "failed"]:
            raise ValueError(
                f"Invalid status: {status}. Must be 'finished' or 'failed'"
            )

        run_dir = self.base_dir / run.run_id
        run_json_path = run_dir / "run.json"

        # Load existing run.json
        try:
            with run_json_path.open("r", encoding="utf-8") as f:
                run_data = json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Experiment run JSON file not found: {run_json_path}") from exc
        except (IOError, OSError) as exc:
            raise IOError(f"Failed to read experiment run JSON from {run_json_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in experiment run file {run_json_path}") from exc

        # Update status
        run_data["status"] = status
        run_data["finished_at"] = datetime.now().isoformat()

        # Save updated run.json
        try:
            with run_json_path.open("w", encoding="utf-8") as f:
                json.dump(run_data, f, indent=2, ensure_ascii=False)
        except (IOError, OSError) as exc:
            logger.error("Failed to update experiment run JSON at %s: %s", run_json_path, exc)
            raise RuntimeError(f"Failed to update experiment run JSON at {run_json_path}") from exc
        except (TypeError, ValueError) as exc:
            logger.error("Failed to serialize experiment run data to JSON: %s", exc)
            raise ValueError(f"Failed to serialize experiment run data to JSON: {run_json_path}") from exc

    def list_runs(self, tags: Sequence[str] | None = None) -> list[ExperimentRun]:
        """List all experiment runs.

        Args:
            tags: Optional filter by tags (runs must have all specified tags)

        Returns:
            List of ExperimentRun instances, sorted by created_at (newest first)
        """
        runs = []

        # Iterate over all run directories
        for run_dir in self.base_dir.iterdir():
            if not run_dir.is_dir():
                continue

            run_json_path = run_dir / "run.json"
            if not run_json_path.exists():
                continue

            # Load run.json
            try:
                try:
                    with run_json_path.open("r", encoding="utf-8") as f:
                        run_data = json.load(f)
                except (IOError, OSError, json.JSONDecodeError) as exc:
                    logger.warning(f"Failed to load experiment run from {run_json_path}: {exc}. Skipping.")
                    continue

                run = ExperimentRun(**run_data)

                # Filter by tags if specified
                if tags:
                    if not all(tag in run.tags for tag in tags):
                        continue

                runs.append(run)
            except (json.JSONDecodeError, KeyError, TypeError):
                # Skip invalid run.json files
                continue

        # Sort by created_at (newest first)
        runs.sort(key=lambda r: r.created_at, reverse=True)

        return runs

    def get_run_metrics(self, run: ExperimentRun) -> pd.DataFrame:
        """Load metrics for an experiment run as DataFrame.

        Args:
            run: ExperimentRun instance

        Returns:
            DataFrame with columns: step, timestamp, metric_name, metric_value
            Empty DataFrame if metrics.csv doesn't exist

        Example:
            >>> metrics_df = tracker.get_run_metrics(run)
            >>> print(metrics_df[metrics_df["metric_name"] == "sharpe"])
        """
        run_dir = self.base_dir / run.run_id
        metrics_csv_path = run_dir / "metrics.csv"

        if not metrics_csv_path.exists():
            return pd.DataFrame(
                columns=["step", "timestamp", "metric_name", "metric_value"]
            )

        return pd.read_csv(metrics_csv_path)
