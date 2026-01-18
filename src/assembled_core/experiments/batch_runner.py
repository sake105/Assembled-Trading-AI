"""Batch runner for executing multiple backtest runs (serial and parallel).

This module provides both serial and parallel execution paths for batch backtests.
Parallel execution uses ProcessPoolExecutor while maintaining deterministic ordering.

Example:
    from src.assembled_core.experiments.batch_config import BatchConfig, load_batch_config
    from src.assembled_core.experiments.batch_runner import expand_run_specs, run_batch_serial, run_batch_parallel

    config = load_batch_config(Path("configs/batch.yaml"))
    run_specs = expand_run_specs(config)
    result = run_batch_serial(run_specs, config.output_root / config.batch_name)
    # Or parallel:
    result = run_batch_parallel(run_specs, config.output_root / config.batch_name, max_workers=4)
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from src.assembled_core.experiments.batch_config import BatchConfig, RunSpec

logger = logging.getLogger(__name__)


def _get_git_commit_hash() -> str | None:
    """Try to get current git commit hash (optional, returns None if git unavailable).

    Returns:
        Commit hash (short, 7 chars) if git is available, None otherwise
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return None


def _compute_config_hash(config_dict: dict[str, Any]) -> str:
    """Compute hash of configuration dictionary for tracking.

    Args:
        config_dict: Configuration as dictionary

    Returns:
        SHA256 hash (first 16 chars) of sorted JSON representation
    """
    # Sort keys for deterministic hashing
    config_json = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(config_json.encode()).hexdigest()[:16]


def _collect_run_artifacts(run_output_dir: Path) -> list[str]:
    """Collect list of artifact files in run output directory.

    Args:
        run_output_dir: Run output directory

    Returns:
        List of relative artifact file paths (sorted for stable ordering)
    """
    artifacts = []
    if not run_output_dir.exists():
        return artifacts

    # Common artifacts from backtest
    artifact_patterns = [
        "**/*.csv",
        "**/*.parquet",
        "**/*.json",
        "**/*.md",
        "**/*.log",
    ]

    for pattern in artifact_patterns:
        for artifact_path in run_output_dir.glob(pattern):
            # Get relative path from run_output_dir
            rel_path = artifact_path.relative_to(run_output_dir)
            artifact_str = rel_path.as_posix()
            if artifact_str not in artifacts:
                artifacts.append(artifact_str)

    # Sort for stable ordering
    return sorted(artifacts)


def _write_run_manifest(
    result: RunResult,
    run_spec: RunSpec,
    base_args: dict[str, Any],
    run_output_dir: Path,
    start_time: datetime,
    end_time: datetime,
) -> None:
    """Write run manifest.json with metadata for a single batch run.

    Args:
        result: RunResult for this run
        run_spec: RunSpec for this run
        base_args: Base arguments used
        run_output_dir: Run output directory
        start_time: Start timestamp
        end_time: End timestamp
    """
    # Compute config hash from merged args
    merged_args = {**base_args, **run_spec.overrides}
    config_hash = _compute_config_hash(merged_args)

    # Get git commit hash
    git_hash = _get_git_commit_hash()

    # Collect artifacts
    artifacts = _collect_run_artifacts(run_output_dir)

    # Build manifest
    manifest = {
        "run_id": result.run_id,
        "run_index": result.run_index,
        "status": result.status,
        "started_at": start_time.isoformat(),
        "finished_at": end_time.isoformat(),
        "runtime_sec": result.runtime_sec,
        "config_hash": config_hash,
        "git_commit_hash": git_hash,
        "run_spec": {
            "bundle_path": str(run_spec.bundle_path),
            "start_date": run_spec.start_date,
            "end_date": run_spec.end_date,
            "tags": run_spec.tags,
            "overrides": run_spec.overrides,
        },
        "base_args": base_args,
        "artifacts": artifacts,
        "error": result.error,
        "created_at": datetime.utcnow().isoformat(),
    }

    # Write manifest
    manifest_path = run_output_dir / "run_manifest.json"
    try:
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=True, sort_keys=True)
        logger.debug(f"Wrote run manifest to {manifest_path}")
    except Exception as exc:
        logger.warning(f"Failed to write run manifest: {exc}", exc_info=True)


def _get_git_commit_hash() -> str | None:
    """Try to get current git commit hash (optional, returns None if git unavailable).

    Returns:
        Commit hash (short, 7 chars) if git is available, None otherwise
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return None


def _compute_config_hash(config_dict: dict[str, Any]) -> str:
    """Compute hash of configuration dictionary for tracking.

    Args:
        config_dict: Configuration as dictionary

    Returns:
        SHA256 hash (first 16 chars) of sorted JSON representation
    """
    # Sort keys for deterministic hashing
    config_json = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(config_json.encode()).hexdigest()[:16]


def _collect_run_artifacts(run_output_dir: Path) -> list[str]:
    """Collect list of artifact files in run output directory.

    Args:
        run_output_dir: Run output directory

    Returns:
        List of relative artifact file paths
    """
    artifacts = []
    if not run_output_dir.exists():
        return artifacts

    # Common artifacts from backtest
    artifact_patterns = [
        "**/*.csv",
        "**/*.parquet",
        "**/*.json",
        "**/*.md",
        "**/*.log",
    ]

    for pattern in artifact_patterns:
        for artifact_path in run_output_dir.glob(pattern):
            # Get relative path from run_output_dir
            rel_path = artifact_path.relative_to(run_output_dir)
            if rel_path.as_posix() not in artifacts:
                artifacts.append(rel_path.as_posix())

    # Sort for stable ordering
    return sorted(artifacts)


@dataclass
class RunResult:
    """Result of a single backtest run.

    Attributes:
        run_id: Unique identifier for the run
        status: Status ("success", "failed", "skipped", "timeout")
        output_dir: Output directory for this run (if successful)
        runtime_sec: Runtime in seconds
        error: Error message (if failed)
        timings_path: Optional path to timings.json (if profiling enabled)
        profile_path: Optional path to profile artifacts (if profiling enabled)
        run_index: Optional index in original run list (for deterministic ordering)
    """

    run_id: str
    status: str
    output_dir: Path | None = None
    runtime_sec: float = 0.0
    error: str | None = None
    timings_path: Path | None = None
    profile_path: Path | None = None
    run_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "status": self.status,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "runtime_sec": self.runtime_sec,
            "error": self.error,
            "timings_path": str(self.timings_path) if self.timings_path else None,
            "profile_path": str(self.profile_path) if self.profile_path else None,
            "run_index": self.run_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunResult:
        """Create RunResult from dictionary."""
        return cls(
            run_id=data["run_id"],
            status=data["status"],
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
            runtime_sec=data.get("runtime_sec", 0.0),
            error=data.get("error"),
            timings_path=Path(data["timings_path"]) if data.get("timings_path") else None,
            profile_path=Path(data["profile_path"]) if data.get("profile_path") else None,
            run_index=data.get("run_index"),
        )


@dataclass
class BatchResult:
    """Result of a batch execution.

    Attributes:
        batch_name: Name of the batch
        started_at: Timestamp when batch started
        finished_at: Timestamp when batch finished
        total_runtime_sec: Total runtime in seconds
        run_results: List of RunResult for each run
        success_count: Number of successful runs
        failed_count: Number of failed runs
        skipped_count: Number of skipped runs
        error_summary: Summary of errors (if any)
    """

    batch_name: str
    started_at: datetime
    finished_at: datetime
    total_runtime_sec: float
    run_results: list[RunResult] = field(default_factory=list)
    success_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    error_summary: str | None = None

    def __post_init__(self) -> None:
        """Compute counts from run_results."""
        self.success_count = sum(1 for r in self.run_results if r.status == "success")
        self.failed_count = sum(1 for r in self.run_results if r.status == "failed")
        self.skipped_count = sum(1 for r in self.run_results if r.status == "skipped")

        # Build error summary
        errors = [r.error for r in self.run_results if r.error]
        if errors:
            self.error_summary = f"{len(errors)} errors: " + "; ".join(errors[:5])
            if len(errors) > 5:
                self.error_summary += f" ... and {len(errors) - 5} more"


def expand_run_specs(config: BatchConfig) -> list[RunSpec]:
    """Expand batch configuration to list of run specifications.

    This function handles both individual runs and grid search expansion.

    Args:
        config: Batch configuration

    Returns:
        List of RunSpec instances (expanded from grid if applicable)

    Raises:
        ValueError: If config is invalid (e.g., both runs and grid specified)
    """
    return config.expand_runs()


def _build_backtest_command(
    run_spec: RunSpec,
    base_args: dict[str, Any],
    output_dir: Path,
    repo_root: Path,
) -> list[str]:
    """Build subprocess command for a single backtest run.

    Args:
        run_spec: Run specification
        base_args: Base arguments for all runs
        output_dir: Output directory for this run
        repo_root: Repository root directory

    Returns:
        List of command arguments
    """
    script_path = repo_root / "scripts" / "run_backtest_strategy.py"

    # Merge base_args with run_spec overrides
    merged_args = {**base_args, **run_spec.overrides}

    cmd: list[str] = [
        sys.executable,
        str(script_path),
        "--freq",
        str(merged_args.get("freq", "1d")),
        "--strategy",
        str(merged_args.get("strategy", "multifactor_long_short")),
        "--start-date",
        run_spec.start_date,
        "--end-date",
        run_spec.end_date,
        "--data-source",
        str(merged_args.get("data_source", "local")),
        "--start-capital",
        str(merged_args.get("start_capital", 100000.0)),
        "--rebalance-freq",
        str(merged_args.get("rebalance_freq", "M")),
        "--max-gross-exposure",
        str(merged_args.get("max_gross_exposure", 1.0)),
        "--bundle-path",
        str(run_spec.bundle_path),
        "--out",
        str(output_dir),
    ]

    # Add optional arguments
    if merged_args.get("symbols_file"):
        cmd.extend(["--symbols-file", str(merged_args["symbols_file"])])
    elif merged_args.get("universe"):
        cmd.extend(["--universe", str(merged_args["universe"])])

    if merged_args.get("generate_report", True):
        cmd.append("--generate-report")

    if merged_args.get("generate_risk_report", False):
        cmd.append("--generate-risk-report")

    if merged_args.get("generate_tca_report", False):
        cmd.append("--generate-tca-report")

    return cmd


def _ensure_unique_output_dir(base_dir: Path, run_id: str, run_index: int | None = None) -> Path:
    """Ensure unique output directory for a run.

    Creates a directory with run_id and optional index/hash to avoid collisions.

    Args:
        base_dir: Base directory for runs
        run_id: Run identifier
        run_index: Optional index in run list (for uniqueness)

    Returns:
        Unique output directory path
    """
    base_dir.mkdir(parents=True, exist_ok=True)

    # Use run_id with index if provided, otherwise use hash of run_id
    if run_index is not None:
        unique_name = f"{run_index:04d}_{run_id}"
    else:
        # Create short hash for uniqueness
        run_hash = hashlib.md5(run_id.encode()).hexdigest()[:8]
        unique_name = f"{run_id}_{run_hash}"

    return base_dir / unique_name


def _setup_run_logging(run_output_dir: Path, run_id: str) -> logging.Logger:
    """Setup logging for a single run.

    Args:
        run_output_dir: Output directory for this run
        run_id: Run identifier

    Returns:
        Logger instance for this run
    """
    run_log_path = run_output_dir / "run.log"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    run_logger = logging.getLogger(f"{__name__}.run.{run_id}")
    run_logger.setLevel(logging.INFO)

    # Remove existing handlers
    run_logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(run_log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    run_logger.addHandler(file_handler)

    return run_logger


def _run_single_backtest_worker(
    run_spec_dict: dict[str, Any],
    base_args: dict[str, Any],
    output_base_dir: str,
    repo_root: str,
    run_index: int,
    timeout_sec: float | None = None,
) -> dict[str, Any]:
    """Worker function for parallel execution (must be top-level for pickling).

    This function is called in a separate process, so all arguments must be serializable.

    Args:
        run_spec_dict: RunSpec as dictionary
        base_args: Base arguments for all runs
        output_base_dir: Base output directory (as string)
        repo_root: Repository root directory (as string)
        run_index: Index of run in original list
        timeout_sec: Optional timeout per run

    Returns:
        RunResult as dictionary
    """
    # Reconstruct RunSpec from dict
    run_spec = RunSpec(
        id=run_spec_dict["id"],
        bundle_path=Path(run_spec_dict["bundle_path"]),
        start_date=run_spec_dict["start_date"],
        end_date=run_spec_dict["end_date"],
        tags=run_spec_dict.get("tags", []),
        overrides=run_spec_dict.get("overrides", {}),
    )

    output_base = Path(output_base_dir)
    repo_root_path = Path(repo_root)

    # Create unique output directory
    run_output_dir = _ensure_unique_output_dir(output_base, run_spec.id, run_index)
    backtest_output_dir = run_output_dir / "backtest"
    backtest_output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging for this run
    run_logger = _setup_run_logging(run_output_dir, run_spec.id)
    run_logger.info(f"Starting backtest run: {run_spec.id}")

    start_time = datetime.utcnow()

    try:
        # Build command
        cmd = _build_backtest_command(run_spec, base_args, backtest_output_dir, repo_root_path)

        run_logger.info(f"Command: {' '.join(str(c) for c in cmd)}")
        run_logger.info(f"Output dir: {backtest_output_dir}")

        # Execute with optional timeout
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root_path),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )

        end_time = datetime.utcnow()
        runtime_sec = (end_time - start_time).total_seconds()

        if proc.returncode == 0:
            run_logger.info(f"Run {run_spec.id} completed successfully in {runtime_sec:.2f}s")

            # Check for timings/profile artifacts
            timings_path = backtest_output_dir / "timings.json"
            profile_path = backtest_output_dir / "profile"

            result = RunResult(
                run_id=run_spec.id,
                status="success",
                output_dir=backtest_output_dir,
                runtime_sec=runtime_sec,
                timings_path=timings_path if timings_path.exists() else None,
                profile_path=profile_path if profile_path.exists() else None,
                run_index=run_index,
            )
        else:
            error_msg = f"Backtest exited with code {proc.returncode}"
            if proc.stderr:
                error_msg += f": {proc.stderr[:200]}"
            run_logger.error(error_msg)

            result = RunResult(
                run_id=run_spec.id,
                status="failed",
                output_dir=backtest_output_dir,
                runtime_sec=runtime_sec,
                error=error_msg,
                run_index=run_index,
            )

    except subprocess.TimeoutExpired:
        end_time = datetime.utcnow()
        runtime_sec = (end_time - start_time).total_seconds()
        error_msg = f"Backtest timed out after {timeout_sec}s"
        run_logger.error(error_msg)

        result = RunResult(
            run_id=run_spec.id,
            status="timeout",
            output_dir=backtest_output_dir,
            runtime_sec=runtime_sec,
            error=error_msg,
            run_index=run_index,
        )

    except Exception as exc:
        end_time = datetime.utcnow()
        runtime_sec = (end_time - start_time).total_seconds()
        error_msg = f"Exception during backtest: {exc}"
        run_logger.error(f"Run {run_spec.id} raised exception: {exc}", exc_info=True)

        result = RunResult(
            run_id=run_spec.id,
            status="failed",
            output_dir=backtest_output_dir,
            runtime_sec=runtime_sec,
            error=error_msg,
            run_index=run_index,
        )

    finally:
        # Clean up logger handlers
        for handler in run_logger.handlers[:]:
            handler.close()
            run_logger.removeHandler(handler)

    return result.to_dict()


def _run_single_backtest(
    run_spec: RunSpec,
    base_args: dict[str, Any],
    output_dir: Path,
    repo_root: Path,
    run_index: int | None = None,
    backtest_fn: Callable[[RunSpec, dict[str, Any], Path], RunResult] | None = None,
) -> RunResult:
    """Execute a single backtest run.

    Args:
        run_spec: Run specification
        base_args: Base arguments for all runs
        output_dir: Output directory for runs
        repo_root: Repository root directory
        run_index: Optional index in run list (for unique output dirs)
        backtest_fn: Optional function to execute backtest (for testing/mocking)

    Returns:
        RunResult with status and metadata
    """
    # Create unique output directory
    run_output_dir = _ensure_unique_output_dir(output_dir, run_spec.id, run_index)
    backtest_output_dir = run_output_dir / "backtest"
    backtest_output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging for this run
    run_logger = _setup_run_logging(run_output_dir, run_spec.id)
    run_logger.info(f"Starting backtest run: {run_spec.id}")

    start_time = datetime.utcnow()

    # Use custom backtest function if provided (for testing)
    if backtest_fn is not None:
        result = backtest_fn(run_spec, base_args, backtest_output_dir)
        result.run_index = run_index
        end_time = datetime.utcnow()
        # Clean up logger handlers
        for handler in run_logger.handlers[:]:
            handler.close()
            run_logger.removeHandler(handler)
        # Write manifest even for mocked runs
        _write_run_manifest(
            result=result,
            run_spec=run_spec,
            base_args=base_args,
            run_output_dir=run_output_dir,
            start_time=start_time,
            end_time=end_time,
        )
        return result

    # Build command
    cmd = _build_backtest_command(run_spec, base_args, backtest_output_dir, repo_root)

    run_logger.info(f"Command: {' '.join(str(c) for c in cmd)}")
    run_logger.info(f"Output dir: {backtest_output_dir}")

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            check=False,
            capture_output=True,
            text=True,
        )

        end_time = datetime.utcnow()
        runtime_sec = (end_time - start_time).total_seconds()

        if proc.returncode == 0:
            run_logger.info(f"Run {run_spec.id} completed successfully in {runtime_sec:.2f}s")

            # Check for timings/profile artifacts
            timings_path = backtest_output_dir / "timings.json"
            profile_path = backtest_output_dir / "profile"

            result = RunResult(
                run_id=run_spec.id,
                status="success",
                output_dir=backtest_output_dir,
                runtime_sec=runtime_sec,
                timings_path=timings_path if timings_path.exists() else None,
                profile_path=profile_path if profile_path.exists() else None,
                run_index=run_index,
            )
        else:
            error_msg = f"Backtest exited with code {proc.returncode}"
            if proc.stderr:
                error_msg += f": {proc.stderr[:200]}"
            run_logger.error(f"Run {run_spec.id} failed: {error_msg}")

            result = RunResult(
                run_id=run_spec.id,
                status="failed",
                output_dir=backtest_output_dir,
                runtime_sec=runtime_sec,
                error=error_msg,
                run_index=run_index,
            )

    except Exception as exc:
        end_time = datetime.utcnow()
        runtime_sec = (end_time - start_time).total_seconds()
        error_msg = f"Exception during backtest: {exc}"
        run_logger.error(f"Run {run_spec.id} raised exception: {exc}", exc_info=True)

        result = RunResult(
            run_id=run_spec.id,
            status="failed",
            output_dir=backtest_output_dir,
            runtime_sec=runtime_sec,
            error=error_msg,
            run_index=run_index,
        )

    finally:
        # Clean up logger handlers
        for handler in run_logger.handlers[:]:
            handler.close()
            run_logger.removeHandler(handler)

    return result


def run_batch_serial(
    run_specs: list[RunSpec],
    batch_name: str,
    output_root: Path,
    base_args: dict[str, Any],
    repo_root: Path | None = None,
    fail_fast: bool = False,
    seed: int | None = None,
    backtest_fn: Callable[[RunSpec, dict[str, Any], Path], RunResult] | None = None,
    config: BatchConfig | None = None,
) -> BatchResult:
    """Execute batch of backtests serially.

    This function executes all runs in sequence, ensuring deterministic ordering.
    No parallelization is used, making this suitable for stable, reproducible results.

    Args:
        run_specs: List of run specifications
        batch_name: Name of the batch
        output_root: Base output directory for batch
        base_args: Base arguments for all runs
        repo_root: Repository root directory (default: auto-detect)
        fail_fast: If True, abort batch on first failure
        seed: Optional random seed (for reproducibility)
        backtest_fn: Optional function to execute backtest (for testing/mocking)

    Returns:
        BatchResult with per-run status and summary
    """
    if repo_root is None:
        # Auto-detect repo root (assume we're in src/assembled_core/experiments/)
        repo_root = Path(__file__).resolve().parents[3]

    # Set random seed if provided
    if seed is not None:
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"Random seed set to {seed}")

    # Create batch output directory
    batch_output_dir = output_root / batch_name
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Batch Backtests: {batch_name}")
    logger.info("=" * 60)
    logger.info(f"Output root: {batch_output_dir}")
    logger.info(f"Runs: {len(run_specs)}")
    logger.info(f"Fail fast: {fail_fast}")
    logger.info("")

    started_at = datetime.utcnow()
    run_results: list[RunResult] = []

    for idx, run_spec in enumerate(run_specs, 1):
        logger.info(f"[{idx}/{len(run_specs)}] Processing run: {run_spec.id}")

        result = _run_single_backtest(
            run_spec=run_spec,
            base_args=base_args,
            output_dir=batch_output_dir / "runs",
            repo_root=repo_root,
            run_index=idx - 1,
            backtest_fn=backtest_fn,
        )

        run_results.append(result)

        # Check fail_fast
        if fail_fast and result.status == "failed":
            logger.warning(f"Fail-fast enabled: aborting batch after run {run_spec.id}")
            break

    finished_at = datetime.utcnow()
    total_runtime_sec = (finished_at - started_at).total_seconds()

    batch_result = BatchResult(
        batch_name=batch_name,
        started_at=started_at,
        finished_at=finished_at,
        total_runtime_sec=total_runtime_sec,
        run_results=run_results,
    )

    # Write summary files
    _write_batch_summary(batch_result, batch_output_dir)

    # Write batch manifest if config provided
    if config is not None:
        _write_batch_manifest(
            batch_result=batch_result,
            config=config,
            batch_output_dir=batch_output_dir,
            run_specs=run_specs,
        )

    logger.info("=" * 60)
    logger.info(f"Batch completed: {batch_result.success_count} success, {batch_result.failed_count} failed")
    logger.info(f"Total runtime: {total_runtime_sec:.2f}s")
    logger.info("=" * 60)

    return batch_result


def collect_backtest_metrics(run_output_dir: Path, freq: str = "1d") -> dict[str, Any]:
    """Collect backtest metrics from run output directory.

    Robust against missing files: returns NaN for missing metrics.

    Args:
        run_output_dir: Run output directory (should contain backtest subdirectory)
        freq: Trading frequency ("1d" or "5min") for annualization

    Returns:
        Dictionary with metrics:
        - sharpe: Sharpe ratio (annualized)
        - deflated_sharpe: Deflated Sharpe ratio (if computable)
        - max_dd: Maximum drawdown (absolute, negative)
        - max_dd_pct: Maximum drawdown (percent, negative)
        - turnover: Portfolio turnover (annualized)
        - start_date: First timestamp (if available)
        - end_date: Last timestamp (if available)
        - strategy: Strategy name (if available from config)
        - params_hash: Config hash (if available from manifest)
    """
    import math

    import pandas as pd

    from src.assembled_core.qa.metrics import (
        compute_all_metrics,
        deflated_sharpe_ratio_from_returns,
    )

    metrics_dict: dict[str, Any] = {
        "sharpe": math.nan,
        "deflated_sharpe": math.nan,
        "max_dd": math.nan,
        "max_dd_pct": math.nan,
        "turnover": math.nan,
        "start_date": None,
        "end_date": None,
        "strategy": None,
        "params_hash": None,
        "data_snapshot_id": None,  # D4: Snapshot ID for reproducibility (wird aus Manifest gelesen)
    }

    # Try to load run manifest for strategy/params and data_snapshot_id (D4)
    manifest_path = run_output_dir / "run_manifest.json"
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
                metrics_dict["params_hash"] = manifest.get("config_hash")
                # D4: Read data_snapshot_id from manifest (wenn vorhanden)
                metrics_dict["data_snapshot_id"] = manifest.get("data_snapshot_id")
                # Extract strategy from base_args or run_spec
                base_args = manifest.get("base_args", {})
                metrics_dict["strategy"] = base_args.get("strategy")
        except Exception:
            pass  # Ignore manifest errors (data_snapshot_id bleibt None)

    # Look for equity curve files (portfolio_equity takes precedence over equity_curve)
    backtest_dir = run_output_dir / "backtest"
    if not backtest_dir.exists():
        backtest_dir = run_output_dir  # Fallback to run_output_dir itself

    # Try portfolio_equity first, then equity_curve
    equity_file = backtest_dir / f"portfolio_equity_{freq}.csv"
    if not equity_file.exists():
        equity_file = backtest_dir / f"equity_curve_{freq}.csv"

    if not equity_file.exists():
        logger.debug(f"No equity file found in {backtest_dir}")
        return metrics_dict

    try:
        equity_df = pd.read_csv(equity_file)
        if "timestamp" not in equity_df.columns or "equity" not in equity_df.columns:
            logger.warning(f"Equity file {equity_file} missing required columns")
            return metrics_dict

        equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"], utc=True, errors="coerce")
        equity_df = equity_df.sort_values("timestamp").reset_index(drop=True)
        equity_df = equity_df.dropna(subset=["timestamp", "equity"])

        if equity_df.empty:
            logger.warning(f"Equity file {equity_file} is empty after cleaning")
            return metrics_dict

        # Extract start/end dates
        metrics_dict["start_date"] = equity_df["timestamp"].iloc[0].isoformat()
        metrics_dict["end_date"] = equity_df["timestamp"].iloc[-1].isoformat()

        # Get start capital from equity curve (first value)
        start_capital = float(equity_df["equity"].iloc[0])

        # Try to load trades for turnover calculation
        trades_file = backtest_dir / f"orders_{freq}.csv"
        trades_df = None
        if trades_file.exists():
            try:
                trades_df = pd.read_csv(trades_file)
                # Ensure required columns for trades
                required_cols = ["timestamp", "symbol", "side", "qty"]
                if all(col in trades_df.columns for col in required_cols):
                    trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"], utc=True, errors="coerce")
                    trades_df = trades_df.dropna(subset=["timestamp"])
                else:
                    trades_df = None  # Missing required columns
            except Exception:
                trades_df = None  # Ignore trade loading errors

        # Compute all metrics
        try:
            metrics = compute_all_metrics(
                equity=equity_df[["timestamp", "equity"]],
                trades=trades_df,
                start_capital=start_capital,
                freq=freq,
                risk_free_rate=0.0,
            )

            metrics_dict["sharpe"] = metrics.sharpe_ratio if metrics.sharpe_ratio is not None else math.nan
            metrics_dict["max_dd"] = metrics.max_drawdown
            metrics_dict["max_dd_pct"] = metrics.max_drawdown_pct
            metrics_dict["turnover"] = metrics.turnover if metrics.turnover is not None else math.nan

            # Compute deflated Sharpe if we have returns
            if metrics.sharpe_ratio is not None and not math.isnan(metrics.sharpe_ratio):
                try:
                    # Extract returns from equity
                    equity_series = equity_df["equity"].astype(float)
                    returns = equity_series.pct_change().dropna()
                    if len(returns) > 1:
                        # Use n_tests=1 as default (single strategy test)
                        # In a grid search context, this could be set to the number of tested parameter combinations
                        deflated_sharpe = deflated_sharpe_ratio_from_returns(
                            returns=returns, freq=freq, n_tests=1, risk_free_rate=0.0
                        )
                        metrics_dict["deflated_sharpe"] = deflated_sharpe
                except Exception as exc:
                    logger.debug(f"Could not compute deflated Sharpe: {exc}")

        except Exception as exc:
            logger.warning(f"Failed to compute metrics from {equity_file}: {exc}")

    except Exception as exc:
        logger.warning(f"Failed to read equity file {equity_file}: {exc}")

    return metrics_dict


def _write_batch_summary(batch_result: BatchResult, output_dir: Path) -> None:
    """Write batch summary files (JSON and CSV).

    Args:
        batch_result: Batch result to write
        output_dir: Output directory for batch
    """
    # Write JSON summary
    summary_json = output_dir / "batch_summary.json"
    try:
        summary_data = {
            "batch_name": batch_result.batch_name,
            "started_at": batch_result.started_at.isoformat(),
            "finished_at": batch_result.finished_at.isoformat(),
            "total_runtime_sec": batch_result.total_runtime_sec,
            "success_count": batch_result.success_count,
            "failed_count": batch_result.failed_count,
            "skipped_count": batch_result.skipped_count,
            "error_summary": batch_result.error_summary,
            "runs": [
                {
                    "run_id": r.run_id,
                    "status": r.status,
                    "output_dir": str(r.output_dir) if r.output_dir else None,
                    "runtime_sec": r.runtime_sec,
                    "error": r.error,
                    "timings_path": str(r.timings_path) if r.timings_path else None,
                    "profile_path": str(r.profile_path) if r.profile_path else None,
                }
                for r in batch_result.run_results
            ],
        }

        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        logger.info(f"Batch summary JSON written to {summary_json}")

    except Exception as exc:
        logger.warning(f"Failed to write batch summary JSON: {exc}", exc_info=True)

    # Write CSV summary with metrics
    summary_csv = output_dir / "batch_summary.csv"
    try:
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Collect metrics for all runs
            all_metrics = []
            for r in batch_result.run_results:
                if r.output_dir and r.output_dir.exists():
                    # Determine freq from output_dir structure or default
                    freq = "1d"  # Default, could be extracted from manifest if needed
                    metrics = collect_backtest_metrics(r.output_dir, freq=freq)
                    all_metrics.append(metrics)
                else:
                    all_metrics.append({})

            # Build header
            header = [
                "run_id",
                "status",
                "strategy",
                "params_hash",
                "data_snapshot_id",  # D4: Snapshot ID for reproducibility
                "start_date",
                "end_date",
                "sharpe",
                "deflated_sharpe",
                "max_dd",
                "max_dd_pct",
                "turnover",
                "runtime_sec",
                "error",
            ]
            writer.writerow(header)

            # Write rows
            for r, metrics in zip(batch_result.run_results, all_metrics, strict=False):
                row = [
                    r.run_id,
                    r.status,
                    metrics.get("strategy", ""),
                    metrics.get("params_hash", ""),
                    metrics.get("data_snapshot_id", "") or "",  # D4: Snapshot ID
                    metrics.get("start_date", ""),
                    metrics.get("end_date", ""),
                    f"{metrics.get('sharpe', math.nan):.4f}" if not math.isnan(metrics.get("sharpe", math.nan)) else "",
                    f"{metrics.get('deflated_sharpe', math.nan):.4f}"
                    if not math.isnan(metrics.get("deflated_sharpe", math.nan))
                    else "",
                    f"{metrics.get('max_dd', math.nan):.2f}" if not math.isnan(metrics.get("max_dd", math.nan)) else "",
                    f"{metrics.get('max_dd_pct', math.nan):.2f}"
                    if not math.isnan(metrics.get("max_dd_pct", math.nan))
                    else "",
                    f"{metrics.get('turnover', math.nan):.2f}"
                    if not math.isnan(metrics.get("turnover", math.nan))
                    else "",
                    f"{r.runtime_sec:.3f}",
                    r.error or "",
                ]
                writer.writerow(row)

        logger.info(f"Batch summary CSV written to {summary_csv}")

    except Exception as exc:
        logger.warning(f"Failed to write batch summary CSV: {exc}", exc_info=True)


def _write_batch_manifest(
    batch_result: BatchResult,
    config: BatchConfig,
    batch_output_dir: Path,
    run_specs: list[RunSpec],
) -> None:
    """Write batch manifest.json with metadata for the entire batch.

    Args:
        batch_result: BatchResult for this batch
        config: BatchConfig used
        batch_output_dir: Batch output directory
        run_specs: List of expanded run specs
    """
    # Get git commit hash
    git_hash = _get_git_commit_hash()

    # Compute config hash
    config_dict = config.model_dump(mode="json")
    config_hash = _compute_config_hash(config_dict)

    # Build expanded runs list (stable ordering)
    expanded_runs = [
        {
            "run_id": spec.id,
            "bundle_path": str(spec.bundle_path),
            "start_date": spec.start_date,
            "end_date": spec.end_date,
            "tags": spec.tags,
            "overrides": spec.overrides,
        }
        for spec in run_specs
    ]

    # Build manifest
    manifest = {
        "batch_name": batch_result.batch_name,
        "description": config.description,
        "started_at": batch_result.started_at.isoformat(),
        "finished_at": batch_result.finished_at.isoformat(),
        "total_runtime_sec": batch_result.total_runtime_sec,
        "config_hash": config_hash,
        "git_commit_hash": git_hash,
        "seed": config.seed,
        "max_workers": getattr(config, "max_workers", None),
        "fail_fast": config.fail_fast,
        "output_root": str(config.output_root),
        "base_args": config.base_args,
        "expanded_runs": expanded_runs,
        "run_results_summary": {
            "total_runs": len(batch_result.run_results),
            "success_count": batch_result.success_count,
            "failed_count": batch_result.failed_count,
            "skipped_count": batch_result.skipped_count,
            "run_ids": [r.run_id for r in batch_result.run_results],
        },
        "versions": {
            "python": sys.version.split()[0],
        },
        "created_at": datetime.utcnow().isoformat(),
    }

    # Write manifest
    manifest_path = batch_output_dir / "batch_manifest.json"
    try:
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=True, sort_keys=True)
        logger.info(f"Batch manifest written to {manifest_path}")
    except Exception as exc:
        logger.warning(f"Failed to write batch manifest: {exc}", exc_info=True)


def run_batch_parallel(
    run_specs: list[RunSpec],
    batch_name: str,
    output_root: Path,
    base_args: dict[str, Any],
    max_workers: int = 4,
    repo_root: Path | None = None,
    fail_fast: bool = False,
    timeout_per_run: float | None = None,
    seed: int | None = None,
    config: BatchConfig | None = None,
) -> BatchResult:
    """Execute batch of backtests in parallel using ProcessPoolExecutor.

    This function executes runs in parallel while maintaining deterministic ordering
    of results. Each run gets a unique output directory and separate log file.

    Args:
        run_specs: List of run specifications
        batch_name: Name of the batch
        output_root: Base output directory for batch
        base_args: Base arguments for all runs
        max_workers: Maximum number of parallel workers (default: 4)
        repo_root: Repository root directory (default: auto-detect)
        fail_fast: If True, cancel remaining runs on first failure
        timeout_per_run: Optional timeout per run in seconds
        seed: Optional random seed (set in main process, not workers)

    Returns:
        BatchResult with per-run status and summary (results in original order)
    """
    if repo_root is None:
        # Auto-detect repo root
        repo_root = Path(__file__).resolve().parents[3]

    # Set random seed if provided (only in main process)
    if seed is not None:
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"Random seed set to {seed}")

    # Create batch output directory
    batch_output_dir = output_root / batch_name
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    runs_output_dir = batch_output_dir / "runs"
    runs_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Batch Backtests (Parallel): {batch_name}")
    logger.info("=" * 60)
    logger.info(f"Output root: {batch_output_dir}")
    logger.info(f"Runs: {len(run_specs)}")
    logger.info(f"Max workers: {max_workers}")
    logger.info(f"Fail fast: {fail_fast}")
    logger.info(f"Timeout per run: {timeout_per_run}s" if timeout_per_run else "No timeout")
    logger.info("")

    started_at = datetime.utcnow()

    # Prepare run specs as dictionaries (for pickling)
    run_spec_dicts = [
        {
            "id": run_spec.id,
            "bundle_path": str(run_spec.bundle_path),
            "start_date": run_spec.start_date,
            "end_date": run_spec.end_date,
            "tags": run_spec.tags,
            "overrides": run_spec.overrides,
        }
        for run_spec in run_specs
    ]

    # Execute in parallel
    run_results: list[RunResult | None] = [None] * len(run_specs)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(
                _run_single_backtest_worker,
                run_spec_dict,
                base_args,
                str(runs_output_dir),
                str(repo_root),
                idx,
                timeout_per_run,
            ): idx
            for idx, run_spec_dict in enumerate(run_spec_dicts)
        }

        # Collect results as they complete (but maintain original order)
        completed = 0
        failed_indices: set[int] = set()

        for future in as_completed(future_to_index):
            run_index = future_to_index[future]
            completed += 1

            try:
                result_dict = future.result()
                result = RunResult.from_dict(result_dict)

                # Store result in original position (for deterministic ordering)
                run_results[run_index] = result

                if result.status == "success":
                    logger.info(
                        f"[{completed}/{len(run_specs)}] Run {result.run_id} completed successfully "
                        f"in {result.runtime_sec:.2f}s"
                    )
                elif result.status == "timeout":
                    logger.warning(
                        f"[{completed}/{len(run_specs)}] Run {result.run_id} timed out "
                        f"after {result.runtime_sec:.2f}s"
                    )
                    failed_indices.add(run_index)
                else:
                    logger.warning(
                        f"[{completed}/{len(run_specs)}] Run {result.run_id} failed: {result.error}"
                    )
                    failed_indices.add(run_index)

                # Check fail_fast
                if fail_fast and result.status in ("failed", "timeout"):
                    logger.warning(
                        f"Fail-fast enabled: cancelling remaining runs after {result.run_id}"
                    )
                    # Cancel remaining futures
                    for remaining_future in future_to_index:
                        if remaining_future != future and not remaining_future.done():
                            remaining_future.cancel()
                    break

            except Exception as exc:
                logger.error(f"Exception while processing run at index {run_index}: {exc}", exc_info=True)
                # Create error result
                run_spec = run_specs[run_index]
                run_results[run_index] = RunResult(
                    run_id=run_spec.id,
                    status="failed",
                    runtime_sec=0.0,
                    error=f"Exception in executor: {exc}",
                    run_index=run_index,
                )
                failed_indices.add(run_index)

    # Filter out None results (if any were cancelled)
    run_results_filtered = [r for r in run_results if r is not None]

    finished_at = datetime.utcnow()
    total_runtime_sec = (finished_at - started_at).total_seconds()

    batch_result = BatchResult(
        batch_name=batch_name,
        started_at=started_at,
        finished_at=finished_at,
        total_runtime_sec=total_runtime_sec,
        run_results=run_results_filtered,
    )

    # Write summary files
    _write_batch_summary(batch_result, batch_output_dir)

    # Write batch manifest if config provided
    if config is not None:
        _write_batch_manifest(
            batch_result=batch_result,
            config=config,
            batch_output_dir=batch_output_dir,
            run_specs=run_specs,
        )

    logger.info("=" * 60)
    logger.info(
        f"Batch completed: {batch_result.success_count} success, "
        f"{batch_result.failed_count} failed, "
        f"Total runtime: {total_runtime_sec:.2f}s"
    )
    logger.info("=" * 60)

    return batch_result

