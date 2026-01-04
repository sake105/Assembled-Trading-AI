#!/usr/bin/env python
"""
Batch Runner MVP (P4) - Reproduzierbare Batch-Backtests mit Manifest.

Dieses Script führt mehrere Backtests basierend auf einer YAML-Konfiguration aus
und erstellt pro Run ein Manifest mit Parametern und Metadaten.

Example:
    python scripts/batch_runner.py --config-file configs/batch_example.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_backtest_strategy import run_backtest_from_args

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file with optional PyYAML dependency."""
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "YAML config requested but PyYAML is not installed. "
            "Install via 'pip install pyyaml' or use JSON config instead."
        ) from exc

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _compute_run_id_hash(run_cfg: RunConfig, seed: int) -> str:
    """Compute deterministic run ID hash from parameters.

    Args:
        run_cfg: Run configuration
        seed: Batch seed

    Returns:
        Short hash (16 chars) for use as run_id
    """
    # Build deterministic params dict (sorted keys)
    params_dict = {
        "strategy": run_cfg.strategy,
        "freq": run_cfg.freq,
        "start_date": run_cfg.start_date,
        "end_date": run_cfg.end_date,
        "universe": run_cfg.universe or "",
        "seed": seed,
    }
    # Include params dict in hash (stable serialization: sorted keys, JSON dumps)
    if run_cfg.params:
        # Sort params keys for deterministic hashing
        params_sorted = {k: v for k, v in sorted(run_cfg.params.items())}
        params_dict["params"] = params_sorted
    
    # Sort keys for deterministic hashing
    params_json = json.dumps(params_dict, sort_keys=True, default=str)
    return hashlib.sha256(params_json.encode()).hexdigest()[:16]


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


class TradingFreq(str, Enum):
    """Supported trading frequencies."""
    DAILY = "1d"
    INTRADAY_5MIN = "5min"


class RunConfig(BaseModel):
    """Configuration for a single backtest run with validation."""

    id: str = Field(..., alias="name", description="Run identifier (unique within batch, alias: 'name')")
    strategy: str = Field(..., min_length=1, description="Strategy name (e.g., 'trend_baseline')")
    freq: TradingFreq = Field(..., description="Trading frequency: '1d' or '5min'")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    universe: str | None = Field(None, description="Path to universe file (optional)")
    start_capital: float = Field(100000.0, gt=0, description="Starting capital (must be > 0)")
    use_factor_store: bool = Field(False, description="Enable factor store")
    factor_store_root: str | None = Field(None, description="Factor store root path (optional)")
    factor_group: str | None = Field(None, description="Factor group name (optional)")
    params: dict[str, Any] = Field(default_factory=dict, description="Additional parameters (mapped to CLI flags)")
    extra_args: dict[str, Any] = Field(default_factory=dict, description="Additional arguments (legacy, use params instead)")

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate date format is YYYY-MM-DD."""
        if not v or not isinstance(v, str):
            raise ValueError("Date must be a non-empty string")
        # Check format YYYY-MM-DD
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        if not date_pattern.match(v):
            raise ValueError(f"Date must be in YYYY-MM-DD format, got: {v}")
        # Try to parse to ensure it's a valid date
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"Invalid date: {v}") from exc
        return v

    @model_validator(mode="after")
    def validate_end_after_start(self) -> "RunConfig":
        """Validate that end_date is after start_date."""
        start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d")
        if end_date < start_date:
            raise ValueError(f"end_date ({self.end_date}) must be >= start_date ({self.start_date})")
        return self

    @field_validator("universe")
    @classmethod
    def validate_universe(cls, v: str | None) -> str | None:
        """Validate universe path if provided."""
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError("universe cannot be empty string (use null/omit if not needed)")
        return v

    @field_validator("factor_store_root", "factor_group")
    @classmethod
    def validate_factor_store_fields(cls, v: str | None) -> str | None:
        """Validate factor store fields if provided."""
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError("factor_store_root and factor_group cannot be empty strings (use null/omit if not needed)")
        return v

    class Config:
        use_enum_values = True  # Use enum values instead of enum objects
        populate_by_name = True  # Allow both 'id' and 'name' (alias) in input


class BatchConfig(BaseModel):
    """Configuration for an entire batch of backtests with validation."""

    batch_name: str = Field(..., min_length=1, description="Batch name (required)")
    output_root: Path = Field(..., description="Output root directory")
    seed: int = Field(42, ge=0, description="Random seed for reproducibility (>= 0)")
    defaults: dict[str, Any] = Field(default_factory=dict, description="Default values for runs (freq, start_date, end_date, universe, start_capital, etc.)")
    runs: list[RunConfig] = Field(..., min_length=1, description="List of run configurations (non-empty)")

    @model_validator(mode="after")
    def validate_run_ids_unique(self) -> "BatchConfig":
        """Validate that all run IDs are unique (after auto-generation)."""
        run_ids = [run.id for run in self.runs]
        duplicates = [rid for rid in run_ids if run_ids.count(rid) > 1]
        if duplicates:
            raise ValueError(f"Duplicate run IDs found: {set(duplicates)}")
        return self

    class Config:
        arbitrary_types_allowed = True  # Allow Path type


def load_batch_config(config_path: Path) -> BatchConfig:
    """Load and validate batch config from YAML file.

    This function performs strict validation using Pydantic models to ensure
    config correctness before execution, especially important for parallel mode
    where errors are harder to debug.

    Args:
        config_path: Path to YAML config file

    Returns:
        Parsed and validated BatchConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid (with detailed error messages)
        pydantic.ValidationError: If Pydantic validation fails (with field-level errors)
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = _load_yaml(config_path)

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping/object")

    # Extract and validate batch-level fields
    batch_name = str(raw.get("batch_name") or "").strip()
    if not batch_name:
        raise ValueError("batch_name must be set in config (non-empty string)")

    output_root_raw = raw.get("output_root") or "output/batch_backtests"
    output_root = (ROOT / output_root_raw).resolve()

    seed_raw = raw.get("seed")
    if seed_raw is not None:
        try:
            seed = int(seed_raw)
            if seed < 0:
                raise ValueError(f"seed must be >= 0, got: {seed}")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"seed must be a non-negative integer, got: {seed_raw}") from exc
    else:
        seed = 42

    # Extract batch-level defaults
    defaults_raw = raw.get("defaults")
    if defaults_raw is not None:
        if not isinstance(defaults_raw, dict):
            raise ValueError("defaults must be a mapping/object if provided")
        defaults = dict(defaults_raw)
    else:
        defaults = {}

    runs_raw = raw.get("runs")
    if not isinstance(runs_raw, list):
        raise ValueError("runs must be a list in config")
    if not runs_raw:
        raise ValueError("runs must be a non-empty list in config")

    # Parse and validate each run config
    runs: list[RunConfig] = []
    for idx, run_raw in enumerate(runs_raw):
        if not isinstance(run_raw, dict):
            raise ValueError(f"runs[{idx}] must be an object/dict")

        # Run-ID can be explicitly set via 'id' or 'name' (alias), or will be auto-generated
        run_id = run_raw.get("id") or run_raw.get("name")
        if run_id:
            run_id = str(run_id).strip()
            if not run_id:
                raise ValueError(f"runs[{idx}]: id/name cannot be empty string (use null/omit for auto-generation)")

        # Merge defaults with run-specific values (run takes precedence)
        merged = dict(defaults)
        merged.update(run_raw)

        # Prepare data for Pydantic model (with type conversions)
        # Apply defaults if fields are missing
        run_data: dict[str, Any] = {
            "id": run_id or "",  # Will be set later if empty
            "strategy": str(merged.get("strategy") or "").strip(),
            "freq": str(merged.get("freq") or "").strip(),
            "start_date": str(merged.get("start_date") or "").strip(),
            "end_date": str(merged.get("end_date") or "").strip(),
        }

        # Optional fields (with defaults fallback)
        if "universe" in merged:
            universe_val = merged["universe"]
            run_data["universe"] = str(universe_val).strip() if universe_val else None

        if "start_capital" in merged:
            try:
                run_data["start_capital"] = float(merged["start_capital"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"runs[{idx}]: start_capital must be a number, got: {merged['start_capital']}") from exc

        if "use_factor_store" in merged:
            run_data["use_factor_store"] = bool(merged["use_factor_store"])

        if "factor_store_root" in merged:
            factor_store_root_val = merged["factor_store_root"]
            run_data["factor_store_root"] = str(factor_store_root_val).strip() if factor_store_root_val else None

        if "factor_group" in merged:
            factor_group_val = merged["factor_group"]
            run_data["factor_group"] = str(factor_group_val).strip() if factor_group_val else None

        # Extract params dict (if present)
        params = {}
        if "params" in merged:
            params_raw = merged["params"]
            if isinstance(params_raw, dict):
                params = dict(params_raw)
            else:
                raise ValueError(f"runs[{idx}]: params must be a mapping/object if provided")
        run_data["params"] = params

        # Collect extra args (legacy, for backward compatibility)
        # Exclude known fields and params
        known_fields = {
            "id", "name", "strategy", "freq", "start_date", "end_date",
            "universe", "start_capital", "use_factor_store",
            "factor_store_root", "factor_group", "params"
        }
        extra_args = {k: v for k, v in run_raw.items() if k not in known_fields}
        run_data["extra_args"] = extra_args

        # Validate using Pydantic (will raise ValidationError with detailed field errors)
        try:
            run_cfg = RunConfig(**run_data)
        except Exception as exc:
            # Re-raise with context about which run failed
            raise ValueError(f"runs[{idx}]: Validation failed: {exc}") from exc

        runs.append(run_cfg)

    # Generate deterministic run IDs for runs without explicit ID
    for run_cfg in runs:
        if not run_cfg.id:
            run_cfg.id = _compute_run_id_hash(run_cfg, seed)

    # Create BatchConfig (will validate run_id uniqueness)
    try:
        batch_cfg = BatchConfig(
            batch_name=batch_name,
            output_root=output_root,
            seed=seed,
            defaults=defaults,
            runs=runs,
        )
    except Exception as exc:
        raise ValueError(f"Batch config validation failed: {exc}") from exc

    return batch_cfg


def _map_params_to_cli_flags(params: dict[str, Any]) -> dict[str, Any]:
    """Map params dict to CLI flags/args.
    
    Args:
        params: Parameters dict (e.g., {"ema_fast": 20, "ema_slow": 50, "verbose": True})
    
    Returns:
        Dict with argparse.Namespace-compatible keys (keep underscore for attribute access)
    """
    args_dict: dict[str, Any] = {}
    
    for key, value in params.items():
        # Keep underscore in key (argparse.Namespace supports underscore attributes)
        # When building CLI command, we'll convert to dash format
        
        if isinstance(value, bool):
            # bool: True => Flag setzen, False => nicht setzen
            if value:
                args_dict[key] = True
        elif isinstance(value, (list, tuple)):
            # list/tuple: comma-join (einheitlich)
            args_dict[key] = ",".join(str(v) for v in value)
        else:
            # Other types: direct value
            args_dict[key] = value
    
    return args_dict


def build_args_from_run_config(run_cfg: RunConfig, output_dir: Path) -> argparse.Namespace:
    """Build argparse.Namespace from RunConfig for run_backtest_from_args.

    Args:
        run_cfg: Run configuration
        output_dir: Output directory for this run

    Returns:
        argparse.Namespace compatible with run_backtest_from_args
    """
    args_dict: dict[str, Any] = {
        "freq": run_cfg.freq,
        "strategy": run_cfg.strategy,
        "start_date": run_cfg.start_date,
        "end_date": run_cfg.end_date,
        "start_capital": run_cfg.start_capital,
        "with_costs": True,  # Default: with costs
        "out": output_dir,
    }

    if run_cfg.universe:
        args_dict["universe"] = Path(run_cfg.universe) if not isinstance(run_cfg.universe, Path) else run_cfg.universe

    if run_cfg.use_factor_store:
        args_dict["use_factor_store"] = True
        if run_cfg.factor_store_root:
            args_dict["factor_store_root"] = Path(run_cfg.factor_store_root) if not isinstance(run_cfg.factor_store_root, Path) else run_cfg.factor_store_root
        if run_cfg.factor_group:
            args_dict["factor_group"] = run_cfg.factor_group

    # Map params to CLI flags
    if run_cfg.params:
        params_flags = _map_params_to_cli_flags(run_cfg.params)
        args_dict.update(params_flags)

    # Merge extra_args (legacy, for backward compatibility)
    args_dict.update(run_cfg.extra_args)

    # Create Namespace object
    return argparse.Namespace(**args_dict)


def write_run_manifest(
    run_id: str,
    run_cfg: RunConfig,
    run_output_dir: Path,
    status: str,
    started_at: datetime,
    finished_at: datetime,
    runtime_sec: float,
    exit_code: int,
    seed: int,
    error: str | None = None,
) -> None:
    """Write run manifest JSON file.

    Args:
        run_id: Run identifier
        run_cfg: Run configuration
        run_output_dir: Run output directory
        status: Run status ("success", "failed", "skipped")
        started_at: Start timestamp
        finished_at: End timestamp
        runtime_sec: Runtime in seconds
        exit_code: Exit code from backtest
        seed: Random seed used for this run (for reproducibility)
        error: Error message (if any)
    """
    git_hash = _get_git_commit_hash()

    # Check if timings file exists
    timings_path = run_output_dir / "run_timings.json"
    timings_path_rel = "run_timings.json" if timings_path.exists() else None

    # Build params dict (include both standard fields and params dict)
    params: dict[str, Any] = {
        "strategy": run_cfg.strategy,
        "freq": run_cfg.freq,
        "start_date": run_cfg.start_date,
        "end_date": run_cfg.end_date,
        "start_capital": run_cfg.start_capital,
        "use_factor_store": run_cfg.use_factor_store,
    }
    if run_cfg.universe:
        params["universe"] = run_cfg.universe
    if run_cfg.factor_store_root:
        params["factor_store_root"] = run_cfg.factor_store_root
    if run_cfg.factor_group:
        params["factor_group"] = run_cfg.factor_group
    
    # Include params dict in manifest (for reproducibility)
    if run_cfg.params:
        params["params"] = run_cfg.params

    manifest = {
        "run_id": run_id,
        "status": status,
        "started_at": started_at.isoformat() + "Z",
        "finished_at": finished_at.isoformat() + "Z",
        "runtime_sec": runtime_sec,
        "seed": seed,  # Explicit seed field for reproducibility
        "params": params,
        "git_commit_hash": git_hash,
        "timings_path": timings_path_rel,
        "output_dir": str(run_output_dir.resolve()),
        "exit_code": exit_code,
    }

    if error:
        manifest["error"] = error

    manifest_path = run_output_dir / "run_manifest.json"
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    except (IOError, OSError) as exc:
        logger.error("Failed to write run manifest to %s: %s", manifest_path, exc)
        raise RuntimeError(f"Failed to write run manifest: {manifest_path}") from exc
    except (TypeError, ValueError) as exc:
        logger.error("Failed to serialize manifest JSON: %s", exc)
        raise ValueError("Manifest contains non-serializable data") from exc

    # Manifest written (logging is done by main process if needed)


def load_existing_manifest(run_output_dir: Path) -> dict[str, Any] | None:
    """Load existing run manifest if it exists.

    Args:
        run_output_dir: Run output directory

    Returns:
        Manifest dict if exists, None otherwise
    """
    manifest_path = run_output_dir / "run_manifest.json"
    if not manifest_path.exists():
        return None

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as exc:
        logger.warning("Failed to load manifest from %s: %s", manifest_path, exc)
        return None


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Sets seeds for:
    - Python random module
    - NumPy random (if available)
    - PyTorch (if available)
    
    Args:
        seed: Random seed value (>= 0)
    """
    random.seed(seed)
    
    if np is not None:
        np.random.seed(seed)
    
    if torch is not None:
        torch.manual_seed(seed)
        # Also set CUDA seeds if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def run_single_backtest(
    run_cfg: RunConfig,
    batch_output_root: Path,
    seed: int,
    dry_run: bool = False,
    resume: bool = False,
    rerun_failed: bool = False,
) -> tuple[str, float, int, str | None]:
    """Execute a single backtest run.

    Args:
        run_cfg: Run configuration
        batch_output_root: Base output directory for batch (output/batch/<batch_name>/)
        seed: Random seed for reproducibility (used for ML/stochastic components)
        dry_run: If True, skip execution
        resume: If True, skip runs that already succeeded
        rerun_failed: If True, rerun failed runs even with resume

    Returns:
        Tuple of (status, runtime_sec, exit_code, error_message)
    """
    # Set random seeds for reproducibility (before any random operations)
    set_random_seeds(seed)
    # Each run gets its own directory: output/batch/<run_id>/
    run_output_dir = batch_output_root / run_cfg.id
    # Create directory atomically to avoid race conditions
    try:
        run_output_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Directory already exists (from parallel execution), continue
        pass

    # Minimal logging in worker (main process will log START/END/SKIP)
    # Check for existing manifest if resume is enabled
    if resume and not dry_run:
        existing_manifest = load_existing_manifest(run_output_dir)
        if existing_manifest:
            existing_status = existing_manifest.get("status")
            if existing_status == "success":
                # Return existing values from manifest (main process will log SKIP)
                existing_runtime = existing_manifest.get("runtime_sec", 0.0)
                existing_exit_code = existing_manifest.get("exit_code", 0)
                return ("skipped", existing_runtime, existing_exit_code, None)
            elif existing_status == "failed" and not rerun_failed:
                # Return existing failed status (main process will log SKIP)
                existing_runtime = existing_manifest.get("runtime_sec", 0.0)
                existing_exit_code = existing_manifest.get("exit_code", 1)
                existing_error = existing_manifest.get("error")
                return ("skipped", existing_runtime, existing_exit_code, existing_error)
            # If rerun_failed=True, continue with execution (main process will log START)

    if dry_run:
        return ("skipped", 0.0, 0, None)

    # Note: Seed is already set at the beginning of the function (line 508)
    # This ensures reproducibility even if resume logic returns early
    
    started_at = datetime.utcnow()

    try:
        # Build args for run_backtest_from_args
        args = build_args_from_run_config(run_cfg, run_output_dir)

        # Call backtest function directly (no subprocess)
        exit_code = run_backtest_from_args(args)

        finished_at = datetime.utcnow()
        runtime_sec = (finished_at - started_at).total_seconds()

        status = "success" if exit_code == 0 else "failed"
        error = None if exit_code == 0 else f"Backtest exited with code {exit_code}"

        # Write manifest (logging is done by main process, not worker)
        write_run_manifest(
            run_id=run_cfg.id,
            run_cfg=run_cfg,
            run_output_dir=run_output_dir,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            runtime_sec=runtime_sec,
            exit_code=exit_code,
            seed=seed,
            error=error,
        )

        return (status, runtime_sec, exit_code, error)

    except Exception as exc:
        finished_at = datetime.utcnow()
        runtime_sec = (finished_at - started_at).total_seconds()
        error = str(exc)
        # Error logging is done by main process, not worker

        # Write manifest with error
        write_run_manifest(
            run_id=run_cfg.id,
            run_cfg=run_cfg,
            run_output_dir=run_output_dir,
            status="failed",
            started_at=started_at,
            finished_at=finished_at,
            runtime_sec=runtime_sec,
            exit_code=1,
            seed=seed,
            error=error,
        )

        return ("failed", runtime_sec, 1, error)


def run_batch_serial(
    batch_cfg: BatchConfig,
    batch_output_root: Path,
    dry_run: bool = False,
    resume: bool = False,
    rerun_failed: bool = False,
) -> int:
    """Execute all runs in the batch serially.

    Args:
        batch_cfg: Batch configuration
        batch_output_root: Base output directory for batch (output/batch/<batch_name>/)
        dry_run: If True, skip execution
        resume: If True, skip runs that already succeeded
        rerun_failed: If True, rerun failed runs even with resume

    Returns:
        Exit code (0 if all successful, 1 otherwise)
    """
    all_success = True
    results = []  # List of (run_cfg, status, runtime_sec, error) tuples

    for run_cfg in batch_cfg.runs:
        # Log START
        logger.info("START  %s", run_cfg.id)
        
        status, runtime_sec, exit_code, error = run_single_backtest(
            run_cfg=run_cfg,
            batch_output_root=batch_output_root,
            seed=batch_cfg.seed,
            dry_run=dry_run,
            resume=resume,
            rerun_failed=rerun_failed,
        )

        results.append((run_cfg, status, runtime_sec, error))
        
        if status != "success":
            all_success = False

        # Log END/SKIP
        if status == "skipped":
            logger.info("SKIP   %s (%.2f sec)", run_cfg.id, runtime_sec)
            if error:
                logger.warning("         Reason: %s", error)
        elif status == "success":
            logger.info("END    %s success (%.2f sec)", run_cfg.id, runtime_sec)
        else:  # failed
            logger.info("END    %s failed (%.2f sec)", run_cfg.id, runtime_sec)
            if error:
                logger.warning("         Error: %s", error)

    # Summary
    success_count = sum(1 for _, status, _, _ in results if status == "success")
    failed_count = sum(1 for _, status, _, _ in results if status == "failed")
    skipped_count = sum(1 for _, status, _, _ in results if status == "skipped")
    
    logger.info("")
    logger.info("Summary: %d success, %d failed, %d skipped", success_count, failed_count, skipped_count)

    return 0 if all_success else 1


def _run_single_backtest_worker(args_tuple: tuple[RunConfig, Path, int, bool, bool, bool]) -> tuple[str, str, float, int, str | None]:
    """Worker function for parallel execution (must be top-level for pickling).

    Args:
        args_tuple: Tuple of (run_cfg, batch_output_root, seed, dry_run, resume, rerun_failed)

    Returns:
        Tuple of (run_id, status, runtime_sec, exit_code, error_message)
    """
    run_cfg, batch_output_root, seed, dry_run, resume, rerun_failed = args_tuple
    status, runtime_sec, exit_code, error = run_single_backtest(
        run_cfg=run_cfg,
        batch_output_root=batch_output_root,
        seed=seed,
        dry_run=dry_run,
        resume=resume,
        rerun_failed=rerun_failed,
    )
    return (run_cfg.id, status, runtime_sec, exit_code, error)


def run_batch_parallel(
    batch_cfg: BatchConfig,
    batch_output_root: Path,
    max_workers: int,
    dry_run: bool = False,
    resume: bool = False,
    rerun_failed: bool = False,
) -> int:
    """Execute all runs in the batch in parallel.

    Args:
        batch_cfg: Batch configuration
        batch_output_root: Base output directory for batch (output/batch/<batch_name>/)
        max_workers: Maximum number of parallel workers
        dry_run: If True, skip execution
        resume: If True, skip runs that already succeeded
        rerun_failed: If True, rerun failed runs even with resume

    Returns:
        Exit code (0 if all successful, 1 otherwise)
    """
    # Prepare work items (stable ordering)
    work_items = [
        (run_cfg, batch_output_root, batch_cfg.seed, dry_run, resume, rerun_failed)
        for run_cfg in batch_cfg.runs
    ]

    # Log START for all runs (before execution starts)
    for run_cfg in batch_cfg.runs:
        logger.info("START  %s", run_cfg.id)

    all_success = True
    completed_results = {}  # run_id -> (status, runtime_sec, exit_code, error)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_run_id = {
            executor.submit(_run_single_backtest_worker, item): item[0].id
            for item in work_items
        }

        # Process completed tasks as they finish (real-time progress)
        for future in as_completed(future_to_run_id):
            run_id = future_to_run_id[future]
            try:
                result_run_id, status, runtime_sec, exit_code, error = future.result()
                completed_results[result_run_id] = (status, runtime_sec, exit_code, error)
                if status != "success":
                    all_success = False
                
                # Log END/SKIP immediately when task completes
                if status == "skipped":
                    logger.info("SKIP   %s (%.2f sec)", result_run_id, runtime_sec)
                    if error:
                        logger.warning("         Reason: %s", error)
                elif status == "success":
                    logger.info("END    %s success (%.2f sec)", result_run_id, runtime_sec)
                else:  # failed
                    logger.info("END    %s failed (%.2f sec)", result_run_id, runtime_sec)
                    if error:
                        logger.warning("         Error: %s", error)
            except Exception as exc:
                logger.error("Run %s raised an exception: %s", run_id, exc, exc_info=True)
                completed_results[run_id] = ("failed", 0.0, 1, str(exc))
                logger.info("END    %s failed (exception)", run_id)
                all_success = False

    # Summary (all runs completed)
    success_count = sum(1 for _, (status, _, _, _) in completed_results.items() if status == "success")
    failed_count = sum(1 for _, (status, _, _, _) in completed_results.items() if status == "failed")
    skipped_count = sum(1 for _, (status, _, _, _) in completed_results.items() if status == "skipped")
    
    logger.info("")
    logger.info("Summary: %d success, %d failed, %d skipped", success_count, failed_count, skipped_count)

    return 0 if all_success else 1


def collect_backtest_metrics(run_output_dir: Path) -> dict[str, Any]:
    """Collect backtest metrics from run output directory.
    
    This function attempts to extract metrics from:
    - metrics.json (preferred, structured format)
    - reports/metrics.json (if metrics.json not found in root)
    - performance_report_*.md (fallback, parsed with regex)
    
    Args:
        run_output_dir: Run output directory
        
    Returns:
        Dictionary with metrics (keys may be missing if files don't exist):
        - final_pf: Final performance factor
        - sharpe: Sharpe ratio (or sharpe_ratio)
        - trades: Number of trades (or total_trades)
        - max_drawdown_pct: Maximum drawdown (percent)
        - total_return: Total return
        - cagr: CAGR (if available)
    """
    metrics: dict[str, Any] = {}
    
    # Try to find metrics.json (preferred format)
    metrics_json_paths = [
        run_output_dir / "metrics.json",  # Root directory
        run_output_dir / "reports" / "metrics.json",  # Reports subdirectory
    ]
    
    for metrics_json_path in metrics_json_paths:
        if metrics_json_path.exists():
            try:
                with metrics_json_path.open("r", encoding="utf-8") as f:
                    metrics_dict = json.load(f)
                
                # Extract metrics (handle both key names for compatibility)
                metrics["final_pf"] = metrics_dict.get("final_pf")
                metrics["sharpe"] = metrics_dict.get("sharpe") or metrics_dict.get("sharpe_ratio")
                metrics["trades"] = metrics_dict.get("trades") or metrics_dict.get("total_trades")
                metrics["max_drawdown_pct"] = metrics_dict.get("max_drawdown_pct")
                metrics["total_return"] = metrics_dict.get("total_return")
                metrics["cagr"] = metrics_dict.get("cagr")
                
                # Also include other available metrics
                for key in ["volatility", "sortino_ratio", "calmar_ratio", "hit_rate", "profit_factor"]:
                    if key in metrics_dict:
                        metrics[key] = metrics_dict[key]
                
                return metrics  # Return immediately if JSON found
            except (IOError, json.JSONDecodeError, KeyError) as exc:
                logger.debug("Failed to read metrics.json %s: %s", metrics_json_path, exc)
                continue
    
    # Fallback: Try to parse performance report Markdown (legacy support)
    report_paths_to_check = []
    # Check root directory
    report_paths_to_check.extend(run_output_dir.glob("performance_report_*.md"))
    # Check reports subdirectory
    reports_dir = run_output_dir / "reports"
    if reports_dir.exists():
        report_paths_to_check.extend(reports_dir.glob("performance_report_*.md"))
    
    for report_file in report_paths_to_check:
        if not report_file.exists():
            continue
        
        try:
            with report_file.open("r", encoding="utf-8") as f:
                content = f.read()
            
            # Extract metrics using regex (similar to sprint9_dashboard.ps1)
            # Pattern: "Final PF: 1.234" or "PF: 1.234"
            pf_match = re.search(r'(?:Final\s+)?PF[:\s]+([0-9\.\-NaN]+)', content, re.IGNORECASE)
            if pf_match:
                try:
                    metrics["final_pf"] = float(pf_match.group(1))
                except (ValueError, AttributeError):
                    pass
            
            # Pattern: "Sharpe: 1.234" or "Sharpe Ratio: 1.234"
            sharpe_match = re.search(r'Sharpe(?:\s+Ratio)?[:\s]+([0-9\.\-NaN]+)', content, re.IGNORECASE)
            if sharpe_match:
                try:
                    metrics["sharpe"] = float(sharpe_match.group(1))
                except (ValueError, AttributeError):
                    pass
            
            # Pattern: "Trades: 123"
            trades_match = re.search(r'Trades[:\s]+([0-9]+)', content, re.IGNORECASE)
            if trades_match:
                try:
                    metrics["trades"] = int(trades_match.group(1))
                except (ValueError, AttributeError):
                    pass
            
            # Pattern: "Max Drawdown: -12.34%" or "Max DD: -12.34%"
            dd_match = re.search(r'Max(?:imum)?\s+(?:Drawdown|DD)[:\s]+([0-9\.\-]+)%?', content, re.IGNORECASE)
            if dd_match:
                try:
                    metrics["max_drawdown_pct"] = float(dd_match.group(1))
                except (ValueError, AttributeError):
                    pass
            
            # Pattern: "Total Return: 12.34%" or "Return: 12.34%"
            return_match = re.search(r'(?:Total\s+)?Return[:\s]+([0-9\.\-]+)%?', content, re.IGNORECASE)
            if return_match:
                try:
                    metrics["total_return"] = float(return_match.group(1))
                except (ValueError, AttributeError):
                    pass
            
            # Pattern: "CAGR: 12.34%"
            cagr_match = re.search(r'CAGR[:\s]+([0-9\.\-NaN]+)%?', content, re.IGNORECASE)
            if cagr_match:
                try:
                    metrics["cagr"] = float(cagr_match.group(1))
                except (ValueError, AttributeError):
                    pass
            
            break  # Use first matching report file
        except (IOError, UnicodeDecodeError) as exc:
            logger.debug("Failed to read performance report %s: %s", report_file, exc)
            continue
    
    return metrics


def write_batch_summary(
    batch_cfg: BatchConfig,
    batch_output_root: Path,
) -> None:
    """Write batch summary CSV and JSON after all runs complete.
    
    This function collects data from all run manifests and writes:
    - output/batch/<batch_name>/summary.csv (CSV format for easy analysis)
    - output/batch/<batch_name>/summary.json (JSON format for programmatic access)
    
    Args:
        batch_cfg: Batch configuration
        batch_output_root: Base output directory for batch (output/batch/<batch_name>/)
    """
    summary_rows = []
    
    # Collect data from all run manifests
    for run_cfg in batch_cfg.runs:
        run_output_dir = batch_output_root / run_cfg.id
        manifest = load_existing_manifest(run_output_dir)
        
        if not manifest:
            # Run without manifest (e.g., skipped or failed early)
            summary_rows.append({
                "run_id": run_cfg.id,
                "strategy": run_cfg.strategy,
                "freq": run_cfg.freq,
                "status": "unknown",
                "runtime_sec": None,
                "exit_code": None,
                "manifest_path": None,
                "timings_path": None,
                "report_path": None,
                "final_pf": None,
                "sharpe": None,
                "trades": None,
                "max_drawdown_pct": None,
                "total_return": None,
                "cagr": None,
            })
            continue
        
        # Extract paths (relative to batch_output_root)
        manifest_path = run_output_dir / "run_manifest.json"
        manifest_path_rel = str(manifest_path.relative_to(batch_output_root)) if manifest_path.exists() else None
        
        timings_path_rel = manifest.get("timings_path")
        if timings_path_rel:
            timings_path = run_output_dir / timings_path_rel
            timings_path_rel = str(timings_path.relative_to(batch_output_root)) if timings_path.exists() else None
        
        # Find report path (performance_report_*.md)
        report_path = None
        for report_file in run_output_dir.glob("performance_report_*.md"):
            report_path = str(report_file.relative_to(batch_output_root))
            break
        
        # Collect metrics
        metrics = collect_backtest_metrics(run_output_dir)
        
        summary_rows.append({
            "run_id": run_cfg.id,
            "strategy": run_cfg.strategy,
            "freq": run_cfg.freq,
            "status": manifest.get("status", "unknown"),
            "runtime_sec": manifest.get("runtime_sec"),
            "exit_code": manifest.get("exit_code"),
            "manifest_path": manifest_path_rel,
            "timings_path": timings_path_rel,
            "report_path": report_path,
            "final_pf": metrics.get("final_pf"),
            "sharpe": metrics.get("sharpe"),
            "trades": metrics.get("trades"),
            "max_drawdown_pct": metrics.get("max_drawdown_pct"),
            "total_return": metrics.get("total_return"),
            "cagr": metrics.get("cagr"),
        })
    
    # Write CSV
    df = pd.DataFrame(summary_rows)
    csv_path = batch_output_root / "summary.csv"
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info("Batch summary CSV written to %s", csv_path)
    except (IOError, OSError) as exc:
        logger.error("Failed to write batch summary CSV to %s: %s", csv_path, exc)
        raise RuntimeError(f"Failed to write batch summary CSV: {csv_path}") from exc
    
    # Write JSON
    summary_json = {
        "batch_name": batch_cfg.batch_name,
        "seed": batch_cfg.seed,
        "total_runs": len(batch_cfg.runs),
        "runs": summary_rows,
    }
    
    json_path = batch_output_root / "summary.json"
    try:
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(summary_json, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Batch summary JSON written to %s", json_path)
    except (IOError, OSError) as exc:
        logger.error("Failed to write batch summary JSON to %s: %s", json_path, exc)
        raise RuntimeError(f"Failed to write batch summary JSON: {json_path}") from exc
    except (TypeError, ValueError) as exc:
        logger.error("Failed to serialize batch summary JSON: %s", exc)
        raise ValueError("Batch summary contains non-serializable data") from exc


def run_batch(
    batch_cfg: BatchConfig,
    max_workers: int = 1,
    dry_run: bool = False,
    resume: bool = False,
    rerun_failed: bool = False,
) -> int:
    """Execute all runs in the batch (serial or parallel).

    Args:
        batch_cfg: Batch configuration
        max_workers: Maximum number of parallel workers (1 = serial)
        dry_run: If True, skip execution
        resume: If True, skip runs that already succeeded
        rerun_failed: If True, rerun failed runs even with resume

    Returns:
        Exit code (0 if all successful, 1 otherwise)
    """
    # Output structure: output/batch/<batch_name>/<run_id>/
    batch_output_root = batch_cfg.output_root / batch_cfg.batch_name
    batch_output_root.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Batch Runner MVP")
    logger.info("=" * 60)
    logger.info("Batch name: %s", batch_cfg.batch_name)
    logger.info("Output root: %s", batch_output_root)
    logger.info("Seed: %d", batch_cfg.seed)
    logger.info("Runs: %d", len(batch_cfg.runs))
    logger.info("Max workers: %d", max_workers)
    if resume:
        logger.info("Resume mode: skipping successful runs")
        if rerun_failed:
            logger.info("Rerun failed: rerunning failed runs")
    if dry_run:
        logger.info("Dry-run mode: commands will not be executed")
    logger.info("")

    exit_code = 0
    if max_workers == 1:
        exit_code = run_batch_serial(batch_cfg, batch_output_root, dry_run=dry_run, resume=resume, rerun_failed=rerun_failed)
    else:
        exit_code = run_batch_parallel(batch_cfg, batch_output_root, max_workers, dry_run=dry_run, resume=resume, rerun_failed=rerun_failed)
    
    # Write batch summary after all runs complete
    if not dry_run:
        try:
            write_batch_summary(batch_cfg, batch_output_root)
        except Exception as exc:
            logger.error("Failed to write batch summary: %s", exc, exc_info=True)
            # Don't fail the entire batch if summary writing fails
    
    return exit_code


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch Runner MVP (P4) - Reproduzierbare Batch-Backtests mit Manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run (Plan anzeigen)
  python scripts/batch_runner.py --config-file configs/batch_example.yaml --dry-run

  # Ausführung (serial)
  python scripts/batch_runner.py --config-file configs/batch_example.yaml

  # Ausführung (parallel mit 4 Workern)
  python scripts/batch_runner.py --config-file configs/batch_example.yaml --max-workers 4

  # Resume (skip erfolgreiche Runs)
  python scripts/batch_runner.py --config-file configs/batch_example.yaml --resume

  # Resume + Rerun Failed
  python scripts/batch_runner.py --config-file configs/batch_example.yaml --resume --rerun-failed
        """,
    )

    parser.add_argument(
        "--config-file",
        type=Path,
        required=True,
        help="Path to YAML config file",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override output_root from config",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of parallel workers (1 = serial execution, default: 1)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs that already succeeded (resume from previous run)",
    )

    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="Rerun failed runs even with --resume (default: skip failed runs)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without executing backtests",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (can be used multiple times)",
    )

    return parser.parse_args()


def _setup_logging(verbosity: int) -> None:
    """Setup basic logging configuration."""
    level = logging.INFO if verbosity <= 0 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    """Main entry point for batch runner."""
    args = parse_args()
    _setup_logging(args.verbose)

    try:
        batch_cfg = load_batch_config(args.config_file)
    except Exception as exc:
        logger.error("Failed to load batch config: %s", exc, exc_info=True)
        return 1

    if args.output_root is not None:
        batch_cfg.output_root = args.output_root.resolve()

    try:
        max_workers = args.max_workers
        if max_workers < 1:
            logger.error("max_workers must be >= 1")
            return 1
        return run_batch(
            batch_cfg,
            max_workers=max_workers,
            dry_run=args.dry_run,
            resume=args.resume,
            rerun_failed=args.rerun_failed,
        )
    except Exception as exc:
        logger.error("Batch execution failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

