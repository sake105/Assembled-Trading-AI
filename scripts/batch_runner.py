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
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

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


@dataclass
class RunConfig:
    """Configuration for a single backtest run."""

    id: str
    strategy: str
    freq: str
    start_date: str
    end_date: str
    universe: str | None = None
    start_capital: float = 100000.0
    use_factor_store: bool = False
    factor_store_root: str | None = None
    factor_group: str | None = None
    # Additional args for run_backtest_from_args
    extra_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchConfig:
    """Configuration for an entire batch of backtests."""

    batch_name: str
    output_root: Path
    seed: int = 42
    runs: list[RunConfig] = field(default_factory=list)


def load_batch_config(config_path: Path) -> BatchConfig:
    """Load and validate batch config from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Parsed BatchConfig instance

    Raises:
        ValueError: If config is invalid
        FileNotFoundError: If config file doesn't exist
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = _load_yaml(config_path)

    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping/object")

    batch_name = str(raw.get("batch_name") or "").strip()
    if not batch_name:
        raise ValueError("batch_name must be set in config")

    output_root_raw = raw.get("output_root") or "output/batch_backtests"
    output_root = (ROOT / output_root_raw).resolve()

    seed = int(raw.get("seed") or 42)

    runs_raw = raw.get("runs")
    if not isinstance(runs_raw, list) or not runs_raw:
        raise ValueError("runs must be a non-empty list in config")

    runs: list[RunConfig] = []
    for idx, run_raw in enumerate(runs_raw):
        if not isinstance(run_raw, dict):
            raise ValueError(f"runs[{idx}] must be an object")

        # Run-ID can be explicitly set, or will be auto-generated from params
        run_id = run_raw.get("id")
        if run_id:
            run_id = str(run_id).strip()
        # If not set, we'll generate it later from params + seed

        strategy = str(run_raw.get("strategy") or "").strip()
        if not strategy:
            raise ValueError(f"runs[{idx}]: strategy must be set")

        freq = str(run_raw.get("freq") or "").strip()
        if freq not in ["1d", "5min"]:
            raise ValueError(f"runs[{idx}]: freq must be '1d' or '5min'")

        start_date = str(run_raw.get("start_date") or "").strip()
        end_date = str(run_raw.get("end_date") or "").strip()
        if not start_date or not end_date:
            raise ValueError(f"runs[{idx}]: start_date and end_date must be set")

        universe = run_raw.get("universe")
        if universe:
            universe = str(universe).strip()

        start_capital = float(run_raw.get("start_capital") or 100000.0)
        use_factor_store = bool(run_raw.get("use_factor_store") or False)
        factor_store_root = run_raw.get("factor_store_root")
        if factor_store_root:
            factor_store_root = str(factor_store_root).strip()
        factor_group = run_raw.get("factor_group")
        if factor_group:
            factor_group = str(factor_group).strip()

        # Collect extra args (for future extension)
        extra_args = {k: v for k, v in run_raw.items() if k not in [
            "id", "strategy", "freq", "start_date", "end_date",
            "universe", "start_capital", "use_factor_store",
            "factor_store_root", "factor_group"
        ]}

        run_cfg = RunConfig(
            id=run_id or "",  # Will be set later if empty
            strategy=strategy,
            freq=freq,
            start_date=start_date,
            end_date=end_date,
            universe=universe,
            start_capital=start_capital,
            use_factor_store=use_factor_store,
            factor_store_root=factor_store_root,
            factor_group=factor_group,
            extra_args=extra_args,
        )
        runs.append(run_cfg)

    # Generate deterministic run IDs for runs without explicit ID
    for run_cfg in runs:
        if not run_cfg.id:
            run_cfg.id = _compute_run_id_hash(run_cfg, seed)

    return BatchConfig(
        batch_name=batch_name,
        output_root=output_root,
        seed=seed,
        runs=runs,
    )


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

    # Merge extra_args
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
        error: Error message (if any)
    """
    git_hash = _get_git_commit_hash()

    # Check if timings file exists
    timings_path = run_output_dir / "run_timings.json"
    timings_path_rel = "run_timings.json" if timings_path.exists() else None

    # Build params dict
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

    manifest = {
        "run_id": run_id,
        "status": status,
        "started_at": started_at.isoformat() + "Z",
        "finished_at": finished_at.isoformat() + "Z",
        "runtime_sec": runtime_sec,
        "params": params,
        "git_commit_hash": git_hash,
        "timings_path": timings_path_rel,
        "output_dir": str(run_output_dir.resolve()),
        "exit_code": exit_code,
    }

    if error:
        manifest["error"] = error

    manifest_path = run_output_dir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info("Run manifest written to %s", manifest_path)


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


def run_single_backtest(
    run_cfg: RunConfig,
    batch_output_root: Path,
    dry_run: bool = False,
    resume: bool = False,
    rerun_failed: bool = False,
) -> tuple[str, float, int, str | None]:
    """Execute a single backtest run.

    Args:
        run_cfg: Run configuration
        batch_output_root: Base output directory for batch (output/batch/<run_id>/)
        dry_run: If True, skip execution
        resume: If True, skip runs that already succeeded
        rerun_failed: If True, rerun failed runs even with resume

    Returns:
        Tuple of (status, runtime_sec, exit_code, error_message)
    """
    # Each run gets its own directory: output/batch/<run_id>/
    run_output_dir = batch_output_root / run_cfg.id
    # Create directory atomically to avoid race conditions
    try:
        run_output_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Directory already exists (from parallel execution), continue
        pass

    logger.info("=" * 60)
    logger.info("Run: %s", run_cfg.id)
    logger.info("  Strategy: %s", run_cfg.strategy)
    logger.info("  Freq: %s", run_cfg.freq)
    logger.info("  Date range: %s to %s", run_cfg.start_date, run_cfg.end_date)
    logger.info("  Output dir: %s", run_output_dir)

    # Check for existing manifest if resume is enabled
    if resume and not dry_run:
        existing_manifest = load_existing_manifest(run_output_dir)
        if existing_manifest:
            existing_status = existing_manifest.get("status")
            if existing_status == "success":
                logger.info("Run %s already succeeded (status: success) - skipping", run_cfg.id)
                # Return existing values from manifest
                existing_runtime = existing_manifest.get("runtime_sec", 0.0)
                existing_exit_code = existing_manifest.get("exit_code", 0)
                return ("success", existing_runtime, existing_exit_code, None)
            elif existing_status == "failed" and not rerun_failed:
                logger.info("Run %s previously failed (status: failed) - skipping (use --rerun-failed to rerun)", run_cfg.id)
                existing_runtime = existing_manifest.get("runtime_sec", 0.0)
                existing_exit_code = existing_manifest.get("exit_code", 1)
                existing_error = existing_manifest.get("error")
                return ("failed", existing_runtime, existing_exit_code, existing_error)
            elif existing_status == "failed" and rerun_failed:
                logger.info("Run %s previously failed - rerunning due to --rerun-failed", run_cfg.id)

    if dry_run:
        logger.info("Dry-run mode: skipping execution")
        return ("skipped", 0.0, 0, None)

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

        if exit_code == 0:
            logger.info("Run %s completed successfully in %.2f sec", run_cfg.id, runtime_sec)
        else:
            logger.warning("Run %s failed: %s", run_cfg.id, error)

        # Write manifest
        write_run_manifest(
            run_id=run_cfg.id,
            run_cfg=run_cfg,
            run_output_dir=run_output_dir,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            runtime_sec=runtime_sec,
            exit_code=exit_code,
            error=error,
        )

        return (status, runtime_sec, exit_code, error)

    except Exception as exc:
        finished_at = datetime.utcnow()
        runtime_sec = (finished_at - started_at).total_seconds()
        error = str(exc)
        logger.error("Run %s raised an exception: %s", run_cfg.id, exc, exc_info=True)

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
        batch_output_root: Base output directory for batch (output/batch/)
        dry_run: If True, skip execution
        resume: If True, skip runs that already succeeded
        rerun_failed: If True, rerun failed runs even with resume

    Returns:
        Exit code (0 if all successful, 1 otherwise)
    """
    all_success = True

    for run_cfg in batch_cfg.runs:
        status, runtime_sec, exit_code, error = run_single_backtest(
            run_cfg=run_cfg,
            batch_output_root=batch_output_root,
            dry_run=dry_run,
            resume=resume,
            rerun_failed=rerun_failed,
        )

        if status != "success":
            all_success = False

        logger.info("")

    return 0 if all_success else 1


def _run_single_backtest_worker(args_tuple: tuple[RunConfig, Path, bool, bool, bool]) -> tuple[str, str, float, int, str | None]:
    """Worker function for parallel execution (must be top-level for pickling).

    Args:
        args_tuple: Tuple of (run_cfg, batch_output_root, dry_run, resume, rerun_failed)

    Returns:
        Tuple of (run_id, status, runtime_sec, exit_code, error_message)
    """
    run_cfg, batch_output_root, dry_run, resume, rerun_failed = args_tuple
    status, runtime_sec, exit_code, error = run_single_backtest(
        run_cfg=run_cfg,
        batch_output_root=batch_output_root,
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
        batch_output_root: Base output directory for batch (output/batch/)
        max_workers: Maximum number of parallel workers
        dry_run: If True, skip execution
        resume: If True, skip runs that already succeeded
        rerun_failed: If True, rerun failed runs even with resume

    Returns:
        Exit code (0 if all successful, 1 otherwise)
    """
    # Prepare work items (stable ordering)
    work_items = [
        (run_cfg, batch_output_root, dry_run, resume, rerun_failed)
        for run_cfg in batch_cfg.runs
    ]

    all_success = True
    completed_results = {}  # run_id -> (status, runtime_sec, exit_code, error)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_run_id = {
            executor.submit(_run_single_backtest_worker, item): item[0].id
            for item in work_items
        }

        # Process completed tasks (maintain order for logging)
        for future in as_completed(future_to_run_id):
            run_id = future_to_run_id[future]
            try:
                result_run_id, status, runtime_sec, exit_code, error = future.result()
                completed_results[result_run_id] = (status, runtime_sec, exit_code, error)
                if status != "success":
                    all_success = False
            except Exception as exc:
                logger.error("Run %s raised an exception: %s", run_id, exc, exc_info=True)
                completed_results[run_id] = ("failed", 0.0, 1, str(exc))
                all_success = False

    # Log results in original order
    for run_cfg in batch_cfg.runs:
        if run_cfg.id in completed_results:
            status, runtime_sec, exit_code, error = completed_results[run_cfg.id]
            logger.info("Run %s: %s (%.2f sec)", run_cfg.id, status, runtime_sec)
            if error:
                logger.warning("  Error: %s", error)

    return 0 if all_success else 1


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
    # Output structure: output/batch/<run_id>/
    batch_output_root = batch_cfg.output_root / "batch"
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

    if max_workers == 1:
        return run_batch_serial(batch_cfg, batch_output_root, dry_run=dry_run, resume=resume, rerun_failed=rerun_failed)
    else:
        return run_batch_parallel(batch_cfg, batch_output_root, max_workers, dry_run=dry_run, resume=resume, rerun_failed=rerun_failed)


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

