#!/usr/bin/env python
"""
Profile common assembled-trading-ai jobs (backtests, factor+ML runs, playbooks).

Usage examples:

    python scripts/profile_jobs.py --job BASIC_BACKTEST --with-cprofile
    python scripts/profile_jobs.py --job FACTOR_ML_JOB
    python scripts/profile_jobs.py --job PLAYBOOK_JOB --with-cprofile --top-n 50

This script is intentionally self-contained and only orchestrates existing
CLI entrypoints / Python functions from the codebase. It does NOT change
any business logic – it only measures runtime and, optionally, produces
cProfile reports.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import pathlib
import subprocess
import sys
import time

try:
    import cProfile
    import pstats
except ImportError:
    cProfile = None  # type: ignore[assignment,unused-ignore]
    pstats = None  # type: ignore[assignment,unused-ignore]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: pathlib.Path) -> pathlib.Path:
    """Ensure directory exists, create if needed."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_timestamp() -> str:
    """Get UTC timestamp string for file naming."""
    return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _setup_logging() -> None:
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Job wrappers
# ---------------------------------------------------------------------------


def run_basic_backtest_job() -> None:
    """
    Run a representative backtest job.

    Uses CLI subprocess to run a medium-size backtest:
    - Multi-Factor strategy
    - Macro World ETFs universe (or similar medium-size universe)
    - Realistic time range (2015-2020)
    """
    root = pathlib.Path(__file__).parent.parent
    cli_script = root / "scripts" / "cli.py"

    logger.info("Running BASIC_BACKTEST job via CLI...")

    cmd = [
        sys.executable,
        str(cli_script),
        "run_backtest",
        "--strategy",
        "multifactor_long_short",
        "--bundle-path",
        "config/factor_bundles/macro_world_etfs_core_bundle.yaml",
        "--symbols-file",
        "config/macro_world_etfs_tickers.txt",
        "--freq",
        "1d",
        "--data-source",
        "local",
        "--start-date",
        "2015-01-01",
        "--end-date",
        "2020-12-31",
        "--rebalance-freq",
        "M",
        "--max-gross-exposure",
        "1.0",
        "--start-capital",
        "100000",
        "--generate-report",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=False,  # Let output go to stdout/stderr
        check=False,
    )

    if result.returncode != 0:
        logger.warning(
            "Backtest job exited with code %d. This may be expected if data is missing.",
            result.returncode,
        )
    else:
        logger.info("Backtest job completed successfully")


def run_factor_ml_job() -> None:
    """
    Run a factor-panel export + ML validation job.

    Chains:
    1. export_factor_panel_for_ml.py (factor panel export)
    2. scripts/cli.py ml_validate_factors (ML validation)

    Note: This is a placeholder that needs to be wired to actual workflows.
    Adjust paths and parameters based on your environment.
    """
    root = pathlib.Path(__file__).parent.parent

    logger.info("Running FACTOR_ML_JOB...")

    # Step 1: Export factor panel
    logger.info("Step 1: Exporting factor panel...")
    export_cmd = [
        sys.executable,
        str(root / "research" / "factors" / "export_factor_panel_for_ml.py"),
        "--freq",
        "1d",
        "--symbols-file",
        "config/macro_world_etfs_tickers.txt",
        "--data-source",
        "local",
        "--factor-set",
        "core+vol_liquidity",
        "--horizon-days",
        "20",
        "--start-date",
        "2015-01-01",
        "--end-date",
        "2020-12-31",
    ]

    result1 = subprocess.run(
        export_cmd,
        cwd=str(root),
        capture_output=False,
        check=False,
    )

    if result1.returncode != 0:
        logger.warning(
            "Factor panel export exited with code %d. This may be expected if data is missing.",
            result1.returncode,
        )
        return

    logger.info("Factor panel export completed")

    # Step 2: ML validation (placeholder - would need actual panel path)
    logger.info("Step 2: ML validation (skipped - would need panel path)")
    # Example command (commented out until we can resolve panel path):
    # ml_cmd = [
    #     sys.executable,
    #     str(cli_script),
    #     "ml_validate_factors",
    #     "--factor-panel-file", "output/factor_panels/...",
    #     "--label-col", "fwd_return_20d",
    #     "--model-type", "ridge",
    #     "--n-splits", "5",
    # ]


def run_playbook_job() -> None:
    """
    Run the AI/Tech research playbook (end-to-end workflow).

    Calls the main() function of the research playbook module.
    """
    logger.info("Running PLAYBOOK_JOB...")

    try:
        from research.playbooks.ai_tech_multifactor_mlalpha_regime_playbook import (
            main as playbook_main,
        )

        playbook_main()
        logger.info("Playbook job completed successfully")
    except ImportError as e:
        logger.warning("Could not import playbook module: %s", e)
        logger.warning(
            "PLAYBOOK_JOB is a placeholder – please ensure playbook is available."
        )
    except Exception as e:
        logger.warning("Playbook job failed: %s", e)
        # Don't raise - we want profiling to continue even if job fails


def run_operations_health_check_job() -> None:
    """
    Run operations health check job.

    Uses CLI subprocess to run health checks:
    - Checks latest backtest directory
    - Validates existence of key files
    - Checks plausibility of performance metrics
    - Outputs health summary (text format)
    """
    root = pathlib.Path(__file__).parent.parent
    cli_script = root / "scripts" / "cli.py"

    logger.info("Running OPERATIONS_HEALTH_CHECK job via CLI...")

    cmd = [
        sys.executable,
        str(cli_script),
        "check_health",
        "--backtests-root",
        "output/backtests/",
        "--format",
        "text",
    ]

    result = subprocess.run(
        cmd, cwd=str(root), capture_output=True, text=True, check=False
    )

    if result.returncode != 0:
        logger.warning(f"Health check completed with exit code {result.returncode}")
        logger.warning(f"stdout: {result.stdout}")
        logger.warning(f"stderr: {result.stderr}")
    else:
        logger.info("Health check completed successfully")
        if result.stdout:
            logger.info(f"Output:\n{result.stdout}")


def run_paper_track_health_check_job() -> None:
    """
    Run paper track health check job.

    Uses CLI subprocess to run health checks with paper track support:
    - Checks latest backtest directory
    - Checks paper track strategies (if available)
    - Validates existence of key files
    - Checks plausibility of performance metrics
    - Outputs health summary (text format)
    """
    root = pathlib.Path(__file__).parent.parent
    cli_script = root / "scripts" / "cli.py"

    logger.info("Running PAPER_TRACK_HEALTH_CHECK job via CLI...")

    cmd = [
        sys.executable,
        str(cli_script),
        "check_health",
        "--backtests-root",
        "output/backtests/",
        "--paper-track-root",
        "output/paper_track/",
        "--paper-track-days",
        "3",
        "--skip-paper-track-if-missing",
        "--format",
        "text",
    ]

    result = subprocess.run(
        cmd, cwd=str(root), capture_output=True, text=True, check=False
    )

    if result.returncode != 0:
        logger.warning(
            f"Paper track health check completed with exit code {result.returncode}"
        )
        logger.warning(f"stdout: {result.stdout}")
        logger.warning(f"stderr: {result.stderr}")
    else:
        logger.info("Paper track health check completed successfully")
        if result.stdout:
            logger.info(f"Output:\n{result.stdout}")


def run_batch_backtest_job() -> None:
    """
    Run a representative batch backtest job via CLI.

    Uses CLI subprocess to run a batch of backtests:
    - Configuration is read from a YAML/JSON batch config file
    - Dry-run mode by default to avoid heavy workloads in profiling
    """
    root = pathlib.Path(__file__).parent.parent
    cli_script = root / "scripts" / "cli.py"

    # Example config path (can be adjusted based on repo conventions)
    batch_config = "configs/batch_backtests/ai_tech_core_vs_mlalpha.yaml"

    logger.info("Running BATCH_BACKTEST job via CLI...")

    cmd = [
        sys.executable,
        str(cli_script),
        "batch_backtest",
        "--config-file",
        batch_config,
        "--dry-run",
    ]

    result = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=False,
        check=False,
    )

    if result.returncode != 0:
        logger.warning(
            "Batch backtest job exited with code %d. This may be expected if config or data is missing.",
            result.returncode,
        )
    else:
        logger.info("Batch backtest job completed successfully")


JOB_MAP = {
    "BASIC_BACKTEST": run_basic_backtest_job,
    "FACTOR_ML_JOB": run_factor_ml_job,
    "PLAYBOOK_JOB": run_playbook_job,
    "BATCH_BACKTEST": run_batch_backtest_job,
    "OPERATIONS_HEALTH_CHECK": run_operations_health_check_job,
    "PAPER_TRACK_HEALTH_CHECK": run_paper_track_health_check_job,
}


# ---------------------------------------------------------------------------
# Profiling logic
# ---------------------------------------------------------------------------


def profile_job(job_name: str, with_cprofile: bool, top_n: int) -> None:
    """
    Profile a job by name.

    Args:
        job_name: Name of the job to profile (must be in JOB_MAP)
        with_cprofile: Whether to enable cProfile profiling
        top_n: Number of top functions to include in stats output
    """
    if job_name not in JOB_MAP:
        raise ValueError(
            f"Unknown job_name={job_name!r}. Valid: {sorted(JOB_MAP.keys())}"
        )

    if with_cprofile and (cProfile is None or pstats is None):
        raise RuntimeError("cProfile/pstats not available. Cannot enable profiling.")

    root = pathlib.Path(".").resolve()
    perf_logs_dir = _ensure_dir(root / "output" / "perf_logs")
    perf_profiles_dir = _ensure_dir(root / "output" / "perf_profiles")

    timestamp = _get_timestamp()
    job_func = JOB_MAP[job_name]

    logger.info("Starting profiled job %s at %s", job_name, timestamp)

    start = time.perf_counter()
    if with_cprofile:
        prof = cProfile.Profile()
        prof.enable()
        try:
            job_func()
        finally:
            prof.disable()
    else:
        job_func()
        prof = None
    total_sec = time.perf_counter() - start

    logger.info("Finished job %s in %.3f seconds", job_name, total_sec)

    # Write log file (simple text log with duration)
    log_path = perf_logs_dir / f"perf_{job_name}_{timestamp}.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"job_name={job_name}\n")
        f.write(f"timestamp={timestamp}\n")
        f.write(f"total_seconds={total_sec:.6f}\n")
        if prof is not None:
            f.write("cprofile_enabled=true\n")
            f.write(
                f"profile_file=output/perf_profiles/profile_{job_name}_{timestamp}.prof\n"
            )

    logger.info("Wrote perf log to %s", log_path)

    if prof is not None:
        prof_path = perf_profiles_dir / f"profile_{job_name}_{timestamp}.prof"
        stats_path = perf_profiles_dir / f"profile_{job_name}_{timestamp}_stats.txt"
        prof.dump_stats(str(prof_path))

        stats = pstats.Stats(prof).sort_stats(pstats.SortKey.CUMULATIVE)
        with stats_path.open("w", encoding="utf-8") as f:
            f.write(f"cProfile stats for {job_name} at {timestamp}\n")
            f.write(f"Total runtime: {total_sec:.6f} seconds\n\n")
            stats.stream = f
            stats.print_stats(top_n)

        logger.info("Wrote cProfile data to %s", prof_path)
        logger.info("Wrote cProfile stats to %s (top %d functions)", stats_path, top_n)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Profile common assembled-trading-ai jobs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/profile_jobs.py --job BASIC_BACKTEST --with-cprofile
  python scripts/profile_jobs.py --job FACTOR_ML_JOB
  python scripts/profile_jobs.py --job PLAYBOOK_JOB --with-cprofile --top-n 50
        """,
    )
    parser.add_argument(
        "--job",
        required=True,
        choices=sorted(JOB_MAP.keys()),
        help="Which predefined job to profile.",
    )
    parser.add_argument(
        "--with-cprofile",
        action="store_true",
        help="Enable cProfile and write .prof + stats files.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top functions to include in cProfile stats output (default: 50).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    _setup_logging()
    args = parse_args(argv)
    try:
        profile_job(
            job_name=args.job, with_cprofile=args.with_cprofile, top_n=args.top_n
        )
        return 0
    except Exception as exc:
        logger.exception("Error while profiling job %s: %s", args.job, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
