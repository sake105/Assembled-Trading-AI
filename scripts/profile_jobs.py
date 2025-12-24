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
from dataclasses import dataclass
from typing import Callable

try:
    import cProfile
    import pstats
except ImportError:
    cProfile = None  # type: ignore[assignment,unused-ignore]
    pstats = None  # type: ignore[assignment,unused-ignore]

try:
    import pyinstrument
except ImportError:
    pyinstrument = None  # type: ignore[assignment,unused-ignore]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Job Registry
# ---------------------------------------------------------------------------


@dataclass
class ReferenceJobSpec:
    """Specification for a reference job used for timing/profiling.

    Attributes:
        name: Unique job name (e.g., "EOD_SMALL")
        description: Short description of what the job does
        job_func: Callable that runs the job (may accept optional args like warm_cache, use_factor_store)
        seed: Optional random seed for determinism (if applicable)
    """

    name: str
    description: str
    job_func: Callable[..., None]  # Accepts variable args for warm_cache, use_factor_store, etc.
    seed: int | None = None


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
# Reference Jobs (P1 Sprint)
# ---------------------------------------------------------------------------


def run_eod_small_job(warm_cache: bool = False, use_factor_store: bool = False) -> None:
    """
    Run EOD_SMALL reference job: 2 years, ~100 symbols, run_daily --freq 1d.

    This job simulates a typical daily order generation workflow with:
    - 2 years of historical data (2023-01-01 to 2024-12-31)
    - ~100 symbols (uses universe file or watchlist.txt if available)
    - Single date execution (last available date in range)
    - Deterministic via fixed seed (if applicable)

    Args:
        warm_cache: If True, run twice: first cold (build cache), then warm (load from cache).
                    If False, run once (cold build).
        use_factor_store: If True, pass --use-factor-store to run_daily.py
    """
    root = pathlib.Path(__file__).parent.parent
    run_daily_script = root / "scripts" / "run_daily.py"

    # Set seed for determinism (if numpy/pandas random state is used)
    import numpy as np
    np.random.seed(42)

    logger.info("Running EOD_SMALL reference job...")
    logger.info("  Spec: 2 years (2023-2024), ~100 symbols, run_daily --freq 1d")
    if warm_cache:
        logger.info("  Mode: warm_cache=True (will run cold build, then warm load)")

    # Use a fixed end date for determinism
    end_date = "2024-12-31"

    if warm_cache and use_factor_store:
        # Run twice: cold build, then warm load
        logger.info("Step 1: Cold build (computing and storing factors)...")
        cmd_cold = [
            sys.executable,
            str(run_daily_script),
            "--date",
            end_date,
            "--use-factor-store",
        ]
        
        result_cold = subprocess.run(
            cmd_cold,
            cwd=str(root),
            capture_output=False,
            check=False,
        )
        
        if result_cold.returncode != 0:
            logger.warning(
                "EOD_SMALL cold build exited with code %d. This may be expected if data is missing.",
                result_cold.returncode,
            )
        else:
            logger.info("EOD_SMALL cold build completed successfully")
        
        logger.info("Step 2: Warm load (loading factors from cache)...")
        cmd_warm = [
            sys.executable,
            str(run_daily_script),
            "--date",
            end_date,
            "--use-factor-store",
        ]
        
        result_warm = subprocess.run(
            cmd_warm,
            cwd=str(root),
            capture_output=False,
            check=False,
        )
        
        if result_warm.returncode != 0:
            logger.warning(
                "EOD_SMALL warm load exited with code %d.",
                result_warm.returncode,
            )
        else:
            logger.info("EOD_SMALL warm load completed successfully")
    else:
        # Single run (cold build or without factor store)
        cmd = [
            sys.executable,
            str(run_daily_script),
            "--date",
            end_date,
        ]
        
        if use_factor_store:
            cmd.append("--use-factor-store")
        
        result = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=False,
            check=False,
        )
        
        if result.returncode != 0:
            logger.warning(
                "EOD_SMALL job exited with code %d. This may be expected if data is missing.",
                result.returncode,
            )
        else:
            logger.info("EOD_SMALL job completed successfully")


def run_backtest_medium_job(use_factor_store: bool = False, enable_timings: bool = False, timings_out: pathlib.Path | None = None) -> None:
    """
    Run BACKTEST_MEDIUM reference job: 10 years, ~200 symbols, trend_baseline backtest.

    This job simulates a medium-size backtest with:
    - 10 years of historical data (2015-01-01 to 2024-12-31)
    - ~200 symbols (uses universe file if available)
    - trend_baseline strategy (EMA crossover)
    - Deterministic via fixed seed

    Args:
        use_factor_store: If True, pass use_factor_store=True to backtest engine
        enable_timings: If True, pass enable_timings=True to backtest runner (generates timings.json)
        timings_out: Optional path to write timings.json (default: output_dir/timings.json in backtest runner)

    Note: This function runs the backtest in-process (not via subprocess) to enable profiling.
    """
    # Set seed for determinism
    import numpy as np
    np.random.seed(42)

    logger.info("Running BACKTEST_MEDIUM reference job...")
    logger.info("  Spec: 10 years (2015-2024), ~200 symbols, trend_baseline strategy")
    if use_factor_store:
        logger.info("  Factor Store: enabled")
    if enable_timings:
        logger.info("  Timings: enabled (will generate timings.json)")

    # Import and call backtest function directly (in-process for profiling)
    try:
        from scripts.run_backtest_strategy import run_backtest_from_args

        # Create a minimal argparse.Namespace-like object with required arguments
        # Capture parameters in closure
        use_fs = use_factor_store
        enable_tim = enable_timings
        timings_out_str = str(timings_out) if timings_out else None
        
        class BacktestArgs:
            freq = "1d"
            strategy = "trend_baseline"
            start_date = "2015-01-01"
            end_date = "2024-12-31"
            start_capital = 100000.0
            generate_report = True
            # Optional/default arguments
            price_file = None
            universe = None
            symbols = None
            symbols_file = None
            data_source = None
            with_costs = True
            commission_bps = None
            spread_w = None
            impact_w = None
            out = None
            track_experiment = False
            experiment_name = None
            experiment_tags = None
            use_meta_model = False
            meta_model_path = None
            meta_min_confidence = 0.5
            meta_ensemble_mode = "filter"
            bundle_path = None  # For multifactor strategy
            # Factor store arguments
            use_factor_store = use_fs
            factor_store_root = None
            factor_group = "core_ta"
            # Timing arguments
            enable_timings = enable_tim
            timings_out = timings_out_str

        args = BacktestArgs()
        exit_code = run_backtest_from_args(args)
        
        if exit_code != 0:
            logger.warning(
                "BACKTEST_MEDIUM job exited with code %d. This may be expected if data is missing.",
                exit_code,
            )
        else:
            logger.info("BACKTEST_MEDIUM job completed successfully")
    except ImportError as e:
        logger.error(f"Could not import run_backtest_from_args: {e}")
        raise
    except Exception as e:
        logger.error(f"BACKTEST_MEDIUM job failed: {e}", exc_info=True)
        raise


def run_ml_job() -> None:
    """
    Run ML_JOB reference job: export_factor_panel_for_ml + lightweight ML validation.

    This job simulates an ML workflow with:
    - Factor panel export (research/factors/export_factor_panel_for_ml.py)
    - Lightweight ML validation (run_ml_factor_validation.py with minimal config)
    - Deterministic via fixed seed
    """
    root = pathlib.Path(__file__).parent.parent

    # Set seed for determinism
    import numpy as np
    np.random.seed(42)

    logger.info("Running ML_JOB reference job...")
    logger.info("  Spec: Factor panel export + ML validation (lightweight)")

    # Step 1: Export factor panel
    logger.info("Step 1: Exporting factor panel...")
    export_script = root / "research" / "factors" / "export_factor_panel_for_ml.py"

    export_cmd = [
        sys.executable,
        str(export_script),
        "--freq",
        "1d",
        "--symbols-file",
        "watchlist.txt",  # Use default universe
        "--factor-set",
        "core",
        "--horizon-days",
        "20",
        "--start-date",
        "2020-01-01",
        "--end-date",
        "2024-12-31",
        "--data-source",
        "local",
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
        # Continue anyway - ML validation might still work with existing panels
    else:
        logger.info("Factor panel export completed")

    # Step 2: Lightweight ML validation
    # For profiling, we use a minimal config (fewer splits, smaller model)
    logger.info("Step 2: Running lightweight ML validation...")
    ml_script = root / "scripts" / "run_ml_factor_validation.py"

    # Try to find the exported panel (this is a best-effort approach)
    # In practice, the panel path would come from the export step
    # For now, we'll skip ML validation if export failed or use a placeholder path
    if result1.returncode == 0:
        # Try common output paths
        possible_panel_paths = [
            root / "output" / "factor_analysis" / "factor_panel_1d.parquet",
            root / "output" / "factor_panels" / "factor_panel_1d.parquet",
        ]

        panel_path = None
        for path in possible_panel_paths:
            if path.exists():
                panel_path = path
                break

        if panel_path:
            ml_cmd = [
                sys.executable,
                str(ml_script),
                "--factor-panel-file",
                str(panel_path),
                "--label-col",
                "fwd_return_20d",
                "--model-type",
                "ridge",
                "--n-splits",
                "3",  # Minimal splits for profiling
                "--output-dir",
                str(root / "output" / "ml_validation"),
            ]

            result2 = subprocess.run(
                ml_cmd,
                cwd=str(root),
                capture_output=False,
                check=False,
            )

            if result2.returncode != 0:
                logger.warning(
                    "ML validation exited with code %d. This may be expected if data/model issues occur.",
                    result2.returncode,
                )
            else:
                logger.info("ML validation completed")
        else:
            logger.warning(
                "Could not find exported factor panel - skipping ML validation step"
            )
    else:
        logger.warning(
            "Factor panel export failed - skipping ML validation step"
        )

    logger.info("ML_JOB completed (some steps may have been skipped if data is missing)")


# ---------------------------------------------------------------------------
# Job wrappers (legacy, kept for backward compatibility)
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


# Reference Jobs Registry (P1 Sprint)
REFERENCE_JOBS = [
    ReferenceJobSpec(
        name="EOD_SMALL",
        description="EOD order generation: 2 years (2023-2024), ~100 symbols, run_daily --freq 1d",
        job_func=run_eod_small_job,
        seed=42,
    ),
    ReferenceJobSpec(
        name="BACKTEST_MEDIUM",
        description="Backtest: 10 years (2015-2024), ~200 symbols, trend_baseline strategy",
        job_func=run_backtest_medium_job,
        seed=42,
    ),
    ReferenceJobSpec(
        name="ML_JOB",
        description="ML workflow: Factor panel export + lightweight ML validation",
        job_func=run_ml_job,
        seed=42,
    ),
]

# Legacy job map (backward compatibility)
JOB_MAP = {
    "BASIC_BACKTEST": run_basic_backtest_job,
    "FACTOR_ML_JOB": run_factor_ml_job,
    "PLAYBOOK_JOB": run_playbook_job,
    "BATCH_BACKTEST": run_batch_backtest_job,
    "OPERATIONS_HEALTH_CHECK": run_operations_health_check_job,
    "PAPER_TRACK_HEALTH_CHECK": run_paper_track_health_check_job,
    # Add reference jobs to legacy map for compatibility
    **{job.name: job.job_func for job in REFERENCE_JOBS},
}


# ---------------------------------------------------------------------------
# Profiling logic
# ---------------------------------------------------------------------------


def profile_job(
    job_name: str,
    profiler: str = "none",
    profile_out: pathlib.Path | None = None,
    top_n: int = 50,
    warm_cache: bool = False,
    use_factor_store: bool = False,
) -> None:
    """
    Profile a job by name using the specified profiler.

    Args:
        job_name: Name of the job to profile (must be in JOB_MAP)
        profiler: Profiler to use ("none", "cprofile", or "pyinstrument")
        profile_out: Optional output directory (default: output/profiles/<job>/<timestamp>/)
        top_n: Number of top functions to include in cProfile stats output
        warm_cache: If True, run EOD_SMALL twice (cold build, then warm load)
        use_factor_store: If True, pass use_factor_store to jobs that support it
    """
    if job_name not in JOB_MAP:
        raise ValueError(
            f"Unknown job_name={job_name!r}. Valid: {sorted(JOB_MAP.keys())}"
        )

    if profiler == "cprofile" and (cProfile is None or pstats is None):
        raise RuntimeError("cProfile/pstats not available. Cannot enable profiling.")
    elif profiler == "pyinstrument" and pyinstrument is None:
        raise RuntimeError(
            "pyinstrument not available (not installed). Cannot enable profiling."
        )

    root = pathlib.Path(".").resolve()
    timestamp = _get_timestamp()

    # Determine output directory
    if profile_out:
        profile_dir = pathlib.Path(profile_out)
    else:
        profile_dir = root / "output" / "profiles" / job_name / timestamp

    profile_dir = _ensure_dir(profile_dir)

    job_func = JOB_MAP[job_name]

    logger.info("Starting profiled job %s at %s (profiler=%s)", job_name, timestamp, profiler)
    if warm_cache:
        logger.info("  warm_cache=True: will run cold build, then warm load")
    if use_factor_store:
        logger.info("  use_factor_store=True: using factor store for caching")

    start = time.perf_counter()
    prof_data = None

    # Prepare job function arguments based on job name
    job_kwargs = {}
    if job_name == "EOD_SMALL":
        job_kwargs["warm_cache"] = warm_cache
        job_kwargs["use_factor_store"] = use_factor_store
    elif job_name == "BACKTEST_MEDIUM":
        job_kwargs["use_factor_store"] = use_factor_store

    if profiler == "cprofile":
        prof = cProfile.Profile()
        prof.enable()
        try:
            job_func(**job_kwargs)
        finally:
            prof.disable()
        prof_data = prof
    elif profiler == "pyinstrument":
        prof = pyinstrument.Profiler()
        prof.start()
        try:
            job_func(**job_kwargs)
        finally:
            prof.stop()
        prof_data = prof
    else:
        job_func(**job_kwargs)

    total_sec = time.perf_counter() - start

    logger.info("Finished job %s in %.3f seconds", job_name, total_sec)

    # Write profiler-specific outputs
    if profiler == "cprofile" and prof_data is not None:
        prof_path = profile_dir / f"profile_{job_name}.prof"
        stats_path = profile_dir / f"profile_{job_name}_stats.txt"
        prof_data.dump_stats(str(prof_path))

        stats = pstats.Stats(prof_data).sort_stats(pstats.SortKey.CUMULATIVE)
        with stats_path.open("w", encoding="utf-8") as f:
            f.write(f"cProfile stats for {job_name} at {timestamp}\n")
            f.write(f"Total runtime: {total_sec:.6f} seconds\n\n")
            stats.stream = f
            stats.print_stats(top_n)

        logger.info("Wrote cProfile data to %s", prof_path)
        logger.info("Wrote cProfile stats to %s (top %d functions)", stats_path, top_n)

    elif profiler == "pyinstrument" and prof_data is not None:
        # Write HTML report
        html_path = profile_dir / f"profile_{job_name}.html"
        html_output = prof_data.output_html()
        with html_path.open("w", encoding="utf-8") as f:
            f.write(html_output)

        # Write text report (if supported)
        try:
            text_path = profile_dir / f"profile_{job_name}.txt"
            text_output = prof_data.output_text()
            with text_path.open("w", encoding="utf-8") as f:
                f.write(text_output)
            logger.info("Wrote pyinstrument text report to %s", text_path)
        except Exception as e:
            logger.warning("Could not write pyinstrument text report: %s", e)

        logger.info("Wrote pyinstrument HTML report to %s", html_path)

    # Write summary log
    log_path = profile_dir / "profile_summary.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"job_name={job_name}\n")
        f.write(f"timestamp={timestamp}\n")
        f.write(f"profiler={profiler}\n")
        f.write(f"total_seconds={total_sec:.6f}\n")
        f.write(f"warm_cache={warm_cache}\n")
        f.write(f"use_factor_store={use_factor_store}\n")
        if warm_cache and use_factor_store:
            f.write("cache_warm=True\n")
        if profiler == "cprofile":
            f.write(f"profile_file=profile_{job_name}.prof\n")
            f.write(f"stats_file=profile_{job_name}_stats.txt\n")
        elif profiler == "pyinstrument":
            f.write(f"html_file=profile_{job_name}.html\n")
            f.write(f"text_file=profile_{job_name}.txt\n")

    logger.info("Wrote profile summary to %s", log_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def list_reference_jobs() -> None:
    """List all registered reference jobs."""
    print("\nReference Jobs (P1 Sprint):")
    print("=" * 80)
    for job in REFERENCE_JOBS:
        seed_info = f" (seed={job.seed})" if job.seed is not None else ""
        print(f"  {job.name}{seed_info}")
        print(f"    {job.description}")
        print()
    print("=" * 80)
    print(f"\nTotal: {len(REFERENCE_JOBS)} reference jobs")
    print("\nAll available jobs (including legacy):")
    all_job_names = sorted(JOB_MAP.keys())
    for name in all_job_names:
        print(f"  - {name}")


def run_job_without_profiling(
    job_name: str, warm_cache: bool = False, use_factor_store: bool = False
) -> None:
    """Run a job without profiling (dry execution for testing).

    Args:
        job_name: Name of the job to run (must be in JOB_MAP)
        warm_cache: If True, run EOD_SMALL twice (cold build, then warm load)
        use_factor_store: If True, pass use_factor_store to jobs that support it

    Raises:
        ValueError: If job_name is not found in JOB_MAP
    """
    if job_name not in JOB_MAP:
        raise ValueError(
            f"Unknown job_name={job_name!r}. Valid: {sorted(JOB_MAP.keys())}"
        )

    job_func = JOB_MAP[job_name]
    
    # Set seed if this is a reference job
    for ref_job in REFERENCE_JOBS:
        if ref_job.name == job_name and ref_job.seed is not None:
            import numpy as np
            np.random.seed(ref_job.seed)
            logger.info(f"Set random seed to {ref_job.seed} for deterministic execution")
            break

    # Prepare job function arguments based on job name
    job_kwargs = {}
    if job_name == "EOD_SMALL":
        job_kwargs["warm_cache"] = warm_cache
        job_kwargs["use_factor_store"] = use_factor_store
    elif job_name == "BACKTEST_MEDIUM":
        job_kwargs["use_factor_store"] = use_factor_store

    logger.info(f"Running job {job_name} (without profiling)...")
    job_func(**job_kwargs)
    logger.info(f"Job {job_name} completed")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Profile common assembled-trading-ai jobs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/profile_jobs.py --list
  python scripts/profile_jobs.py --job EOD_SMALL
  python scripts/profile_jobs.py --job BACKTEST_MEDIUM --profiler cprofile
  python scripts/profile_jobs.py --job BACKTEST_MEDIUM --profiler pyinstrument
  python scripts/profile_jobs.py --job ML_JOB --profiler cprofile --top-n 50
  python scripts/profile_jobs.py --job BACKTEST_MEDIUM --profiler cprofile --profile-out output/my_profiles
        """,
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available reference jobs and exit.",
    )
    parser.add_argument(
        "--job",
        required=False,
        choices=sorted(JOB_MAP.keys()),
        help="Which predefined job to run (use --with-cprofile to enable profiling).",
    )
    parser.add_argument(
        "--profiler",
        type=str,
        choices=["none", "cprofile", "pyinstrument"],
        default="none",
        help="Profiler to use: 'none' (no profiling), 'cprofile' (cProfile), or 'pyinstrument' (default: none)",
    )
    parser.add_argument(
        "--profile-out",
        type=str,
        default=None,
        help="Output directory for profile reports (default: output/profiles/<job>/<timestamp>/)",
    )
    parser.add_argument(
        "--with-cprofile",
        action="store_true",
        help="[DEPRECATED] Use --profiler cprofile instead. Enable cProfile and write .prof + stats files (requires --job).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top functions to include in cProfile stats output (default: 50).",
    )
    parser.add_argument(
        "--warm-cache",
        action="store_true",
        default=False,
        help="For EOD_SMALL: Run twice (cold build, then warm load from cache). Enables factor store automatically.",
    )
    parser.add_argument(
        "--use-factor-store",
        action="store_true",
        default=False,
        help="Use factor store for caching (supported by EOD_SMALL and BACKTEST_MEDIUM).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    _setup_logging()
    args = parse_args(argv)
    
    # Handle --list option
    if args.list:
        list_reference_jobs()
        return 0
    
    # Require --job if not listing
    if not args.job:
        parser = argparse.ArgumentParser(description="Profile common assembled-trading-ai jobs.")
        parser.error("Either --list or --job must be specified")
        return 1
    
    try:
        # Determine profiler (backward compatibility: --with-cprofile maps to --profiler cprofile)
        profiler = args.profiler
        if args.with_cprofile:
            logger.warning(
                "--with-cprofile is deprecated, use --profiler cprofile instead"
            )
            profiler = "cprofile"

        profile_out = pathlib.Path(args.profile_out) if args.profile_out else None
        
        # Determine use_factor_store: --warm-cache implies --use-factor-store for EOD_SMALL
        use_factor_store = args.use_factor_store or (args.warm_cache and args.job == "EOD_SMALL")

        if profiler == "none":
            # Run job without profiling
            run_job_without_profiling(
                args.job,
                warm_cache=args.warm_cache,
                use_factor_store=use_factor_store,
            )
        else:
            # Run job with profiling
            profile_job(
                job_name=args.job,
                profiler=profiler,
                profile_out=profile_out,
                top_n=args.top_n,
                warm_cache=args.warm_cache,
                use_factor_store=use_factor_store,
            )
        return 0
    except Exception as exc:
        logger.exception("Error while running job %s: %s", args.job, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
