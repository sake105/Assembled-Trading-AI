# scripts/profile_job.py
"""Profile reference benchmark jobs (EOD_SMALL, BACKTEST_MEDIUM, ML_JOB).

This script runs standard benchmark jobs with profiling enabled and stores
profiling outputs for analysis.

Example usage:
    python scripts/profile_job.py --job EOD_SMALL --profiler cprofile
    python scripts/profile_job.py --job BACKTEST_MEDIUM --profiler pyinstrument
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Import core modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.config.settings import get_settings


def run_eod_small(output_dir: Path) -> int:
    """Run EOD_SMALL benchmark job.
    
    Args:
        output_dir: Output directory for profiling artifacts
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    from scripts.run_daily import run_daily_eod
    
    try:
        # Run with timings enabled
        safe_path = run_daily_eod(
            date_str=None,  # Use today
            universe_file=None,  # Use default
            price_file=None,
            output_dir=None,  # Use default
            total_capital=1.0,
            top_n=None,
            ma_fast=20,
            ma_slow=50,
            min_score=0.0,
            disable_pre_trade_checks=False,
            ignore_kill_switch=False,
            enable_timings=True,
            timings_out=output_dir / "timings.json",
            use_factor_store=False,
            factor_store_root=None,
            factor_group="core_ta",
        )
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


def run_backtest_medium(output_dir: Path) -> int:
    """Run BACKTEST_MEDIUM benchmark job.
    
    Args:
        output_dir: Output directory for profiling artifacts
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    from scripts.run_backtest_strategy import run_backtest_from_args
    import argparse
    
    # Create args namespace
    args = argparse.Namespace()
    args.freq = "1d"
    args.strategy = "trend_baseline"
    args.start_capital = 10000.0
    args.with_costs = True
    args.price_file = None
    args.universe = None
    args.symbols = None
    args.symbols_file = None
    args.data_source = None
    args.start_date = None
    args.end_date = None
    args.out = None
    args.generate_report = False
    args.use_meta_model = False
    args.meta_model_path = None
    args.meta_min_confidence = 0.5
    args.meta_ensemble_mode = "filter"
    args.track_experiment = False
    args.experiment_name = None
    args.experiment_tags = None
    args.bundle_path = None
    args.top_quantile = 0.2
    args.bottom_quantile = 0.2
    args.rebalance_freq = "M"
    args.max_gross_exposure = 1.0
    args.use_regime_overlay = False
    args.regime_config_file = None
    args.enable_timings = True
    args.timings_out = output_dir / "timings.json"
    args.commission_bps = None
    args.spread_w = None
    args.impact_w = None
    
    try:
        return run_backtest_from_args(args)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


def run_ml_job(output_dir: Path) -> int:
    """Run ML_JOB benchmark (build_ml_dataset + train_meta_model).
    
    Args:
        output_dir: Output directory for profiling artifacts
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    from scripts.cli import build_ml_dataset_subcommand, train_meta_model_subcommand
    import argparse
    
    # Step 1: Build ML dataset
    args_build = argparse.Namespace()
    args_build.strategy = "trend_baseline"
    args_build.freq = "1d"
    args_build.price_file = None
    args_build.universe = None
    args_build.symbols = None
    args_build.symbols_file = None
    args_build.start_date = None
    args_build.end_date = None
    args_build.label_horizon_days = 10
    args_build.success_threshold = 0.05
    args_build.out = None
    args_build.start_capital = 10000.0
    
    try:
        exit_code_build = build_ml_dataset_subcommand(args_build)
        if exit_code_build != 0:
            return exit_code_build
    except Exception as e:
        print(f"ERROR in build_ml_dataset: {e}", file=sys.stderr)
        return 1
    
    # Step 2: Train meta-model
    args_train = argparse.Namespace()
    args_train.dataset_path = None  # Will build on-the-fly
    args_train.strategy = "trend_baseline"
    args_train.freq = "1d"
    args_train.start_date = None
    args_train.end_date = None
    args_train.symbols = None
    args_train.model_type = "gradient_boosting"
    args_train.output_model_path = None
    args_train.label_horizon_days = 10
    args_train.success_threshold = 0.05
    args_train.track_experiment = False
    args_train.experiment_name = None
    args_train.experiment_tags = None
    
    try:
        exit_code_train = train_meta_model_subcommand(args_train)
        return exit_code_train
    except Exception as e:
        print(f"ERROR in train_meta_model: {e}", file=sys.stderr)
        return 1


def profile_with_cprofile(job_func, output_dir: Path) -> tuple[int, Path, float]:
    """Run job with cProfile profiling.
    
    Args:
        job_func: Function that runs the job (takes output_dir, returns exit code)
        output_dir: Output directory for profiling artifacts
        
    Returns:
        Tuple of (exit_code, profile_file_path, total_runtime_seconds)
    """
    import time
    
    profile_file = output_dir / "profile.prof"
    
    # Create profiler and run job with timing
    profiler = cProfile.Profile()
    
    start_time = time.time()
    try:
        profiler.enable()
        exit_code = job_func(output_dir)
        profiler.disable()
    finally:
        # Always save profile, even on error
        profiler.dump_stats(str(profile_file))
    
    end_time = time.time()
    total_runtime = end_time - start_time
    
    # Generate text report (top functions by cumulative time)
    stats = pstats.Stats(str(profile_file))
    stats.sort_stats("cumulative")
    
    report_file = output_dir / "profile_report.txt"
    with report_file.open("w", encoding="utf-8") as f:
        stats.print_stats(50, file=f)  # Top 50 functions
    
    # Generate call graph (optional, but useful)
    callgraph_file = output_dir / "profile_callgraph.txt"
    with callgraph_file.open("w", encoding="utf-8") as f:
        stats.print_callers(30, file=f)  # Top 30 callers
    
    return exit_code, profile_file, total_runtime


def profile_with_pyinstrument(job_func, output_dir: Path) -> tuple[int, Path, float]:
    """Run job with pyinstrument profiling.
    
    Args:
        job_func: Function that runs the job (takes output_dir, returns exit code)
        output_dir: Output directory for profiling artifacts
        
    Returns:
        Tuple of (exit_code, profile_file_path, total_runtime_seconds)
    """
    import time
    
    try:
        from pyinstrument import Profiler
    except ImportError:
        print("ERROR: pyinstrument not installed. Install with: pip install pyinstrument", file=sys.stderr)
        return 1, output_dir / "profile.html", 0.0
    
    profile_file = output_dir / "profile.html"
    
    profiler = Profiler()
    
    start_time = time.time()
    try:
        profiler.start()
        exit_code = job_func(output_dir)
        profiler.stop()
    finally:
        # Always save profile, even on error
        profiler.save_html(str(profile_file))
    
    end_time = time.time()
    total_runtime = end_time - start_time
    
    # Also generate text output
    report_file = output_dir / "profile_report.txt"
    with report_file.open("w", encoding="utf-8") as f:
        f.write(profiler.output_text(unicode=True, color=False))
    
    return exit_code, profile_file, total_runtime


def extract_top_hotspots_from_cprofile(profile_file: Path, top_n: int = 3) -> list[dict[str, Any]]:
    """Extract top N hotspots from cProfile output.
    
    Args:
        profile_file: Path to .prof file
        top_n: Number of hotspots to extract
        
    Returns:
        List of dicts with keys: function, cumulative_time, calls, per_call_time
    """
    if not profile_file.exists():
        return []
    
    stats = pstats.Stats(str(profile_file))
    stats.sort_stats("cumulative")
    
    hotspots = []
    # Iterate over stats (stats.stats is a dict of (filename, lineno, funcname) -> (ncalls, tottime, cumtime, ...))
    for (filename, lineno, funcname), (ncalls, tottime, cumtime, callers) in stats.stats.items():
        # Format function name
        func_name = f"{filename}:{lineno}({funcname})"
        cumulative_time = cumtime  # cumulative time
        calls = ncalls  # call count (can be tuple (total, primitive) if recursive)
        
        # Handle tuple calls (recursive functions)
        if isinstance(calls, tuple):
            calls = calls[0]
        
        # Skip entries with very low time (likely noise)
        if cumulative_time < 0.01:
            continue
        
        hotspots.append({
            "function": func_name,
            "cumulative_time": cumulative_time,
            "calls": calls,
            "per_call_time": cumulative_time / calls if calls > 0 else 0.0,
        })
    
    # Sort by cumulative time and return top N
    hotspots.sort(key=lambda x: x["cumulative_time"], reverse=True)
    return hotspots[:top_n]


def extract_top_hotspots_from_pyinstrument(profile_file: Path, top_n: int = 3) -> list[dict[str, Any]]:
    """Extract top N hotspots from pyinstrument output.
    
    Args:
        profile_file: Path to .html file
        top_n: Number of hotspots to extract
        
    Returns:
        List of dicts with keys: function, time, percentage
    """
    try:
        from pyinstrument import Profiler
    except ImportError:
        return []
    
    # Load profile and extract top functions
    # pyinstrument doesn't have a simple API for this, so we parse the text output
    # For now, return empty list (can be enhanced later)
    return []


def update_performance_profile_md(
    job_id: str,
    hotspots: list[dict[str, Any]],
    profiler: str,
    total_runtime: float | None = None,
    phase: str = "P3",
) -> None:
    """Append hotspots and runtime to docs/PERFORMANCE_PROFILE.md (optional feature).
    
    Args:
        job_id: Job ID (EOD_SMALL, BACKTEST_MEDIUM, ML_JOB)
        hotspots: List of hotspot dicts
        profiler: Profiler name (cprofile, pyinstrument)
        total_runtime: Total runtime in seconds (optional)
        phase: Phase identifier (e.g., "P3", "Before P3", "After P3")
    """
    profile_doc = ROOT / "docs" / "PERFORMANCE_PROFILE.md"
    
    if not profile_doc.exists():
        return  # Skip if doc doesn't exist
    
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    with profile_doc.open("a", encoding="utf-8") as f:
        f.write(f"\n## {job_id} - {phase} (profiled with {profiler}) - {timestamp}\n\n")
        
        if total_runtime is not None:
            f.write(f"**Total Runtime:** {total_runtime:.2f}s\n\n")
        
        if hotspots:
            f.write("**Top-3 Hotspots:**\n\n")
            for i, hotspot in enumerate(hotspots, 1):
                func = hotspot.get("function", "unknown")
                cumulative_time = hotspot.get("cumulative_time", 0.0)
                calls = hotspot.get("calls", 0)
                
                f.write(f"{i}. **{func}**\n")
                f.write(f"   - Cumulative time: {cumulative_time:.3f}s\n")
                f.write(f"   - Calls: {calls}\n")
                if calls > 0:
                    f.write(f"   - Per-call time: {cumulative_time/calls:.6f}s\n")
                f.write("\n")
        else:
            f.write("**Top-3 Hotspots:** (none extracted)\n\n")
        
        f.write("\n")


def main() -> int:
    """Main entry point for profiling script."""
    parser = argparse.ArgumentParser(
        description="Profile reference benchmark jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile EOD_SMALL with cProfile
  python scripts/profile_job.py --job EOD_SMALL --profiler cprofile
  
  # Profile BACKTEST_MEDIUM with pyinstrument
  python scripts/profile_job.py --job BACKTEST_MEDIUM --profiler pyinstrument
  
  # Profile ML_JOB with cProfile (dry-run)
  python scripts/profile_job.py --job ML_JOB --profiler cprofile --dry-run
        """,
    )
    
    parser.add_argument(
        "--job",
        type=str,
        required=True,
        choices=["EOD_SMALL", "BACKTEST_MEDIUM", "ML_JOB"],
        help="Job ID to profile",
    )
    
    parser.add_argument(
        "--profiler",
        type=str,
        required=True,
        choices=["cprofile", "pyinstrument"],
        help="Profiler to use",
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Dry run: print command but don't execute",
    )
    
    parser.add_argument(
        "--update-doc",
        action="store_true",
        default=False,
        help="Update docs/PERFORMANCE_PROFILE.md with top hotspots (append-only)",
    )
    
    args = parser.parse_args()
    
    # Determine job function
    job_funcs = {
        "EOD_SMALL": run_eod_small,
        "BACKTEST_MEDIUM": run_backtest_medium,
        "ML_JOB": run_ml_job,
    }
    
    job_func = job_funcs[args.job]
    
    # Create output directory
    settings = get_settings()
    profiles_root = settings.output_dir / "profiles" / args.job
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = profiles_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Profiling job: {args.job}")
    print(f"Profiler: {args.profiler}")
    print(f"Output directory: {output_dir}")
    
    if args.dry_run:
        print("DRY RUN: Would execute profiling (skipped)")
        return 0
    
    # Run profiling
    total_runtime = None
    if args.profiler == "cprofile":
        exit_code, profile_file, total_runtime = profile_with_cprofile(job_func, output_dir)
        hotspots = extract_top_hotspots_from_cprofile(profile_file, top_n=3) if profile_file.exists() else []
    elif args.profiler == "pyinstrument":
        exit_code, profile_file, total_runtime = profile_with_pyinstrument(job_func, output_dir)
        hotspots = extract_top_hotspots_from_pyinstrument(profile_file, top_n=3) if profile_file.exists() else []
    else:
        print(f"ERROR: Unknown profiler: {args.profiler}", file=sys.stderr)
        return 1
    
    # Write summary
    summary_file = output_dir / "summary.txt"
    with summary_file.open("w", encoding="utf-8") as f:
        f.write(f"Job: {args.job}\n")
        f.write(f"Profiler: {args.profiler}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Exit Code: {exit_code}\n")
        if total_runtime is not None:
            f.write(f"Total Runtime: {total_runtime:.2f}s\n")
        f.write(f"Profile File: {profile_file.name}\n")
        f.write("\nTop 3 Hotspots:\n")
        for i, hotspot in enumerate(hotspots, 1):
            f.write(f"{i}. {hotspot.get('function', 'unknown')}\n")
    
    print(f"Profile saved: {profile_file}")
    if total_runtime is not None:
        print(f"Total runtime: {total_runtime:.2f}s")
    print(f"Summary saved: {summary_file}")
    
    if exit_code != 0:
        print(f"WARNING: Job exited with code {exit_code}", file=sys.stderr)
    
    # Update performance profile doc if requested
    if args.update_doc:
        # Determine phase based on job_id and profiler (can be enhanced with CLI arg later)
        phase = "P3"  # Default phase
        update_performance_profile_md(args.job, hotspots, args.profiler, total_runtime, phase=phase)
        print(f"Updated docs/PERFORMANCE_PROFILE.md with hotspots and runtime")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

