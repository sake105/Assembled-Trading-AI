# scripts/commands/ops.py
"""Ops subcommands: check_health, paper_track, walk_forward, run_phase4_tests."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def check_health_subcommand(args: argparse.Namespace) -> int:
    """Check backend health status subcommand."""
    from scripts.check_health import run_health_checks_from_cli

    logger = logging.getLogger(__name__)

    try:
        return run_health_checks_from_cli(args)
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return 1


def paper_track_subcommand(args: argparse.Namespace) -> int:
    """Run paper track subcommand."""
    from scripts.run_paper_track import (
        find_config_by_strategy_name,
        list_paper_track_configs,
        run_paper_track_from_cli,
    )

    logger = logging.getLogger(__name__)

    if args.list:
        return list_paper_track_configs()

    config_file = None
    if args.strategy_name:
        config_file = find_config_by_strategy_name(args.strategy_name)
        if config_file is None:
            logger.error(
                f"Config not found for strategy '{args.strategy_name}'. "
                "Run with --list to see available strategies."
            )
            return 1
        logger.info(f"Auto-discovered config for '{args.strategy_name}': {config_file}")
    elif args.config_file:
        config_file = args.config_file
    else:
        logger.error(
            "Either --config-file or --strategy-name must be provided. "
            "Run with --list to see available strategies."
        )
        return 1

    try:
        return run_paper_track_from_cli(
            config_file=config_file,
            as_of=args.as_of,
            start_date=args.start_date,
            end_date=args.end_date,
            catch_up=getattr(args, "catch_up", False),
            dry_run=args.dry_run,
            fail_fast=args.fail_fast,
            rerun=getattr(args, "rerun", False),
            generate_risk_report=getattr(args, "generate_risk_report", False),
            risk_report_frequency=getattr(args, "risk_report_frequency", "weekly"),
            benchmark_symbol=getattr(args, "benchmark_symbol", None),
            factor_returns_file=getattr(args, "factor_returns_file", None),
        )
    except Exception as e:
        logger.error(f"Paper track failed: {e}", exc_info=True)
        return 1


def walk_forward_subcommand(args: argparse.Namespace) -> int:
    """Run walk-forward analysis subcommand."""
    from scripts.run_walk_forward_analysis import run_walk_forward_analysis

    return run_walk_forward_analysis(args)


def run_phase4_tests_subcommand(args: argparse.Namespace) -> int:
    """Run Phase-4 test suite subcommand."""
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Phase-4 Test Suite (run_phase4_tests)")
    logger.info("=" * 60)

    pytest_args = [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "phase4",
        "-q",
        "--maxfail=1",
        "--tb=short",
    ]

    if args.verbose:
        pytest_args = [arg for arg in pytest_args if arg != "-q"]
        pytest_args.append("-vv")

    if args.durations:
        pytest_args.append(f"--durations={args.durations}")

    logger.info(f"Running: {' '.join(pytest_args[2:])}")
    logger.info("")

    try:
        result = subprocess.run(
            pytest_args,
            cwd=str(ROOT),
            check=False,
        )
        return result.returncode
    except Exception as e:
        logger.error(f"Failed to run pytest: {e}", exc_info=True)
        return 1


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register ops subcommands."""
    # check_health
    check_health_parser = subparsers.add_parser(
        "check_health",
        help="Check backend health status (read-only, operations monitoring)",
        description="Health checks for backend operations (existence, plausibility, status interpretation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic health check
  python scripts/cli.py check_health --backtests-root output/backtests/

  # With custom thresholds
  python scripts/cli.py check_health --backtests-root output/backtests/ --min-sharpe 0.5 --max-drawdown-min -0.3

  # JSON output format
  python scripts/cli.py check_health --backtests-root output/backtests/ --format json

  # With paper track checks
  python scripts/cli.py check_health --backtests-root output/backtests/ --paper-track-root output/paper_track/ --paper-track-days 3

  # Skip paper track if missing
  python scripts/cli.py check_health --backtests-root output/backtests/ --skip-paper-track-if-missing
        """,
    )
    check_health_parser.add_argument(
        "--backtests-root",
        type=Path,
        default=Path("output/backtests/"),
        help="Root directory containing backtest outputs (default: output/backtests/)",
    )
    check_health_parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="Lookback window in days for historical comparison (default: 60)",
    )
    check_health_parser.add_argument(
        "--benchmark-symbol",
        type=str,
        default=None,
        help="Benchmark symbol (e.g., 'SPY') for correlation checks",
    )
    check_health_parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=None,
        help="Path to benchmark file (CSV/Parquet with timestamp, returns/close)",
    )
    check_health_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for health reports (default: output/health/)",
    )
    check_health_parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json", "both"],
        default="text",
        help="Output format: 'text' for human-readable, 'json' for machine-readable, 'both' for both (default: text)",
    )
    check_health_parser.add_argument(
        "--min-sharpe",
        type=float,
        default=0.0,
        help="Minimum acceptable Sharpe ratio (default: 0.0)",
    )
    check_health_parser.add_argument(
        "--max-drawdown-min",
        type=float,
        default=-0.40,
        help="Minimum acceptable max drawdown (more negative = worse, default: -0.40)",
    )
    check_health_parser.add_argument(
        "--max-drawdown-max",
        type=float,
        default=0.0,
        help="Maximum acceptable max drawdown (less negative = better, default: 0.0)",
    )
    check_health_parser.add_argument(
        "--max-turnover",
        type=float,
        default=10.0,
        help="Maximum acceptable turnover (default: 10.0)",
    )
    check_health_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    check_health_parser.add_argument(
        "--paper-track-root",
        type=Path,
        default=None,
        help="Root directory for paper track outputs (default: auto-detect under output/paper_track/)",
    )
    check_health_parser.add_argument(
        "--paper-track-days",
        type=int,
        default=3,
        help="Maximum allowed age in days for paper track runs (default: 3)",
    )
    check_health_parser.add_argument(
        "--skip-paper-track-if-missing",
        action="store_true",
        default=False,
        help="Skip paper track checks if paper track directory doesn't exist (default: False = WARN)",
    )
    check_health_parser.add_argument(
        "--paper-track-max-daily-pnl-pct",
        type=float,
        default=10.0,
        help="Maximum acceptable daily PnL percentage for plausibility check (default: 10.0%%)",
    )
    check_health_parser.add_argument(
        "--paper-track-max-drawdown-min",
        type=float,
        default=-0.25,
        help="Minimum acceptable max drawdown for paper track (default: -0.25 = -25%%)",
    )
    check_health_parser.add_argument(
        "--paper-track-max-gap-days",
        type=int,
        default=5,
        help="Maximum allowed gap in business days for paper track equity curve (default: 5)",
    )
    check_health_parser.add_argument(
        "--batch-root",
        type=Path,
        default=None,
        help="Root directory containing batch backtest outputs (default: auto-detect under output/)",
    )
    check_health_parser.add_argument(
        "--batch-max-failure-rate",
        type=float,
        default=0.2,
        help="Maximum acceptable failure rate for batch runs (default: 0.2 = 20%%)",
    )
    check_health_parser.add_argument(
        "--skip-batch-if-missing",
        action="store_true",
        default=False,
        help="Skip batch checks if batch directory doesn't exist (default: False = WARN)",
    )
    check_health_parser.set_defaults(func=check_health_subcommand)

    # paper_track
    paper_track_parser = subparsers.add_parser(
        "paper_track",
        help="Run paper track for a single day or date range",
        description="Runs paper track execution for trading strategies, executing the complete daily flow: load state -> compute signals -> size positions -> simulate fills -> update state -> write artifacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available strategies
  python scripts/cli.py paper_track --list

  # Run for single day (with strategy name)
  python scripts/cli.py paper_track --strategy-name trend_baseline_example --as-of 2025-01-15

  # Run for single day (with config file)
  python scripts/cli.py paper_track --config-file configs/paper_track/strategy.yaml --as-of 2025-01-15

  # Run for date range
  python scripts/cli.py paper_track --strategy-name trend_baseline_example --start-date 2025-01-15 --end-date 2025-01-20

  # Catch-up mode: automatically run from last_run_date+1 to today (or --as-of)
  python scripts/cli.py paper_track --strategy-name trend_baseline_example --catch-up
  python scripts/cli.py paper_track --strategy-name trend_baseline_example --catch-up --as-of 2025-01-20

  # Dry run (no files written)
  python scripts/cli.py paper_track --strategy-name trend_baseline_example --as-of 2025-01-15 --dry-run

  # Fail fast on errors
  python scripts/cli.py paper_track --strategy-name trend_baseline_example --start-date 2025-01-15 --end-date 2025-01-20 --fail-fast
        """,
    )
    paper_track_parser.add_argument(
        "--list",
        action="store_true",
        default=False,
        help="List all available paper track configs and strategies (exits immediately)",
    )
    config_group = paper_track_parser.add_mutually_exclusive_group(required=False)
    config_group.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="Path to paper track config file (YAML/JSON)",
    )
    config_group.add_argument(
        "--strategy-name",
        type=str,
        default=None,
        help="Strategy name (will auto-discover config from configs/paper_track/{name}.yaml or output/paper_track/{name}/config.yaml)",
    )
    paper_track_parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Single date to run (YYYY-MM-DD). Mutually exclusive with --start-date/--end-date",
    )
    paper_track_parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for range (YYYY-MM-DD, inclusive). Requires --end-date",
    )
    paper_track_parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for range (YYYY-MM-DD, inclusive). Requires --start-date",
    )
    paper_track_parser.add_argument(
        "--catch-up",
        action="store_true",
        default=False,
        help=(
            "Catch-up mode: automatically compute date range from state last_run_date. "
            "If no --start-date/--end-date specified, starts from last_run_date+1 and ends at --as-of (or today). "
            "If no state exists, falls back to --as-of (single day) or errors."
        ),
    )
    paper_track_parser.add_argument(
        "--dry-run", action="store_true", help="Dry run mode: don't write any files"
    )
    paper_track_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first error (default: continue and log errors)",
    )
    paper_track_parser.add_argument(
        "--rerun",
        action="store_true",
        help="Re-run days even if run directory already exists (default: skip existing days)",
    )
    paper_track_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    paper_track_parser.set_defaults(func=paper_track_subcommand)

    # walk_forward
    # Note: walk_forward_subcommand arguments are registered in run_walk_forward_analysis.py
    # We only set up a minimal parser here; actual args are handled by the script
    walk_forward_parser = subparsers.add_parser(
        "walk_forward",
        help="Run walk-forward analysis (out-of-sample validation, research tool)",
        description="Run walk-forward analysis for out-of-sample strategy validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    walk_forward_parser.set_defaults(func=walk_forward_subcommand)

    # run_phase4_tests
    tests_parser = subparsers.add_parser(
        "run_phase4_tests",
        help="Run Phase-4 test suite",
        description="Runs the Phase-4 regression test suite (~13s, 110 tests).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/cli.py run_phase4_tests
  python scripts/cli.py run_phase4_tests --verbose
  python scripts/cli.py run_phase4_tests --durations 5
  python scripts/cli.py run_phase4_tests --verbose --durations 10
        """,
    )
    tests_parser.add_argument(
        "--durations",
        type=int,
        default=None,
        metavar="N",
        help="Show N slowest tests (e.g., 5 for --durations=5)",
    )
    tests_parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show verbose test output (-vv instead of -q)",
    )
    tests_parser.set_defaults(func=run_phase4_tests_subcommand)
