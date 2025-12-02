# scripts/cli.py
"""Central CLI for Assembled Trading AI Backend.

This script provides a unified command-line interface for the most important backend operations:
- run_daily: Daily EOD pipeline
- run_backtest: Strategy backtest
- run_phase4_tests: Phase-4 test suite
- info: Show project information

Usage:
    python scripts/cli.py run_daily --freq 1d
    python scripts/cli.py run_backtest --freq 1d --universe watchlist.txt
    python scripts/cli.py run_phase4_tests
    python scripts/cli.py info
    python scripts/cli.py --version
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Import core modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.logging_utils import setup_logging

logger = setup_logging(level="INFO")

# Project version (from pyproject.toml / __init__.py)
__version__ = "0.0.1"


def get_git_branch() -> str | None:
    """Try to get current git branch.
    
    Returns:
        Branch name if git is available, None otherwise
    """
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return None


def print_version() -> None:
    """Print version and project information."""
    branch = get_git_branch()
    print(f"Assembled Trading AI - Central CLI")
    print(f"Version: {__version__}")
    if branch:
        print(f"Git Branch: {branch}")
    print(f"Status: Phase-4/Phase-6 ready")
    print(f"Python: {sys.version.split()[0]}")


def info_subcommand(args: argparse.Namespace) -> int:
    """Show project information subcommand.
    
    Args:
        args: Parsed command-line arguments (unused)
        
    Returns:
        Exit code (always 0)
    """
    print("=" * 60)
    print("Assembled Trading AI - Project Information")
    print("=" * 60)
    print()
    print("Main Subcommands:")
    print("  run_daily          - Run daily EOD pipeline (execute, backtest, portfolio, QA)")
    print("  run_backtest       - Run strategy backtest with portfolio-level engine")
    print("  run_phase4_tests   - Run Phase-4 regression test suite (~13s, 110 tests)")
    print("  info               - Show this information")
    print()
    print("Documentation:")
    print("  - Backend Architecture: docs/ARCHITECTURE_BACKEND.md")
    print("  - Legacy Overview: docs/LEGACY_OVERVIEW.md")
    print("  - Legacy Mapping: docs/LEGACY_TO_CORE_MAPPING.md")
    print("  - PowerShell Wrappers: docs/POWERSHELL_WRAPPERS.md")
    print("  - Testing Commands: docs/TESTING_COMMANDS.md")
    print()
    print("Examples:")
    print("  python scripts/cli.py run_daily --freq 1d")
    print("  python scripts/cli.py run_backtest --freq 1d --universe watchlist.txt --generate-report")
    print("  python scripts/cli.py run_phase4_tests --verbose --durations 5")
    print()
    print("For detailed help on a subcommand:")
    print("  python scripts/cli.py <subcommand> --help")
    print()
    return 0


def run_daily_subcommand(args: argparse.Namespace) -> int:
    """Run daily EOD pipeline subcommand.
    
    Args:
        args: Parsed command-line arguments for run_daily
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger.info("=" * 60)
    logger.info("EOD Pipeline (run_daily)")
    logger.info("=" * 60)
    
    # Import here to avoid circular imports
    from scripts.run_eod_pipeline import run_eod_from_args
    
    try:
        run_eod_from_args(args)
        return 0
    except RuntimeError:
        # Expected error from run_eod_from_args when pipeline fails
        return 1
    except Exception as e:
        logger.error(f"EOD pipeline failed: {e}", exc_info=True)
        return 1


def run_backtest_subcommand(args: argparse.Namespace) -> int:
    """Run strategy backtest subcommand.
    
    Args:
        args: Parsed command-line arguments for run_backtest
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger.info("=" * 60)
    logger.info("Strategy Backtest (run_backtest)")
    logger.info("=" * 60)
    
    # Import here to avoid circular imports
    from scripts.run_backtest_strategy import run_backtest_from_args
    
    try:
        return run_backtest_from_args(args)
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        return 1


def run_phase4_tests_subcommand(args: argparse.Namespace) -> int:
    """Run Phase-4 test suite subcommand.
    
    Args:
        args: Parsed command-line arguments for run_phase4_tests
        
    Returns:
        Exit code from pytest (0 for success, non-zero for failure)
    """
    logger.info("=" * 60)
    logger.info("Phase-4 Test Suite (run_phase4_tests)")
    logger.info("=" * 60)
    
    # Build pytest command
    pytest_args = [
        sys.executable,
        "-m", "pytest",
        "-m", "phase4",
        "-q",
        "--maxfail=1",
        "--tb=short"
    ]
    
    # Remove -q if verbose is requested
    if args.verbose:
        pytest_args = [arg for arg in pytest_args if arg != "-q"]
        pytest_args.append("-vv")
    
    if args.durations:
        pytest_args.append(f"--durations={args.durations}")
    
    logger.info(f"Running: {' '.join(pytest_args[2:])}")  # Exclude python and -m pytest for cleaner log
    logger.info("")
    
    # Run pytest
    try:
        result = subprocess.run(
            pytest_args,
            cwd=str(ROOT),
            check=False  # Don't raise on non-zero exit
        )
        return result.returncode
    except Exception as e:
        logger.error(f"Failed to run pytest: {e}", exc_info=True)
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands.
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Assembled Trading AI - Central CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run daily EOD pipeline
  python scripts/cli.py run_daily --freq 1d
  
  # Run strategy backtest
              python scripts/cli.py run_backtest --freq 1d --universe watchlist.txt --generate-report
              python scripts/cli.py run_backtest --freq 1d --strategy event_insider_shipping --generate-report
  
  # Run Phase-4 tests
  python scripts/cli.py run_phase4_tests
  
  # Run Phase-4 tests with verbose output and durations
  python scripts/cli.py run_phase4_tests --verbose --durations 5
  
  # Show project information
  python scripts/cli.py info
  
  # Show version
  python scripts/cli.py --version
        """
    )
    
    # Global --version flag
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show version and exit"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run", required=True)
    
    # run_daily subcommand
    daily_parser = subparsers.add_parser(
        "run_daily",
        help="Run daily EOD pipeline (execute, backtest, portfolio, QA)",
        description="Runs the full EOD pipeline: execute, backtest, portfolio simulation, and QA checks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/cli.py run_daily --freq 1d
  python scripts/cli.py run_daily --freq 1d --universe watchlist.txt --start-capital 50000
  python scripts/cli.py run_daily --freq 5min --price-file data/sample/eod_sample.parquet
        """
    )
    daily_parser.add_argument(
        "--freq",
        type=str,
        required=True,
        choices=["1d", "5min"],
        help="Trading frequency: '1d' for daily or '5min' for 5-minute bars"
    )
    daily_parser.add_argument(
        "--universe",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to universe file (default: watchlist.txt in repo root)"
    )
    daily_parser.add_argument(
        "--price-file",
        type=str,
        default=None,
        metavar="FILE",
        help="Optional explicit path to price file (overrides default path)"
    )
    daily_parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Start date for price data filtering (optional)"
    )
    daily_parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="End date for price data filtering (optional)"
    )
    daily_parser.add_argument(
        "--start-capital",
        type=float,
        default=10000.0,
        metavar="AMOUNT",
        help="Starting capital in USD (default: 10000.0)"
    )
    daily_parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip backtest step in pipeline"
    )
    daily_parser.add_argument(
        "--skip-portfolio",
        action="store_true",
        help="Skip portfolio simulation step"
    )
    daily_parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="Skip QA checks step"
    )
    daily_parser.add_argument(
        "--commission-bps",
        type=float,
        default=None,
        metavar="BPS",
        help="Commission in basis points (overrides default cost model)"
    )
    daily_parser.add_argument(
        "--spread-w",
        type=float,
        default=None,
        metavar="WEIGHT",
        help="Spread weight for cost model (overrides default)"
    )
    daily_parser.add_argument(
        "--impact-w",
        type=float,
        default=None,
        metavar="WEIGHT",
        help="Market impact weight for cost model (overrides default)"
    )
    daily_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory (default: from config.OUTPUT_DIR)"
    )
    daily_parser.set_defaults(func=run_daily_subcommand)
    
    # run_backtest subcommand
    backtest_parser = subparsers.add_parser(
        "run_backtest",
        help="Run strategy backtest",
        description="Runs a strategy backtest using the portfolio-level backtest engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/cli.py run_backtest --freq 1d --universe watchlist.txt
  python scripts/cli.py run_backtest --freq 1d --price-file data/sample/eod_sample.parquet --generate-report
  python scripts/cli.py run_backtest --freq 5min --start-capital 50000 --no-costs
        """
    )
    backtest_parser.add_argument(
        "--freq",
        type=str,
        required=True,
        choices=["1d", "5min"],
        help="Trading frequency: '1d' for daily or '5min' for 5-minute bars"
    )
    backtest_parser.add_argument(
        "--price-file",
        type=Path,
        default=None,
        metavar="FILE",
        help="Explicit path to price file (overrides default path)"
    )
    backtest_parser.add_argument(
        "--universe",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to universe file (default: watchlist.txt in repo root)"
    )
    backtest_parser.add_argument(
        "--strategy",
        type=str,
        default="trend_baseline",
        choices=["trend_baseline", "event_insider_shipping"],
        metavar="NAME",
        help="Strategy name: 'trend_baseline' (EMA crossover) or 'event_insider_shipping' (Phase 6 event-based)"
    )
    backtest_parser.add_argument(
        "--start-capital",
        type=float,
        default=10000.0,
        metavar="AMOUNT",
        help="Starting capital in USD (default: 10000.0)"
    )
    backtest_parser.add_argument(
        "--with-costs",
        action="store_true",
        default=True,
        help="Include transaction costs in backtest (default: True)"
    )
    backtest_parser.add_argument(
        "--no-costs",
        action="store_false",
        dest="with_costs",
        help="Disable transaction costs (use cost-free simulation)"
    )
    backtest_parser.add_argument(
        "--commission-bps",
        type=float,
        default=None,
        metavar="BPS",
        help="Commission in basis points (overrides default cost model)"
    )
    backtest_parser.add_argument(
        "--spread-w",
        type=float,
        default=None,
        metavar="WEIGHT",
        help="Spread weight for cost model (overrides default)"
    )
    backtest_parser.add_argument(
        "--impact-w",
        type=float,
        default=None,
        metavar="WEIGHT",
        help="Market impact weight for cost model (overrides default)"
    )
    backtest_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory (default: from config.OUTPUT_DIR)"
    )
    backtest_parser.add_argument(
        "--generate-report",
        action="store_true",
        default=False,
        help="Generate QA report after backtest"
    )
    backtest_parser.set_defaults(func=run_backtest_subcommand)
    
    # run_phase4_tests subcommand
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
        """
    )
    tests_parser.add_argument(
        "--durations",
        type=int,
        default=None,
        metavar="N",
        help="Show N slowest tests (e.g., 5 for --durations=5)"
    )
    tests_parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Show verbose test output (-vv instead of -q)"
    )
    tests_parser.set_defaults(func=run_phase4_tests_subcommand)
    
    # info subcommand
    info_parser = subparsers.add_parser(
        "info",
        help="Show project information",
        description="Shows project information, available subcommands, and documentation links."
    )
    info_parser.set_defaults(func=info_subcommand)
    
    return parser


def main() -> int:
    """Main entry point for central CLI.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle --version flag (before subcommand routing)
    if hasattr(args, "version") and args.version:
        print_version()
        return 0
    
    # Route to appropriate subcommand
    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.error("No subcommand specified. Use --help for usage.")


if __name__ == "__main__":
    sys.exit(main())
