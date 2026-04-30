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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.logging_config import generate_run_id, setup_logging

_run_id = generate_run_id(prefix="cli")
setup_logging(run_id=_run_id, level="INFO")
import logging

logger = logging.getLogger(__name__)

__version__ = "0.0.1"

# Import all command modules
from scripts.commands import backtest as _backtest_mod
from scripts.commands import info as _info_mod
from scripts.commands import ml as _ml_mod
from scripts.commands import news as _news_mod
from scripts.commands import ops as _ops_mod
from scripts.commands import paper as _paper_mod
from scripts.commands import reports as _reports_mod
from scripts.commands import run_daily as _run_daily_mod

_COMMAND_MODULES = [
    _info_mod,
    _run_daily_mod,
    _ml_mod,
    _backtest_mod,
    _news_mod,
    _reports_mod,
    _paper_mod,
    _ops_mod,
]

# Re-export handler functions for backward-compatible imports (e.g., tests)
from scripts.commands.info import print_version
from scripts.commands.ml import _run_backtest_for_ml_dataset
from scripts.commands.paper import run_paper_experiment_subcommand, summarize_intel_activity_subcommand
from scripts.commands.backtest import batch_backtest_subcommand


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
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
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show version and exit",
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Subcommand to run", required=True
    )

    for mod in _COMMAND_MODULES:
        mod.register(subparsers)

    return parser


def main() -> int:
    """Main entry point for central CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if hasattr(args, "version") and args.version:
        print_version()
        return 0

    if hasattr(args, "func"):
        return args.func(args)
    else:
        parser.error("No subcommand specified. Use --help for usage.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
