# scripts/commands/info.py
"""Info subcommand: project information display."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

__version__ = "0.0.1"


def get_git_branch() -> str | None:
    """Try to get current git branch."""
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=str(ROOT),
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


def print_version() -> None:
    """Print version and project information."""
    branch = get_git_branch()
    print("Assembled Trading AI - Central CLI")
    print(f"Version: {__version__}")
    if branch:
        print(f"Git Branch: {branch}")
    print("Status: Phase-4/Phase-6 ready")
    print(f"Python: {sys.version.split()[0]}")


def info_subcommand(args: argparse.Namespace) -> int:
    """Show project information subcommand."""
    print("=" * 60)
    print("Assembled Trading AI - Project Information")
    print("=" * 60)
    print()
    print("Main Subcommands:")
    print(
        "  run_daily          - Run daily EOD pipeline (execute, backtest, portfolio, QA)"
    )
    print("  run_backtest       - Run strategy backtest with portfolio-level engine")
    print(
        "  batch_backtest     - Run batch of strategy backtests from config file (recommended)"
    )
    print("  batch_run          - Run batch backtests (alias, same as batch_backtest)")
    print(
        "  leaderboard        - Rank and display best runs from batch backtest results"
    )
    print(
        "  walk_forward       - Run walk-forward analysis (out-of-sample validation, research tool)"
    )
    print(
        "  risk_report        - Generate comprehensive risk report from backtest results"
    )
    print(
        "  tca_report         - Generate transaction cost analysis (TCA) report from backtest results"
    )
    print("  factor_report      - Generate factor analysis report (IC/IR statistics)")
    print(
        "  analyze_factors    - Comprehensive factor analysis (IC + Portfolio evaluation)"
    )
    print(
        "  ml_validate_factors - ML validation on factor panels (predict forward returns)"
    )
    print(
        "  ml_model_zoo       - Compare multiple ML models on factor panels (model zoo)"
    )
    print(
        "  check_health       - Check backend health status (read-only, operations monitoring)"
    )
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
    print(
        "  python scripts/cli.py run_backtest --freq 1d --universe watchlist.txt --generate-report"
    )
    print("  python scripts/cli.py run_phase4_tests --verbose --durations 5")
    print()
    print("For detailed help on a subcommand:")
    print("  python scripts/cli.py <subcommand> --help")
    print()
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the info subcommand."""
    info_parser = subparsers.add_parser(
        "info",
        help="Show project information",
        description="Shows project information, available subcommands, and documentation links.",
    )
    info_parser.set_defaults(func=info_subcommand)
