"""
run_daily_scheduler.py — CLI runner for the autonomous daily operations cycle.

Usage:
    python scripts/run_daily_scheduler.py [--date YYYY-MM-DD] [--output-dir PATH]
                                           [--dry-run] [--once]
                                           [--interval-hours FLOAT]

Exit codes:
    0 — all workers completed with status ok or skip
    1 — one or more workers completed with status error
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

# Ensure project root is importable (matches pattern in other scripts)
_root_path = Path(__file__).resolve().parents[1]
if str(_root_path) not in sys.path:
    sys.path.insert(0, str(_root_path))

from src.assembled_core.ops.daily_scheduler import (  # noqa: E402  # noqa: E402
    build_cycle_summary,
    run_daily_cycle,
    schedule_loop,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assembled-Trading-AI — autonomous daily operations cycle"
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Date to run cycle for (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for workers. Defaults to 'output'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run in dry-run mode (workers skip actual writes).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        default=False,
        help="Run cycle once and exit (default: continuous loop).",
    )
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=24.0,
        help="Hours between cycles in continuous mode. Defaults to 24.0.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.once:
        results = run_daily_cycle(
            date_str=args.date,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
        summary = build_cycle_summary(results)
        logger.info("[OK] cycle_summary %s", summary)
        return 1 if summary["error"] > 0 else 0
    else:
        # Continuous loop — only exits via KeyboardInterrupt
        try:
            schedule_loop(
                interval_hours=args.interval_hours,
                output_dir=args.output_dir,
                dry_run=args.dry_run,
            )
        except KeyboardInterrupt:
            logger.info("[OK] scheduler stopped by user")
        return 0


if __name__ == "__main__":
    sys.exit(main())
