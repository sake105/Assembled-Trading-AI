"""Refresh cached earnings calendar for paper cycle.

Writes ``output/intel/earnings/calendar_latest.parquet`` that
``paper/intel_context.py`` reads into ``ctx.earnings_calendar`` when
``signal_generation.earnings_guard.enabled=true``.

Run periodically (e.g. weekly cron) — yfinance is slow per-symbol so the
paper cycle must not fetch inline.

Usage:
    python scripts/fetch_earnings_calendar.py --universe configs/universe.txt
    python scripts/fetch_earnings_calendar.py --symbols AAPL,MSFT,NVDA
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _load_universe(path: Path) -> list[str]:
    syms: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            syms.append(s.split(",")[0].split()[0].upper())
    return sorted(set(syms))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh earnings calendar cache")
    parser.add_argument(
        "--universe",
        type=Path,
        default=None,
        help="Path to universe file (one symbol per line)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols (alternative to --universe)",
    )
    parser.add_argument("--days-ahead", type=int, default=90)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/intel/earnings/calendar_latest.parquet"),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    log = logging.getLogger("fetch_earnings")

    if args.symbols:
        symbols = sorted(
            {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        )
    elif args.universe and args.universe.exists():
        symbols = _load_universe(args.universe)
    else:
        log.error("Must provide --symbols or --universe")
        return 2

    log.info(
        "Fetching earnings for %d symbols, days_ahead=%d", len(symbols), args.days_ahead
    )

    from src.assembled_core.data.sources.earnings_calendar_source import (
        EarningsCalendarSource,
    )

    source = EarningsCalendarSource()
    cal = source.fetch_calendar(symbols=symbols, days_ahead=args.days_ahead)

    if cal is None or cal.empty:
        log.warning("No earnings data fetched — cache NOT written")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cal.to_parquet(args.output, index=False)
    log.info("[OK] wrote %d rows to %s", len(cal), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
