"""Seed QuestDB tick store from existing historical CSV/Parquet price data.

Usage:
    python scripts/seed_questdb_from_csv.py \
        --file data/prices.csv \
        [--url postgresql://localhost:8812/qdb] \
        [--symbol-col symbol] \
        [--date-col timestamp] \
        [--batch-size 1000]

Expected CSV columns: timestamp (or date), symbol, open, high, low, close, volume
Parquet files are auto-detected by extension.

Requires QuestDB running on localhost:8812 (default) or the URL specified.
Falls back to a dry-run count when QuestDB is not reachable.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import timezone

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed QuestDB from CSV/Parquet price data")
    parser.add_argument("--file", required=True, help="Path to CSV or Parquet file")
    parser.add_argument("--url", default="postgresql://localhost:8812/qdb", help="QuestDB PG-wire URL")
    parser.add_argument("--symbol-col", default="symbol")
    parser.add_argument("--date-col", default="timestamp")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", help="Parse only, do not write to QuestDB")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError:
        log.error("pandas required — pip install pandas")
        sys.exit(1)

    # Load data
    if not os.path.isfile(args.file):
        log.error("File not found: %s", args.file)
        sys.exit(1)

    log.info("Loading %s ...", args.file)
    if args.file.endswith(".parquet"):
        df = pd.read_parquet(args.file)
    else:
        df = pd.read_csv(args.file)

    log.info("Loaded %d rows, columns: %s", len(df), list(df.columns))

    # Normalize columns
    required = {args.symbol_col, args.date_col, "close"}
    missing = required - set(df.columns)
    if missing:
        log.error("Missing required columns: %s", missing)
        sys.exit(1)

    df = df.rename(columns={args.symbol_col: "symbol", args.date_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    for col in ("open", "high", "low", "volume"):
        if col not in df.columns:
            df[col] = df["close"] if col != "volume" else 0.0

    from src.assembled_core.data.tick_store import (
        QUESTDB_DRIVER_AVAILABLE,
        OHLCVTick,
        TickStore,
    )

    if not QUESTDB_DRIVER_AVAILABLE:
        log.warning("QuestDB driver not installed (psycopg2/pg8000). Install one to connect.")

    store = TickStore(url=args.url)
    reachable = store.ping()
    if not reachable:
        log.warning("QuestDB not reachable at %s — running in dry-run mode", args.url)
        args.dry_run = True

    if not args.dry_run:
        store.ensure_table()

    total_written = 0
    batch: list[OHLCVTick] = []

    for row in df.itertuples(index=False):
        ts = row.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        tick = OHLCVTick(
            symbol=str(row.symbol),
            ts=ts,
            open=float(getattr(row, "open", row.close)),
            high=float(getattr(row, "high", row.close)),
            low=float(getattr(row, "low", row.close)),
            close=float(row.close),
            volume=float(getattr(row, "volume", 0)),
        )
        batch.append(tick)

        if len(batch) >= args.batch_size:
            if args.dry_run:
                total_written += len(batch)
                log.info("[DRY-RUN] would write batch of %d ticks (total so far: %d)", len(batch), total_written)
            else:
                n = store.write_ticks(batch)
                total_written += n
                log.info("[OK] wrote batch of %d ticks (total: %d)", n, total_written)
            batch = []

    # Final batch
    if batch:
        if args.dry_run:
            total_written += len(batch)
            log.info("[DRY-RUN] would write final batch of %d ticks", len(batch))
        else:
            n = store.write_ticks(batch)
            total_written += n
            log.info("[OK] wrote final batch of %d ticks", n)

    mode = "dry-run" if args.dry_run else "written"
    log.info("[DONE] %s %d ticks total from %s", mode, total_written, args.file)


if __name__ == "__main__":
    main()
