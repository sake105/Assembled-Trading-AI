"""Refresh output/aggregates/daily.parquet from data/sample/master_universe_panel.parquet.

Bridges the data freshness gap that breaks the paper pilot when yfinance is
rate-limited at 21:30. The pilot's `_load_prices` in scripts/run_live_paper.py
checks if daily.parquet is <= 3 days old; if not, it falls through to a
sequential per-symbol yfinance fetch that — with 197 symbols and a 15-minute
Task Scheduler ExecutionTimeLimit — gets hard-terminated by Windows.

The master_universe_panel.parquet is built earlier in the daily cycle and
typically contains fresher OHLCV. This script copies its newer rows into
daily.parquet so the pilot's cache-fresh path stays satisfied.

Schemas:
- daily.parquet: [timestamp, symbol, open, high, low, close, adj_close, volume]
- master_universe_panel.parquet: [timestamp, symbol, open, high, low, close, volume]
  (no adj_close — we default it to close for the appended rows)

Idempotent: appends only rows with timestamp > cache.timestamp.max(); drops
exact (symbol, timestamp) duplicates as a final safety net.

Usage:
    python scripts/ops/refresh_daily_cache_from_panel.py
    python scripts/ops/refresh_daily_cache_from_panel.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

CACHE_PATH = ROOT / "output" / "aggregates" / "daily.parquet"
PANEL_PATH = ROOT / "data" / "sample" / "master_universe_panel.parquet"


def refresh(cache_path: Path, panel_path: Path, *, dry_run: bool) -> int:
    """Append fresher panel rows into the daily cache. Returns rows appended."""
    import pandas as pd

    if not cache_path.exists():
        logger.error("[refresh-cache] cache not found: %s", cache_path)
        return -1
    if not panel_path.exists():
        logger.error("[refresh-cache] panel not found: %s", panel_path)
        return -1

    cache = pd.read_parquet(cache_path)
    panel = pd.read_parquet(panel_path)

    cache["timestamp"] = pd.to_datetime(cache["timestamp"], utc=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)

    cache_latest = cache["timestamp"].max()
    panel_latest = panel["timestamp"].max()
    logger.info(
        "[refresh-cache] cache latest=%s, panel latest=%s",
        cache_latest,
        panel_latest,
    )

    # F-RX-2 (audit 2026-05-21): per-symbol comparison instead of global max.
    # Even if global panel.max() == global cache.max(), individual symbols
    # may have stale rows in cache that the panel has refreshed. Compare
    # per-symbol latest timestamps so heterogeneous freshness is fixed at
    # the per-symbol level.
    very_old = pd.Timestamp("1900-01-01", tz="UTC")
    cache_per_sym = cache.groupby("symbol")["timestamp"].max().rename("_cache_max")

    # Merge each panel row with the cache's per-symbol max; absent symbols
    # default to very_old so all their rows are treated as new.
    cache_max_df = cache_per_sym.reset_index()
    panel_with_cmax = panel.merge(cache_max_df, on="symbol", how="left")
    panel_with_cmax["_cache_max"] = panel_with_cmax["_cache_max"].fillna(very_old)
    new_rows = panel_with_cmax[
        panel_with_cmax["timestamp"] > panel_with_cmax["_cache_max"]
    ].drop(columns=["_cache_max"])
    if new_rows.empty:
        logger.info(
            "[refresh-cache] no panel rows strictly newer than per-symbol cache max"
        )
        return 0

    n_syms = new_rows["symbol"].nunique()
    ts_min = new_rows["timestamp"].min()
    ts_max = new_rows["timestamp"].max()
    logger.info(
        "[refresh-cache] %d rows to append for %d symbols, ts %s..%s",
        len(new_rows),
        n_syms,
        ts_min,
        ts_max,
    )

    if dry_run:
        logger.info("[refresh-cache] --dry-run set, not writing")
        return len(new_rows)

    # Panel lacks adj_close; default to close. Acceptable for the short
    # horizon (last few days have no splits/dividends realistic enough to
    # materially affect signals before the next full cache rebuild).
    if "adj_close" not in new_rows.columns:
        new_rows["adj_close"] = new_rows["close"]

    # Reorder to match cache schema (drop any extra cols panel may carry).
    new_rows = new_rows[cache.columns.tolist()]

    merged = pd.concat([cache, new_rows], ignore_index=True)
    merged = (
        merged.sort_values(["symbol", "timestamp"])
        .drop_duplicates(subset=["symbol", "timestamp"], keep="last")
        .reset_index(drop=True)
    )

    logger.info(
        "[refresh-cache] merged total rows=%d (%d symbols), new latest=%s",
        len(merged),
        merged["symbol"].nunique(),
        merged["timestamp"].max(),
    )

    tmp = cache_path.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=False)
    shutil.move(str(tmp), str(cache_path))
    logger.info("[refresh-cache] wrote %s", cache_path)
    return len(new_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without writing",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    rc = refresh(CACHE_PATH, PANEL_PATH, dry_run=args.dry_run)
    return 0 if rc >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
