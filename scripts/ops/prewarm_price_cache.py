"""Pre-warm output/aggregates/daily.parquet with missing watchlist symbols.

Pilot R6-followup: after switching the watchlist to the 195-symbol master
universe, 125 of those symbols are missing from the EOD price cache.
The pilot's load_eod_prices reads daily.parquet first and only falls back
to yfinance when stale — but missing symbols still need a one-time fetch.

This script:
1. Reads watchlist.txt (skipping comment lines)
2. Loads existing cache at output/aggregates/daily.parquet
3. Computes the gap (watchlist - cache)
4. Fetches gap symbols via yfinance (2-year history by default)
5. Merges + sorts + writes back to cache atomically (tmp + replace)

Usage:
    python scripts/ops/prewarm_price_cache.py            # default: ~2y history
    python scripts/ops/prewarm_price_cache.py --years 5  # longer history
    python scripts/ops/prewarm_price_cache.py --dry-run  # show gap only
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

WATCHLIST_PATH = ROOT / "watchlist.txt"
CACHE_PATH = ROOT / "output" / "aggregates" / "daily.parquet"


def load_watchlist(path: Path = WATCHLIST_PATH) -> list[str]:
    """Read watchlist.txt, skipping comments + blanks."""
    if not path.exists():
        raise FileNotFoundError(f"Watchlist not found: {path}")
    return [
        s.strip()
        for s in path.read_text(encoding="utf-8").splitlines()
        if s.strip() and not s.startswith("#")
    ]


def cache_symbols(path: Path = CACHE_PATH) -> set[str]:
    """Return symbols currently in the price cache."""
    if not path.exists():
        return set()
    import pandas as pd

    df = pd.read_parquet(path, columns=["symbol"])
    return set(df["symbol"].unique())


def stale_cache_symbols(
    watchlist: list[str],
    max_age_days: int,
    path: Path = CACHE_PATH,
) -> list[str]:
    """Return watchlist symbols in the cache whose own latest bar is > max_age_days old.

    F-RX-6 §9.12 (d) follow-up: prewarm previously refreshed only MISSING
    symbols (watchlist - cache). Symbols PRESENT in cache but stale per-symbol
    (e.g. KO/PEP/BRK-B/PG @ 2026-05-01 while panel-refreshed peers are at
    2026-05-18) stayed frozen forever — refresh_daily_cache_from_panel.py
    can't fix them because they're not in the master_universe_panel. This
    helper surfaces them so the prewarm path can yfinance-refresh them too.

    Returns symbols sorted by ascending freshness (oldest first), so a
    --max-symbols budget caps the work to the most-urgent ones.
    """
    if not path.exists():
        return []
    import pandas as pd

    df = pd.read_parquet(path, columns=["symbol", "timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[df["symbol"].isin(watchlist)]
    per_sym = df.groupby("symbol")["timestamp"].max()
    today = pd.Timestamp.now("UTC").normalize()
    ages = (today - per_sym.dt.normalize()).dt.days
    stale = ages[ages > max_age_days].sort_values(ascending=False)
    return list(stale.index)


def fetch_missing(missing: list[str], years: int) -> "pd.DataFrame":
    """Fetch the missing symbols via yfinance."""
    from src.assembled_core.data.sources.yfinance_source import fetch_prices_yfinance

    end = datetime.now(tz=timezone.utc).date()
    start = end - timedelta(days=int(years * 366))  # buffer for leap years
    logger.info(
        "[prewarm] fetching %d symbols from yfinance (%s to %s)",
        len(missing),
        start.isoformat(),
        end.isoformat(),
    )
    df = fetch_prices_yfinance(missing, start.isoformat(), end.isoformat())
    if df.empty:
        logger.error("[prewarm] yfinance returned EMPTY DataFrame for all symbols")
        return df
    got = set(df["symbol"].unique())
    failed = set(missing) - got
    if failed:
        logger.warning(
            "[prewarm] %d/%d symbols had no data: %s",
            len(failed),
            len(missing),
            sorted(failed)[:20],
        )
    logger.info(
        "[prewarm] fetched %d rows for %d/%d symbols",
        len(df),
        len(got),
        len(missing),
    )
    return df


def merge_and_save(new_df: "pd.DataFrame", cache_path: Path = CACHE_PATH) -> int:
    """Merge new rows into cache. Returns rows-after-merge count."""
    import pandas as pd

    if cache_path.exists():
        existing = pd.read_parquet(cache_path)
    else:
        existing = pd.DataFrame(columns=new_df.columns)

    combined = pd.concat([existing, new_df], ignore_index=True)
    # Dedupe on (symbol, timestamp) — last-write-wins favors the fresh fetch
    combined = combined.drop_duplicates(
        subset=["symbol", "timestamp"], keep="last"
    ).sort_values(["symbol", "timestamp"])

    # Atomic write
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    combined.to_parquet(tmp, index=False)
    tmp.replace(cache_path)
    return len(combined)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--years",
        type=int,
        default=2,
        help="History years to fetch for missing symbols (default 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report the gap, do not fetch or write",
    )
    parser.add_argument(
        "--max-stale-days",
        type=int,
        default=3,
        help=(
            "Refresh cache-present watchlist symbols whose own latest bar is "
            "older than this many calendar days (default 3, aligned with "
            "_drop_per_symbol_stale_rows max_age_days in run_live_paper.py so "
            "there is no silent dead-zone of stale-but-not-prewarmed symbols "
            "— F-RX-FU-4). Set to 0 to skip stale-row refresh entirely."
        ),
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=30,
        help=(
            "Hard budget on the number of symbols yfinance will be asked for "
            "in one invocation (default 30). Caps wall-clock time when "
            "rate-limited so the Task Scheduler ExecutionTimeLimit isn't hit. "
            "Stale symbols are processed oldest-first."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    watchlist = load_watchlist()
    cached = cache_symbols()
    missing = sorted(set(watchlist) - cached)
    stale = (
        stale_cache_symbols(watchlist, max_age_days=args.max_stale_days)
        if args.max_stale_days > 0
        else []
    )

    print(
        f"[prewarm] watchlist={len(watchlist)} cached={len(cached)} "
        f"missing={len(missing)} stale(>{args.max_stale_days}d)={len(stale)}"
    )

    if not missing and not stale:
        print("[prewarm] no gap, no stale rows — cache fully fresh")
        return 0

    # Budget: missing first (truly absent), then stale (refresh-eligible).
    targets = missing + [s for s in stale if s not in missing]
    if args.max_symbols > 0 and len(targets) > args.max_symbols:
        print(
            f"[prewarm] {len(targets)} targets exceeds --max-symbols={args.max_symbols} "
            f"budget; deferring tail to next invocation"
        )
        targets = targets[: args.max_symbols]

    print(f"[prewarm] will fetch (first 20): {targets[:20]}")
    if args.dry_run:
        print("[prewarm] DRY RUN — no fetch performed")
        return 0

    df = fetch_missing(targets, years=args.years)
    if df.empty:
        print("[prewarm] no data fetched — aborting merge")
        return 1

    total = merge_and_save(df)
    print(f"[prewarm] cache updated: {total:,} total rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
