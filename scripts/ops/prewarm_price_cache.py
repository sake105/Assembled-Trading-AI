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
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    watchlist = load_watchlist()
    cached = cache_symbols()
    missing = sorted(set(watchlist) - cached)

    print(
        f"[prewarm] watchlist={len(watchlist)} cached={len(cached)} gap={len(missing)}"
    )
    if not missing:
        print("[prewarm] no gap — cache already has all watchlist symbols")
        return 0

    print(f"[prewarm] missing (first 20): {missing[:20]}")
    if args.dry_run:
        print("[prewarm] DRY RUN — no fetch performed")
        return 0

    df = fetch_missing(missing, years=args.years)
    if df.empty:
        print("[prewarm] no data fetched — aborting merge")
        return 1

    total = merge_and_save(df)
    print(f"[prewarm] cache updated: {total:,} total rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
