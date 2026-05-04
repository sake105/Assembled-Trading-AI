"""Update local price cache with latest data from Polygon or yfinance.

Run this before the daily paper trading cycle to ensure fresh prices.

Usage:
  python scripts/update_prices.py                    # update last 10 days
  python scripts/update_prices.py --days 30          # update last 30 days
  python scripts/update_prices.py --source yfinance  # force yfinance
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _load_watchlist() -> list[str]:
    wl = ROOT / "watchlist.txt"
    if not wl.exists():
        logger.error("watchlist.txt not found")
        return []
    symbols = [
        s.strip()
        for s in wl.read_text(encoding="utf-8").splitlines()
        if s.strip() and not s.strip().startswith("#")
    ]
    # US symbols only (no .DE, .AX etc.)
    return [s for s in symbols if "." not in s]


def _fetch_polygon(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    from src.assembled_core.data.sources.polygon_source import fetch_prices_polygon

    return fetch_prices_polygon(symbols, start, end)


def _fetch_yfinance(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    from src.assembled_core.data.sources.yfinance_source import fetch_prices_yfinance

    return fetch_prices_yfinance(symbols, start, end)


def _get_cache_path() -> Path:
    from src.assembled_core.data.prices_ingest import get_default_price_path

    return get_default_price_path("1d")


def main():
    parser = argparse.ArgumentParser(description="Update local price cache")
    parser.add_argument("--days", type=int, default=10, help="Days of history to fetch")
    parser.add_argument(
        "--source",
        choices=["polygon", "yfinance", "auto"],
        default="auto",
        help="Data source (auto tries polygon first)",
    )
    parser.add_argument("--full", action="store_true", help="Full re-download (1 year)")
    args = parser.parse_args()

    symbols = _load_watchlist()
    if not symbols:
        sys.exit(1)

    logger.info("Updating prices for %d US symbols", len(symbols))

    days = 400 if args.full else args.days
    end_date = (pd.Timestamp.now("UTC") + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.now("UTC") - pd.DateOffset(days=days)).strftime(
        "%Y-%m-%d"
    )

    new_data = pd.DataFrame()

    if args.source in ("polygon", "auto"):
        try:
            logger.info("Fetching from Polygon (%s to %s)...", start_date, end_date)
            new_data = _fetch_polygon(symbols, start_date, end_date)
            if not new_data.empty:
                logger.info(
                    "Polygon: fetched %d rows for %d symbols",
                    len(new_data),
                    new_data["symbol"].nunique(),
                )
        except Exception as exc:
            logger.warning("Polygon failed: %s", exc)

    if new_data.empty and args.source in ("yfinance", "auto"):
        try:
            logger.info("Fetching from yfinance (%s to %s)...", start_date, end_date)
            new_data = _fetch_yfinance(symbols, start_date, end_date)
            if not new_data.empty:
                logger.info(
                    "yfinance: fetched %d rows for %d symbols",
                    len(new_data),
                    new_data["symbol"].nunique(),
                )
        except Exception as exc:
            logger.warning("yfinance failed: %s", exc)

    if new_data.empty:
        logger.error("No data fetched from any source")
        sys.exit(1)

    # Merge with existing cache
    cache_path = _get_cache_path()
    if cache_path.exists():
        logger.info("Loading existing cache: %s", cache_path)
        existing = pd.read_parquet(cache_path)
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
        new_data["timestamp"] = pd.to_datetime(new_data["timestamp"], utc=True)

        # Remove overlapping dates from existing, keep new data
        overlap_dates = new_data["timestamp"].unique()
        existing_clean = existing[~existing["timestamp"].isin(overlap_dates)]

        merged = pd.concat([existing_clean, new_data], ignore_index=True)
        merged = merged.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        merged = merged.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    else:
        merged = new_data.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Save
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(cache_path, index=False)
    latest = merged["timestamp"].max()
    logger.info(
        "Cache updated: %d rows, %d symbols, latest: %s -> %s",
        len(merged),
        merged["symbol"].nunique(),
        latest.date() if hasattr(latest, "date") else latest,
        cache_path,
    )


if __name__ == "__main__":
    main()
