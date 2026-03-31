#!/usr/bin/env python
"""Fetch real daily OHLCV data from Yahoo Finance for the runner universe.

Writes a single runner-compatible Parquet file with all symbols combined.

Usage:
    python scripts/fetch_real_daily.py
    python scripts/fetch_real_daily.py --start 2025-01-01 --end 2025-10-31
    python scripts/fetch_real_daily.py --universe watchlist.txt --output output/aggregates/daily_real.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def fetch_symbol(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch daily OHLCV for a single symbol via yfinance."""
    import yfinance as yf

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval="1d", auto_adjust=True)
        if df.empty:
            return None

        df = df.reset_index()
        df = df.rename(columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        df["symbol"] = symbol
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        available = [c for c in cols if c in df.columns]
        df = df[available]

        for c in ["open", "high", "low", "close"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(float)

        return df
    except Exception as exc:
        logger.warning(f"Failed to fetch {symbol}: {exc}")
        return None


def validate_ohlc(df: pd.DataFrame) -> int:
    """Count and log rows with invalid OHLC relationships. Returns count."""
    bad = (
        (df["high"] < df["low"])
        | (df["high"] < df["close"])
        | (df["high"] < df["open"])
        | (df["low"] > df["close"])
        | (df["low"] > df["open"])
    )
    n_bad = int(bad.sum())
    if n_bad > 0:
        logger.warning(f"OHLC validation: {n_bad} rows with invalid relationships ({n_bad/len(df)*100:.1f}%)")
    else:
        logger.info("OHLC validation: all rows consistent")
    return n_bad


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch real daily data from Yahoo Finance")
    parser.add_argument("--start", type=str, default="2025-01-01")
    parser.add_argument("--end", type=str, default="2025-12-31")
    parser.add_argument("--universe", type=str, default="watchlist.txt")
    parser.add_argument("--output", type=str, default="output/aggregates/daily_real.parquet")
    parser.add_argument("--sleep", type=float, default=0.3, help="Seconds between requests")
    args = parser.parse_args()

    universe_path = ROOT / args.universe
    if not universe_path.exists():
        logger.error(f"Universe file not found: {universe_path}")
        sys.exit(1)

    symbols = []
    with open(universe_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                symbols.append(line)

    logger.info(f"Fetching {len(symbols)} symbols from {args.start} to {args.end}")

    all_dfs = []
    loaded = 0
    failed = []

    for i, sym in enumerate(symbols):
        logger.info(f"[{i+1}/{len(symbols)}] Fetching {sym}...")
        df = fetch_symbol(sym, args.start, args.end)
        if df is not None and not df.empty:
            all_dfs.append(df)
            loaded += 1
            logger.info(f"  {sym}: {len(df)} rows")
        else:
            failed.append(sym)
            logger.warning(f"  {sym}: no data")
        if i < len(symbols) - 1:
            time.sleep(args.sleep)

    if not all_dfs:
        logger.error("No data fetched for any symbol")
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    n_bad = validate_ohlc(combined)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)

    n_days = combined["timestamp"].dt.date.nunique()
    min_date = combined["timestamp"].min().date()
    max_date = combined["timestamp"].max().date()

    logger.info("=== FETCH COMPLETE ===")
    logger.info(f"Symbols requested: {len(symbols)}")
    logger.info(f"Symbols loaded:    {loaded}")
    logger.info(f"Symbols failed:    {len(failed)} {failed if failed else ''}")
    logger.info(f"Date range:        {min_date} to {max_date}")
    logger.info(f"Trading days:      {n_days}")
    logger.info(f"Total rows:        {len(combined)}")
    logger.info(f"OHLC invalid:      {n_bad}")
    logger.info(f"Written to:        {out_path}")


if __name__ == "__main__":
    main()
