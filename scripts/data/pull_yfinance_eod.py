#!/usr/bin/env python
"""Fetch EOD price data from Yahoo Finance via yfinance.

Usage:
    py -3 scripts/data/pull_yfinance_eod.py [--symbols AAPL,MSFT,...] [--period 5y] [--out-dir data/raw/equities_eod/yfinance]

Default universe: Top-30 US stocks + core macro/sector ETFs.
Output: One parquet file per symbol in out-dir, then reassemble via assemble_eod_daily.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

REPO = Path(__file__).resolve().parents[2]

# Default universe: liquid US equities + macro ETFs
DEFAULT_SYMBOLS = [
    # Top US equities (diversified sectors)
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "BRK-B",
    "JPM",
    "JNJ",
    "V",
    "UNH",
    "XOM",
    "PG",
    "MA",
    "HD",
    "CVX",
    "MRK",
    "ABBV",
    "LLY",
    "PEP",
    "KO",
    "COST",
    "AVGO",
    "WMT",
    "MCD",
    "CRM",
    "TMO",
    "ADBE",
    "NFLX",
    # Broad market ETFs
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    # Sector ETFs
    "XLF",
    "XLK",
    "XLE",
    "XLV",
    "XLI",
    "XLU",
    "XLP",
    "XLY",
    # International / EM
    "EFA",
    "EEM",
    "VWO",
    # Bonds / rates
    "TLT",
    "IEF",
    "SHY",
    "HYG",
    # Commodities / alternatives
    "GLD",
    "SLV",
    "USO",
]


def fetch_symbol(symbol: str, period: str = "5y") -> pd.DataFrame | None:
    """Fetch OHLCV data for a single symbol."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, auto_adjust=True)
        if df.empty:
            print(f"[yfinance] WARN: No data for {symbol}", file=sys.stderr)
            return None

        df = df.reset_index()
        df = df.rename(
            columns={
                "Date": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        # Ensure UTC timezone
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["symbol"] = symbol

        # Keep only standard OHLCV columns
        cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        df = df[[c for c in cols if c in df.columns]]

        # Drop rows with NaN close
        df = df.dropna(subset=["close"])

        return df

    except Exception as e:
        print(f"[yfinance] ERROR fetching {symbol}: {e}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch EOD prices from Yahoo Finance")
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols (default: built-in universe)",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="5y",
        help="yfinance period string (default: 5y)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO / "data" / "raw" / "equities_eod" / "yfinance"),
        help="Output directory for per-symbol parquet files",
    )
    args = parser.parse_args()

    symbols = args.symbols.split(",") if args.symbols else DEFAULT_SYMBOLS
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[yfinance] Fetching {len(symbols)} symbols, period={args.period}")
    print(f"[yfinance] Output: {out_dir}")

    success = 0
    failed = []

    for i, symbol in enumerate(symbols, 1):
        print(f"[yfinance] [{i}/{len(symbols)}] {symbol}...", end=" ")
        df = fetch_symbol(symbol, period=args.period)
        if df is not None and not df.empty:
            out_path = out_dir / f"{symbol}.parquet"
            df.to_parquet(out_path, index=False)
            print(
                f"OK ({len(df)} rows, {df['timestamp'].min().date()} to {df['timestamp'].max().date()})"
            )
            success += 1
        else:
            print("FAILED")
            failed.append(symbol)

    print(f"\n[yfinance] [DONE] {success}/{len(symbols)} symbols fetched")
    if failed:
        print(f"[yfinance] Failed: {', '.join(failed)}")

    # Also create combined file for quick loading
    all_files = list(out_dir.glob("*.parquet"))
    if all_files:
        frames = [pd.read_parquet(f) for f in all_files]
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp", "symbol"]).sort_values(
            ["symbol", "timestamp"]
        )
        combined_path = out_dir / "_combined.parquet"
        combined.to_parquet(combined_path, index=False)
        print(
            f"[yfinance] Combined: {combined_path} ({len(combined)} rows, {combined['symbol'].nunique()} symbols)"
        )


if __name__ == "__main__":
    main()
