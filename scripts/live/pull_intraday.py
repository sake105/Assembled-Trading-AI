#!/usr/bin/env python
"""
Pull 1m intraday bars via yfinance and write Parquet files compatible with Sprint8+ pipeline.

Output: data/raw/1min/<SYMBOL>.parquet
Columns: timestamp (UTC, tz-aware), symbol, open, high, low, close, volume
"""

import argparse
import sys
import os
from datetime import datetime, timedelta, timezone

def _ensure_deps():
    try:
        import yfinance  # noqa: F401
    except ImportError:
        print("[LIVE] Missing dependency 'yfinance'. Install with: pip install yfinance==0.2.54", file=sys.stderr)
        sys.exit(2)

_ensure_deps()

import pandas as pd
import numpy as np
import yfinance as yf  # type: ignore


def fetch_1m(symbol: str, days: int = 5) -> pd.DataFrame:
    """
    Fetch last `days` of 1-minute data from Yahoo Finance.
    """
    # yfinance supports period='5d' at 1m granularity
    period = f"{days}d"
    df = yf.download(symbol, period=period, interval="1m", auto_adjust=False, progress=False, prepost=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # Normalize columns
    df = df.rename(columns=str.lower)
    # df index is DatetimeIndex (tz-aware? often UTC or exchange tz). Force UTC.
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone.utc)
    else:
        df.index = df.index.tz_convert(timezone.utc)

    df = df.reset_index().rename(columns={"index": "timestamp"})
    # yfinance names 'Datetime' sometimes; handle both
    if "datetime" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"datetime": "timestamp"})

    keep = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df[keep].copy()
    df.insert(1, "symbol", symbol.upper())

    # basic clean
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "high", "low", "close"])
    df = df.sort_values("timestamp")
    return df


def write_parquet(df: pd.DataFrame, symbol: str, root: str) -> str:
    outdir = os.path.join(root, "data", "raw", "1min")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{symbol.upper()}.parquet")
    # Use pyarrow if available (it is pinned in your requirements)
    df.to_parquet(path, index=False)
    return path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", type=str, default="AAPL,MSFT", help="Comma separated symbols list.")
    p.add_argument("--days", type=int, default=5, help="How many days of 1m data to fetch (<=7 for 1m on Yahoo).")
    p.add_argument("--repo-root", type=str, default=".", help="Repo root where data/ exists.")
    return p.parse_args()


def main():
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("[LIVE] No symbols provided", file=sys.stderr)
        sys.exit(1)

    wrote = []
    for s in symbols:
        print(f"[LIVE] Fetching {s} 1m last {args.days}d…")
        df = fetch_1m(s, days=args.days)
        if df.empty:
            print(f"[LIVE] WARN: no data for {s}")
            continue
        path = write_parquet(df, s, args.repo_root)
        print(f"[LIVE] OK: wrote {path} rows={len(df)}")
        wrote.append(path)

    if not wrote:
        print("[LIVE] ERROR: nothing was written", file=sys.stderr)
        sys.exit(3)
    print("[LIVE] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
