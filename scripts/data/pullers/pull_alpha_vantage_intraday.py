#!/usr/bin/env python
# coding: utf-8
"""
pull_alpha_intraday.py
Free intraday via yfinance (keine API Keys). Writes one Parquet per symbol.
Usage:
  python pull_alpha_intraday.py --symbols "AAPL,MSFT" --interval 5m --days 5 --out data/raw/intraday/alphavantage/5min
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import pandas as pd
import yfinance as yf


def dl_intraday(sym: str, interval: str, days: int) -> pd.DataFrame:
    # yfinance: period must be like "5d", "7d", "60d", "730d"
    period = f"{days}d" if days > 0 else "5d"
    df = yf.download(
        sym, period=period, interval=interval, auto_adjust=False, progress=False
    )
    if df.empty:
        return df
    df = df.reset_index().rename(columns=str.lower)
    df["symbol"] = sym
    # expected: DatetimeIndex column name 'Datetime' on some versions -> after reset it's 'datetime'
    # unify:
    if "datetime" in df.columns:
        df = df.rename(columns={"datetime": "timestamp"})
    elif "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    cols = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
    df = df[[c for c in cols if c in df.columns]]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--days", type=int, default=5, help="how many days to pull")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    any_ok = False

    for s in syms:
        try:
            df = dl_intraday(s, args.interval, args.days)
            if df.empty:
                print(f"[INTRA] WARN empty: {s}", file=sys.stderr)
                continue
            fp = out / f"{s}_{args.interval}.parquet"
            df.to_parquet(fp, index=False)
            print(f"[INTRA] OK {s} → {fp}")
            any_ok = True
            time.sleep(0.2)
        except Exception as e:
            print(f"[INTRA] ERR {s}: {e}", file=sys.stderr)

    if not any_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
