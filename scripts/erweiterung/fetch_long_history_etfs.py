#!/usr/bin/env python
"""Fetch 19-Jahres-ETF-Historie für Cross-Asset-Backtest.

Universum:
- Equity: SPY, QQQ, IWM, EFA, EEM
- Bonds: AGG, TLT, HYG
- Commodities: GLD, SLV, DBC

Cache in data/cache/yfinance_long/ (separater Cache, modifiziert nicht
data/cache/yfinance/ — bleibt Mainline-Asset).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

UNIVERSE = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "AGG",
    "TLT",
    "HYG",
    "GLD",
    "SLV",
    "DBC",
]

START = "2007-01-01"
END = "2026-05-08"
OUT_DIR = Path("data/cache/yfinance_long")


def main():
    try:
        import time
        import yfinance as yf
    except ImportError:
        print("yfinance not installed")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(UNIVERSE)} ETFs {START} -> {END} ...")

    # Use yf.download batch — fewer API calls
    print("Trying batch download ...")
    try:
        batch = yf.download(
            UNIVERSE,
            start=START,
            end=END,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=False,
        )
    except Exception as e:
        print(f"Batch download failed: {e}")
        batch = None

    success = []
    failed = []
    if batch is not None and not batch.empty:
        for sym in UNIVERSE:
            out_path = OUT_DIR / f"{sym}.parquet"
            try:
                sub = (
                    batch[sym].dropna(how="all")
                    if isinstance(batch.columns, pd.MultiIndex)
                    else batch
                )
                if sub.empty or len(sub) < 1000:
                    print(f"  {sym}: empty or too short ({len(sub)} rows)")
                    failed.append(sym)
                    continue
                sub.columns = [c.lower() for c in sub.columns]
                sub["symbol"] = sym
                cols_keep = [
                    c
                    for c in ["open", "high", "low", "close", "volume", "symbol"]
                    if c in sub.columns
                ]
                sub = sub[cols_keep]
                sub.index.name = "date"
                sub.to_parquet(out_path)
                print(
                    f"  {sym}: {len(sub)} rows ({sub.index.min().date()} -> {sub.index.max().date()})"
                )
                success.append(sym)
            except Exception as e:
                print(f"  {sym}: ERROR processing {e}")
                failed.append(sym)
    else:
        # Fallback: sequential with delays
        print("Falling back to sequential fetch with 3s delays ...")
        for sym in UNIVERSE:
            out_path = OUT_DIR / f"{sym}.parquet"
            if out_path.exists():
                existing = pd.read_parquet(out_path)
                if not existing.empty and len(existing) > 1000:
                    print(f"  {sym}: already cached -> skip")
                    success.append(sym)
                    continue
            try:
                time.sleep(3)
                t = yf.Ticker(sym)
                df = t.history(start=START, end=END, auto_adjust=True)
                if df.empty or len(df) < 1000:
                    failed.append(sym)
                    continue
                df.columns = [c.lower() for c in df.columns]
                df["symbol"] = sym
                cols_keep = [
                    c
                    for c in ["open", "high", "low", "close", "volume", "symbol"]
                    if c in df.columns
                ]
                df = df[cols_keep]
                df.index.name = "date"
                df.to_parquet(out_path)
                print(f"  {sym}: {len(df)} rows")
                success.append(sym)
            except Exception as e:
                print(f"  {sym}: ERROR {e}")
                failed.append(sym)

    print(f"\nSuccess: {len(success)}/{len(UNIVERSE)}")
    print(f"Failed: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
