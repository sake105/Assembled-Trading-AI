#!/usr/bin/env python
"""Resilient yfinance-Long-History-Fetcher (rate-limit aware).

Kleinere Batches (5), Sleeps zwischen Batches, Retry-on-Rate-Limit.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

START = "2007-01-01"
END = "2026-05-08"
OUT_DIR = Path("data/cache/yfinance_long")
BATCH_SIZE = 5
SLEEP_BETWEEN_BATCHES = 8.0  # seconds
RATE_LIMIT_SLEEP = 90.0  # backoff on rate limit
MAX_RATE_LIMIT_RETRIES = 3


def fetch_one(yf, sym: str, retries: int = 0) -> pd.DataFrame | None:
    try:
        df = yf.Ticker(sym).history(
            start=START, end=END, auto_adjust=True, raise_errors=False
        )
    except Exception as e:
        msg = str(e).lower()
        if "rate" in msg and retries < MAX_RATE_LIMIT_RETRIES:
            print(
                f"  {sym}: rate-limited, sleeping {RATE_LIMIT_SLEEP}s "
                f"(retry {retries + 1}/{MAX_RATE_LIMIT_RETRIES})"
            )
            time.sleep(RATE_LIMIT_SLEEP)
            return fetch_one(yf, sym, retries + 1)
        print(f"  {sym}: ERROR {str(e)[:80]}")
        return None
    if df is None or df.empty or len(df) < 500:
        return None
    df.columns = [c.lower() for c in df.columns]
    df["symbol"] = sym
    cols_keep = [
        c
        for c in ["open", "high", "low", "close", "volume", "symbol"]
        if c in df.columns
    ]
    df = df[cols_keep]
    df.index.name = "date"
    return df


def main():
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed")
        return 1

    universe_file = Path("configs/universes/full_us_universe.txt")
    universe = [
        line.strip()
        for line in universe_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pending = []
    cached = 0
    for sym in universe:
        out_path = OUT_DIR / f"{sym}.parquet"
        if out_path.exists():
            try:
                existing = pd.read_parquet(out_path)
                if not existing.empty and len(existing) > 1000:
                    cached += 1
                    continue
            except Exception:
                pass
        pending.append(sym)
    print(f"Universe: {len(universe)}, cached: {cached}, pending: {len(pending)}")

    if not pending:
        return 0

    success = 0
    failed = []
    total_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_i in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_i : batch_i + BATCH_SIZE]
        print(f"\nBatch {batch_i // BATCH_SIZE + 1}/{total_batches}: {batch}")
        for sym in batch:
            df = fetch_one(yf, sym)
            if df is not None:
                df.to_parquet(OUT_DIR / f"{sym}.parquet")
                success += 1
                print(
                    f"  {sym}: {len(df)} days ({df.index.min().date()} to {df.index.max().date()})"
                )
            else:
                failed.append(sym)
            time.sleep(1.0)  # gentle per-symbol pause
        if batch_i + BATCH_SIZE < len(pending):
            print(f"  ... batch done, sleeping {SLEEP_BETWEEN_BATCHES}s")
            time.sleep(SLEEP_BETWEEN_BATCHES)

    print(
        f"\nFinal: cached {cached} + new success {success} = {cached + success} total"
    )
    if failed:
        print(f"Failed: {len(failed)} symbols: {failed[:20]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
