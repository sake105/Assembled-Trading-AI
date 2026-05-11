#!/usr/bin/env python
"""Fetch volles 200-Ticker-Universe für 2007-2026.

Cache in data/cache/yfinance_long/. Tickers aus configs/universes/full_us_universe.txt.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

START = "2007-01-01"
END = "2026-05-08"
OUT_DIR = Path("data/cache/yfinance_long")


def main():
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed")
        return 1

    universe_file = Path("configs/universes/full_us_universe.txt")
    if not universe_file.exists():
        print(f"ERROR: {universe_file} not found")
        return 1
    universe = [
        line.strip()
        for line in universe_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    print(f"Universe size: {len(universe)} tickers")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Skip already cached
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
    print(f"Already cached: {cached}/{len(universe)}")
    print(f"Pending: {len(pending)}")

    if not pending:
        print("All cached — nothing to fetch.")
        return 0

    # Batch fetch (yfinance erlaubt bis ~100 Symbole, splitten)
    batch_size = 50
    success = 0
    failed = []
    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        print(
            f"\nBatch {i // batch_size + 1}/{(len(pending) + batch_size - 1) // batch_size}: {batch[:5]}..."
        )
        try:
            data = yf.download(
                batch,
                start=START,
                end=END,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=False,
            )
        except Exception as e:
            print(f"  Batch failed: {e}")
            failed.extend(batch)
            continue

        if data is None or data.empty:
            failed.extend(batch)
            continue

        for sym in batch:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    sub = (
                        data[sym].dropna(how="all")
                        if sym in data.columns.levels[0]
                        else None
                    )
                else:
                    sub = data
                if sub is None or sub.empty or len(sub) < 500:
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
                sub.to_parquet(OUT_DIR / f"{sym}.parquet")
                success += 1
            except Exception as e:
                print(f"  {sym}: ERROR {e}")
                failed.append(sym)

    print(f"\nNew Success: {success}/{len(pending)}")
    print(f"Total cached: {cached + success}/{len(universe)}")
    if failed:
        print(f"Failed (first 20): {failed[:20]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
