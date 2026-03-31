#!/usr/bin/env python
"""Generate synthetic daily price data for paper track demo/testing.

Creates output/aggregates/daily.parquet with realistic-looking random-walk
prices for symbols in watchlist.txt over a configurable date range.

Usage:
    python scripts/generate_demo_daily.py
    python scripts/generate_demo_daily.py --start 2025-06-01 --end 2025-10-31 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def generate_demo_daily(
    start: str = "2025-06-01",
    end: str = "2025-10-31",
    seed: int = 42,
    universe_file: str = "watchlist.txt",
    output_path: str | None = None,
) -> pd.DataFrame:
    """Generate synthetic daily OHLCV data.

    Produces a deterministic random walk with slight upward drift,
    realistic daily ranges (~1-3%), and synthetic volume.
    """
    rng = np.random.RandomState(seed)

    universe_path = ROOT / universe_file
    if not universe_path.exists():
        symbols = ["AAPL", "MSFT"]
    else:
        symbols = []
        with open(universe_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    symbols.append(line.upper())

    bdays = pd.bdate_range(start=start, end=end, freq="B")
    if len(bdays) == 0:
        raise ValueError(f"No business days in range {start} to {end}")

    rows = []
    for sym in symbols:
        base_price = 50.0 + rng.random() * 200.0
        price = base_price
        for day in bdays:
            daily_return = rng.normal(0.0003, 0.015)
            price *= 1.0 + daily_return
            price = max(price, 1.0)

            intra_range = abs(rng.normal(0.0, 0.01))
            high = price * (1.0 + intra_range)
            low = price * (1.0 - intra_range)
            open_price = price * (1.0 + rng.normal(0, 0.003))
            volume = int(rng.lognormal(12, 1.5))

            rows.append(
                {
                    "timestamp": pd.Timestamp(day, tz="UTC"),
                    "symbol": sym,
                    "open": round(open_price, 4),
                    "high": round(high, 4),
                    "low": round(low, 4),
                    "close": round(price, 4),
                    "volume": volume,
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    if output_path is None:
        output_path = str(ROOT / "output" / "aggregates" / "daily.parquet")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    n_days = df["timestamp"].dt.date.nunique()
    n_symbols = df["symbol"].nunique()
    print(f"Generated {len(df)} rows: {n_symbols} symbols × {n_days} days")
    print(
        f"Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
    )
    print(f"Written to: {out}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo daily price data")
    parser.add_argument("--start", type=str, default="2025-06-01")
    parser.add_argument("--end", type=str, default="2025-10-31")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    generate_demo_daily(
        start=args.start,
        end=args.end,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
