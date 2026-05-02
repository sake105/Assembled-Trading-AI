#!/usr/bin/env python
"""Inspect available price data for paper track runner.

Usage:
    python scripts/inspect_paper_track_data.py
    python scripts/inspect_paper_track_data.py --freq 1d
    python scripts/inspect_paper_track_data.py --freq 5min
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.utils.paths import get_default_price_path


def inspect_data(freq: str = "1d") -> dict:
    """Inspect price data coverage for a given frequency."""
    price_path = get_default_price_path(freq)

    result = {
        "schema_version": "paper.track.data_coverage.v1",
        "freq": freq,
        "price_file": str(price_path),
        "exists": price_path.exists(),
    }

    if not price_path.exists():
        result.update(
            {
                "min_date": None,
                "max_date": None,
                "n_days": 0,
                "n_symbols": 0,
                "symbols": [],
                "recommended_ranges": {},
                "note": f"Price file not found: {price_path}",
            }
        )
        return result

    df = pd.read_parquet(price_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    symbols = sorted(df["symbol"].unique().tolist())
    dates = sorted(df["timestamp"].dt.normalize().unique())
    if not dates:
        result["error"] = "no valid timestamps in price data"
        return result
    min_date = dates[0].strftime("%Y-%m-%d")
    max_date = dates[-1].strftime("%Y-%m-%d")

    result.update(
        {
            "min_date": min_date,
            "max_date": max_date,
            "n_days": len(dates),
            "n_symbols": len(symbols),
            "symbols": symbols,
        }
    )

    ranges = {}
    for window in [10, 30, 60, 90]:
        if len(dates) >= window:
            start = dates[-window].strftime("%Y-%m-%d")
            end = dates[-1].strftime("%Y-%m-%d")
            ranges[f"last_{window}_days"] = {"start": start, "end": end, "days": window}
        else:
            ranges[f"last_{window}_days"] = None

    if len(dates) >= 2:
        ranges["full"] = {"start": min_date, "end": max_date, "days": len(dates)}

    result["recommended_ranges"] = ranges
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect paper track data coverage")
    parser.add_argument("--freq", type=str, default="1d", choices=["1d", "5min"])
    args = parser.parse_args()

    result = inspect_data(args.freq)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
