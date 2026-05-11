#!/usr/bin/env python
"""GDELT Weekly Backfill 2020-2026.

~365 Daten-Samples (1/Woche) → max praktikable Resolution ohne Daily-12GB-Download.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_gdelt_geo_aggregates import (  # noqa: E402
    OUT_DIR,
    compute_day_aggregates,
    fetch_gdelt_day,
)


def sample_weekly(start: str = "20200101", end: str | None = None) -> list[str]:
    """Generate weekly samples — Wednesday of each ISO week."""
    if end is None:
        end_dt = datetime.utcnow()
    else:
        end_dt = datetime.strptime(end, "%Y%m%d")
    start_dt = datetime.strptime(start, "%Y%m%d")
    # Find next Wednesday
    cur = start_dt + timedelta(days=(2 - start_dt.weekday()) % 7)
    samples = []
    while cur < end_dt:
        samples.append(cur.strftime("%Y%m%d"))
        cur += timedelta(weeks=1)
    return samples


def main():
    out_path = OUT_DIR / "weekly_aggregates.parquet"
    existing_dates: set[str] = set()
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        existing_dates = set(existing["sample_date"].astype(str).tolist())
        print(f"Existing weekly cache: {len(existing_dates)} dates")
    else:
        existing = pd.DataFrame()

    sample_dates = sample_weekly("20200101")
    pending = [d for d in sample_dates if d not in existing_dates]
    print(f"Weekly 2020-2026: {len(sample_dates)} targets, {len(pending)} pending")

    new_rows = []
    for i, ymd in enumerate(pending):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(pending)} ({ymd})")
        df = fetch_gdelt_day(ymd)
        agg = compute_day_aggregates(df)
        if agg:
            agg["sample_date"] = ymd
            new_rows.append(agg)
        time.sleep(0.3)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = (
            pd.concat([existing, new_df], ignore_index=True)
            if not existing.empty
            else new_df
        )
        combined = combined.drop_duplicates(subset="sample_date").sort_values(
            "sample_date"
        )
        combined.to_parquet(out_path)
        print(
            f"\nTotal: {len(combined)} weekly samples, "
            f"{combined['sample_date'].iloc[0]} to {combined['sample_date'].iloc[-1]}"
        )
        print(f"Saved -> {out_path}")
    else:
        print("\nNo new data fetched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
