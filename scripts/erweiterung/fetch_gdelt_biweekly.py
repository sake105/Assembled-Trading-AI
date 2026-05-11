#!/usr/bin/env python
"""GDELT Bi-Weekly Backfill 2020-2026 für höhere Composite-Frequenz.

Strategie: 2 Tage/Monat (15ter + 1ster) für 2020-2026 → ~150 zusätzliche Datenpunkte.
Output: data/cache/gdelt/biweekly_aggregates.parquet
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_gdelt_geo_aggregates import (  # noqa: E402
    OUT_DIR,
    compute_day_aggregates,
    fetch_gdelt_day,
)


def sample_biweekly(start_year: int = 2020, end_year: int = 2026) -> list[str]:
    samples = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            samples.append(f"{y:04d}{m:02d}01")
            samples.append(f"{y:04d}{m:02d}15")
    today = datetime.utcnow()
    return [s for s in samples if datetime.strptime(s, "%Y%m%d") < today]


def main():
    out_path = OUT_DIR / "biweekly_aggregates.parquet"
    existing_dates: set[str] = set()
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        existing_dates = set(existing["sample_date"].astype(str).tolist())
        print(f"Existing biweekly cache: {len(existing_dates)} dates")
    else:
        existing = pd.DataFrame()

    sample_dates = sample_biweekly(2020, 2026)
    pending = [d for d in sample_dates if d not in existing_dates]
    print(f"Bi-weekly 2020-2026: {len(sample_dates)} targets, {len(pending)} pending")

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
            f"\nTotal: {len(combined)} biweekly samples, "
            f"{combined['sample_date'].iloc[0]} to {combined['sample_date'].iloc[-1]}"
        )
        print(f"Saved -> {out_path}")
    else:
        print("\nNo new data fetched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
