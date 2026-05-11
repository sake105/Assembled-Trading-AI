#!/usr/bin/env python
"""GDELT-Multi-Decade Geo-Aggregates Backfill.

Strategie
---------
Statt FULL-density (96 files × 365 days × 14y = 484k Files) holen wir
**1 Tag/Monat** aus GDELT 1.0 events. Pro Tag: 5-10MB. Extrahieren nur
Aggregate (mean tone, mean goldstein, event count). Output: Monthly-Series.

Schema GDELT 1.0 Events:
- Col 1: GLOBALEVENTID
- Col 2: SQLDATE (YYYYMMDD)
- Col 27: EventBaseCode (CAMEO-Codes 14-20 = Conflict)
- Col 29: NumMentions
- Col 30: NumSources
- Col 31: NumArticles
- Col 32: AvgTone (sentiment -100 to +100)
- Col 35: GoldsteinScale (-10 to +10, conflict intensity)

Cache: data/cache/gdelt/monthly_aggregates.parquet
"""

from __future__ import annotations

import io
import time
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

OUT_DIR = Path("data/cache/gdelt")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Conflict-CAMEO-Codes (14=protest, 15=force exhibition, 16=relations reduction,
# 17=coerce, 18=assault, 19=fight, 20=mass violence)
CONFLICT_CAMEO_PREFIXES = ("14", "15", "16", "17", "18", "19", "20")

GDELT_DAILY_URL = "http://data.gdeltproject.org/events/{ymd}.export.CSV.zip"


def fetch_gdelt_day(ymd: str, timeout: int = 30) -> pd.DataFrame | None:
    """Fetch ein GDELT-Daily-Events-File und parse zu Aggregates."""
    url = GDELT_DAILY_URL.format(ymd=ymd)
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return None
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            names = z.namelist()
            if not names:
                return None
            with z.open(names[0]) as f:
                df = pd.read_csv(
                    f,
                    sep="\t",
                    header=None,
                    usecols=[1, 26, 28, 29, 30, 31, 34],
                    names=[
                        "sqldate",
                        "eventcode",
                        "num_mentions",
                        "num_sources",
                        "num_articles",
                        "avg_tone",
                        "goldstein",
                    ],
                    dtype={"eventcode": str},
                    low_memory=False,
                    on_bad_lines="skip",
                )
        return df
    except Exception as e:
        print(f"  {ymd}: ERROR {str(e)[:60]}")
        return None


def compute_day_aggregates(df: pd.DataFrame) -> dict:
    """Berechne Tages-Aggregate aus GDELT-Events."""
    if df is None or df.empty:
        return {}
    # Numeric coercion gegen korrupte Rows
    for c in ("num_mentions", "avg_tone", "goldstein"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Conflict-Events
    if "eventcode" in df.columns:
        is_conflict = df["eventcode"].fillna("").str[:2].isin(CONFLICT_CAMEO_PREFIXES)
    else:
        is_conflict = pd.Series(False, index=df.index)
    return {
        "n_events": int(len(df)),
        "n_conflict_events": int(is_conflict.sum()),
        "mean_tone": (
            float(df["avg_tone"].mean()) if "avg_tone" in df.columns else float("nan")
        ),
        "mean_goldstein": (
            float(df["goldstein"].mean()) if "goldstein" in df.columns else float("nan")
        ),
        "min_goldstein": (
            float(df["goldstein"].min()) if "goldstein" in df.columns else float("nan")
        ),
        "total_mentions": (
            float(df["num_mentions"].sum()) if "num_mentions" in df.columns else 0.0
        ),
        "conflict_share": float(is_conflict.mean()) if len(df) > 0 else 0.0,
    }


def sample_monthly(start_year: int = 2010, end_year: int = 2026) -> list[str]:
    """Generate 1 Tag/Monat: 15ter des Monats (mittig)."""
    samples = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            samples.append(f"{y:04d}{m:02d}15")
    # Filter future dates
    today = datetime.utcnow()
    samples = [s for s in samples if datetime.strptime(s, "%Y%m%d") < today]
    return samples


def main():
    out_path = OUT_DIR / "monthly_aggregates.parquet"
    existing_dates = set()
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        existing_dates = set(existing["sample_date"].astype(str).tolist())
        print(f"Existing cache: {len(existing_dates)} dates")
    else:
        existing = pd.DataFrame()

    sample_dates = sample_monthly(start_year=2010, end_year=2026)
    pending = [d for d in sample_dates if d not in existing_dates]
    print(
        f"Sampling 1 Tag/Monat 2010-2026 to {len(sample_dates)} dates, "
        f"{len(pending)} pending"
    )

    new_rows = []
    for i, ymd in enumerate(pending):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(pending)} ({ymd})")
        df = fetch_gdelt_day(ymd)
        agg = compute_day_aggregates(df)
        if agg:
            agg["sample_date"] = ymd
            new_rows.append(agg)
        time.sleep(0.3)  # gentle rate limiting

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if not existing.empty:
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined = combined.drop_duplicates(subset="sample_date").sort_values(
            "sample_date"
        )
        combined.to_parquet(out_path)
        print(
            f"\nTotal: {len(combined)} monthly samples, "
            f"{combined['sample_date'].iloc[0]} to {combined['sample_date'].iloc[-1]}"
        )
        print(f"Saved -> {out_path}")
    else:
        print("\nNo new data fetched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
