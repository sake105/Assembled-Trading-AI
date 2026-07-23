"""Ground-truth inventory: what intraday price + timestamped event data do we ACTUALLY
have to test an event-driven intraday strategy? Read-only."""

from __future__ import annotations
import glob
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)


def inspect(path, label=""):
    if not os.path.exists(path):
        print(f"[MISSING] {path}")
        return
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"[ERR] {path}: {e}")
        return
    tcol = next(
        (
            c
            for c in df.columns
            if c.lower() in ("timestamp", "date", "datetime", "time", "published_utc")
        ),
        None,
    )
    rng = res = ""
    if tcol:
        t = pd.to_datetime(df[tcol], utc=True, errors="coerce")
        rng = f"{t.min()}..{t.max()}"
        # intraday resolution: median gap within a day
        if t.notna().sum() > 10:
            d = t.dropna().sort_values()
            gaps = d.diff().dt.total_seconds().dropna()
            gaps = gaps[(gaps > 0) & (gaps < 86400)]
            res = (
                f" intraday_gap_median={gaps.median():.0f}s"
                if len(gaps)
                else " (daily?)"
            )
    scol = next((c for c in df.columns if c.lower() in ("symbol", "ticker")), None)
    ns = df[scol].nunique() if scol else "?"
    print(
        f"[OK] {label or os.path.basename(path)}: shape={df.shape} syms={ns} cols={list(df.columns)[:8]}"
    )
    print(f"       range={rng}{res}")


print("=== INTRADAY PRICE DATA ===")
for p in (
    "output/aggregates/5min.parquet",
    "output/assembled_intraday/assembled_intraday.parquet",
    "output/aggregates/assembled_intraday_60min.parquet",
    "output/features/base_5min.parquet",
    "output/features/base_1min.parquet",
):
    inspect(p)

print("\n=== EVENT / NEWS / GEOPOLITICAL SOURCES ===")
for p in (
    "output/news_raw.parquet",
    "output/news_sentiment_daily.parquet",
    "data/cache/gpr/sheet1.parquet",
    "data/cache/gdelt/weekly_aggregates.parquet",
):
    inspect(p)

print("\n=== news_alpha module files ===")
for f in sorted(glob.glob("src/assembled_core/events/news_alpha/*.py")):
    n = sum(1 for _ in open(f, encoding="utf-8"))
    print(f"  {os.path.relpath(f)} ({n} lines)")
