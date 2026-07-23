"""Resolve the TRUE price-history span available for the mill. Read-only."""

from __future__ import annotations
import glob
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)


def cov(path):
    if not os.path.exists(path):
        print(f"[MISSING] {path}")
        return
    df = pd.read_parquet(path)
    tcol = (
        "timestamp"
        if "timestamp" in df.columns
        else ("date" if "date" in df.columns else None)
    )
    print(f"\n[OK] {path} shape={df.shape} cols={list(df.columns)}")
    if tcol:
        t = pd.to_datetime(df[tcol], utc=True, errors="coerce")
        print(f"  {tcol}: {t.min()} -> {t.max()}")
    if "symbol" in df.columns and tcol:
        g = df.assign(_t=t).groupby("symbol")["_t"].agg(["min", "max", "count"])
        print("  n symbols:", len(g))
        print("  per-symbol START date distribution:")
        print(g["min"].dt.year.value_counts().sort_index())
        print("  per-symbol row-count describe:\n", g["count"].describe())
        # how many symbols have >=1000 trading days (~4y)?
        print("  symbols with >=1000 rows:", int((g["count"] >= 1000).sum()))
        print("  symbols with >=1500 rows:", int((g["count"] >= 1500).sum()))


print("=== output/aggregates/daily.parquet ===")
cov("output/aggregates/daily.parquet")

# hunt for any other/larger price caches
print("\n=== hunting other price parquets ===")
cands = []
for pat in (
    "data/**/*.parquet",
    "output/**/*.parquet",
    "**/daily*.parquet",
    "**/prices*.parquet",
):
    cands += glob.glob(pat, recursive=True)
cands = sorted(set(cands))
for p in cands:
    if "raw/fundamentals" in p.replace("\\", "/") or "form4_by_symbol" in p.replace(
        "\\", "/"
    ):
        continue
    try:
        df = pd.read_parquet(p, columns=None)
        tcol = (
            "timestamp"
            if "timestamp" in df.columns
            else ("date" if "date" in df.columns else None)
        )
        rng = ""
        if tcol:
            t = pd.to_datetime(df[tcol], utc=True, errors="coerce")
            rng = f"{t.min().date() if t.notna().any() else '?'}..{t.max().date() if t.notna().any() else '?'}"
        print(f"  {p}  shape={df.shape} time={rng}")
    except Exception as e:
        print(f"  {p}  [unreadable: {e}]")
