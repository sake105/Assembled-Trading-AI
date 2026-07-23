"""Mandat data pull #2 — EDGAR XBRL company facts for current S&P 500 members.

PIT-correct via the existing production ingester (available_at from filing
acceptance instants). Free EDGAR data, rate-limited inside the ingester.
Skips symbols already covered by the 2026-06-12 pull (data/raw/fundamentals/
fundamentals_xbrl_full.parquet, 178 symbols) and merges both at the end.

Output: research/mandat/data/fundamentals_sp500.parquet
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:  # SEC UA lives in .env; load it without reading the file into the session
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass
os.environ.setdefault("SEC_USER_AGENT", "Assembled-Trading-AI hans.oertel2@gmail.com")

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT = DATA_DIR / "fundamentals_sp500.parquet"
EXISTING = ROOT / "data" / "raw" / "fundamentals" / "fundamentals_xbrl_full.parquet"
CONSTITUENTS = ROOT / "research" / "mandat_data_constituents.csv"


def main() -> int:
    from src.assembled_core.data.fundamentals_xbrl_ingest import (
        ingest_fundamentals_xbrl,
    )

    cons = pd.read_csv(CONSTITUENTS)
    # EDGAR cik map keys are plain tickers; share classes use '-' there too
    wanted_syms = sorted({s.replace(".", "-") for s in cons["Symbol"].astype(str)})

    existing = pd.read_parquet(EXISTING)
    have = set(existing["symbol"].unique())
    todo = [s for s in wanted_syms if s not in have]
    print(
        f"[START] {len(wanted_syms)} wanted, {len(have)} already pulled, {len(todo)} to fetch",
        flush=True,
    )

    fresh = ingest_fundamentals_xbrl(todo, out_path=DATA_DIR / "_fresh_pull.parquet")
    print(
        f"[OK] fresh pull: {fresh['symbol'].nunique() if len(fresh) else 0} symbols, {len(fresh)} rows",
        flush=True,
    )

    merged = pd.concat([existing, fresh], ignore_index=True)
    merged.to_parquet(OUT, index=False)
    print(
        f"[DONE] merged {merged['symbol'].nunique()} symbols, {len(merged)} rows -> {OUT}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
