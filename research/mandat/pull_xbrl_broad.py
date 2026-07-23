"""XBRL-Breitpull — companyfacts fuer alle jemals-S&P-Mitglieder ohne XBRL-Daten.

Erweitert fundamentals_sp500.parquet (540 Symbole) auf das volle jemals-Universum.
KEIN Trial — Datenbeschaffung. XBRL-Historie beginnt ~2009 (Pflicht) — dokumentiert.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass
os.environ.setdefault("SEC_USER_AGENT", "Assembled-Trading-AI hans.oertel2@gmail.com")

DATA = Path(__file__).resolve().parent / "data"
FUND = DATA / "fundamentals_sp500.parquet"
CSV_PATH = DATA / "sp500_historical_constituents.csv"


def main() -> int:
    from src.assembled_core.data.fundamentals_xbrl_ingest import (
        ingest_fundamentals_xbrl,
    )

    ever: set[str] = set()
    with open(CSV_PATH, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ever.update(t.strip() for t in row["tickers"].split(",") if t.strip())
    existing = pd.read_parquet(FUND)
    have = set(existing["symbol"].unique())
    todo = sorted(s.replace(".", "-") for s in ever if s not in have)
    print(
        f"[START] {len(ever)} ever-members, {len(have)} have XBRL, {len(todo)} to fetch",
        flush=True,
    )

    fresh = ingest_fundamentals_xbrl(todo, out_path=DATA / "_xbrl_broad_fresh.parquet")
    print(
        f"[OK] fresh: {fresh['symbol'].nunique() if len(fresh) else 0} symbols, {len(fresh)} rows",
        flush=True,
    )
    merged = pd.concat([existing, fresh], ignore_index=True)
    merged.to_parquet(FUND, index=False)
    print(
        f"[DONE] merged {merged['symbol'].nunique()} symbols, {len(merged)} rows -> {FUND}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
