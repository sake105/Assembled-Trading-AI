"""Form-4-Breitpull (Vorbereitung §4.6.1-Insider-Patrone) — tranchenweise, resumefähig.

Zieht Form-4-Historie (~30 J via lookback) fuer jemals-S&P-Mitglieder, die im
bestehenden 260-Symbol-Pull fehlen. Pro Aufruf EINE Tranche (default 40 Symbole),
Ergebnis als eigenes Parquet unter data/form4_broad/ — wiederholtes Aufrufen
arbeitet die Liste ab (Resume ueber vorhandene Tranchen-Dateien).
KEIN Trial — reine Datenbeschaffung. SEC-Rate-Limit im Ingester.
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
OUT_DIR = DATA / "form4_broad"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXISTING = ROOT / "data" / "raw" / "insider_congress" / "form4_insider_full.parquet"
CSV_PATH = DATA / "sp500_historical_constituents.csv"
TRANCHE = 40
LOOKBACK_DAYS = 11000  # ~30y


def main() -> int:
    from src.assembled_core.data.edgar_form4_ingest import ingest_form4_for_symbols

    ever: set[str] = set()
    with open(CSV_PATH, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ever.update(t.strip() for t in row["tickers"].split(",") if t.strip())
    have = set(pd.read_parquet(EXISTING, columns=["symbol"])["symbol"].unique())
    done: set[str] = set()
    for p in OUT_DIR.glob("tranche_*.parquet"):
        done.update(pd.read_parquet(p, columns=["symbol"])["symbol"].unique())
        done.update(Path(p).stem.split("__")[1:])  # symbols with zero rows
    todo = sorted(ever - have - done)
    if not todo:
        print("[DONE] nothing left to pull", flush=True)
        return 0
    batch = todo[:TRANCHE]
    n_t = len(list(OUT_DIR.glob("tranche_*.parquet")))
    print(
        f"[START] tranche {n_t + 1}: {len(batch)} symbols ({len(todo)} remaining total)",
        flush=True,
    )
    df = ingest_form4_for_symbols(
        batch,
        lookback_days=LOOKBACK_DAYS,
        out_path=OUT_DIR / "_last_raw.parquet",
    )
    name = f"tranche_{n_t + 1:03d}__" + "__".join(batch)
    # filename length guard
    out = OUT_DIR / (name[:180] + ".parquet")
    df.to_parquet(out, index=False)
    print(
        f"[DONE] tranche {n_t + 1}: {len(df)} rows, {df['symbol'].nunique() if len(df) else 0} symbols -> {out.name}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
