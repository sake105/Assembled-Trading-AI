#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LEGACY assembler — überschreibt output/aggregates/daily.parquet aus data/raw.

GEFAHR (GESAMTBEWERTUNG 2026-07-23): data/raw/equities_eod enthält heute nur
noch 2 Symbole (AAPL/MSFT-Reste) — ein Aufruf würde den 197-Symbol-Live-Cache
des Paper-Piloten mit diesem Restbestand ÜBERSCHREIBEN. Der Pfad ist tot; die
Live-Writer sind scripts/ops/refresh_daily_cache_from_eodhd.py bzw.
refresh_daily_cache_from_panel.py. Ausführung erfordert --force-legacy.
"""

import glob
import sys
from pathlib import Path

import pandas as pd

if "--force-legacy" not in sys.argv:
    print(
        "[ABBRUCH] assemble_eod_daily.py ist ein Legacy-Writer und wuerde "
        "daily.parquet mit dem 2-Symbol-Raw-Restbestand ueberschreiben. "
        "Live-Refresh: scripts/ops/refresh_daily_cache_from_eodhd.py. "
        "Wirklich gewollt: --force-legacy anhaengen.",
        file=sys.stderr,
    )
    sys.exit(1)
sys.argv.remove("--force-legacy")

REPO = Path(__file__).resolve().parents[2]  # <repo>/scripts/data/ -> up 2
RAW_ROOTS = [
    REPO / "data" / "raw" / "equities_eod" / "stooq",
    REPO / "data" / "raw" / "equities_eod" / "yfinance",
]
OUT_DIR = REPO / "output" / "aggregates"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "daily.parquet"


def load_one_parquet(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    # minimal required cols normalisieren
    # gängige Varianten abfangen (adj close etc. sind optional)
    for need in ["timestamp", "symbol", "open", "high", "low", "close"]:
        if need not in df.columns:
            raise ValueError(f"{p}: benötigte Spalte fehlt: {need}")

    # Timestamp -> UTC tz-aware
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = ts

    # Optional: volume, adjclose normalisieren
    if "adjclose" in df.columns and "adj_close" not in df.columns:
        df = df.rename(columns={"adjclose": "adj_close"})
    if "adj_close" not in df.columns and "close" in df.columns:
        # wenn keine adj_close vorhanden, setze = close (harmlos für Start)
        df["adj_close"] = df["close"]

    keep = [
        "timestamp",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def main():
    files = []
    for root in RAW_ROOTS:
        files += glob.glob(str(root / "*.parquet"))

    if not files:
        print("[EOD] Keine Parquet-Dateien gefunden unter:", RAW_ROOTS, file=sys.stderr)
        sys.exit(2)

    frames = []
    for f in files:
        try:
            frames.append(load_one_parquet(Path(f)))
        except Exception as e:
            print(f"[EOD] WARN skip {f}: {e}", file=sys.stderr)

    if not frames:
        print("[EOD] Keine gültigen Dateien nach Parsing.", file=sys.stderr)
        sys.exit(2)

    df = pd.concat(frames, axis=0, ignore_index=True)
    # Deduplikation + Sortierung
    df = df.drop_duplicates(subset=["timestamp", "symbol"]).sort_values(
        ["symbol", "timestamp"]
    )

    # Normalize timestamps to midnight UTC (yfinance uses 04:00 UTC for US markets)
    if not df.empty:
        df["timestamp"] = df["timestamp"].dt.normalize()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)
    print(
        f"[EOD] [OK] written: {OUT_FILE} | rows={len(df)} symbols={df['symbol'].nunique()}"
    )


if __name__ == "__main__":
    main()
