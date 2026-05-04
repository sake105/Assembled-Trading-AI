from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import pandas as pd
from common.io_utils import http_get_text, to_parquet

# SDW CSV Example (EUR->USD Daily):
# https://sdw.ecb.europa.eu/quickviewexport.do?trans=N&node=2018794&SERIES_KEY=120.EXR.D.USD.EUR.SP00.A&type=csv
# Wir nutzen die generische SDW API (CSV), hier Demo: EURUSD, EURGBP.


SERIES = {
    "EURUSD": "120.EXR.D.USD.EUR.SP00.A",
    "EURGBP": "120.EXR.D.GBP.EUR.SP00.A",
}


BASE = "https://sdw.ecb.europa.eu/quickviewexport.do?trans=N&node=2018794&SERIES_KEY={series}&type=csv"


def fetch(series_key: str) -> pd.DataFrame:
    url = BASE.format(series=series_key)
    csv_txt = http_get_text(url)
    # CSV hat Headerzeilen mit Metadaten -> ab der letzten Headerzeile einlesen
    lines = [  # noqa: F841
        ln
        for ln in csv_txt.splitlines()
        if ";" in ln and ln.split(";")[0].strip().isdigit() is False or ln[0].isdigit()
    ]
    # Simplifizierter Parser: wir suchen die Data-Section -- als Fallback lesen wir alles und filtern spaeter
    df = pd.read_csv(StringIO(csv_txt), sep=",")
    # Heuristik: Spaltennamen finden
    cand_date = [
        c
        for c in df.columns
        if c.lower().startswith("time")
        or c.lower().startswith("period")
        or c.lower().startswith("date")
    ]
    cand_val = [
        c
        for c in df.columns
        if c.lower().startswith("obs") or c.lower().startswith("value")
    ]
    if not cand_date or not cand_val:
        # alternativer Versuch: direkt die letzten 2 Spalten
        cand_date = [df.columns[0]]
        cand_val = [df.columns[-1]]
    df = df[[cand_date[0], cand_val[0]]].rename(
        columns={cand_date[0]: "timestamp", cand_val[0]: "close"}
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["open"] = df["high"] = df["low"] = df["close"]
    df["volume"] = 0.0
    return df


def main():
    if len(sys.argv) < 3:
        print("Usage: python pull_ecb_fx.py <pairs_csv> <out_dir>")
        sys.exit(2)
    pairs = sys.argv[1].split(",")
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    dfs = []
    for p in pairs:
        key = SERIES.get(p.upper())
        if not key:
            print(f"WARN: Pair {p} nicht im Demo-Mapping, überspringe.")
            continue
        df = fetch(key)
        df["symbol"] = p.upper()
        df["provider"] = "ecb_sdw"
        df = (
            df[
                [
                    "timestamp",
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "provider",
                ]
            ]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        to_parquet(df, out_dir / f"{p.upper()}.parquet")
        dfs.append(df)
    if dfs:
        big = pd.concat(dfs, ignore_index=True)
        to_parquet(big, out_dir / "fx_ref.parquet")


if __name__ == "__main__":
    main()
