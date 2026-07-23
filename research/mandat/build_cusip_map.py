"""CUSIP->Ticker-Map aus SEC Fails-to-Deliver-Dateien (2 je Jahr, 2013-2026).

FTD-Dateien listen CUSIP|SYMBOL|DESCRIPTION fuer alle CNS-Titel inkl. toter.
Zeitgestempelt (year), damit Ticker-Recycling aufloesbar bleibt.
Output: data/cusip_ticker_map.parquet (cusip, symbol, year, description).
KEIN Trial.
"""

from __future__ import annotations

import io
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

UA = {"User-Agent": "Assembled-Trading-AI hans.oertel2@gmail.com"}
DATA = Path(__file__).resolve().parent / "data"
OUT = DATA / "cusip_ticker_map.parquet"


def fetch(year: int, month: int) -> pd.DataFrame | None:
    url = f"https://www.sec.gov/files/data/fails-deliver-data/cnsfails{year}{month:02d}a.zip"
    try:
        raw = urllib.request.urlopen(
            urllib.request.Request(url, headers=UA), timeout=120
        ).read()
        z = zipfile.ZipFile(io.BytesIO(raw))
        df = pd.read_csv(
            z.open(z.namelist()[0]),
            sep="|",
            dtype=str,
            usecols=["CUSIP", "SYMBOL", "DESCRIPTION"],
            on_bad_lines="skip",
            encoding_errors="ignore",
        )
        df = df.dropna(subset=["CUSIP", "SYMBOL"]).drop_duplicates(["CUSIP", "SYMBOL"])
        df["year"] = year
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] {year}-{month:02d}: {exc}", flush=True)
        return None


def main() -> int:
    frames = []
    for year in range(2013, 2027):
        for month in (3, 9):
            df = fetch(year, month)
            if df is not None:
                frames.append(df)
                print(
                    f"[OK] {year}-{month:02d}: {len(df)} cusip-symbol pairs", flush=True
                )
            time.sleep(1)
    out = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["CUSIP", "SYMBOL", "year"]
    )
    out.to_parquet(OUT, index=False)
    print(
        f"[DONE] {out['CUSIP'].nunique()} cusips, {len(out)} rows -> {OUT}", flush=True
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
