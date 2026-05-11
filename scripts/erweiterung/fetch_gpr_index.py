#!/usr/bin/env python
"""Fetch Caldara-Iacoviello Geopolitical Risk Index (GPR).

Source: https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls

Coverage:
- GPR: Country-aggregate GPR (Daily, 1985+)
- GPRH: Historical GPR (Monthly, 1900+)
- GPRD: Daily GPR (~1985+)

Caldara & Iacoviello (2018), "Measuring Geopolitical Risk", FED Working Paper.

Cache: data/cache/gpr/gpr.parquet
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
OUT_DIR = Path("data/cache/gpr")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {URL} ...")
    r = requests.get(URL, timeout=30)
    if r.status_code != 200:
        print(f"FAIL: HTTP {r.status_code}")
        return 1
    print(f"OK: {len(r.content)} bytes")

    # Excel hat mehrere Sheets — lass mich alle anschauen
    xls = pd.ExcelFile(BytesIO(r.content))
    print(f"Sheets: {xls.sheet_names}")

    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        print(f"\n--- Sheet: {sheet} ---")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)[:10]}")
        print(df.head(3))

        # Normalize: ensure date column
        date_cols = [c for c in df.columns if "date" in str(c).lower() or "month" in str(c).lower() or "year" in str(c).lower()]
        if date_cols:
            df["date"] = pd.to_datetime(df[date_cols[0]], errors="coerce")
        elif isinstance(df.iloc[0, 0], pd.Timestamp):
            df["date"] = pd.to_datetime(df.iloc[:, 0])
        else:
            print(f"  WARN: no date column found in {sheet}")
            continue

        out_path = OUT_DIR / f"{sheet.lower().replace(' ', '_').replace('-', '_')}.parquet"
        df.to_parquet(out_path)
        print(f"  Saved -> {out_path}")

    print("\n[OK] GPR data cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
