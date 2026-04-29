#!/usr/bin/env python
# coding: utf-8
"""
pull_ecb_fxref.py
Fetch ECB reference rates (EUR base). Produces daily time series for requested pairs.
Usage:
  python pull_ecb_fxref.py --pairs "EURUSD,EURGBP" --out data/raw/fx/ecb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ECB_URL = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pairs", required=True, help="e.g. EURUSD,EURGBP (EUR must be base)"
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    want_ccys = [p[3:] if p.startswith("EUR") else None for p in pairs]
    if any(x is None for x in want_ccys):
        print("[ECB] Only EUR-base pairs supported (EURXXX).", file=sys.stderr)
        sys.exit(2)

    # Load CSV
    df = pd.read_csv(ECB_URL)
    df = df.rename(columns={"Date": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp")

    for p, ccy in zip(pairs, want_ccys):
        if ccy not in df.columns:
            print(f"[ECB] WARN currency not available: {p}", file=sys.stderr)
            continue
        out_df = df[["timestamp", ccy]].rename(columns={ccy: "rate"})
        out_df["pair"] = p
        out_df.to_parquet(out / f"{p}.parquet", index=False)
        print(f"[ECB] OK {p} → {(out / f'{p}.parquet')}")


if __name__ == "__main__":
    main()
