"""Weltmarkt-/Regional-Aktien-ETFs für H-052 (global tax-aware rebalancing). KEIN Trial.

URTH=MSCI World, ACWI=All-Country, EFA=Dev-ex-US, VGK=Europe, EWJ=Japan, EEM=EM.
Alle sind Aktien-ETFs (18,46 % Teilfreistellung). Output: data/prices_world_etf.parquet.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
TOK = os.environ["EODHD_API_TOKEN"]
DATA = Path(__file__).resolve().parent / "data"
SYMS = ["URTH", "ACWI", "EFA", "VGK", "EWJ", "EEM", "SPY"]


def main() -> int:
    frames = []
    for s in SYMS:
        rows = json.loads(
            urllib.request.urlopen(
                urllib.request.Request(
                    f"https://eodhd.com/api/eod/{s}.US?api_token={TOK}&fmt=json&from=1990-01-01",
                    headers={"User-Agent": "research"},
                ),
                timeout=60,
            )
            .read()
            .decode()
        )
        df = pd.DataFrame(rows)[["date", "adjusted_close"]]
        df.columns = ["timestamp", "close"]
        df["symbol"] = s
        frames.append(df)
        print(
            f"[OK] {s}: {len(df)} rows {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}",
            flush=True,
        )
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out.to_parquet(DATA / "prices_world_etf.parquet", index=False)
    print(f"[DONE] {out['symbol'].nunique()} ETFs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
