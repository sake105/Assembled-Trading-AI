"""Pull Welle 28: VIX-Index, EURUSD, Faktor-ETFs. KEIN Trial."""

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

SYMS = [
    "VIX.INDX",
    "EURUSD.FOREX",
    "MTUM.US",
    "USMV.US",
    "QUAL.US",
    "VLUE.US",
    "SCHD.US",
    "NOBL.US",
    "SPMO.US",
]

frames = []
for s in SYMS:
    rows = json.loads(
        urllib.request.urlopen(
            urllib.request.Request(
                f"https://eodhd.com/api/eod/{s}?api_token={TOK}&fmt=json&from=2000-01-01",
                headers={"User-Agent": "research"},
            ),
            timeout=60,
        )
        .read()
        .decode()
    )
    df = pd.DataFrame(rows)[["date", "adjusted_close"]]
    df.columns = ["timestamp", "close"]
    df["symbol"] = s.split(".")[0]
    frames.append(df)
    print(f"[OK] {s}: {len(df)} rows", flush=True)
out = pd.concat(frames, ignore_index=True)
out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
out.to_parquet(DATA / "prices_w28.parquet", index=False)
print(f"[DONE] {out['symbol'].nunique()} symbols -> prices_w28.parquet", flush=True)
