"""FX-Majors für den Mega-Search-FX-Strang. KEIN Trial."""

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

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"]
frames = []
for p in PAIRS:
    rows = json.loads(
        urllib.request.urlopen(
            urllib.request.Request(
                f"https://eodhd.com/api/eod/{p}.FOREX?api_token={TOK}&fmt=json&from=2002-01-01",
                headers={"User-Agent": "research"},
            ),
            timeout=60,
        )
        .read()
        .decode()
    )
    df = pd.DataFrame(rows)[["date", "adjusted_close"]]
    df.columns = ["timestamp", "close"]
    df["symbol"] = p
    frames.append(df)
    print(f"[OK] {p}: {len(df)}", flush=True)
out = pd.concat(frames, ignore_index=True)
out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
out.to_parquet(DATA / "prices_fx_majors.parquet", index=False)
print(f"[DONE] {out['symbol'].nunique()} pairs", flush=True)
