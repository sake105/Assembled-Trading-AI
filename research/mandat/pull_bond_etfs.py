"""Bond-ETFs (TLT/IEF/SHY) für Portfolio-Konstruktion Welle 25. KEIN Trial."""

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

for s in ("TLT", "IEF", "SHY", "SLV"):
    rows = json.loads(
        urllib.request.urlopen(
            urllib.request.Request(
                f"https://eodhd.com/api/eod/{s}.US?api_token={TOK}&fmt=json&from=2000-01-01",
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
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.to_parquet(DATA / f"bond_{s}.parquet", index=False)
    print(
        f"[OK] {s}: {len(df)} rows {df['timestamp'].min().date()} -> {df['timestamp'].max().date()}",
        flush=True,
    )
