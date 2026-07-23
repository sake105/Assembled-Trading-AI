"""EU-Blue-Chips (XETRA/LSE/Euronext) für den Welt-Indikator-Sweep. KEIN Trial."""

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
    "SAP.XETRA",
    "SIE.XETRA",
    "ALV.XETRA",
    "BAS.XETRA",
    "BMW.XETRA",
    "DTE.XETRA",
    "MUV2.XETRA",
    "ADS.XETRA",
    "RWE.XETRA",
    "IFX.XETRA",
    "MC.PA",
    "OR.PA",
    "TTE.PA",
    "AIR.PA",
    "BP.LSE",
    "SHEL.LSE",
    "AZN.LSE",
    "HSBA.LSE",
    "ULVR.LSE",
    "RIO.LSE",
]

frames = []
for s in SYMS:
    try:
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
        df["symbol"] = s
        frames.append(df)
        print(f"[OK] {s}: {len(df)}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] {s}: {str(e)[:60]}", flush=True)
out = pd.concat(frames, ignore_index=True)
out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
out.to_parquet(DATA / "prices_eu_stocks.parquet", index=False)
print(f"[DONE] {out['symbol'].nunique()} EU stocks", flush=True)
