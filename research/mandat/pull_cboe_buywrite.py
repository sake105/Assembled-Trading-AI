"""CBOE-Strategie-Indizes BXM/BXMD/PUT (+SPX-TR) — via EODHD, Fallback CBOE-CSV. KEIN Trial."""

from __future__ import annotations

import io
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

CAND = {
    "BXM": ["BXM.INDX"],
    "BXMD": ["BXMD.INDX"],
    "PUT": ["PUT.INDX"],
    "SP500TR": ["SP500TR.INDX", "SPXTR.INDX"],
    "GSPC": ["GSPC.INDX"],
}
CBOE_CSV = {
    "BXM": "https://cdn.cboe.com/api/global/us_indices/daily_prices/BXM_History.csv",
    "BXMD": "https://cdn.cboe.com/api/global/us_indices/daily_prices/BXMD_History.csv",
    "PUT": "https://cdn.cboe.com/api/global/us_indices/daily_prices/PUT_History.csv",
    "PPUT": "https://cdn.cboe.com/api/global/us_indices/daily_prices/PPUT_History.csv",
    "CLL": "https://cdn.cboe.com/api/global/us_indices/daily_prices/CLL_History.csv",
    "CNDR": "https://cdn.cboe.com/api/global/us_indices/daily_prices/CNDR_History.csv",
}
CAND.update({"PPUT": [], "CLL": [], "CNDR": []})


def eodhd(sym):
    url = f"https://eodhd.com/api/eod/{sym}?api_token={TOK}&fmt=json&from=1980-01-01"
    rows = json.loads(
        urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "research"}), timeout=60
        )
        .read()
        .decode()
    )
    df = pd.DataFrame(rows)[["date", "adjusted_close"]]
    df.columns = ["timestamp", "close"]
    return df


def cboe(url):
    raw = (
        urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 research"}),
            timeout=60,
        )
        .read()
        .decode()
    )
    df = pd.read_csv(io.StringIO(raw))
    df.columns = [c.strip().lower() for c in df.columns]
    dcol = [c for c in df.columns if "date" in c][0]
    vcol = [c for c in df.columns if c != dcol][0]
    df = df[[dcol, vcol]]
    df.columns = ["timestamp", "close"]
    return df


frames = []
for name, syms in CAND.items():
    got, src = None, None
    if name in CBOE_CSV:  # volle Historie bevorzugt
        try:
            got, src = cboe(CBOE_CSV[name]), "CBOE"
        except Exception as e:  # noqa: BLE001
            print(f"[..] CBOE {name}: {str(e)[:60]}", flush=True)
    if got is None:
        for s in syms:
            try:
                got, src = eodhd(s), f"EODHD:{s}"
                break
            except Exception as e:  # noqa: BLE001
                print(f"[..] {s}: {str(e)[:60]}", flush=True)
    if got is None:
        print(f"[FAIL] {name}", flush=True)
        continue
    got["timestamp"] = pd.to_datetime(
        got["timestamp"], utc=True, errors="coerce", format="mixed"
    )
    got["close"] = pd.to_numeric(got["close"], errors="coerce")
    got = got.dropna()
    got["symbol"] = name
    frames.append(got)
    print(
        f"[OK-{src}] {name}: {len(got)} rows {got['timestamp'].iloc[0].date()} -> {got['timestamp'].iloc[-1].date()}",
        flush=True,
    )

out = pd.concat(frames, ignore_index=True)
out.to_parquet(DATA / "prices_cboe_buywrite.parquet", index=False)
print(f"[DONE] {sorted(out['symbol'].unique())}", flush=True)
