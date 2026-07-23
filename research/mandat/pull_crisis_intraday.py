"""Intraday 5m fuer Crisis-ETFs (H-043): XLE/GLD/ITA/SPY, ~Okt 2020 -> Jul 2026.

EODHD intraday: from/to als UNIX, gefensterte Chunks (5m: sichere 120-Tage-Fenster).
Output: data/intraday_crisis_5m.parquet (symbol, datetime[UTC], open, close). KEIN Trial.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
TOK = os.environ["EODHD_API_TOKEN"]
DATA = Path(__file__).resolve().parent / "data"

SYMS = ["XLE", "GLD", "ITA", "SPY"]
START = 1601510400  # 2020-10-01 UTC
END = 1783641600  # 2026-07-10 UTC
WIN = 120 * 86400  # 120-day windows (safe for 5m)


def get(sym: str, frm: int, to: int):
    url = (
        f"https://eodhd.com/api/intraday/{sym}.US?interval=5m"
        f"&from={frm}&to={to}&api_token={TOK}&fmt=json"
    )
    try:
        return json.loads(
            urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "research"}),
                timeout=60,
            )
            .read()
            .decode()
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] {sym} {frm}: {str(exc)[:80]}", flush=True)
        return []


def main() -> int:
    frames = []
    for sym in SYMS:
        n = 0
        frm = START
        while frm < END:
            to = min(frm + WIN, END)
            rows = get(sym, frm, to)
            if isinstance(rows, list) and rows:
                df = pd.DataFrame(rows)[["datetime", "open", "close"]]
                df["symbol"] = sym
                frames.append(df)
                n += len(df)
            frm = to
            time.sleep(0.15)
        print(f"[OK] {sym}: {n} bars", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True)
    out = out.drop_duplicates(subset=["symbol", "datetime"]).sort_values(
        ["symbol", "datetime"]
    )
    out.to_parquet(DATA / "intraday_crisis_5m.parquet", index=False)
    print(
        f"[DONE] {len(out)} bars, {out['datetime'].min()} -> {out['datetime'].max()}, "
        f"syms {sorted(out['symbol'].unique())}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
