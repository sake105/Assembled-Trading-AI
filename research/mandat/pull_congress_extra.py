"""Pull fehlender Congress-Symbole (Non-S&P) via EODHD — fuer H-034. KEIN Trial."""

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
OUT = DATA / "prices_congress_extra.parquet"


def main() -> int:
    c = pd.read_parquet(
        ROOT / "data" / "raw" / "insider_congress" / "congress_trades_full.parquet"
    )
    c = c[c["type"] == "buy"]
    have = set(
        pd.read_parquet(DATA / "prices_verdict.parquet", columns=["symbol"])[
            "symbol"
        ].unique()
    )
    todo = sorted(
        {s for s in c["symbol"].dropna().unique() if s not in have and s.isascii()}
    )
    print(f"[START] {len(todo)} non-S&P congress symbols", flush=True)
    frames, fails = [], 0
    for i, sym in enumerate(todo):
        esym = sym.replace(".", "-") + ".US"
        url = (
            f"https://eodhd.com/api/eod/{esym}?api_token={TOK}&fmt=json&from=2012-01-01"
        )
        try:
            rows = json.loads(
                urllib.request.urlopen(
                    urllib.request.Request(url, headers={"User-Agent": "research"}),
                    timeout=45,
                )
                .read()
                .decode()
            )
            if rows:
                df = pd.DataFrame(rows)[["date", "adjusted_close", "volume"]]
                df.columns = ["timestamp", "close", "volume"]
                df["symbol"] = sym
                frames.append(df)
        except Exception:  # noqa: BLE001
            fails += 1
        if (i + 1) % 200 == 0:
            print(f"[OK] {i + 1}/{len(todo)} ({fails} fails)", flush=True)
        time.sleep(0.06)
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out.to_parquet(OUT, index=False)
    print(
        f"[DONE] {out['symbol'].nunique()} symbols, {len(out)} rows -> {OUT} | fails {fails}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
