"""Sentiment-Pull (EODHD) fuer das Verdict-Universum — fuer H-038. KEIN Trial.

Output: research/mandat/data/sentiment.parquet (date UTC, symbol, normalized, count).
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
OUT = DATA / "sentiment.parquet"


def main() -> int:
    syms = sorted(
        pd.read_parquet(DATA / "prices_verdict.parquet", columns=["symbol"])[
            "symbol"
        ].unique()
    )
    print(f"[START] sentiment for {len(syms)} symbols", flush=True)
    frames, empty = [], 0
    for i, sym in enumerate(syms):
        esym = sym.replace(".", "-") + ".US"
        url = f"https://eodhd.com/api/sentiments?s={esym}&from=2011-01-01&to=2026-07-09&api_token={TOK}&fmt=json"
        try:
            d = json.loads(
                urllib.request.urlopen(
                    urllib.request.Request(url, headers={"User-Agent": "research"}),
                    timeout=45,
                )
                .read()
                .decode()
            )
            rows = d.get(esym, []) if isinstance(d, dict) else []
            if rows:
                df = pd.DataFrame(rows)[["date", "normalized", "count"]]
                df["symbol"] = sym
                frames.append(df)
            else:
                empty += 1
        except Exception:  # noqa: BLE001
            empty += 1
        if (i + 1) % 200 == 0:
            print(f"[OK] {i + 1}/{len(syms)} ({empty} empty)", flush=True)
        time.sleep(0.05)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], utc=True)
    out["normalized"] = pd.to_numeric(out["normalized"], errors="coerce")
    out = out.dropna().sort_values(["symbol", "date"]).reset_index(drop=True)
    out.to_parquet(OUT, index=False)
    print(
        f"[DONE] {out['symbol'].nunique()} symbols, {len(out)} rows -> {OUT} | empty {empty}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
