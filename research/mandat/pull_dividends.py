"""Dividenden-Pull (EODHD /div) fuer alle Verdict-Symbole + Internationals.

Output: research/mandat/data/dividends.parquet (ex_date UTC, symbol, dividend/share).
Basis fuer die Dividendensteuer-Erweiterung der Engines. KEIN Trial.
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
OUT = DATA / "dividends.parquet"
EXTRA = ["SPY", "EFA", "EEM", "IEF", "AGG"]


def main() -> int:
    prices = pd.read_parquet(DATA / "prices_verdict.parquet", columns=["symbol"])
    syms = sorted(set(prices["symbol"].unique()) | set(EXTRA))
    print(f"[START] dividends for {len(syms)} symbols", flush=True)
    frames, fails = [], []
    for i, sym in enumerate(syms):
        esym = sym.replace(".", "-") + ".US"
        url = (
            f"https://eodhd.com/api/div/{esym}?api_token={TOK}&fmt=json&from=1995-01-01"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research"})
            rows = json.loads(urllib.request.urlopen(req, timeout=45).read().decode())
            if rows:
                df = pd.DataFrame(rows)
                # value = adjusted (per current shares), unadjustedValue = declared
                col = "value" if "value" in df.columns else "dividends"
                df = df[["date", col]].rename(
                    columns={"date": "ex_date", col: "dividend"}
                )
                df["symbol"] = sym
                frames.append(df)
        except Exception:  # noqa: BLE001
            fails.append(sym)
        if (i + 1) % 200 == 0:
            print(f"[OK] {i + 1}/{len(syms)} ({len(fails)} fails)", flush=True)
        time.sleep(0.06)
    out = pd.concat(frames, ignore_index=True)
    out["ex_date"] = pd.to_datetime(out["ex_date"], utc=True)
    out["dividend"] = pd.to_numeric(out["dividend"], errors="coerce")
    out = out.dropna().sort_values(["symbol", "ex_date"]).reset_index(drop=True)
    out.to_parquet(OUT, index=False)
    print(
        f"[DONE] {out['symbol'].nunique()} symbols, {len(out)} dividend events -> {OUT} | fails {len(fails)}: {fails[:12]}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
