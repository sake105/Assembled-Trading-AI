"""Verdict-Universum-Pull — EODHD EOD-Historie fuer ALLE jemals-S&P-500-Mitglieder.

Universum: Union aller Ticker aus der historischen Konstituenten-CSV (1996-2026,
fja05680). EODHD liefert auch DELISTED (verifiziert: SIVB bis 2023-03-09, BBBY).
adjusted_close = split+dividend-adjustiert (Total-Return-nah).

Outputs:
  research/mandat/data/prices_verdict.parquet  (timestamp, symbol, close=adj, volume)
  research/mandat/data/pull_eodhd_report.json  (coverage/fails — Datenqualitaets-Report)
"""

from __future__ import annotations

import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"
OUT = DATA / "prices_verdict.parquet"
REPORT = DATA / "pull_eodhd_report.json"
CSV_PATH = DATA / "sp500_historical_constituents.csv"

sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
import os  # noqa: E402

TOK = os.environ["EODHD_API_TOKEN"]


def all_ever_members() -> list[str]:
    tickers: set[str] = set()
    with open(CSV_PATH, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            tickers.update(t.strip() for t in row["tickers"].split(",") if t.strip())
    return sorted(tickers)


def fetch(sym: str) -> list[dict]:
    # EODHD wants '-' for share classes (BRK.B -> BRK-B.US)
    esym = sym.replace(".", "-") + ".US"
    url = f"https://eodhd.com/api/eod/{esym}?api_token={TOK}&fmt=json&from=1995-01-01"
    req = urllib.request.Request(url, headers={"User-Agent": "research"})
    return json.loads(urllib.request.urlopen(req, timeout=45).read().decode())


def main() -> int:
    tickers = all_ever_members()
    print(f"[START] {len(tickers)} ever-members", flush=True)
    frames: list[pd.DataFrame] = []
    fails: list[str] = []
    for i, sym in enumerate(tickers):
        try:
            rows = fetch(sym)
            if not rows:
                fails.append(sym)
                continue
            df = pd.DataFrame(rows)[["date", "adjusted_close", "volume"]]
            df.columns = ["timestamp", "close", "volume"]
            df["symbol"] = sym
            frames.append(df)
        except Exception:  # noqa: BLE001
            fails.append(sym)
        if (i + 1) % 100 == 0:
            print(f"[OK] {i + 1}/{len(tickers)} ({len(fails)} fails)", flush=True)
        time.sleep(0.07)  # ~14/s, well under 1000/min

    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    out.to_parquet(OUT, index=False)

    g = out.groupby("symbol")["timestamp"].agg(["min", "max"])
    # delisted proxy: history ends >30 days before the pull date
    cutoff = out["timestamp"].max() - pd.Timedelta(days=30)
    n_delisted = int((g["max"] < cutoff).sum())
    report = {
        "tickers_wanted": len(tickers),
        "tickers_got": int(out["symbol"].nunique()),
        "tickers_failed": len(fails),
        "failed_sample": fails[:40],
        "rows": int(len(out)),
        "n_delisted_like": n_delisted,
        "earliest": str(g["min"].min().date()),
        "latest": str(g["max"].max().date()),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[DONE] {report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
