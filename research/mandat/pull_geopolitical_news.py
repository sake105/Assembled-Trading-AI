"""Geopolitische News-Intensitaet aus EODHD-Artikeln (fuer H-039). KEIN Trial.

Union der Konflikt-Tags, paginiert (offset), dedupe per link -> tageszaehler.
Output: research/mandat/data/geopol_intensity.parquet (date, n_articles).
Zusatz: Crisis-ETF-Preise (XLE/GLD/ITA/SPY) -> data/prices_crisis.parquet.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
TOK = os.environ["EODHD_API_TOKEN"]
DATA = Path(__file__).resolve().parent / "data"

TAGS = [
    "WAR",
    "RUSSIA",
    "UKRAINE",
    "MIDDLE EAST",
    "IRAN",
    "ISRAEL",
    "SANCTIONS",
    "MILITARY",
    "GEOPOLITICAL RISK",
]
CRISIS = ["XLE", "GLD", "ITA", "SPY"]


def get(url):
    return json.loads(
        urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "research"}), timeout=90
        )
        .read()
        .decode()
    )


def pull_news() -> pd.DataFrame:
    seen = {}
    for tag in TAGS:
        off, got_total = 0, 0
        while off < 20000:
            url = (
                f"https://eodhd.com/api/news?t={urllib.parse.quote(tag)}"
                f"&limit=1000&offset={off}&from=2015-01-01&to=2026-07-09&api_token={TOK}&fmt=json"
            )
            try:
                rows = get(url)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] {tag} off{off}: {exc}", flush=True)
                break
            if not rows:
                break
            for r in rows:
                seen[r.get("link", r.get("date", "") + r.get("title", ""))] = r["date"][
                    :10
                ]
            got_total += len(rows)
            off += 1000
            time.sleep(0.1)
            if len(rows) < 1000:
                break
        print(f"[OK] {tag}: {got_total} articles cumulated", flush=True)
    df = pd.DataFrame({"date": list(seen.values())})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    daily = df.groupby("date").size().rename("n_articles").reset_index()
    daily.to_parquet(DATA / "geopol_intensity.parquet", index=False)
    print(
        f"[DONE] intensity: {len(daily)} days {daily['date'].min().date()} -> {daily['date'].max().date()}, {len(seen)} unique articles",
        flush=True,
    )
    return daily


def pull_crisis():
    frames = []
    for s in CRISIS:
        rows = get(
            f"https://eodhd.com/api/eod/{s}.US?api_token={TOK}&fmt=json&from=2004-01-01"
        )
        df = pd.DataFrame(rows)[["date", "adjusted_close"]]
        df.columns = ["timestamp", "close"]
        df["symbol"] = s
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out.to_parquet(DATA / "prices_crisis.parquet", index=False)
    print(f"[DONE] crisis prices: {out['symbol'].unique().tolist()}", flush=True)


def main() -> int:
    pull_news()
    pull_crisis()
    return 0


if __name__ == "__main__":
    sys.exit(main())
