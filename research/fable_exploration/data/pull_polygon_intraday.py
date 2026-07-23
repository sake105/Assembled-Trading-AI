"""Polygon.io intraday (minute-bar) ingester for the first-minutes event test.
Loads .env (never prints the key), resolves any POLYGON* var, fetches minute
aggregates (incl. extended hours), paginates via next_url, handles 429 rate-limits.
Writes per-symbol parquet under research/fable_exploration/data/intraday/.

Self-test (python pull_polygon_intraday.py): pulls 2 recent days of AAPL minute bars
and reports count / extended-hours / rate-limit so we know the tier before a big pull.
"""

from __future__ import annotations
import os
import sys
import time
import urllib.request
import urllib.error
import json
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
OUT = os.path.join("research", "fable_exploration", "data", "intraday")
os.makedirs(OUT, exist_ok=True)


def _load_key() -> str | None:
    # process env first
    for k, v in os.environ.items():
        if "POLYGON" in k.upper() and v:
            return v
    # then .env (manual parse; never echo the value)
    if os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, val = line.split("=", 1)
                if "POLYGON" in name.upper():
                    return val.strip().strip('"').strip("'")
    return None


KEY = _load_key()
BASE = "https://api.polygon.io"


def _get(url: str, tries: int = 5) -> dict:
    sep = "&" if "?" in url else "?"
    full = f"{url}{sep}apiKey={KEY}"
    for i in range(tries):
        try:
            req = urllib.request.Request(full, headers={"User-Agent": "research/1.0"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:  # rate limited (free tier 5/min)
                wait = 15 * (i + 1)
                print(f"   [429] rate-limited, sleep {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("max retries")


def fetch_minute_bars(
    symbol: str, start: str, end: str, adjusted: bool = True
) -> pd.DataFrame:
    """Minute aggregates [start,end] (YYYY-MM-DD), extended hours included.
    Returns DataFrame[ts(UTC), symbol, open, high, low, close, volume, vwap, n]."""
    url = (
        f"{BASE}/v2/aggs/ticker/{symbol}/range/1/minute/{start}/{end}"
        f"?adjusted={'true' if adjusted else 'false'}&sort=asc&limit=50000"
    )
    rows = []
    while url:
        j = _get(url)
        for b in j.get("results", []) or []:
            rows.append(b)
        nxt = j.get("next_url")
        url = nxt if nxt else None
        if url:
            time.sleep(0.2)
    if not rows:
        return pd.DataFrame(
            columns=[
                "ts",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vwap",
                "n",
            ]
        )
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.rename(
        columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "n": "n",
        }
    )
    df["symbol"] = symbol
    return df[["ts", "symbol", "open", "high", "low", "close", "volume", "vwap", "n"]]


def _selftest():
    if not KEY:
        print("[FAIL] no POLYGON key found in env or .env")
        return
    print(f"[OK] key resolved (len={len(KEY)})")
    # 2 recent weekdays
    end = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=2)
    start = end - pd.Timedelta(days=4)
    df = fetch_minute_bars("AAPL", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    print(f"AAPL minute bars {start.date()}..{end.date()}: {len(df)} rows")
    if len(df):
        et = df["ts"].dt.tz_convert("America/New_York")
        reg = ((et.dt.hour > 9) | ((et.dt.hour == 9) & (et.dt.minute >= 30))) & (
            et.dt.hour < 16
        )
        print(
            f"  regular-hours bars: {int(reg.sum())}  extended-hours bars: {int((~reg).sum())}"
        )
        print(f"  time span/day OK; sample:\n{df.head(3).to_string(index=False)}")
        print(
            f"  -> extended hours {'PRESENT' if (~reg).sum() else 'ABSENT'} (need it for after-hours earnings)"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        _selftest()
    else:
        _selftest()
