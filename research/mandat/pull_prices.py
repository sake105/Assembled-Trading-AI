"""Mandat data pull #1 — Alpaca daily bars (adjustment=all) for current S&P 500.

EXPLORATIV: survivorship-biased by construction (current members only; delisted
names absent). Alpaca free tier reaches back to 2016-01-04 → ~10.5y of history.
yfinance was hard rate-limited (YFRateLimitError) at pull time; Stooq is behind
a PoW anti-bot wall (not circumvented on purpose). Alpaca is the legitimate,
key-authenticated source this project already uses for its OOS scripts.

Output: research/mandat/data/prices_sp500.parquet
  columns: timestamp (UTC, normalized to date), symbol, open, high, low,
  close (split+dividend adjusted), volume
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT = DATA_DIR / "prices_sp500.parquet"
CONSTITUENTS = ROOT / "research" / "mandat_data_constituents.csv"

START = dt.datetime(2016, 1, 1)
END = dt.datetime(2026, 7, 4)
BATCH = 100


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    from alpaca.data import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_API_SECRET"],
    )

    cons = pd.read_csv(CONSTITUENTS)
    # Alpaca uses '.' class notation (BRK.B) — keep as-is
    tickers = sorted(set(cons["Symbol"].astype(str)))
    tickers.append("SPY")
    print(f"[START] {len(tickers)} tickers, {START.date()} -> {END.date()}", flush=True)

    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i : i + BATCH]
        for attempt in range(3):
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=batch,
                    timeframe=TimeFrame.Day,
                    start=START,
                    end=END,
                    adjustment="all",
                )
                df = client.get_stock_bars(req).df.reset_index()
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] batch {i} attempt {attempt}: {exc}", flush=True)
                time.sleep(15 * (attempt + 1))
        else:
            failed.extend(batch)
            continue
        got = set(df["symbol"].unique())
        failed.extend([s for s in batch if s not in got])
        frames.append(
            df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]
        )
        print(
            f"[OK] batch {i}-{i + len(batch)}: {len(got)} syms, {len(df)} rows",
            flush=True,
        )
        time.sleep(1)

    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True).dt.normalize()
    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    out.to_parquet(OUT, index=False)
    g = out.groupby("symbol")["timestamp"].min()
    print(
        f"[DONE] {out['symbol'].nunique()} symbols, {len(out)} rows -> {OUT}\n"
        f"       starting <=2016-01-10: {(g <= pd.Timestamp('2016-01-10', tz='UTC')).sum()}\n"
        f"       failed/empty ({len(failed)}): {failed[:20]}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
