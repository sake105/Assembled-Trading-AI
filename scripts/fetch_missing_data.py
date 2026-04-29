"""Fetch missing market data: SH, PSQ, VIX, VIX3M, UUP for M16 grand backtest.

These instruments are referenced by existing modules but not yet in local data:
- SH  : ProShares Short S&P500 (inverse ETF, no borrow risk)
- PSQ : ProShares Short QQQ (inverse ETF, tech short)
- VIX : CBOE Volatility Index (^VIX)
- VIX3M: CBOE 3-Month VIX (^VIX3M) — for term structure
- UUP : Invesco DB US Dollar Index Bullish Fund (USD strength proxy)
"""
from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import yfinance as yf

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "equities_eod", "yfinance")
START = "2023-01-01"

TICKERS = {
    "SH":    "SH",      # ProShares Short S&P500
    "PSQ":   "PSQ",     # ProShares Short QQQ
    "VIX":   "^VIX",    # CBOE VIX spot
    "VIX3M": "^VIX3M",  # CBOE 3-Month VIX
    "UUP":   "UUP",     # Dollar index ETF
}


def fetch_and_save(out_name: str, ticker: str, start: str = START) -> bool:
    out_path = os.path.join(OUT_DIR, f"{out_name}.parquet")
    if os.path.exists(out_path):
        df_existing = pd.read_parquet(out_path)
        _log.info("[SKIP] %s already exists (%d rows)", out_name, len(df_existing))
        return True

    _log.info("[FETCH] %s  (%s) from %s ...", out_name, ticker, start)
    try:
        raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if raw.empty:
            _log.warning("[WARN] %s: no data returned", out_name)
            return False

        # Flatten MultiIndex columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]

        raw = raw.reset_index()
        raw = raw.rename(columns={"Date": "timestamp", "date": "timestamp"})
        raw["timestamp"] = pd.to_datetime(raw["timestamp"])
        raw["symbol"] = out_name
        raw = raw.sort_values("timestamp").reset_index(drop=True)

        os.makedirs(OUT_DIR, exist_ok=True)
        raw.to_parquet(out_path, index=False)
        _log.info("[OK]   %s: %d rows saved to %s", out_name, len(raw), out_path)
        return True

    except Exception as exc:
        _log.error("[ERROR] %s: %s", out_name, exc)
        return False


def main() -> None:
    print("=" * 60)
    print("FETCHING MISSING DATA FOR M16 GRAND BACKTEST")
    print("=" * 60)

    results = {}
    for out_name, ticker in TICKERS.items():
        results[out_name] = fetch_and_save(out_name, ticker)

    print("\nSummary:")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name:<8} {status}")

    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print(f"\nWARNING: {len(failed)} fetch(es) failed: {failed}")
        print("Backtest will use proxy values for missing data.")
    else:
        print("\nAll data fetched successfully.")


if __name__ == "__main__":
    main()
