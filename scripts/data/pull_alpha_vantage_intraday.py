from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from common.io_utils import RateBucket, http_get_json, normalize_ohlc, to_parquet

API = (
    "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY"
    "&symbol={sym}&interval={interval}&apikey={key}&outputsize=full&datatype=json"
)

bucket = RateBucket(calls_per_minute=5)

MAP = {
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "30min": "30min",
    "60min": "60min",
}


def pull_one(symbol: str, interval: str, api_key: str) -> pd.DataFrame:
    bucket.consume()
    url = API.format(sym=symbol, interval=interval, key=api_key)
    js = http_get_json(url)
    meta_key = "Meta Data"  # noqa: F841
    ts_key = next((k for k in js.keys() if k.startswith("Time Series")), None)
    if not ts_key:
        raise RuntimeError(
            f"No time series in response for {symbol}: {list(js.keys())}"
        )
    ts = js[ts_key]
    rows = []
    for ts_str, v in ts.items():
        rows.append(
            {
                "timestamp": ts_str,
                "open": float(v.get("1. open")),
                "high": float(v.get("2. high")),
                "low": float(v.get("3. low")),
                "close": float(v.get("4. close")),
                "volume": float(v.get("5. volume", 0.0)),
            }
        )
    df = pd.DataFrame(rows)
    df = normalize_ohlc(df, symbol, provider="alphavantage")
    return df


def main():
    if len(sys.argv) < 5:
        print(
            "Usage: python pull_alpha_vantage_intraday.py <symbols_csv> <interval> <api_key> <out_dir>"
        )
        sys.exit(2)
    symbols = sys.argv[1].split(",")
    interval = sys.argv[2]
    api_key = sys.argv[3] or os.environ.get("ALPHAVANTAGE_API_KEY", "")
    out_dir = Path(sys.argv[4])
    if interval not in MAP:
        raise SystemExit(f"interval must be one of {list(MAP)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    dfs = []
    for s in symbols:
        df = pull_one(s, interval, api_key)
        to_parquet(df, out_dir / f"{s}_{interval}.parquet")
        dfs.append(df)
    if dfs:
        big = pd.concat(dfs, ignore_index=True)
        to_parquet(big, out_dir / f"intraday_{interval}_all.parquet")


if __name__ == "__main__":
    main()
