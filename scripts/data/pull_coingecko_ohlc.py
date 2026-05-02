from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from common.io_utils import http_get_json, normalize_ohlc, to_parquet

# OHLC endpoint: /coins/{id}/ohlc?vs_currency=usd&days=1/7/30/90/180/365/max
BASE = "https://api.coingecko.com/api/v3/coins/{cid}/ohlc?vs_currency={ccy}&days={days}"


CID_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
}


def pull_one(symbol: str, days: int, ccy: str = "usd") -> pd.DataFrame:
    cid = CID_MAP.get(symbol.upper())
    if not cid:
        raise SystemExit(f"Unbekanntes Symbol für Demo: {symbol}")
    url = BASE.format(cid=cid, ccy=ccy, days=days)
    data = http_get_json(url)
    # Antwort: [[ts, open, high, low, close], ...] (ms-epoch)
    df = pd.DataFrame(data, columns=["ts_ms", "open", "high", "low", "close"])  # kein Volume hier
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df["volume"] = 0.0
    df = df.drop(columns=["ts_ms"])
    df = normalize_ohlc(df, symbol, provider="coingecko")
    return df


def main():
    if len(sys.argv) < 4:
        print("Usage: python pull_coingecko_ohlc.py <symbols_csv> <days> <out_dir>")
        sys.exit(2)
    symbols = sys.argv[1].split(',')
    try:
        days = int(sys.argv[2])
    except ValueError:
        print(f"Error: <days> must be an integer, got {sys.argv[2]!r}")
        sys.exit(2)
    out_dir = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    dfs = []
    for s in symbols:
        df = pull_one(s, days)
        to_parquet(df, out_dir / f"{s}_ohlc.parquet")
        dfs.append(df)
    if dfs:
        big = pd.concat(dfs, ignore_index=True)
        to_parquet(big, out_dir / "crypto_ohlc_all.parquet")


if __name__ == "__main__":
    main()
