#!/usr/bin/env python
# coding: utf-8
"""
pull_coingecko_ohlc.py
Fetch OHLC from CoinGecko free endpoint.
Usage:
  python pull_coingecko_ohlc.py --coins "BTC,ETH" --vs USD --days 30 --out data/raw/crypto/coingecko
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

DEFAULT_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "ADA": "cardano",
    "XRP": "ripple",
    "DOGE": "dogecoin",
}


def fetch_ohlc(coin_id: str, vs: str, days: int):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {"vs_currency": vs.lower(), "days": str(days)}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    # format: [[ts, open, high, low, close], ...]
    cols = ["timestamp", "open", "high", "low", "close"]
    df = pd.DataFrame(data, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--coins", required=True, help="comma separated tickers, e.g. BTC,ETH"
    )
    ap.add_argument("--vs", default="USD")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    any_ok = False

    for c in coins:
        coin_id = DEFAULT_MAP.get(c, c.lower())
        try:
            df = fetch_ohlc(coin_id, args.vs, args.days)
            if df.empty:
                print(f"[CGK] WARN empty: {c}/{coin_id}", file=sys.stderr)
                continue
            df["symbol"] = c
            fp = out / f"{c}_{args.vs.upper()}.parquet"
            df.to_parquet(fp, index=False)
            print(f"[CGK] OK {c} → {fp}")
            any_ok = True
            time.sleep(0.2)
        except Exception as e:
            print(f"[CGK] ERR {c}: {e}", file=sys.stderr)

    if not any_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
