from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

import pandas as pd
from common.io_utils import http_get_text, normalize_ohlc, to_parquet

"""
Stooq CSV EoD Download.
URL-Format (US): https://stooq.com/q/d/l/?s=aapl.us&i=d
Hinweis: Symbole je nach Börse suffixen (z. B. .us, .de). Für Demo nutzen wir .us.
"""


BASE = "https://stooq.com/q/d/l/?i=d&s={symbol}"


def fetch(symbol: str) -> pd.DataFrame:
    url = BASE.format(symbol=symbol)
    txt = http_get_text(url)
    df = pd.read_csv(StringIO(txt))
    # stooq columns: Date,Open,High,Low,Close,Volume
    df = df.rename(
        columns={
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    return df


def main():
    if len(sys.argv) < 4:
        print("Usage: python pull_stooq_eod.py <symbols_csv> <out_dir> <suffix>")
        print("Example: python pull_stooq_eod.py AAPL,MSFT data/raw/eod/stooq .us")
        sys.exit(2)
    symbols = sys.argv[1].split(",")
    out_dir = Path(sys.argv[2])
    suffix = sys.argv[3]
    out_dir.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    for s in symbols:
        sym = f"{s}{suffix}"
        df = fetch(sym)
        df = normalize_ohlc(df, s, provider="stooq")
        to_parquet(df, out_dir / f"{s}.parquet")
        all_dfs.append(df)
    if all_dfs:
        big = pd.concat(all_dfs, ignore_index=True)
        to_parquet(big, out_dir / "eod_all.parquet")


if __name__ == "__main__":
    main()
