"""Bulk yfinance downloader for full_us_universe.yaml.

Usage:
    python scripts/download_master_universe_data.py --start 2021-01-01
    python scripts/download_master_universe_data.py --start 2021-01-01 --end 2026-05-01
    python scripts/download_master_universe_data.py --refresh-only AAPL,MSFT,NVDA
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yfinance as yf

from src.assembled_core.data.master_universe_loader import load_master_universe

CACHE_DIR = ROOT / "data" / "cache" / "yfinance"
PANEL_OUT = ROOT / "data" / "sample" / "master_universe_panel.parquet"


def fetch_one(symbol: str, start: str, end: str) -> pd.DataFrame | None:
    """Download OHLCV for one symbol via yfinance."""
    try:
        raw = yf.download(
            symbol, start=start, end=end, auto_adjust=True, progress=False
        )
        if raw.empty:
            return None
        # Flatten multi-index columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]
        # Normalize column names
        col_map = {
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "adj close": "close",
            "adj_close": "close",
        }
        raw = raw.rename(columns=col_map)
        keep = [
            c for c in ["open", "high", "low", "close", "volume"] if c in raw.columns
        ]
        df = raw[keep].copy()
        df.index = pd.to_datetime(df.index)
        # F-dl-1 (2026-05-19): use the canonical column name `timestamp` at
        # the producer boundary so per-symbol cache files match the consumer
        # contract in load_eod_prices (data/prices_ingest.py:82). The old
        # `date` name leaked through into 195 cache parquets and forced a
        # defensive rename in every reader.
        df.index.name = "timestamp"
        df["symbol"] = symbol
        return df
    except Exception as exc:
        print(f"  [WARN] {symbol}: {exc}")
        return None


def consolidate(symbols: list[str]) -> pd.DataFrame:
    """Load all per-symbol parquets and concatenate into a panel."""
    frames = []
    for sym in symbols:
        fp = CACHE_DIR / f"{sym}.parquet"
        if fp.exists():
            df = pd.read_parquet(fp)
            if "symbol" not in df.columns:
                df["symbol"] = sym
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=False)
    panel = panel.reset_index()
    # Legacy fallback (F-dl-1): older caches were written with index.name="date".
    # Producer now writes "timestamp", so this branch only matters for caches
    # built before 2026-05-19. Belt-and-suspenders.
    if "timestamp" not in panel.columns and "date" in panel.columns:
        panel = panel.rename(columns={"date": "timestamp"})
    # F-dl-3 (2026-05-19): de-dup on (timestamp, symbol) so an overlapping
    # re-run of fetch_one doesn't silently double rows in the panel. keep=last
    # preserves the most recent download.
    panel = panel.drop_duplicates(["timestamp", "symbol"], keep="last")
    panel = panel.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return panel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bulk yfinance downloader for master universe"
    )
    parser.add_argument("--universe", default="configs/universes/full_us_universe.yaml")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--refresh-only", default=None, help="Comma-separated subset to refresh"
    )
    parser.add_argument(
        "--delay", type=float, default=0.3, help="Seconds between API calls"
    )
    args = parser.parse_args(argv)

    end = args.end or pd.Timestamp.now().strftime("%Y-%m-%d")

    symbols, meta = load_master_universe(args.universe)

    if args.refresh_only:
        refresh_set = {s.strip().upper() for s in args.refresh_only.split(",")}
        symbols = [s for s in symbols if s in refresh_set]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"[START] Downloading {len(symbols)} symbols ({args.start} to {end})")

    ok, fail = 0, 0
    for i, sym in enumerate(symbols, 1):
        out_path = CACHE_DIR / f"{sym}.parquet"
        df = fetch_one(sym, args.start, end)
        if df is not None and not df.empty:
            df.to_parquet(out_path)
            ok += 1
            if i % 20 == 0:
                print(f"  [{i}/{len(symbols)}] {ok} ok, {fail} fail")
        else:
            fail += 1
        time.sleep(args.delay)

    print(f"[OK] Download complete: {ok} ok, {fail} failed")

    print("[START] Consolidating panel...")
    panel = consolidate(symbols)
    if not panel.empty:
        panel.to_parquet(PANEL_OUT, index=False)
        print(
            f"[OK] Panel saved: {PANEL_OUT} — {len(panel):,} rows, {panel['symbol'].nunique()} symbols"
        )
    else:
        print("[WARN] No data to consolidate")

    return 0


if __name__ == "__main__":
    sys.exit(main())
