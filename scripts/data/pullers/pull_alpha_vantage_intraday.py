#!/usr/bin/env python
# coding: utf-8
"""
pull_alpha_intraday.py
Free intraday via yfinance (keine API Keys). Writes one Parquet per symbol.
Usage:
  python pull_alpha_intraday.py --symbols "AAPL,MSFT" --interval 5m --days 5 --out data/raw/intraday/alphavantage/5min
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf


def dl_intraday(sym: str, interval: str, days: int) -> pd.DataFrame:
    # yfinance: period must be like "5d", "7d", "60d", "730d"
    period = f"{days}d" if days > 0 else "5d"
    df = yf.download(
        sym, period=period, interval=interval, auto_adjust=False, progress=False
    )
    if df.empty:
        return df
    df = df.reset_index().rename(columns=str.lower)
    df["symbol"] = sym
    # expected: DatetimeIndex column name 'Datetime' on some versions -> after reset it's 'datetime'
    # unify:
    if "datetime" in df.columns:
        df = df.rename(columns={"datetime": "timestamp"})
    elif "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    cols = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
    df = df[[c for c in cols if c in df.columns]]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--days", type=int, default=5, help="how many days to pull")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    any_ok = False

    # E-112: every ingest writes a request protocol — key, window, status, bar
    # count — ALSO AND ESPECIALLY for empty results. Before this, an empty
    # frame produced a WARN on stderr and `continue`: no file, no record, no
    # way to later distinguish "the vendor does not cover this symbol" from
    # "this symbol was never requested". That confusion is what produced a
    # published and wrong coverage figure once already.
    plog = None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from src.assembled_core.data.pull_log import PullLog

        plog = PullLog(source="yfinance_intraday")
    except Exception as exc:  # pragma: no cover - bookkeeping must not block a pull
        print(f"[INTRA] WARN pull_log unavailable: {exc}", file=sys.stderr)

    # window bleibt leer: window_start/-end sind ZEITpunkte, keine Dauer und
    # kein Intervall. Beides geht als Extra-Feld an record().
    window = None

    try:
        for s in syms:
            try:
                df = dl_intraday(s, args.interval, args.days)
                if df.empty:
                    print(f"[INTRA] WARN empty: {s}", file=sys.stderr)
                    if plog is not None:
                        # No http_status: yfinance does not surface one, and inventing a 200
                        # on the EMPTY path would assert exactly the thing the
                        # protocol is supposed to establish (E-112).
                        plog.record(
                            s,
                            window=window,
                            n_rows=0,
                            interval=args.interval,
                            lookback_days=args.days,
                        )
                    continue
                fp = out / f"{s}_{args.interval}.parquet"
                df.to_parquet(fp, index=False)
                # ASCII statt "→": unter Windows-cp1252 wirft der Pfeil einen
                # UnicodeEncodeError - und zwar NACH to_parquet, aber VOR
                # plog.record. Der Erfolgspfad landete dadurch als
                # Anbieterfehler im Protokoll und der Lauf mit rc=2 (E-151).
                print(f"[INTRA] OK {s} -> {fp}")
                if plog is not None:
                    plog.record(
                        s,
                        window=window,
                        n_rows=len(df),
                        interval=args.interval,
                        lookback_days=args.days,
                    )
                any_ok = True
                time.sleep(0.2)
            except Exception as e:
                print(f"[INTRA] ERR {s}: {e}", file=sys.stderr)
                if plog is not None:
                    plog.record(
                        s,
                        window=window,
                        error=f"{type(e).__name__}: {e}",
                        interval=args.interval,
                        lookback_days=args.days,
                    )

    finally:
        if plog is not None:
            plog.write()

    if not any_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
