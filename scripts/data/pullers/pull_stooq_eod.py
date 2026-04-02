#!/usr/bin/env python
# coding: utf-8
"""
pull_stooq_eod.py  —  Free EoD via Yahoo (yfinance) und/oder Stooq CSV

Exit codes:
  0 = mindestens eine Symbol-Datei geschrieben
  2 = alles fehlgeschlagen/leer
"""
from __future__ import annotations
import argparse, sys, time, random, io
from pathlib import Path
from typing import Optional, List

import pandas as pd
import numpy as np
import requests
import yfinance as yf

# ---------- Helpers


def _find_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    cols = [c.lower() for c in df.columns]
    # direkte Kandidaten
    for cand in ("timestamp", "datetime", "date"):
        if cand in cols:
            return df.columns[cols.index(cand)]
    # häufige Fälle bei reset_index(): "index" oder "unnamed: 0"
    for cand in ("index", "unnamed: 0"):
        if cand in cols:
            c = df.columns[cols.index(cand)]
            # nur nehmen, wenn es wie ein Datum/Datetime aussieht
            try:
                pd.to_datetime(df[c], utc=True)
                return c
            except Exception:
                pass
    return None


def _clean_df(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    """
    Vereinheitliche Spalten: timestamp, open, high, low, close, adj_close?, volume, symbol
    df kann von yfinance (Index=Datetime) oder Stooq (CSV) kommen.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance: meist DatetimeIndex; nach reset_index() heißt die Zeitspalte oft "Date" oder "index"
    if isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        df = df.reset_index()

    # lower-case columns
    df = df.rename(columns=lambda x: str(x).strip().lower())

    # Zeitspalte finden/erzeugen
    ts_col = _find_timestamp_column(df)
    if ts_col is None:
        # letzte Eskalation: versuche, den ersten Datetime-artigen Vektor zu finden
        for c in df.columns:
            try:
                cand = pd.to_datetime(df[c], utc=True, errors="raise")
                df["timestamp"] = cand
                break
            except Exception:
                continue
        else:
            raise KeyError("timestamp")
    else:
        df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")

    df = df.dropna(subset=["timestamp"])

    # Normspalten
    rename_map = {
        "adj close": "adj_close",
    }
    df = df.rename(columns=rename_map)

    want_cols = ["timestamp", "open", "high", "low", "close", "adj_close", "volume"]
    have_cols = [c for c in want_cols if c in df.columns]
    df = df[["timestamp"] + [c for c in have_cols if c != "timestamp"]].copy()

    df["symbol"] = sym.upper()
    # Sortierung & Deduplikation
    df = (
        df.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp", "symbol"], keep="last")
        .reset_index(drop=True)
    )
    return df


def _stooq_symbol(sym: str) -> str:
    """
    Stooq erwartet für US-Aktien oft das Suffix '.us' (aapl.us, msft.us).
    Für ETFs etc. können andere Suffixe nötig sein – hier US default.
    """
    s = sym.lower()
    if "." not in s:
        s = f"{s}.us"
    return s


def _stooq_fetch(sym: str, session: requests.Session) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={_stooq_symbol(sym)}&i=d"
    r = session.get(url, timeout=15)
    # Stooq gibt bei Fehlern HTML zurück → dann leer
    if r.status_code != 200 or not r.text or r.text.lstrip().startswith("<"):
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(r.text))
    if df.empty:
        return pd.DataFrame()
    df = df.rename(columns=lambda x: str(x).strip().lower())
    if "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})
    return _clean_df(df, sym)


def _yahoo_fetch(sym: str, period: str, session: requests.Session) -> pd.DataFrame:
    df = yf.download(
        tickers=sym,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        session=session,
    )
    # yfinance gibt leeres DF zurück → _clean_df gibt dann leer zurück
    return _clean_df(df, sym)


def dl_yahoo_with_retry(
    sym: str,
    years: int,
    session: requests.Session,
    max_retries: int = 3,
    base_sleep: float = 1.0,
    jitter: float = 0.5,
) -> pd.DataFrame:
    periods: List[str] = ([f"{years}y"] if years > 0 else []) + [
        "5y",
        "2y",
        "1y",
        "max",
    ]
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        period = periods[min(attempt, len(periods) - 1)]
        try:
            df = _yahoo_fetch(sym, period, session)
            if not df.empty:
                return df
        except Exception as e:
            last_err = e
            # 429/Rate-Limit/KeyErrors etc. → Backoff und nächster Versuch
        # kurzer Backoff
        sleep_s = base_sleep * (2**attempt) + random.uniform(0, jitter)
        time.sleep(sleep_s)
    if last_err:
        print(f"[EOD] WARN {sym} yahoo_last: {last_err}", file=sys.stderr)
    return pd.DataFrame()


# ---------- Main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--symbols", required=True, help="comma separated tickers, e.g. AAPL,MSFT"
    )
    ap.add_argument("--out", required=True, help="output folder for parquet files")
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument(
        "--source",
        choices=["auto", "yahoo", "stooq"],
        default="auto",
        help="auto: erst kurz Yahoo, dann Stooq; yahoo: nur Yahoo; stooq: nur Stooq",
    )
    ap.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Yahoo: Anzahl Versuche (kurz halten)",
    )
    ap.add_argument(
        "--base-sleep", type=float, default=1.0, help="Yahoo: Basis-Backoff"
    )
    ap.add_argument(
        "--jitter", type=float, default=0.5, help="Yahoo: zusätzlicher Zufalls-Jitter"
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    any_ok = False

    # Reuse-Session für beide Quellen
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0 Safari/537.36"
            )
        }
    )

    for s in syms:
        df = pd.DataFrame()

        try:
            if args.source == "yahoo":
                df = dl_yahoo_with_retry(
                    s,
                    args.years,
                    session,
                    max_retries=args.max_retries,
                    base_sleep=args.base_sleep,
                    jitter=args.jitter,
                )

            elif args.source == "stooq":
                df = _stooq_fetch(s, session)

            else:  # auto
                # 1 kurzer Yahoo-Schuss + 1 Retry → dann sofort Stooq
                df = dl_yahoo_with_retry(
                    s,
                    args.years,
                    session,
                    max_retries=min(2, args.max_retries),
                    base_sleep=args.base_sleep,
                    jitter=args.jitter,
                )
                if df.empty:
                    df = _stooq_fetch(s, session)

        except Exception as e:
            print(f"[EOD] WARN {s} fetch: {e}", file=sys.stderr)

        if df.empty:
            print(f"[EOD] WARN empty: {s}", file=sys.stderr)
            continue

        fp = out / f"{s}.parquet"
        try:
            df.to_parquet(fp, index=False)
            print(f"[EOD] OK {s} → {fp}")
            any_ok = True
        except Exception as e:
            print(f"[EOD] ERR write {s}: {e}", file=sys.stderr)

        # leichte Drossel
        time.sleep(0.2)

    sys.exit(0 if any_ok else 2)


if __name__ == "__main__":
    main()
