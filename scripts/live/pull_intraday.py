# scripts/live/pull_intraday.py
from __future__ import annotations

import argparse
import sys
import time
import random
import pathlib as pl
from typing import List, Dict

import pandas as pd

# yfinance ist in requirements (0.2.54)
import yfinance as yf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pull intraday data via yfinance.")
    p.add_argument("--symbols", type=str, default="AAPL,MSFT",
                   help="Comma-separated tickers, e.g. 'AAPL,MSFT'")
    p.add_argument("--days", type=int, default=2,
                   help="Lookback window in days (yfinance period)")
    p.add_argument("--repo-root", type=str, required=True,
                   help="Path to repository root")
    p.add_argument("--interval", type=str, default=None,
                   help="Force interval (default: choose 1m if days<=7 else 5m)")
    p.add_argument("--max-retries", type=int, default=4,
                   help="Max retries on errors/rate limits")
    p.add_argument("--base-sleep", type=float, default=2.5,
                   help="Base sleep seconds for backoff")
    p.add_argument("--jitter", type=float, default=1.5,
                   help="Random jitter seconds added to sleep")
    return p.parse_args()


def decide_interval(days: int, forced: str | None) -> str:
    if forced:
        return forced
    # 1m geht bei yfinance i.d.R. nur ca. 7 Tage zurück
    return "1m" if days <= 7 else "5m"


def normalize_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Bringt yfinance-DF in Standardform:
    columns: timestamp (UTC tz-aware), symbol, open, high, low, close, volume
    """
    # yfinance liefert bei download() OHLCV-Spalten
    cols_map = {}
    for c in df.columns:
        lc = str(c).lower()
        if "open" == lc:
            cols_map[c] = "open"
        elif "high" == lc:
            cols_map[c] = "high"
        elif "low" == lc:
            cols_map[c] = "low"
        elif "close" == lc:
            cols_map[c] = "close"
        elif "volume" == lc:
            cols_map[c] = "volume"

    df = df.rename(columns=cols_map)
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    if not keep:
        return pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])

    out = df[keep].copy()

    # Index -> timestamp (UTC)
    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex):
        # Fallback: versuchen aus 'datetime' Spalte
        if "datetime" in out.columns:
            idx = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
        else:
            # kein Datum → leer
            return pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])

    if idx.tz is None:
        idx = idx.tz_localize("UTC")

    out.insert(0, "timestamp", idx)
    out.insert(1, "symbol", symbol)
    # fehlende Spalten auffüllen
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in out.columns:
            out[c] = pd.NA

    out = out.dropna(subset=["close"]).reset_index(drop=True)
    return out[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]


def download_batch(tickers: List[str], period: str, interval: str) -> Dict[str, pd.DataFrame]:
    """
    Lädt mehrere Ticker via yfinance.download. Gibt dict(symbol -> DataFrame) zurück.
    """
    # yfinance.download gibt:
    # - bei mehreren Tickern MultiIndex-Spalten (level 0 = Feld, level 1 = Ticker) ODER
    #   (je nach Version) level 0 = Ticker, level 1 = Feld. Wir fangen beide Fälle ab.
    data = yf.download(
        tickers=" ".join(tickers),
        period=period,
        interval=interval,
        auto_adjust=False,
        prepost=False,
        threads=True,
        progress=False,
        group_by="ticker",  # erzwingt Spalten gruppiert je Ticker in neuerem yfinance
    )

    out: Dict[str, pd.DataFrame] = {}

    # Falls nur ein Ticker: data ist ein normales DF
    if isinstance(data, pd.DataFrame) and not isinstance(data.columns, pd.MultiIndex):
        # single symbol Fall → wir wissen aber nicht, welcher, daher nehmen tickers[0]
        out[tickers[0]] = data
        return out

    # MultiIndex Fälle
    if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
        # Heuristik: checke, ob Level 0 Felder ("Open", "Close", …) enthält
        lvl0 = [str(x).lower() for x in data.columns.get_level_values(0)]
        looks_like_fields_first = any(k in lvl0 for k in ["open", "high", "low", "close", "volume"])

        if looks_like_fields_first:
            # Spalten sind (Field, Ticker)
            fields = sorted(set(data.columns.get_level_values(0)))
            symbols = sorted(set([str(s) for s in data.columns.get_level_values(1)]))
            for sym in tickers:
                if sym in symbols:
                    sub = pd.DataFrame(index=data.index)
                    for f in fields:
                        if (f, sym) in data.columns:
                            sub[f] = data[(f, sym)]
                    out[sym] = sub
        else:
            # Spalten sind (Ticker, Field)
            symbols = sorted(set([str(s) for s in data.columns.get_level_values(0)]))
            for sym in tickers:
                if sym in symbols:
                    sub = pd.DataFrame(index=data.index)
                    for f in ["Open", "High", "Low", "Close", "Volume"]:
                        if (sym, f) in data.columns:
                            sub[f.lower()] = data[(sym, f)]
                    out[sym] = sub

    return out


def main() -> None:
    a = parse_args()
    symbols = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    if not symbols:
        print("[LIVE] WARN: keine Symbole übergeben.", file=sys.stderr)
        sys.exit(0)

    interval = decide_interval(a.days, a.interval)
    period = f"{a.days}d"

    repo = pl.Path(a.repo_root).resolve()
    out_dir = repo / "data" / "raw" / "1min"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LIVE] Symbols: {','.join(symbols)}  Days:{a.days}  Interval:{interval}")
    print(f"[LIVE] Out:     {out_dir}")

    wrote = 0
    attempt = 0
    while attempt < a.max_retries:
        attempt += 1
        try:
            by_sym = download_batch(symbols, period=period, interval=interval)
            empty_syms = []
            for sym, df in by_sym.items():
                norm = normalize_df(df, sym)
                if len(norm) == 0:
                    empty_syms.append(sym)
                    continue
                # speichern
                dest = out_dir / f"{sym}.parquet"
                norm.to_parquet(dest, index=False)
                wrote += 1
                print(f"[LIVE] OK {sym} → {dest} rows={len(norm)}")

            # Symbole, die gar nicht im Result waren (Rate-Limit o.ä.)
            missing = [s for s in symbols if s not in by_sym]
            all_empty = set(empty_syms) | set(missing)
            if wrote > 0:
                break

            msg = f"{len(all_empty)} Failed downloads:\n{list(all_empty)}: YFRateLimitError('Too Many Requests. Rate limited. Try after a while.')"
            print(msg)
            if attempt < a.max_retries:
                # Exponential Backoff + Jitter
                sleep_s = a.base_sleep * (2 ** (attempt - 1)) + random.uniform(0, a.jitter)
                print(f"[LIVE] WARN empty all (try#{attempt}) → sleep {sleep_s:.1f}s")
                time.sleep(sleep_s)
            else:
                break

        except Exception as e:
            # Andere Fehler (Netz, Parsing, …): ebenfalls backoff
            print(f"[LIVE] ERROR: {type(e).__name__}: {e}")
            if attempt < a.max_retries:
                sleep_s = a.base_sleep * (2 ** (attempt - 1)) + random.uniform(0, a.jitter)
                print(f"[LIVE] WARN exception (try#{attempt}) → sleep {sleep_s:.1f}s")
                time.sleep(sleep_s)
            else:
                break

    if wrote == 0:
        print("[LIVE] DONE but no files written (Rate-Limit/keine Daten?).")
        # Exit 0, damit das PS1 nicht abbricht; Log reicht zur Diagnose.
        sys.exit(0)

    print("[LIVE] DONE")


if __name__ == "__main__":
    # yfinance fix: sicherstellen, dass pandas tz-handling erwartbar ist
    pd.options.mode.use_inf_as_na = True
    main()
