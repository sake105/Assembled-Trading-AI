# scripts/live/pull_intraday_av.py
from __future__ import annotations
import argparse
import os
import sys
import time
import random
import pathlib as pl
import pandas as pd
import requests

API_URL = "https://www.alphavantage.co/query"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", type=str, default="AAPL,MSFT")
    p.add_argument("--days", type=int, default=2)
    p.add_argument("--interval", type=str, default="5min", choices=["1min","5min","15min","30min","60min"])
    p.add_argument("--repo-root", type=str, default="")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--apikey", type=str, default=os.environ.get("ALPHAVANTAGE_API_KEY",""))
    return p.parse_args()

def _out_dir(a) -> pl.Path:
    if a.out:
        return pl.Path(a.out)
    root = pl.Path(a.repo_root) if a.repo_root else pl.Path(__file__).resolve().parents[2]
    return root / "data" / "raw" / "1min"

def _fetch(symbol: str, interval: str, apikey: str) -> pd.DataFrame:
    # TIME_SERIES_INTRADAY (compact ~ 100 Punkte, full ~ 2 Monate je nach Intervall)
    # Wir nutzen "full", um genug Historie zu haben; free-tier limitiert Requests/min & Tag.
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "outputsize": "full",
        "datatype": "json",
        "apikey": apikey or "demo",  # "demo" funktioniert v.a. für MSFT
    }
    r = requests.get(API_URL, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    # Fehler-/Limit-Indikatoren abfangen
    if any(k in js for k in ["Note", "Information", "Error Message"]):
        return pd.DataFrame()
    # Knotenname ist z.B. "Time Series (5min)"
    key = next((k for k in js.keys() if k.startswith("Time Series")), None)
    if not key or key not in js:
        return pd.DataFrame()
    d = js[key]
    if not isinstance(d, dict) or not d:
        return pd.DataFrame()
    df = (
        pd.DataFrame.from_dict(d, orient="index")
        .rename(columns={
            "1. open":"open","2. high":"high","3. low":"low","4. close":"close","5. volume":"volume"
        })
        .reset_index().rename(columns={"index":"timestamp"})
    )
    # Typen & Sortierung
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["symbol"] = symbol
    df = df.dropna(subset=["timestamp","close"]).sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp","symbol","open","high","low","close","volume"]]

def main():
    a = parse_args()
    out_dir = _out_dir(a)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not a.apikey:
        print("[AV] WARN: Kein API Key gesetzt. Setze $env:ALPHAVANTAGE_API_KEY oder nutze --apikey. (Für Test: 'demo' → MSFT)", flush=True)

    symbols = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    wrote = 0
    empty = []

    # Free-Tier Limits: 5 req/min, 25 req/Tag → konservativ: ~13–15s Pause
    base_sleep = 13.0

    for i, sym in enumerate(symbols):
        df = _fetch(sym, interval=a.interval, apikey=a.apikey)
        if df.empty:
            empty.append(sym)
        else:
            out = out_dir / f"{sym}_{a.interval}.parquet"
            df.to_parquet(out, index=False)
            wrote += 1
            print(f"[AV] OK {sym} → {out} rows={len(df)}", flush=True)
        # Rate-Limit sanft respektieren (auch nach leerem Ergebnis)
        if i < len(symbols)-1:
            time.sleep(base_sleep + random.uniform(0, 1.5))

    if wrote == 0:
        print(f"[AV] DONE but no files written (Limit/kein Key/keine Daten). Empty: {','.join(empty) if empty else '-'}", flush=True)
        sys.exit(0)  # weich, damit Pipeline weiterläuft
    else:
        if empty:
            print(f"[AV] INFO leer: {','.join(empty)}", flush=True)
        print(f"[AV] DONE wrote={wrote}", flush=True)
        sys.exit(0)

if __name__ == "__main__":
    main()
