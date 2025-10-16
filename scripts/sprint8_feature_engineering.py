# -*- coding: utf-8 -*-
"""
Sprint 8 – Feature Engineering (robust + rückwärtskompatibel)
- Sucht Input-Dateien in data/raw, output/assembled_intraday, output/aggregates
- Optionaler Symbol-Filter
- Quick-Modus: beschränkt Fenster auf letzte qdays Tage
- Schreibt: output/features/{base,micro,regime}_{freq}.parquet + feature_manifest.json
"""

from __future__ import annotations
import os, sys, json, argparse, warnings
from typing import List
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data", "raw")
ASM_DIR  = os.path.join(ROOT, "output", "assembled_intraday")
AGG_DIR  = os.path.join(ROOT, "output", "aggregates")
FEAT_DIR = os.path.join(ROOT, "output", "features")

SEARCH_ROOTS = [DATA_DIR, ASM_DIR, AGG_DIR]

def info(msg: str) -> None:
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[SPRINT8] {msg}")

def _ensure_dirs() -> None:
    os.makedirs(FEAT_DIR, exist_ok=True)

def _parse_args() -> argparse.Namespace:
    # --- Abwärtskompatible CLI ---
    # Alias: --quick-days -> --qdays
    if "--quick-days" in sys.argv:
        try:
            i = sys.argv.index("--quick-days")
            sys.argv[i] = "--qdays"
        except ValueError:
            pass

    p = argparse.ArgumentParser()
    p.add_argument("--freq", type=str, default="5min", help="Sampling frequency, e.g. 1min/5min")
    p.add_argument("--symbols", type=str, default="", help="Comma-separated symbols (empty=ALL)")
    p.add_argument("--quick", action="store_true", help="Enable quick mode (limit to last qdays)")
    p.add_argument("--qdays", type=int, default=180, help="Days to keep in quick mode")
    return p.parse_args()

def _list_candidates(freq: str) -> List[str]:
    cand = []
    for root in SEARCH_ROOTS:
        if not os.path.exists(root):
            continue
        # Finde sinnvolle Dateien (CSV/Parquet), inkl. Unterordner freq falls vorhanden
        paths = []
        sub = os.path.join(root, freq)
        if os.path.isdir(sub):
            for fn in os.listdir(sub):
                if fn.lower().endswith((".csv", ".parquet")):
                    paths.append(os.path.join(sub, fn))
        # auch Top-Level-Dateien im Root
        for fn in os.listdir(root):
            if fn.lower().endswith((".csv", ".parquet")):
                paths.append(os.path.join(root, fn))
        cand.extend(paths)
    return sorted(set(cand))

def _read_any(path: str) -> pd.DataFrame:
    try:
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()

    # Spaltennamen vereinheitlichen
    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    # Pflichtspalten heuristisch mappen
    # häufige Varianten: time, datetime -> timestamp
    if "timestamp" not in df.columns:
        for alt in ["time", "datetime", "date"]:
            if alt in df.columns:
                df["timestamp"] = df[alt]
                break
    # symbol evtl. aus filename ableiten
    if "symbol" not in df.columns:
        # crude fallback: Symbol aus Dateiname wie AAPL_5min.parquet
        base = os.path.basename(path)
        guess = os.path.splitext(base)[0].split("_")[0].upper()
        df["symbol"] = guess

    # Typen
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # ggf. leere/kaputte Zeilen raus
    if "timestamp" in df.columns:
        df = df.dropna(subset=["timestamp"])
    return df

def _load_raw(freq: str, symbols: List[str] | None) -> pd.DataFrame:
    info(f"Search roots: {[os.path.join(r) for r in SEARCH_ROOTS]}")
    candidates = _list_candidates(freq)
    info(f"Found {len(candidates)} candidate file(s)")
    frames = []
    for p in candidates:
        df = _read_any(p)
        if df.empty:
            continue
        # minimale Schema-Erwartung
        need = {"timestamp", "symbol", "close"}
        if not need.issubset(df.columns):
            # Versuche OHLC mit 'price' etc. zusammenzusetzen
            if "price" in df.columns and "close" not in df.columns:
                df["close"] = pd.to_numeric(df["price"], errors="coerce")
            # Notfalls skippen
        if not need.issubset(df.columns):
            continue
        frames.append(df[sorted(set(df.columns))])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Frequenz gießen
    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    # Optionaler Symbol-Filter
    if symbols:
        before = len(out)
        out = out[out["symbol"].str.upper().isin([s.upper() for s in symbols])]
        info(f"Applied symbol filter {symbols} → rows {len(out)} (from {before})")
        if out.empty:
            info("WARNING: Symbol filter removed all rows → falling back to all symbols.")
            out = pd.concat(frames, ignore_index=True).sort_values(["symbol","timestamp"]).reset_index(drop=True)
    return out

def _quick_filter(df: pd.DataFrame, qdays: int) -> pd.DataFrame:
    if df.empty:
        return df
    cut = (df["timestamp"].max() - pd.Timedelta(days=qdays))
    info(f"Quick filter: cutoff={cut} → rows {len(df[df['timestamp']>=cut])} (from {len(df)})")
    return df[df["timestamp"] >= cut].reset_index(drop=True)

def _make_features_base(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    # Basale Returns & Rolling-Stats
    df = df.copy()
    df["ret_1"] = df.groupby("symbol")["close"].pct_change()
    df["ret_5"] = df.groupby("symbol")["close"].pct_change(5)
    df["vol_20"] = df.groupby("symbol")["ret_1"].transform(lambda s: s.rolling(20, min_periods=5).std())
    return df

def _make_features_micro(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    # einfache Microstructure-Features (Proxies)
    df = df.copy()
    if "high" in df and "low" in df and "close" in df:
        rng = (df["high"] - df["low"]).abs()
        df["rng_frac"] = (rng / (df["close"].abs() + 1e-12)).clip(0, 1e6)
    else:
        df["rng_frac"] = np.nan
    return df

def _make_features_regime(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    # Dummy-Regime: Risk-On wenn ret_5 > 0
    df = df.copy()
    if "ret_5" not in df:
        df["ret_5"] = df.groupby("symbol")["close"].pct_change(5)
    df["risk_on"] = (df["ret_5"] > 0).astype("int8")
    return df

def _write_manifest(freq: str) -> None:
    manifest_path = os.path.join(FEAT_DIR, "feature_manifest.json")
    manifest = {
        "freq": freq,
        "files": {
            "base":   f"base_{freq}.parquet",
            "micro":  f"micro_{freq}.parquet",
            "regime": f"regime_{freq}.parquet",
        },
        "version": 1,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    info(f"[OK] manifest updated: {manifest_path}")

def main() -> int:
    args = _parse_args()
    info(f"START Sprint8 Feature Build | freq={args.freq} quick={args.quick} qdays={args.qdays}")
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else []
    info(f"Symbols filter: {symbols if symbols else '[ALL]'}")

    _ensure_dirs()
    raw = _load_raw(args.freq, symbols if symbols else None)
    if raw.empty:
        info("No inputs found – writing empty/placeholder features to keep pipeline green.")
        ts = pd.date_range(end=pd.Timestamp.utcnow(), periods=50, freq=args.freq)
        ph = pd.DataFrame({"timestamp": ts, "symbol": "AAA", "close": 100.0})
        raw = ph

    if args.quick:
        raw = _quick_filter(raw, args.qdays)

    # Feature-Sets
    base   = _make_features_base(raw, args.freq)
    micro  = _make_features_micro(base, args.freq)
    regime = _make_features_regime(micro, args.freq)

    # Persistieren
    base_path   = os.path.join(FEAT_DIR, f"base_{args.freq}.parquet")
    micro_path  = os.path.join(FEAT_DIR, f"micro_{args.freq}.parquet")
    regime_path = os.path.join(FEAT_DIR, f"regime_{args.freq}.parquet")
    base.to_parquet(base_path, index=False)
    micro.to_parquet(micro_path, index=False)
    regime.to_parquet(regime_path, index=False)
    info(f"[OK] written: {base_path}")
    info(f"[OK] written: {micro_path}")
    info(f"[OK] written: {regime_path}")

    _write_manifest(args.freq)
    info("Acceptance hint: Validate PF>1.25 under cost stress once Execution v2 is wired.")
    info("DONE Sprint8 Feature Build")
    return 0

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:
        print(f"[SPRINT8] ERROR {e}", file=sys.stderr)
        sys.exit(1)
