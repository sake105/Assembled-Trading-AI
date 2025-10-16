# -*- coding: utf-8 -*-
"""
Sprint 8 Feature Engineering – Produktionsklar
"""

from __future__ import annotations
import os, sys, json, argparse, warnings
from typing import List, Iterable
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "output")
OUT_FEAT = os.path.join(OUT_DIR, "features")

SEARCH_ROOTS = [
    os.path.join(DATA_DIR, "raw"),
    os.path.join(OUT_DIR, "assembled_intraday"),
    os.path.join(OUT_DIR, "aggregates"),
]

REQUIRED_COLS = ["timestamp", "symbol", "open", "high", "low", "close"]
OPTIONAL_COLS = ["volume", "vwap"]

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _find_candidates(freq: str) -> list[str]:
    exts = (".parquet", ".csv")
    hits: list[str] = []
    for r in SEARCH_ROOTS:
        if not os.path.exists(r):
            continue
        for root, _, files in os.walk(r):
            for f in files:
                lf = f.lower()
                if lf.endswith(exts) and (freq in lf or lf.endswith(f"{freq}.parquet") or lf.endswith(f"{freq}.csv")):
                    hits.append(os.path.join(root, f))
    return hits

def _read_any(path: str) -> pd.DataFrame:
    try:
        if path.lower().endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Spaltennamen → lower
    lower = {c: str(c).lower() for c in df.columns}
    df = df.rename(columns=lower)
    # Aliasse
    alias = {"time": "timestamp", "date": "timestamp", "dt": "timestamp"}
    for a, tgt in alias.items():
        if a in df.columns and "timestamp" not in df.columns:
            df = df.rename(columns={a: "timestamp"})
    if "ticker" in df.columns and "symbol" not in df.columns:
        df = df.rename(columns={"ticker": "symbol"})
    # Typisierung
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in ["open","high","low","close","volume","vwap"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df = df.replace([np.inf, -np.inf], np.nan)
    if "timestamp" in df.columns:
        df = df.dropna(subset=["timestamp"]).sort_values(["timestamp", "symbol"] if "symbol" in df.columns else ["timestamp"])
    return df.reset_index(drop=True)

def _assert_schema(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}")

def _concat_nonempty(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if isinstance(f, pd.DataFrame) and not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def _clip_numbers(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].ffill().bfill().fillna(0.0).clip(-1e12, 1e12)
    return df

def _compute_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df["ret_1"] = df.groupby("symbol")["close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["hlc3"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["vol_20"] = df.groupby("symbol")["ret_1"].rolling(20).std().reset_index(level=0, drop=True).fillna(0.0)
    tr = (df["high"] - df["low"]).abs()
    df["atr_14"] = tr.rolling(14).mean().fillna(0.0)
    df["ma_fast"] = df.groupby("symbol")["close"].rolling(10).mean().reset_index(level=0, drop=True)
    df["ma_slow"] = df.groupby("symbol")["close"].rolling(30).mean().reset_index(level=0, drop=True)
    df["risk_on"] = (df["ma_fast"] > df["ma_slow"]).astype("int8")

    base_cols = list({*REQUIRED_COLS, "volume", "vwap", "ret_1", "hlc3", "vol_20", "atr_14"})
    base_cols = [c for c in base_cols if c in df.columns]
    base = df[base_cols].copy()
    micro = df[["symbol", "timestamp", "ret_1", "vol_20", "atr_14"]].copy()
    regime = df[["symbol", "timestamp", "ma_fast", "ma_slow", "risk_on"]].copy()
    return base, micro, regime

def build_features(freq: str, symbols: list[str], quick: bool, qdays: int) -> None:
    print(f"[SPRINT8] START Sprint8 Feature Build | freq={freq} quick={quick} qdays={qdays}")
    print(f"[SPRINT8] Symbols filter: {symbols if symbols else ['ALL']}")
    print(f"[SPRINT8] Search roots: {SEARCH_ROOTS}")

    files = _find_candidates(freq)
    print(f"[SPRINT8] Found {len(files)} candidate file(s)")
    if not files:
        raise FileNotFoundError("No input files found in " + ", ".join(SEARCH_ROOTS))

    frames = []
    for f in files:
        df = _standardize(_read_any(f))
        if df.empty or "timestamp" not in df.columns:
            continue
        if "symbol" not in df.columns:
            df["symbol"] = "NAN"
        frames.append(df)

    all_df = _concat_nonempty(frames)
    if all_df.empty:
        raise ValueError("[RAW_ALL] DataFrame is empty")

    if quick:
        cutoff = all_df["timestamp"].max() - pd.Timedelta(days=qdays)
        all_df = all_df[all_df["timestamp"] >= cutoff]
        print(f"[SPRINT8] Quick filter: cutoff={cutoff} → rows {len(all_df)} (from {sum(len(x) for x in frames)})")

    if symbols and symbols != ["ALL"]:
        before = len(all_df)
        all_df = all_df[all_df["symbol"].isin(symbols)]
        print(f"[SPRINT8] Applied symbol filter {symbols} → rows {len(all_df)} (from {before})")
        if all_df.empty:
            print("[SPRINT8] WARNING: Symbol filter removed all rows → falling back to ALL.")
            all_df = _concat_nonempty(frames)
            if quick:
                cutoff = all_df["timestamp"].max() - pd.Timedelta(days=qdays)
                all_df = all_df[all_df["timestamp"] >= cutoff]

    # Pflichtspalten ggf. minimal auffüllen
    for c in REQUIRED_COLS:
        if c not in all_df.columns:
            if c in ("open", "high", "low") and "close" in all_df.columns:
                all_df[c] = all_df["close"]
            elif c in ("timestamp", "symbol"):
                raise ValueError(f"[RAW_ALL] Missing required column: {c}")
            else:
                all_df[c] = np.nan
    _assert_schema(all_df, "RAW_ALL")

    all_df = _clip_numbers(all_df)

    base, micro, regime = _compute_features(all_df)

    _ensure_dir(OUT_FEAT)
    base_path   = os.path.join(OUT_FEAT, f"base_{freq}.parquet")
    micro_path  = os.path.join(OUT_FEAT, f"micro_{freq}.parquet")
    regime_path = os.path.join(OUT_FEAT, f"regime_{freq}.parquet")
    base.to_parquet(base_path)
    micro.to_parquet(micro_path)
    regime.to_parquet(regime_path)
    print(f"[SPRINT8] [OK] written: {base_path}")
    print(f"[SPRINT8] [OK] written: {micro_path}")
    print(f"[SPRINT8] [OK] written: {regime_path}")

    manifest = os.path.join(OUT_FEAT, "feature_manifest.json")
    try:
        info = {"freq": freq, "rows": int(len(all_df)), "symbols": sorted(list(map(str, all_df["symbol"].dropna().unique())))}
        if os.path.exists(manifest):
            with open(manifest, "r", encoding="utf-8") as f:
                obj = json.load(f)
        else:
            obj = {}
        obj[f"features_{freq}"] = info
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        print(f"[SPRINT8] [OK] manifest updated: {manifest}")
    except Exception:
        pass

    print("[SPRINT8] Acceptance hint: Validate PF>1.25 under cost stress once Execution v2 is wired.")
    print("[SPRINT8] DONE Sprint8 Feature Build")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freq", type=str, default="5min")
    p.add_argument("--symbols", type=str, default="ALL")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--qdays", type=int, default=180)
    # Alias für Kompatibilität zu run_sprint8_rehydrate.ps1
    p.add_argument("--quick-days", dest="qdays", type=int)
    args = p.parse_args()

    syms = [s.strip() for s in (args.symbols or "ALL").split(",")] if args.symbols else ["ALL"]
    warnings.filterwarnings("ignore", category=FutureWarning)
    build_features(args.freq, syms, args.quick, args.qdays)

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"[SPRINT8] [FAIL] {e}", file=sys.stderr)
        sys.exit(1)
