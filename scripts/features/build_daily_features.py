#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "output" / "aggregates" / "daily.parquet"
OUT_DIR = REPO / "output" / "features"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = OUT_DIR / "feature_manifest.json"

# einfache Helfer
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi(s: pd.Series, window: int = 14) -> pd.Series:
    diff = s.diff()
    up = diff.clip(lower=0.0)
    down = -diff.clip(upper=0.0)
    ma_up = up.rolling(window, min_periods=window).mean()
    ma_down = down.rolling(window, min_periods=window).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def realized_vol(s: pd.Series, window: int) -> pd.Series:
    ret = np.log(s).diff()
    return ret.rolling(window, min_periods=window).std() * np.sqrt(252)

def main():
    if not SRC.exists():
        raise FileNotFoundError(f"{SRC} nicht gefunden. Bitte zuerst assemble_eod_daily.py ausführen.")

    df = pd.read_parquet(SRC)
    # Sicherstellen: richtige Sortierung / Index mit UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # --- BASE (Prices + simple returns)
    base = df.copy()
    base["ret1"] = (
        base.groupby("symbol")["adj_close"]
        .apply(lambda s: np.log(s).diff())
        .fillna(0.0)
        .values
    )

    # --- MICRO (Momentum/RSI)
    micro = df.copy()
    micro["ema20"] = micro.groupby("symbol")["adj_close"].apply(lambda s: ema(s, 20)).values
    micro["ema60"] = micro.groupby("symbol")["adj_close"].apply(lambda s: ema(s, 60)).values
    micro["rsi14"] = micro.groupby("symbol")["adj_close"].apply(lambda s: rsi(s, 14)).values
    micro["mom20"] = micro.groupby("symbol")["adj_close"].apply(lambda s: s.pct_change(20)).fillna(0.0).values

    # --- REGIME (Volatilität/Drawdown)
    regime = df.copy()
    regime["vol20"] = regime.groupby("symbol")["adj_close"].apply(lambda s: realized_vol(s, 20)).values
    regime["vol60"] = regime.groupby("symbol")["adj_close"].apply(lambda s: realized_vol(s, 60)).values
    # rollierendes Hoch/Tief (einfacher Drawdown-Proxy)
    def dd(s: pd.Series, w: int) -> pd.Series:
        roll_max = s.rolling(w, min_periods=w).max()
        return (s / roll_max - 1.0)
    regime["dd60"] = regime.groupby("symbol")["adj_close"].apply(lambda s: dd(s, 60)).values

    # Schreiben
    base_out   = OUT_DIR / "base_1d.parquet"
    micro_out  = OUT_DIR / "micro_1d.parquet"
    regime_out = OUT_DIR / "regime_1d.parquet"
    base.to_parquet(base_out, index=False)
    micro.to_parquet(micro_out, index=False)
    regime.to_parquet(regime_out, index=False)
    print(f"[FEAT] [OK] written:\n - {base_out}\n - {micro_out}\n - {regime_out}")

    # Manifest aktualisieren
    manifest = {}
    if MANIFEST.exists():
        try:
            manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    manifest["daily"] = {
        "base":   str(base_out.relative_to(REPO)),
        "micro":  str(micro_out.relative_to(REPO)),
        "regime": str(regime_out.relative_to(REPO)),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[FEAT] [OK] manifest updated: {MANIFEST}")

if __name__ == "__main__":
    main()
