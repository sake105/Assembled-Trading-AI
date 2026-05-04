"""Leakage audit for ML v3 features. Saves JSON report to output/."""
import sys
import json
import datetime
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

# -- load panel --
panel_path = Path("data/sample/watchlist_2020_2026.parquet")
if not panel_path.exists():
    sys.exit(f"Panel not found: {panel_path}")

prices = pd.read_parquet(panel_path)
prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
prices = prices.sort_values(["symbol", "timestamp"])
syms = prices["symbol"].unique()
print(f"Panel: {len(prices)} rows, {len(syms)} symbols", flush=True)

# -- compute features per symbol --
rows = []
for sym, grp in prices.groupby("symbol", sort=False):
    grp = grp.sort_values("timestamp").copy()
    c = grp["close"].astype(float)
    v = grp["volume"].astype(float) if "volume" in grp.columns else pd.Series(np.nan, index=grp.index)
    ts = grp["timestamp"]
    n = len(grp)

    # Log return
    lr = np.log(c / c.shift(1))

    # RSI-14
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # MACD histogram (12/26/9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - signal_line

    # Bollinger %B and bandwidth (20)
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_pctb = (c - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    bb_bw = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

    # ATR-14
    if "high" in grp.columns and "low" in grp.columns:
        h = grp["high"].astype(float)
        lo = grp["low"].astype(float)
        tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    else:
        tr = (c - c.shift()).abs()
    atr14 = tr.ewm(span=14, adjust=False).mean()

    # ADX-14 (simplified: use ATR-normalized directional movement)
    if "high" in grp.columns and "low" in grp.columns:
        h = grp["high"].astype(float)
        lo = grp["low"].astype(float)
        dm_plus = (h - h.shift()).clip(lower=0)
        dm_minus = (lo.shift() - lo).clip(lower=0)
        di_plus = dm_plus.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan) * 100
        di_minus = dm_minus.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan) * 100
        dx = ((di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan) * 100)
        adx = dx.ewm(span=14, adjust=False).mean()
    else:
        adx = pd.Series(np.nan, index=grp.index)

    # Realized vol
    rv20 = lr.rolling(20).std() * np.sqrt(252)
    rv60 = lr.rolling(60).std() * np.sqrt(252)
    vov = rv20 / rv60.replace(0, np.nan)

    # Volume z-score (20d)
    vol_z = (v - v.rolling(20).mean()) / v.rolling(20).std()

    # Amihud illiquidity
    dollar_vol = c * v
    amihud = lr.abs() / dollar_vol.replace(0, np.nan)
    amihud20 = amihud.rolling(20).mean()

    # Forward returns
    ret5 = c.pct_change(5)
    ret20 = c.pct_change(20)
    fwd_ret5 = ret5.shift(-5)

    tmp = pd.DataFrame({
        "timestamp": ts.values,
        "symbol": sym,
        "ta_log_return_v1": lr.values,
        "ta_rsi_14_v1": rsi.values,
        "ta_macd_hist_v1": macd_hist.values,
        "ta_bb_pctb_v1": bb_pctb.values,
        "ta_bb_bandwidth_v1": bb_bw.values,
        "ta_adx_v1": adx.values,
        "ta_atr_14_v1": atr14.values,
        "rv_20": rv20.values,
        "rv_60": rv60.values,
        "vov_20_60": vov.values,
        "volume_zscore": vol_z.values,
        "amihud_illiq_20d": amihud20.values,
        "ret_5d": ret5.values,
        "ret_20d": ret20.values,
        "fwd_ret_5d": fwd_ret5.values,
    })
    rows.append(tmp)

feat = pd.concat(rows, ignore_index=True)
feat["timestamp"] = pd.to_datetime(feat["timestamp"], utc=True)
feat = feat.dropna()
print(f"Features computed: {len(feat)} samples, {feat['symbol'].nunique()} symbols", flush=True)

ml_cols = [
    "ta_log_return_v1","ta_rsi_14_v1","ta_macd_hist_v1","ta_bb_pctb_v1",
    "ta_bb_bandwidth_v1","ta_adx_v1","ta_atr_14_v1","rv_20","rv_60",
    "vov_20_60","volume_zscore","amihud_illiq_20d","ret_5d","ret_20d",
]

train_cutoff = pd.Timestamp("2024-01-01", tz="UTC")
train_f = feat[feat["timestamp"] < train_cutoff][ml_cols]
test_f  = feat[feat["timestamp"] >= train_cutoff][ml_cols]
target_train = feat[feat["timestamp"] < train_cutoff]["fwd_ret_5d"]
all_features = feat[ml_cols]
all_target   = feat["fwd_ret_5d"]

print(f"Train: {len(train_f)}, Test: {len(test_f)}", flush=True)

from src.assembled_core.qa.leakage_analyzer import LeakageAnalyzer
la = LeakageAnalyzer()

print("Running check_lookahead...", flush=True)
r1 = la.check_lookahead(all_features, all_target)
print(f"  Lookahead findings: {len(r1)}", flush=True)

print("Running check_recursive...", flush=True)
r2 = la.check_recursive(all_features, all_target)
print(f"  Recursive findings: {len(r2)}", flush=True)

print("Running check_normalization_leakage...", flush=True)
r3 = la.check_normalization_leakage(train_f, test_f)
print(f"  Normalization findings: {len(r3)}", flush=True)


def finding_to_dict(f):
    d = {}
    for k in dir(f):
        if k.startswith("_"):
            continue
        try:
            v = getattr(f, k)
            if callable(v):
                continue
            if isinstance(v, float):
                d[k] = round(v, 6)
            elif isinstance(v, (int, bool, type(None))):
                d[k] = v
            else:
                d[k] = str(v).replace("≈", "~").replace("→", "->")
        except Exception:
            pass
    return d


report = {
    "generated": datetime.date.today().isoformat(),
    "panel": str(panel_path),
    "n_samples_total": len(feat),
    "n_train": len(train_f),
    "n_test": len(test_f),
    "train_cutoff": str(train_cutoff.date()),
    "ml_features": ml_cols,
    "lookahead": {
        "n_findings": len(r1),
        "verdict": "CLEAN" if len(r1) == 0 else "FLAGGED",
        "findings": [finding_to_dict(f) for f in r1],
    },
    "recursive": {
        "n_findings": len(r2),
        "verdict": "CLEAN" if len(r2) == 0 else "FLAGGED",
        "findings": [finding_to_dict(f) for f in r2],
    },
    "normalization": {
        "n_findings": len(r3),
        "verdict": "CLEAN" if len(r3) == 0 else "FLAGGED",
        "findings": [finding_to_dict(f) for f in r3],
    },
    "summary": {
        "total_issues": len(r1) + len(r2) + len(r3),
        "safe_for_training": (len(r1) + len(r2) == 0),
        "normalization_action_needed": len(r3) > 0,
    },
}

out = Path("output/leakage_report_ml_features_2026-05-03.json")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
print(f"Report saved: {out}", flush=True)
print(f"Summary: lookahead={len(r1)} recursive={len(r2)} normalization={len(r3)}", flush=True)

if r3:
    print("\nNormalization findings:", flush=True)
    for f in r3:
        d = finding_to_dict(f)
        feature = d.get("feature", d.get("column", "?"))
        detail = d.get("detail", d.get("message", str(d)))
        print(f"  [{feature}] {detail}", flush=True)
