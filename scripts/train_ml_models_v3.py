#!/usr/bin/env python3
"""
train_ml_models_v3.py — Meta-model retrained on panel-available feature names.

Key change from v2: feature_cols are exact panel column names so they match
at inference time in apply_meta_model_filter().

Usage:
    python scripts/train_ml_models_v3.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PANEL_FILE = ROOT / "output" / "factor_panels" / "full_panel_7y.parquet"
MODEL_OUT = ROOT / "models" / "meta_model_lgbm_v3.joblib"

# Features that exist in the panel at inference time
FEATURE_COLS = [
    "ta_log_return_v1",
    "ta_rsi_14_v1",
    "ta_macd_hist_v1",
    "ta_bb_pctb_v1",
    "ta_bb_bandwidth_v1",
    "ta_adx_v1",
    "ta_atr_14_v1",
    "rv_20",
    "rv_60",
    "vov_20_60",
    "volume_zscore",
    "amihud_illiq_20d",
    "avg_corr_short",
    "avg_corr_long",
    "corr_regime_zscore",
    "return_dispersion",
    # lagged returns computed below from close
    "ret_5d",
    "ret_20d",
]

TRAIN_CUTOFF = "2025-01-01"
TARGET_COL = "fwd_return_20d"


def compute_lagged_returns(df: pd.DataFrame) -> pd.DataFrame:
    if "close" not in df.columns:
        df["ret_5d"] = np.nan
        df["ret_20d"] = np.nan
        return df
    df = df.sort_values(["symbol", "date"])

    def _lret(g, n):
        return np.log(g["close"].clip(lower=1e-9) / g["close"].shift(n).clip(lower=1e-9))

    df["ret_5d"] = df.groupby("symbol", group_keys=False).apply(lambda g: _lret(g, 5))
    df["ret_20d"] = df.groupby("symbol", group_keys=False).apply(lambda g: _lret(g, 20))
    return df


def main():
    print(f"[v3] Loading panel: {PANEL_FILE}")
    df = pd.read_parquet(PANEL_FILE)
    print(f"     {len(df):,} rows, {df['symbol'].nunique()} symbols")
    print(f"     columns: {list(df.columns)}")

    # Normalise date
    if "date" not in df.columns:
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    df = compute_lagged_returns(df)

    # Drop rows without target
    df = df.dropna(subset=[TARGET_COL])
    print(f"[v3] After dropping NaN target: {len(df):,} rows")

    df["target"] = (df[TARGET_COL] > 0).astype(int)
    print(f"[v3] Target positive rate: {df['target'].mean():.3f}")

    available = [f for f in FEATURE_COLS if f in df.columns]
    missing = [f for f in FEATURE_COLS if f not in df.columns]
    if missing:
        print(f"[WARN] Features not in panel (skipped): {missing}")
    print(f"[v3] Using {len(available)} features: {available}")

    X = df[available].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["target"]

    train_mask = df["date"] < TRAIN_CUTOFF
    val_mask = df["date"] >= TRAIN_CUTOFF

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    print(f"[v3] Train: {len(X_train):,} | Val: {len(X_val):,}")

    try:
        import lightgbm as lgb
    except ImportError:
        print("[ERROR] lightgbm not installed")
        sys.exit(1)

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        is_unbalance=True,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]

    train_acc = accuracy_score(y_train, (train_proba > 0.5).astype(int))
    val_acc = accuracy_score(y_val, (val_proba > 0.5).astype(int))
    try:
        val_auc = roc_auc_score(y_val, val_proba)
    except Exception:
        val_auc = float("nan")

    # calibrate threshold on val
    best_threshold, best_f1 = 0.5, 0.0
    for t in np.arange(0.40, 0.70, 0.02):
        preds = (val_proba > t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t

    val_acc_cal = accuracy_score(y_val, (val_proba > best_threshold).astype(int))
    baseline_acc = max(y_val.mean(), 1 - y_val.mean())

    print(f"[v3] Train acc:          {train_acc:.4f}")
    print(f"[v3] Val acc (0.5):      {val_acc:.4f}")
    print(f"[v3] Val AUC:            {val_auc:.4f}")
    print(f"[v3] Best threshold:     {best_threshold:.2f}  F1={best_f1:.4f}  acc={val_acc_cal:.4f}")
    print(f"[v3] Baseline (majority):{baseline_acc:.4f}")

    importances = dict(zip(available, model.feature_importances_))
    top = sorted(importances.items(), key=lambda x: -x[1])[:10]
    print("[v3] Top-10 feature importances:")
    for feat, imp in top:
        print(f"     {feat}: {imp:.1f}")

    import joblib

    artifact = {
        "model": model,
        "feature_cols": available,
        "decision_threshold": float(best_threshold),
        "oos_accuracy": float(val_acc),
        "oos_accuracy_calibrated": float(val_acc_cal),
        "oos_auc": float(val_auc),
        "baseline_accuracy": float(baseline_acc),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "target": TARGET_COL,
        "train_cutoff": TRAIN_CUTOFF,
        "version": "v3",
    }
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_OUT)
    print(f"[v3] Saved → {MODEL_OUT}")


if __name__ == "__main__":
    main()
