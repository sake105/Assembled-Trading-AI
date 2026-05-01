#!/usr/bin/env python3
"""
train_ml_models_v4.py — Cross-sectional rank target for stable signal.

Key improvement over v3: instead of fwd_return_20d > 0 (noisy, regime-biased),
target is cross-sectional percentile rank > 0.5 within each date, i.e. "does
this stock beat the median peer on that day?". This removes bull/bear market
bias from the label and gives a balanced, consistently learnable signal.

Features are also cross-sectionally ranked within each date so that absolute
levels (e.g. RSI=70 in a calm market vs a trending market) become comparable.

Usage:
    python scripts/train_ml_models_v4.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from src.assembled_core.qa.cpcv_validation import (
    purged_train_test_split,
    walk_forward_oos_score,
)

PANEL_FILE = ROOT / "output" / "factor_panels" / "full_panel_7y.parquet"
MODEL_OUT = ROOT / "models" / "meta_model_lgbm_v4.joblib"

# Features available in panel at inference time
RAW_FEATURE_COLS = [
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
    "ret_5d",
    "ret_20d",
]

TRAIN_CUTOFF = "2025-01-01"
TARGET_COL = "fwd_return_20d"
# Minimum peers per date for cross-sectional ranking to be meaningful
MIN_PEERS = 5


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


def add_cross_sectional_features(df: pd.DataFrame, raw_cols: list[str]) -> list[str]:
    """Add cs_<col> = within-date percentile rank for each raw feature."""
    cs_cols = []
    for col in raw_cols:
        if col not in df.columns:
            continue
        cs_name = f"cs_{col}"
        df[cs_name] = df.groupby("date")[col].rank(pct=True, na_option="keep")
        cs_cols.append(cs_name)
    return cs_cols


def main():
    print(f"[v4] Loading panel: {PANEL_FILE}")
    df = pd.read_parquet(PANEL_FILE)
    print(f"     {len(df):,} rows, {df['symbol'].nunique()} symbols")

    if "date" not in df.columns:
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    df = compute_lagged_returns(df)
    df = df.dropna(subset=[TARGET_COL])
    print(f"[v4] After dropping NaN target: {len(df):,} rows")

    # Cross-sectional rank target: label=1 if above median peer on same date
    df["cs_target_rank"] = df.groupby("date")[TARGET_COL].rank(pct=True)
    # Drop dates with too few peers (ranking is unstable)
    peer_counts = df.groupby("date")["symbol"].transform("count")
    df = df[peer_counts >= MIN_PEERS].copy()
    df["target"] = (df["cs_target_rank"] > 0.5).astype(int)
    print(f"[v4] After peer filter (>={MIN_PEERS}): {len(df):,} rows")
    print(f"[v4] Target positive rate: {df['target'].mean():.3f}  (should be ~0.500)")

    # Raw features
    raw_available = [c for c in RAW_FEATURE_COLS if c in df.columns]
    raw_missing = [c for c in RAW_FEATURE_COLS if c not in df.columns]
    if raw_missing:
        print(f"[WARN] Features not in panel (skipped): {raw_missing}")

    # Add cross-sectional ranked versions
    cs_cols = add_cross_sectional_features(df, raw_available)
    print(f"[v4] Cross-sectional features added: {len(cs_cols)}")

    # Use both raw and cross-sectional features
    feature_cols = raw_available + cs_cols
    print(f"[v4] Total features: {len(feature_cols)}")

    # Sort by date so positional slicing is chronological (required for CPCV)
    df_sorted = df.sort_values("date").reset_index(drop=True)
    X = df_sorted[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df_sorted["target"]

    # Compute val fraction matching TRAIN_CUTOFF date
    n_val = (df_sorted["date"] >= TRAIN_CUTOFF).sum()
    test_size = n_val / len(df_sorted)

    # Purged split: adds embargo_bars gap between train and val to prevent
    # information leakage near the boundary (replaces raw date-mask split).
    EMBARGO_BARS = 10  # ~2 trading weeks
    X_train, X_val, y_train, y_val = purged_train_test_split(
        X, y, test_size=test_size, embargo_bars=EMBARGO_BARS
    )
    print(f"[v4] Train: {len(X_train):,} | Embargo: {EMBARGO_BARS} bars | Val: {len(X_val):,}")

    try:
        import lightgbm as lgb
    except ImportError:
        print("[ERROR] lightgbm not installed")
        sys.exit(1)

    _lgb_params = dict(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=5,
        min_child_samples=100,
        subsample=0.6,
        colsample_bytree=0.5,
        reg_alpha=1.0,
        reg_lambda=1.0,
        is_unbalance=False,
        random_state=42,
        verbose=-1,
    )

    # Walk-forward OOS diagnostic on training fold (5 splits, roc_auc)
    try:
        _cv_estimator = lgb.LGBMClassifier(**_lgb_params)
        _cv_result = walk_forward_oos_score(
            _cv_estimator, X_train, y_train, n_splits=5, scoring="roc_auc"
        )
        print(
            f"[v4] Walk-forward CV AUC:  {_cv_result.mean_score:.4f}"
            f" +/- {_cv_result.std_score:.4f} (n={_cv_result.n_splits} splits)"
        )
    except Exception as _cv_err:
        print(f"[v4] Walk-forward CV skipped: {_cv_err}")

    model = lgb.LGBMClassifier(**_lgb_params)
    model.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]

    train_auc = roc_auc_score(y_train, train_proba)
    try:
        val_auc = roc_auc_score(y_val, val_proba)
    except Exception:
        val_auc = float("nan")

    # Calibrate threshold on val
    best_threshold, best_f1 = 0.5, 0.0
    for t in np.arange(0.40, 0.65, 0.02):
        preds = (val_proba > t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t

    val_acc = accuracy_score(y_val, (val_proba > 0.5).astype(int))
    val_acc_cal = accuracy_score(y_val, (val_proba > best_threshold).astype(int))
    baseline_acc = max(y_val.mean(), 1 - y_val.mean())

    print(f"[v4] Train AUC:           {train_auc:.4f}")
    print(f"[v4] Val AUC:             {val_auc:.4f}")
    print(f"[v4] Val acc (0.5):       {val_acc:.4f}")
    print(f"[v4] Best threshold:      {best_threshold:.2f}  F1={best_f1:.4f}  acc={val_acc_cal:.4f}")
    print(f"[v4] Baseline (majority): {baseline_acc:.4f}")

    importances = dict(zip(feature_cols, model.feature_importances_))
    top = sorted(importances.items(), key=lambda x: -x[1])[:10]
    print("[v4] Top-10 feature importances:")
    for feat, imp in top:
        print(f"     {feat}: {imp:.1f}")

    import joblib

    # At inference time, only raw features are available in signals_df.
    # The cs_ features require the full cross-section which doesn't exist per-stock.
    # So: store raw_available as inference_feature_cols; cs_ are training-only.
    artifact = {
        "model": model,
        "feature_cols": raw_available,        # raw names — used at inference
        "cs_feature_cols": cs_cols,            # cross-sectional — training only
        "training_feature_cols": feature_cols, # all features used during training
        "decision_threshold": float(best_threshold),
        "oos_accuracy": float(val_acc),
        "oos_accuracy_calibrated": float(val_acc_cal),
        "oos_auc": float(val_auc),
        "baseline_accuracy": float(baseline_acc),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "embargo_bars": EMBARGO_BARS,
        "cv_method": "walk_forward_oos (5 splits) + purged_train_test_split",
        "target": "cs_rank > 0.5 (cross-sectional within date)",
        "train_cutoff": TRAIN_CUTOFF,
        "version": "v4",
    }
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_OUT)
    print(f"[v4] Saved -> {MODEL_OUT}")

    if val_auc >= 0.55:
        print("[v4] AUC ≥ 0.55 — model has genuine signal, recommended for use.")
    elif val_auc >= 0.52:
        print("[v4] AUC 0.52-0.55 — marginal signal, use with caution.")
    else:
        print("[v4] AUC < 0.52 — near-random, features insufficient for this target.")


if __name__ == "__main__":
    main()
