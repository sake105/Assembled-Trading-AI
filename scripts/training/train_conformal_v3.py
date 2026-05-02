#!/usr/bin/env python3
"""
train_conformal_v3.py — Conformal position-sizer v3 with panel-native feature names.

Key improvement over v2:
  v2 was trained with short names (rsi_14, macd_hist, ...) that required a runtime
  name-map translation in _tc_sizing.py. At inference only 7/13 features resolved;
  the other 6 were zero-filled, making intervals too wide and multipliers hit the
  0.25 floor. v3 uses panel-native names directly (ta_rsi_14_v1, etc.) so all 13
  features resolve at inference with zero translation overhead.

Model architecture: three LightGBM quantile regressors (q05, q50, q95) predicting
  fwd_return_20d. Interval width = q95 - q05; size multiplier = median_width / width,
  clipped to [0.25, 2.0].

Usage:
    python scripts/training/train_conformal_v3.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

PANEL_FILE = ROOT / "output" / "factor_panels" / "full_panel_7y.parquet"
MODEL_OUT  = ROOT / "models" / "conformal_position_v3.joblib"

# Panel-native feature names — match _tc_features output exactly.
# These are guaranteed to be present in prices_with_features at inference time.
FEATURE_COLS = [
    "ta_rsi_14_v1",
    "ta_macd_hist_v1",
    "ta_bb_pctb_v1",
    "ta_bb_bandwidth_v1",
    "ta_atr_14_v1",
    "ta_adx_v1",
    "ta_log_return_v1",
    "rv_20",
    "rv_60",
    "vov_20_60",
    "volume_zscore",
    "amihud_illiq_20d",
    "avg_corr_short",
]

TARGET_COL = "fwd_return_20d"
TRAIN_CUTOFF = "2025-01-01"
ALPHA_LOW  = 0.05   # q05
ALPHA_HIGH = 0.95   # q95


def main() -> None:
    print(f"[v3] Loading panel: {PANEL_FILE}")
    if not PANEL_FILE.exists():
        print(f"[ERROR] Panel not found: {PANEL_FILE}")
        sys.exit(1)

    df = pd.read_parquet(PANEL_FILE)
    print(f"     {len(df):,} rows, {df['symbol'].nunique()} symbols")

    if "date" not in df.columns:
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    df = df.dropna(subset=[TARGET_COL])
    print(f"[v3] After dropping NaN target: {len(df):,} rows")

    avail = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"[WARN] Missing features (will use available): {missing}")
    print(f"[v3] Using {len(avail)}/{len(FEATURE_COLS)} features")

    df_sorted = df.sort_values("date").reset_index(drop=True)
    X = df_sorted[avail].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df_sorted[TARGET_COL]

    # Chronological train/val split at TRAIN_CUTOFF
    train_mask = df_sorted["date"] < TRAIN_CUTOFF
    X_train, X_val = X[train_mask], X[~train_mask]
    y_train, y_val = y[train_mask], y[~train_mask]
    print(f"[v3] Train: {len(X_train):,} | Val: {len(X_val):,}")

    try:
        import lightgbm as lgb
    except ImportError:
        print("[ERROR] lightgbm not installed")
        sys.exit(1)

    _base_params = dict(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=50,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=42,
        verbose=-1,
    )

    models: dict = {}
    for alpha, label in [(ALPHA_LOW, "q05"), (0.50, "q50"), (ALPHA_HIGH, "q95")]:
        print(f"[v3] Training {label} (alpha={alpha})...")
        m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **_base_params)
        m.fit(X_train, y_train)
        models[f"{label}_model"] = m

    # Compute empirical coverage on val set
    q_lo  = models["q05_model"].predict(X_val)
    q_hi  = models["q95_model"].predict(X_val)
    widths = (q_hi - q_lo).clip(1e-8)

    coverage = float(((y_val.values >= q_lo) & (y_val.values <= q_hi)).mean())
    med_width = float(np.median(widths))
    width_std = float(widths.std())
    print(f"[v3] Val coverage:         {coverage:.4f}  (target >= 0.85)")
    print(f"[v3] Median interval width: {med_width:.6f}")
    print(f"[v3] Width std:            {width_std:.6f}")

    # Size-multiplier distribution diagnostic
    _q50_pred = models["q50_model"].predict(X_val)
    med_w = np.median(widths)
    multipliers = np.clip(med_w / widths, 0.25, 2.0)
    print(f"[v3] Multiplier — mean: {multipliers.mean():.3f}, "
          f"p25: {np.percentile(multipliers, 25):.3f}, "
          f"p75: {np.percentile(multipliers, 75):.3f}, "
          f"at-floor (0.25): {(multipliers == 0.25).mean():.2%}")

    artifact = {
        **models,
        "feature_cols": avail,
        "target_col": TARGET_COL,
        "alpha_low": ALPHA_LOW,
        "alpha_high": ALPHA_HIGH,
        "actual_coverage": coverage,
        "median_interval_width": med_width,
        "interval_width_std": width_std,
        "train_cutoff": TRAIN_CUTOFF,
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "model_type": "QuantileRegressionInterval_v2",
        "model_version": "v3",
        "notes": "Panel-native feature names; no runtime name-map translation required",
    }

    import joblib
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_OUT)
    print(f"[v3] Saved -> {MODEL_OUT}")

    if coverage >= 0.85:
        print("[v3] Coverage >= 0.85 -- model ready for deployment.")
        print("[v3] Next: set policy.yaml conformal_sizing.model_path = models/conformal_position_v3.joblib")
    else:
        print(f"[WARN][v3] Coverage {coverage:.3f} < 0.85 -- increase n_estimators or widen alpha.")


if __name__ == "__main__":
    main()
