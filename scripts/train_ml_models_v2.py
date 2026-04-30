"""Retrain ML models v2: meta-model + conformal sizing.

Improvements over v1:
- Meta-model: HMM regime as feature, multi-timeframe momentum, is_unbalance=True,
  decision-threshold calibration on validation set
- Conformal: q05/q95 quantiles → targets ~90% empirical coverage (was 78% with q10/q90)
"""
from __future__ import annotations

import logging
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

YF_DIR = ROOT / "data" / "raw" / "equities_eod" / "yfinance"
TRAIN_CUTOFF = pd.Timestamp("2024-01-01", tz="UTC")
VAL_CUTOFF = pd.Timestamp("2024-07-01", tz="UTC")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_price_panel() -> pd.DataFrame:
    files = sorted(YF_DIR.glob("*.parquet"))
    log.info("Loading %d symbol files ...", len(files))
    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            if "timestamp" in df.columns and "close" in df.columns:
                frames.append(df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]])
        except Exception:
            pass
    panel = pd.concat(frames, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    panel = panel.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    log.info("Panel: %d rows, %d symbols", len(panel), panel["symbol"].nunique())
    return panel


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features_per_symbol(grp: pd.DataFrame) -> pd.DataFrame:
    g = grp.sort_values("timestamp").copy()
    c = g["close"]
    v = g["volume"].replace(0, np.nan)

    # Returns
    g["ret_1d"] = np.log(c / c.shift(1))
    g["ret_5d"] = np.log(c / c.shift(5))
    g["ret_10d"] = np.log(c / c.shift(10))
    g["ret_20d"] = np.log(c / c.shift(20))
    g["ret_60d"] = np.log(c / c.shift(60))

    # Volatility
    g["vol_10d"] = g["ret_1d"].rolling(10).std()
    g["vol_20d"] = g["ret_1d"].rolling(20).std()
    g["vol_60d"] = g["ret_1d"].rolling(60).std()
    g["vol_ratio"] = g["vol_10d"] / (g["vol_60d"] + 1e-8)
    g["vol_zscore"] = (g["vol_20d"] - g["vol_20d"].rolling(60).mean()) / (g["vol_20d"].rolling(60).std() + 1e-8)

    # MAs and position
    for w in [20, 50, 200]:
        ma = c.rolling(w).mean()
        g[f"ma{w}"] = ma
        g[f"pos_ma{w}"] = (c - ma) / (ma + 1e-8)

    # RSI
    for p in [7, 14, 21]:
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(p).mean()
        loss = (-delta.clip(upper=0)).rolling(p).mean()
        rs = gain / (loss + 1e-8)
        g[f"rsi_{p}"] = 100 - 100 / (1 + rs)

    # Bollinger band position
    bb_ma = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    g["bb_pos"] = (c - bb_ma) / (2 * bb_std + 1e-8)
    g["bb_width"] = (4 * bb_std) / (bb_ma + 1e-8)

    # ATR
    hl = g["high"] - g["low"]
    hpc = (g["high"] - c.shift(1)).abs()
    lpc = (g["low"] - c.shift(1)).abs()
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    g["atr_14"] = tr.rolling(14).mean()
    g["atr_norm"] = g["atr_14"] / (c + 1e-8)

    # Volume features
    vol_ma = v.rolling(20).mean()
    g["vol_zscore_20"] = (v - vol_ma) / (v.rolling(20).std() + 1e-8)
    g["vol_ratio_5_20"] = v.rolling(5).mean() / (vol_ma + 1e-8)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    g["macd_hist"] = (macd - signal) / (c + 1e-8)
    g["macd_sign"] = np.sign(macd - signal)

    return g


def add_market_regime(panel: pd.DataFrame) -> pd.DataFrame:
    """Add HMM regime label as a feature using the pre-trained model."""
    try:
        import joblib
        from src.assembled_core.ml.regime_hmm import MultiFeatureRegimeHMM

        hmm_path = MODELS_DIR / "regime_hmm_4state_spy.joblib"
        if not hmm_path.exists():
            log.warning("HMM artifact not found — skipping regime feature")
            panel["regime"] = "sideways"
            return panel

        hmm = MultiFeatureRegimeHMM.load(hmm_path)

        # Build equal-weight market return + vol
        px = panel.pivot_table(index="timestamp", columns="symbol", values="close", aggfunc="last")
        mkt = px.mean(axis=1).sort_index()
        log_ret = np.log(mkt / mkt.shift(1)).dropna()
        vol_20d = log_ret.rolling(20).std().dropna()
        log_ret = log_ret.loc[vol_20d.index]

        feat = pd.DataFrame(
            {"daily_return": log_ret.values, "realized_vol": vol_20d.values},
            index=log_ret.index,
        )
        regimes = hmm.predict_regime(feat)
        regime_map = regimes.to_dict()

        # Map regime to integer encoding
        regime_enc = {"bull": 3, "sideways": 2, "bear": 1, "crisis": 0}
        panel["regime"] = panel["timestamp"].map(regime_map).fillna("sideways")
        panel["regime_enc"] = panel["regime"].map(regime_enc).fillna(2).astype(int)
        log.info("Regime distribution: %s", panel["regime"].value_counts().to_dict())
        return panel
    except Exception as e:
        log.warning("Regime feature skipped: %s", e)
        panel["regime"] = "sideways"
        panel["regime_enc"] = 2
        return panel


# ---------------------------------------------------------------------------
# Triple-barrier labeling
# ---------------------------------------------------------------------------

def triple_barrier_labels(
    grp: pd.DataFrame,
    profit_target: float = 0.02,
    stop_loss: float = 0.01,
    max_horizon: int = 10,
) -> pd.DataFrame:
    closes = grp["close"].values
    n = len(closes)
    labels = np.full(n, np.nan)

    for i in range(n - 1):
        entry = closes[i]
        pt = entry * (1 + profit_target)
        sl = entry * (1 - stop_loss)
        for j in range(i + 1, min(i + max_horizon + 1, n)):
            if closes[j] >= pt:
                labels[i] = 1
                break
            if closes[j] <= sl:
                labels[i] = 0
                break
        else:
            labels[i] = 0  # timeout → no profit target hit

    grp = grp.copy()
    grp["label"] = labels
    return grp


# ---------------------------------------------------------------------------
# Meta-model v2 training
# ---------------------------------------------------------------------------

META_FEAT_COLS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "vol_10d", "vol_20d", "vol_60d", "vol_ratio", "vol_zscore",
    "pos_ma20", "pos_ma50", "pos_ma200",
    "rsi_7", "rsi_14", "rsi_21",
    "bb_pos", "bb_width",
    "atr_norm",
    "vol_zscore_20", "vol_ratio_5_20",
    "macd_hist", "macd_sign",
    "regime_enc",
]


def train_meta_model_v2(panel: pd.DataFrame) -> dict[str, Any]:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    log.info("=== Training meta-model v2 ===")

    # Build per-symbol features + labels
    log.info("Computing features ...")
    frames = []
    for sym, grp in panel.groupby("symbol"):
        feat = build_features_per_symbol(grp)
        feat = triple_barrier_labels(feat)
        frames.append(feat)
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["label"] + [c for c in META_FEAT_COLS if c in data.columns])

    # Align available features
    avail_feat = [c for c in META_FEAT_COLS if c in data.columns]
    log.info("Features: %d  Rows: %d", len(avail_feat), len(data))

    # Split
    train = data[data["timestamp"] < TRAIN_CUTOFF]
    val = data[(data["timestamp"] >= TRAIN_CUTOFF) & (data["timestamp"] < VAL_CUTOFF)]
    oos = data[data["timestamp"] >= VAL_CUTOFF]

    X_tr, y_tr = train[avail_feat].values, train["label"].values.astype(int)
    X_val, y_val = val[avail_feat].values, val["label"].values.astype(int)
    X_oos, y_oos = oos[avail_feat].values, oos["label"].values.astype(int)

    log.info("Train: %d  Val: %d  OOS: %d", len(X_tr), len(X_val), len(X_oos))
    log.info("Label distribution — Train: %.3f  Val: %.3f  OOS: %.3f",
             y_tr.mean(), y_val.mean(), y_oos.mean())

    # Train with class balancing
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        is_unbalance=True,   # key fix: handle class imbalance
        random_state=42,
        n_jobs=2,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
    )

    # Calibrate decision threshold on validation set
    val_proba = model.predict_proba(X_val)[:, 1]
    best_thresh, best_acc = 0.5, 0.0
    for thresh in np.arange(0.3, 0.7, 0.02):
        preds = (val_proba >= thresh).astype(int)
        acc = (preds == y_val).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    log.info("Optimal threshold on val: %.2f → val_acc=%.4f", best_thresh, best_acc)

    # OOS evaluation
    oos_proba = model.predict_proba(X_oos)[:, 1]
    oos_preds = (oos_proba >= best_thresh).astype(int)
    oos_acc = (oos_preds == y_oos).mean()
    oos_auc = roc_auc_score(y_oos, oos_proba) if len(np.unique(y_oos)) > 1 else 0.5
    baseline_acc = max(y_oos.mean(), 1 - y_oos.mean())

    log.info("OOS accuracy: %.4f  (baseline: %.4f  delta: %+.4f)",
             oos_acc, baseline_acc, oos_acc - baseline_acc)
    log.info("OOS AUC:      %.4f", oos_auc)

    # Precision at top 10% by confidence
    top10 = int(len(oos_proba) * 0.1)
    if top10 > 0:
        top_idx = np.argsort(oos_proba)[-top10:]
        prec_top10 = y_oos[top_idx].mean()
        log.info("Precision@top10%%: %.4f  (baseline: %.4f)", prec_top10, y_oos.mean())
    else:
        prec_top10 = float("nan")

    bundle = {
        "model": model,
        "feature_cols": avail_feat,
        "target": "triple_barrier_binary (profit_target=2%, stop_loss=1%, horizon=10)",
        "decision_threshold": best_thresh,
        "oos_accuracy": oos_acc,
        "oos_auc": oos_auc,
        "precision_top10pct": prec_top10,
        "baseline_accuracy": baseline_acc,
        "train_cutoff": str(TRAIN_CUTOFF.date()),
        "val_cutoff": str(VAL_CUTOFF.date()),
        "profit_target": 0.02,
        "stop_loss": 0.01,
        "max_horizon": 10,
        "model_version": "v2",
    }
    return bundle


# ---------------------------------------------------------------------------
# Conformal v2: q05/q95 for better coverage
# ---------------------------------------------------------------------------

CONF_FEAT_COLS = [
    "ret_5d", "ret_20d", "vol_20d", "vol_ratio",
    "pos_ma20", "pos_ma50", "rsi_14", "bb_pos", "bb_width", "atr_norm",
    "vol_zscore_20", "macd_hist", "regime_enc",
]


def train_conformal_v2(panel: pd.DataFrame) -> dict[str, Any]:
    import lightgbm as lgb

    log.info("=== Training conformal v2 (q05/q95) ===")

    frames = []
    for sym, grp in panel.groupby("symbol"):
        feat = build_features_per_symbol(grp)
        g = feat.copy()
        g["fwd_return_5d"] = np.log(g["close"].shift(-5) / g["close"])
        frames.append(g)
    data = pd.concat(frames, ignore_index=True)

    avail_feat = [c for c in CONF_FEAT_COLS if c in data.columns]
    data = data.dropna(subset=["fwd_return_5d"] + avail_feat)

    train = data[data["timestamp"] < TRAIN_CUTOFF]
    oos = data[data["timestamp"] >= VAL_CUTOFF]

    X_tr = train[avail_feat].values.astype(float)
    y_tr = train["fwd_return_5d"].values.astype(float)
    X_oos = oos[avail_feat].values.astype(float)
    y_oos = oos["fwd_return_5d"].values.astype(float)

    log.info("Train: %d  OOS: %d", len(X_tr), len(X_oos))

    base_params = dict(
        learning_rate=0.05, max_depth=5, n_estimators=200,
        colsample_bytree=0.7, subsample=0.8, n_jobs=2, verbose=-1, random_state=42,
    )

    # q05 model
    q05 = lgb.LGBMRegressor(objective="quantile", alpha=0.05, **base_params)
    q05.fit(X_tr, y_tr)

    # median model (q50)
    q50 = lgb.LGBMRegressor(objective="quantile", alpha=0.5, **base_params)
    q50.fit(X_tr, y_tr)

    # q95 model
    q95 = lgb.LGBMRegressor(objective="quantile", alpha=0.95, **base_params)
    q95.fit(X_tr, y_tr)

    # Evaluate empirical coverage on OOS
    pred_lo = q05.predict(X_oos)
    pred_hi = q95.predict(X_oos)
    pred_med = q50.predict(X_oos)
    widths = (pred_hi - pred_lo).clip(1e-8)
    coverage = float(((y_oos >= pred_lo) & (y_oos <= pred_hi)).mean())
    median_width = float(np.median(widths))
    width_std = float(widths.std())

    log.info("OOS coverage: %.4f  (target: ≥0.85)", coverage)
    log.info("Median interval width: %.6f  std: %.6f", median_width, width_std)

    bundle = {
        "q05_model": q05,
        "q50_model": q50,
        "q95_model": q95,
        "feature_cols": avail_feat,
        "target_col": "fwd_return_5d",
        "alpha_low": 0.05,
        "alpha_high": 0.95,
        "actual_coverage": coverage,
        "median_interval_width": median_width,
        "interval_width_std": width_std,
        "train_cutoff": str(TRAIN_CUTOFF.date()),
        "model_type": "QuantileRegressionInterval_v2",
        "model_version": "v2",
    }
    return bundle


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import joblib

    panel = load_price_panel()
    panel = add_market_regime(panel)

    # Meta-model v2
    meta_bundle = train_meta_model_v2(panel)
    meta_path = MODELS_DIR / "meta_model_lgbm_v2.joblib"
    joblib.dump(meta_bundle, meta_path)
    log.info("Saved meta-model v2 → %s", meta_path)

    # Conformal v2
    conf_bundle = train_conformal_v2(panel)
    conf_path = MODELS_DIR / "conformal_position_v2.joblib"
    joblib.dump(conf_bundle, conf_path)
    log.info("Saved conformal v2 → %s", conf_path)

    log.info("=== Training complete ===")
    log.info("meta_model_lgbm_v2: OOS acc=%.4f  baseline=%.4f  AUC=%.4f",
             meta_bundle["oos_accuracy"], meta_bundle["baseline_accuracy"], meta_bundle["oos_auc"])
    log.info("conformal_v2: coverage=%.4f  width_std=%.6f",
             conf_bundle["actual_coverage"], conf_bundle["interval_width_std"])


if __name__ == "__main__":
    main()
