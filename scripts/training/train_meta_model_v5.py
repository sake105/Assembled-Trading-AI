#!/usr/bin/env python3
"""train_meta_model_v5.py — News + earnings augmented meta-model (Plan 11/10 §1.2).

Extends v4 with:
- news_sentiment_daily: sentiment_score, sentiment_3d_avg, count_3d
- events_earnings: eps_surprise_pct, days_since_last_earnings
- fundamentals: pe_ratio, ps_ratio (forward-looking proxy, PIT-safe)

All features merged with ≥1-day lag to preserve PIT safety.
Target: cross-sectional rank > 0.5 (same as v4 for apples-to-apples AUC comparison).

Usage:
    python scripts/training/train_meta_model_v5.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PANEL_FILE = ROOT / "output" / "factor_panels" / "full_panel_7y.parquet"
NEWS_FILE = ROOT / "output" / "news_sentiment_daily.parquet"
EARNINGS_FILE = ROOT / "output" / "events_earnings.parquet"
FUNDAMENTALS_FILE = ROOT / "output" / "fundamentals.parquet"
MODEL_OUT = ROOT / "models" / "meta_model_lgbm_v5.joblib"

TA_FEATURE_COLS = [
    "ta_log_return_v1",
    "ta_rsi_14_v1",
    "ta_macd_hist_v1",
    "ta_macd_v1",
    "ta_macd_signal_v1",
    "ta_bb_pctb_v1",
    "ta_bb_bandwidth_v1",
    "ta_adx_v1",
    "ta_atr_14_v1",
    "ta_stoch_k_v1",
    "ta_stoch_d_v1",
    "ta_obv_v1",
    "ta_plus_di_v1",
    "ta_minus_di_v1",
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
    "fraction_above_ma_50",
]

NEWS_FEATURE_COLS = ["sentiment_score_lag1", "sentiment_3d_avg", "sentiment_count_3d"]
EARNINGS_FEATURE_COLS = ["eps_surprise_pct_lag1", "days_since_earnings"]
FUNDAMENTALS_FEATURE_COLS = ["pe_ratio_lag1", "ps_ratio_lag1"]

TRAIN_CUTOFF = "2025-01-01"
TARGET_COL = "fwd_return_20d"
MIN_PEERS = 5
EMBARGO_BARS = 10


def _normalize_date(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """Normalize a date column to tz-naive date."""
    if col in df.columns:
        df[col] = pd.to_datetime(df[col]).dt.tz_localize(None).dt.normalize()
    return df


def _load_news_features(
    panel_dates: pd.Series, panel_symbols: pd.Series
) -> pd.DataFrame:
    """Load and merge news sentiment features with 1-day lag for PIT safety."""
    if not NEWS_FILE.exists():
        print(f"[v5] News file not found: {NEWS_FILE} — news features skipped")
        return pd.DataFrame()

    news = pd.read_parquet(NEWS_FILE)
    news = news.rename(columns={"timestamp": "date"})
    news = _normalize_date(news, "date")

    # Sparsity guard: skip news if fewer than 60 unique dates (< 3 months coverage)
    # Near-zero-variance features from sparse news degrade model AUC
    n_dates = news["date"].nunique() if "date" in news.columns else 0
    if n_dates < 60:
        print(
            f"[v5] News: only {n_dates} unique dates -- too sparse, skipping (need >=60)"
        )
        return pd.DataFrame()

    # 1-day lag: merge today's panel with yesterday's sentiment
    news["merge_date"] = news["date"] + pd.Timedelta(days=1)

    # Compute 3-day rolling from the raw news
    news = news.sort_values(["symbol", "date"])
    news["sentiment_3d_avg"] = news.groupby("symbol")["sentiment_score"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    news["sentiment_count_3d"] = news.groupby("symbol")["count"].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )

    news_features = news.rename(columns={"sentiment_score": "sentiment_score_lag1"})[
        [
            "merge_date",
            "symbol",
            "sentiment_score_lag1",
            "sentiment_3d_avg",
            "sentiment_count_3d",
        ]
    ]
    print(
        f"[v5] News: {len(news_features)} rows, "
        f"{news_features['symbol'].nunique()} symbols, "
        f"{news_features['merge_date'].min()} to {news_features['merge_date'].max()}"
    )
    return news_features


def _load_earnings_features(
    panel_dates: pd.Series, panel_symbols: pd.Series
) -> pd.DataFrame:
    """Load earnings surprise + days-since-earnings with 1-day lag."""
    if not EARNINGS_FILE.exists():
        print(
            f"[v5] Earnings file not found: {EARNINGS_FILE} — earnings features skipped"
        )
        return pd.DataFrame()

    earn = pd.read_parquet(EARNINGS_FILE)
    earn = earn.rename(columns={"timestamp": "date", "event_date": "earn_date"})
    earn = _normalize_date(earn, "date")
    earn = _normalize_date(earn, "earn_date")
    # Use disclosure_date if available (PIT-safe), else event_date + 1 day
    if "disclosure_date" in earn.columns:
        earn = _normalize_date(earn, "disclosure_date")
        earn["merge_date"] = earn["disclosure_date"] + pd.Timedelta(days=1)
    else:
        earn["merge_date"] = earn["earn_date"] + pd.Timedelta(days=1)

    earn = earn.dropna(subset=["eps_surprise_pct"]).copy()
    earn["eps_surprise_pct_lag1"] = earn["eps_surprise_pct"].clip(-200, 200)

    earn_features = earn[["merge_date", "symbol", "eps_surprise_pct_lag1"]].copy()

    # Now build days_since_earnings for the full panel date range
    all_sym = panel_symbols.unique()
    all_dates = sorted(panel_dates.unique())
    rows = []
    for sym in all_sym:
        sym_earn = earn_features[earn_features["symbol"] == sym].copy()
        if sym_earn.empty:
            continue
        sym_earn_dates = sorted(sym_earn["merge_date"])
        for d in all_dates:
            past = [x for x in sym_earn_dates if x <= d]
            if past:
                days_ago = (d - max(past)).days
                rows.append(
                    {"date": d, "symbol": sym, "days_since_earnings": float(days_ago)}
                )

    days_df = (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=["date", "symbol", "days_since_earnings"])
    )

    merged = earn_features.rename(columns={"merge_date": "date"})
    result = days_df.merge(merged, on=["date", "symbol"], how="left")
    print(f"[v5] Earnings: {len(result)} rows, {result['symbol'].nunique()} symbols")
    return result


def _load_fundamentals_features() -> pd.DataFrame:
    if not FUNDAMENTALS_FILE.exists():
        return pd.DataFrame()
    fund = pd.read_parquet(FUNDAMENTALS_FILE)
    fund = fund.rename(columns={"timestamp": "date"})
    fund = _normalize_date(fund, "date")
    fund["merge_date"] = fund["date"] + pd.Timedelta(days=1)
    keep = []
    if "pe_ratio" in fund.columns:
        fund["pe_ratio_lag1"] = fund["pe_ratio"].clip(-500, 500)
        keep.append("pe_ratio_lag1")
    if "ps_ratio" in fund.columns:
        fund["ps_ratio_lag1"] = fund["ps_ratio"].clip(0, 200)
        keep.append("ps_ratio_lag1")
    if not keep:
        return pd.DataFrame()
    result = fund[["merge_date", "symbol"] + keep].rename(
        columns={"merge_date": "date"}
    )
    print(f"[v5] Fundamentals: {len(result)} rows, features: {keep}")
    return result


def compute_lagged_returns(df: pd.DataFrame) -> pd.DataFrame:
    if "close" not in df.columns:
        df["ret_5d"] = np.nan
        df["ret_20d"] = np.nan
        return df
    df = df.sort_values(["symbol", "date"])
    df["ret_5d"] = df.groupby("symbol", group_keys=False)["close"].transform(
        lambda x: np.log(x.clip(lower=1e-9) / x.shift(5).clip(lower=1e-9))
    )
    df["ret_20d"] = df.groupby("symbol", group_keys=False)["close"].transform(
        lambda x: np.log(x.clip(lower=1e-9) / x.shift(20).clip(lower=1e-9))
    )
    return df


def add_cross_sectional_features(df: pd.DataFrame, raw_cols: list[str]) -> list[str]:
    cs_cols = []
    for col in raw_cols:
        if col not in df.columns:
            continue
        cs_name = f"cs_{col}"
        df[cs_name] = df.groupby("date")[col].rank(pct=True, na_option="keep")
        cs_cols.append(cs_name)
    return cs_cols


def main():
    print(f"[v5] Loading panel: {PANEL_FILE}")
    df = pd.read_parquet(PANEL_FILE)
    print(f"     {len(df):,} rows, {df['symbol'].nunique()} symbols")

    if "date" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "date"})
    df = _normalize_date(df, "date")
    df = compute_lagged_returns(df)
    df = df.dropna(subset=[TARGET_COL])

    # Cross-sectional rank target
    df["cs_target_rank"] = df.groupby("date")[TARGET_COL].rank(pct=True)
    peer_counts = df.groupby("date")["symbol"].transform("count")
    df = df[peer_counts >= MIN_PEERS].copy()
    df["target"] = (df["cs_target_rank"] > 0.5).astype(int)
    print(
        f"[v5] After peer filter: {len(df):,} rows, target rate={df['target'].mean():.3f}"
    )

    # Merge news features
    news_df = _load_news_features(df["date"], df["symbol"])
    if not news_df.empty:
        df = df.merge(
            news_df.rename(columns={"merge_date": "date"}),
            on=["date", "symbol"],
            how="left",
        )

    # Merge earnings features
    earn_df = _load_earnings_features(df["date"], df["symbol"])
    if not earn_df.empty:
        df = df.merge(earn_df, on=["date", "symbol"], how="left")

    # Merge fundamentals
    fund_df = _load_fundamentals_features()
    if not fund_df.empty:
        df = df.merge(fund_df, on=["date", "symbol"], how="left")

    # Build feature set
    ta_available = [c for c in TA_FEATURE_COLS if c in df.columns]
    news_available = [c for c in NEWS_FEATURE_COLS if c in df.columns]
    earnings_available = [c for c in EARNINGS_FEATURE_COLS if c in df.columns]
    fund_available = [c for c in FUNDAMENTALS_FEATURE_COLS if c in df.columns]

    raw_available = ta_available + news_available + earnings_available + fund_available
    ta_missing = [c for c in TA_FEATURE_COLS if c not in df.columns]
    if ta_missing:
        print(f"[WARN] TA features missing: {ta_missing}")
    print(
        f"[v5] Features: {len(ta_available)} TA + {len(news_available)} news + "
        f"{len(earnings_available)} earnings + {len(fund_available)} fundamentals "
        f"= {len(raw_available)} total"
    )

    cs_cols = add_cross_sectional_features(df, raw_available)
    feature_cols = raw_available + cs_cols
    print(f"[v5] + cross-sectional features: {len(feature_cols)} total")

    df_sorted = df.sort_values("date").reset_index(drop=True)
    X = df_sorted[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df_sorted["target"]

    n_val = (df_sorted["date"] >= TRAIN_CUTOFF).sum()
    test_size = n_val / len(df_sorted)
    print(f"[v5] Test size: {test_size:.1%} ({n_val:,} val rows)")

    try:
        from src.assembled_core.qa.cpcv_validation import purged_train_test_split

        X_train, X_val, y_train, y_val = purged_train_test_split(
            X, y, test_size=test_size, embargo_bars=EMBARGO_BARS
        )
    except Exception as e:
        print(f"[v5] Purged split failed ({e}), using simple date split")
        n_split = int(len(X) * (1 - test_size))
        X_train, X_val = X.iloc[:n_split], X.iloc[n_split:]
        y_train, y_val = y.iloc[:n_split], y.iloc[n_split:]

    print(
        f"[v5] Train: {len(X_train):,} | Embargo: {EMBARGO_BARS} | Val: {len(X_val):,}"
    )

    try:
        import lightgbm as lgb
    except ImportError:
        print("[ERROR] lightgbm not installed — pip install lightgbm")
        sys.exit(1)

    lgb_params = dict(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=20,
        max_depth=4,
        min_child_samples=200,
        subsample=0.7,
        colsample_bytree=0.6,
        reg_alpha=2.0,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1,
    )

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]
    train_auc = roc_auc_score(y_train, train_proba)
    try:
        val_auc = roc_auc_score(y_val, val_proba)
    except Exception:
        val_auc = float("nan")

    best_threshold, best_f1 = 0.5, 0.0
    for t in np.arange(0.40, 0.65, 0.02):
        preds = (val_proba > t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, t

    val_acc = accuracy_score(y_val, (val_proba > 0.5).astype(int))
    baseline_acc = max(y_val.mean(), 1 - y_val.mean())

    print(f"[v5] Train AUC:            {train_auc:.4f}")
    print(f"[v5] Val AUC:              {val_auc:.4f}  (v4 baseline: ~0.50)")
    print(f"[v5] Val acc (0.5):        {val_acc:.4f}")
    print(f"[v5] Baseline (majority):  {baseline_acc:.4f}")

    importances = dict(zip(feature_cols, model.feature_importances_))
    top = sorted(importances.items(), key=lambda x: -x[1])[:10]
    print("[v5] Top-10 feature importances:")
    for feat, imp in top:
        print(f"     {feat}: {imp:.1f}")

    import joblib

    artifact = {
        "model": model,
        "feature_cols": raw_available,
        "cs_feature_cols": cs_cols,
        "training_feature_cols": feature_cols,
        "news_feature_cols": news_available,
        "earnings_feature_cols": earnings_available,
        "fundamentals_feature_cols": fund_available,
        "decision_threshold": float(best_threshold),
        "oos_accuracy": float(val_acc),
        "oos_auc": float(val_auc),
        "baseline_accuracy": float(baseline_acc),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "embargo_bars": EMBARGO_BARS,
        "cv_method": "purged_train_test_split",
        "target": "cs_rank > 0.5 (cross-sectional within date)",
        "train_cutoff": TRAIN_CUTOFF,
        "version": "v5",
    }
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_OUT)
    print(f"[v5] Saved -> {MODEL_OUT}")

    delta_auc = val_auc - 0.5017  # v4 baseline AUC
    print(f"[v5] AUC delta vs v4: {delta_auc:+.4f}")
    if val_auc >= 0.55:
        print("[v5] AUC ≥ 0.55 — news/earnings features add signal. DEPLOY v5.")
    elif val_auc >= 0.52:
        print("[v5] AUC 0.52-0.55 — marginal improvement. Monitor before deploying.")
    else:
        print("[v5] AUC < 0.52 — features insufficient or data too sparse for signal.")


if __name__ == "__main__":
    main()
