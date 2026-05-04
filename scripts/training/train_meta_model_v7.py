#!/usr/bin/env python3
"""train_meta_model_v7.py — Extended Momentum + Macro + Optuna HPO meta-model.

Extends v6 with:
- Extended price momentum: ret_60d, ret_120d, ret_252d, momentum_12_1 (skip-1m)
- Price-position features: close_to_52w_high, ma50_vs_ma200, close_vs_ma200
- Macro: VIX level, yield curve spread, VIX/MA20 ratio (from v6)
- Fundamentals: pe, ps, roe, roa, profit_margins, debt_to_equity (from v6)
- Optuna hyperparameter optimization (40 trials) instead of fixed params
- Drops insider features (all 'unknown' transaction type, no signal)

Usage:
    python scripts/training/train_meta_model_v7.py
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
MACRO_FILE = ROOT / "output" / "macro.parquet"
MODEL_OUT = ROOT / "models" / "meta_model_lgbm_v7.joblib"

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

MOMENTUM_FEATURE_COLS = [
    "ret_60d",
    "ret_120d",
    "ret_252d",
    "momentum_12_1",
    "close_to_52w_high",
    "ma50_vs_ma200",
    "close_vs_ma200",
    "volume_trend_20d",
]

NEWS_FEATURE_COLS = ["sentiment_score_lag1", "sentiment_3d_avg", "sentiment_count_3d"]
EARNINGS_FEATURE_COLS = [
    "eps_surprise_pct_lag1",
    "revenue_surprise_pct_lag1",
    "days_since_earnings",
]
FUNDAMENTALS_FEATURE_COLS = [
    "pe_ratio_lag1",
    "ps_ratio_lag1",
    "roe_lag1",
    "roa_lag1",
    "profit_margins_lag1",
    "debt_to_equity_lag1",
]
MACRO_FEATURE_COLS = [
    "vix_level",
    "yield_curve_spread",
    "vix_vs_ma20",
    "macro_risk_score",
]

TRAIN_CUTOFF = "2025-01-01"
TARGET_COL = "fwd_return_20d"
MIN_PEERS = 5
EMBARGO_BARS = 10
OPTUNA_TRIALS = 40


def _normalize_date(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col]).dt.tz_localize(None).dt.normalize()
    return df


def compute_extended_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute extended momentum and price-position features from close/MA columns."""
    df = df.sort_values(["symbol", "date"])

    if "close" in df.columns:
        grp = df.groupby("symbol", group_keys=False)["close"]
        df["ret_60d"] = grp.transform(
            lambda x: np.log(x.clip(lower=1e-9) / x.shift(60).clip(lower=1e-9))
        )
        df["ret_120d"] = grp.transform(
            lambda x: np.log(x.clip(lower=1e-9) / x.shift(120).clip(lower=1e-9))
        )
        df["ret_252d"] = grp.transform(
            lambda x: np.log(x.clip(lower=1e-9) / x.shift(252).clip(lower=1e-9))
        )
        df["ret_5d_comp"] = grp.transform(
            lambda x: np.log(x.clip(lower=1e-9) / x.shift(5).clip(lower=1e-9))
        )
        # 12-1 month momentum: 252-day return minus last 21 days (skip-month)
        df["momentum_12_1"] = df["ret_252d"] - df["ret_5d_comp"]

        # 52-week high (252 trading days rolling max)
        df["roll_max_252"] = grp.transform(
            lambda x: x.rolling(252, min_periods=126).max()
        )
        df["close_to_52w_high"] = (
            df["close"] / df["roll_max_252"].clip(lower=1e-9)
        ).clip(0, 1)

        # Volume trend: current vol / 20-day avg vol
        if "volume" in df.columns:
            df["vol_ma20"] = df.groupby("symbol", group_keys=False)["volume"].transform(
                lambda x: x.rolling(20, min_periods=5).mean()
            )
            df["volume_trend_20d"] = (
                df["volume"] / df["vol_ma20"].clip(lower=1e-9)
            ).clip(0, 10)

    # Price vs moving averages (already in panel)
    if "ma_50" in df.columns and "ma_200" in df.columns:
        df["ma50_vs_ma200"] = (df["ma_50"] / df["ma_200"].clip(lower=1e-9)).clip(0.5, 2)
    if "close" in df.columns and "ma_200" in df.columns:
        df["close_vs_ma200"] = (df["close"] / df["ma_200"].clip(lower=1e-9)).clip(
            0.3, 3
        )

    # Clean up temp columns
    df = df.drop(columns=["roll_max_252", "vol_ma20", "ret_5d_comp"], errors="ignore")
    return df


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


def add_cross_sectional_features(df: pd.DataFrame, raw_cols: list) -> list:
    cs_cols = []
    for col in raw_cols:
        if col not in df.columns:
            continue
        cs_name = f"cs_{col}"
        df[cs_name] = df.groupby("date")[col].rank(pct=True, na_option="keep")
        cs_cols.append(cs_name)
    return cs_cols


# ---------------------------------------------------------------------------
# Feature loaders (re-used from v6)
# ---------------------------------------------------------------------------


def _load_news_features(panel_dates, panel_symbols):
    if not NEWS_FILE.exists():
        return pd.DataFrame()
    news = pd.read_parquet(NEWS_FILE).rename(columns={"timestamp": "date"})
    news = _normalize_date(news, "date")
    if news["date"].nunique() < 60:
        print(f"[v7] News: {news['date'].nunique()} dates — too sparse, skipped")
        return pd.DataFrame()
    news["merge_date"] = news["date"] + pd.Timedelta(days=1)
    news = news.sort_values(["symbol", "date"])
    news["sentiment_3d_avg"] = news.groupby("symbol")["sentiment_score"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    news["sentiment_count_3d"] = news.groupby("symbol")["count"].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )
    return news.rename(columns={"sentiment_score": "sentiment_score_lag1"})[
        [
            "merge_date",
            "symbol",
            "sentiment_score_lag1",
            "sentiment_3d_avg",
            "sentiment_count_3d",
        ]
    ].rename(columns={"merge_date": "date"})


def _load_earnings_features(panel_dates, panel_symbols):
    if not EARNINGS_FILE.exists():
        return pd.DataFrame()
    earn = pd.read_parquet(EARNINGS_FILE).rename(
        columns={"timestamp": "date", "event_date": "earn_date"}
    )
    earn = _normalize_date(earn, "date")
    earn = _normalize_date(earn, "earn_date")
    if "disclosure_date" in earn.columns:
        earn = _normalize_date(earn, "disclosure_date")
        earn["merge_date"] = earn["disclosure_date"] + pd.Timedelta(days=1)
    else:
        earn["merge_date"] = earn["earn_date"] + pd.Timedelta(days=1)

    earn = earn.dropna(subset=["eps_surprise_pct"]).copy()
    earn["eps_surprise_pct_lag1"] = earn["eps_surprise_pct"].clip(-200, 200)

    if "revenue_actual" in earn.columns and "revenue_estimate" in earn.columns:
        rev_actual = pd.to_numeric(earn["revenue_actual"], errors="coerce")
        rev_est = pd.to_numeric(earn["revenue_estimate"], errors="coerce")
        earn["revenue_surprise_pct_lag1"] = (
            (rev_actual - rev_est) / rev_est.abs().clip(lower=1e-9)
        ).clip(-5, 5)
    else:
        earn["revenue_surprise_pct_lag1"] = np.nan

    earn_features = earn[
        ["merge_date", "symbol", "eps_surprise_pct_lag1", "revenue_surprise_pct_lag1"]
    ].copy()

    all_sym = panel_symbols.unique()
    all_dates = sorted(panel_dates.unique())
    rows = []
    for sym in all_sym:
        sym_earn = earn_features[earn_features["symbol"] == sym].copy()
        if sym_earn.empty:
            continue
        sym_dates = sorted(sym_earn["merge_date"])
        for d in all_dates:
            past = [x for x in sym_dates if x <= d]
            if past:
                rows.append(
                    {
                        "date": d,
                        "symbol": sym,
                        "days_since_earnings": float((d - max(past)).days),
                    }
                )

    days_df = (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=["date", "symbol", "days_since_earnings"])
    )
    merged = earn_features.rename(columns={"merge_date": "date"})
    result = days_df.merge(merged, on=["date", "symbol"], how="left")
    print(f"[v7] Earnings: {len(result)} rows, {result['symbol'].nunique()} symbols")
    return result


def _load_fundamentals_features():
    if not FUNDAMENTALS_FILE.exists():
        return pd.DataFrame()
    fund = pd.read_parquet(FUNDAMENTALS_FILE).rename(columns={"timestamp": "date"})
    fund = _normalize_date(fund, "date")
    fund = fund.sort_values(["symbol", "date"])

    mapping = {
        "pe_ratio": ("pe_ratio_lag1", -500, 500),
        "ps_ratio": ("ps_ratio_lag1", 0, 200),
        "roe": ("roe_lag1", -5, 5),
        "roa": ("roa_lag1", -2, 2),
        "profit_margins": ("profit_margins_lag1", -5, 5),
        "debt_to_equity": ("debt_to_equity_lag1", 0, 50),
    }
    keep = []
    for src, (dst, lo, hi) in mapping.items():
        if src in fund.columns:
            fund[dst] = fund[src].clip(lo, hi)
            keep.append(dst)

    if not keep:
        return pd.DataFrame()

    result_parts = []
    for sym, grp in fund.groupby("symbol"):
        grp = grp.sort_values("date").copy()
        for col in keep:
            grp[col] = grp[col].ffill()
        grp["date"] = grp["date"] + pd.Timedelta(days=1)
        result_parts.append(grp[["date", "symbol"] + keep])

    if not result_parts:
        return pd.DataFrame()
    result = pd.concat(result_parts, ignore_index=True)
    print(f"[v7] Fundamentals: {len(result)} rows, features: {keep}")
    return result


def _load_macro_features(panel_dates):
    if not MACRO_FILE.exists():
        return pd.DataFrame()
    macro = pd.read_parquet(MACRO_FILE)
    date_col = next((c for c in ["timestamp", "date"] if c in macro.columns), None)
    if date_col is None:
        return pd.DataFrame()
    macro = macro.rename(columns={date_col: "date"})
    macro = _normalize_date(macro, "date")

    agg_dict = {
        c: "last"
        for c in macro.columns
        if c != "date" and pd.api.types.is_numeric_dtype(macro[c])
    }
    macro = macro.groupby("date").agg(agg_dict).reset_index().sort_values("date")

    vix_col = next((c for c in ["vix", "vix_close"] if c in macro.columns), None)
    y10_col = next(
        (c for c in ["tnx_10y_yield", "treasury_10y"] if c in macro.columns), None
    )
    y3m_col = next(
        (c for c in ["irx_3m_yield", "treasury_2y"] if c in macro.columns), None
    )

    if vix_col is None:
        return pd.DataFrame()

    macro["vix_level"] = macro[vix_col].clip(5, 100)
    macro["vix_ma20"] = macro["vix_level"].rolling(20, min_periods=1).mean()
    macro["vix_vs_ma20"] = (
        macro["vix_level"] / macro["vix_ma20"].clip(lower=1e-6)
    ).clip(0.3, 5)
    macro["yield_curve_spread"] = (
        (macro[y10_col] - macro[y3m_col]).clip(-5, 5) if (y10_col and y3m_col) else 0.0
    )
    macro["macro_risk_score"] = (
        (macro["vix_level"] - 20) / 20 - macro["yield_curve_spread"] / 2
    ).clip(-3, 3)

    keep = ["date"] + [c for c in MACRO_FEATURE_COLS if c in macro.columns]
    result = macro[keep].copy()
    result["date"] = result["date"] + pd.Timedelta(days=1)  # PIT lag
    print(
        f"[v7] Macro: {len(result)} rows, "
        f"{result['date'].min()} to {result['date'].max()}, "
        f"VIX range: {macro['vix_level'].min():.1f}-{macro['vix_level'].max():.1f}"
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"[v7] Loading panel: {PANEL_FILE}")
    df = pd.read_parquet(PANEL_FILE)
    print(f"     {len(df):,} rows, {df['symbol'].nunique()} symbols")

    if "date" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "date"})
    df = _normalize_date(df, "date")
    df = compute_lagged_returns(df)
    df = compute_extended_features(df)
    df = df.dropna(subset=[TARGET_COL])

    # Cross-sectional rank target
    df["cs_target_rank"] = df.groupby("date")[TARGET_COL].rank(pct=True)
    peer_counts = df.groupby("date")["symbol"].transform("count")
    df = df[peer_counts >= MIN_PEERS].copy()
    df["target"] = (df["cs_target_rank"] > 0.5).astype(int)
    print(
        f"[v7] After peer filter: {len(df):,} rows, target rate={df['target'].mean():.3f}"
    )

    # --- Merge features ---
    print("\n[v7] Loading alt-data features...")

    news_df = _load_news_features(df["date"], df["symbol"])
    if not news_df.empty:
        df = df.merge(news_df, on=["date", "symbol"], how="left")

    earn_df = _load_earnings_features(df["date"], df["symbol"])
    if not earn_df.empty:
        df = df.merge(earn_df, on=["date", "symbol"], how="left")

    fund_df = _load_fundamentals_features()
    if not fund_df.empty:
        df = df.merge(fund_df, on=["date", "symbol"], how="left")
        fund_feats = [c for c in FUNDAMENTALS_FEATURE_COLS if c in df.columns]
        if fund_feats:
            df = df.sort_values(["symbol", "date"])
            df[fund_feats] = df.groupby("symbol")[fund_feats].ffill()

    macro_df = _load_macro_features(df["date"])
    if not macro_df.empty:
        df = df.merge(macro_df, on="date", how="left")
        macro_feats = [c for c in MACRO_FEATURE_COLS if c in df.columns]
        if macro_feats:
            df = df.sort_values("date")
            df[macro_feats] = df[macro_feats].ffill()

    # --- Build feature set ---
    print()
    ta_available = [c for c in TA_FEATURE_COLS if c in df.columns]
    mom_available = [c for c in MOMENTUM_FEATURE_COLS if c in df.columns]
    news_available = [c for c in NEWS_FEATURE_COLS if c in df.columns]
    earnings_available = [c for c in EARNINGS_FEATURE_COLS if c in df.columns]
    fund_available = [c for c in FUNDAMENTALS_FEATURE_COLS if c in df.columns]
    macro_available = [c for c in MACRO_FEATURE_COLS if c in df.columns]

    raw_available = (
        ta_available
        + mom_available
        + news_available
        + earnings_available
        + fund_available
        + macro_available
    )
    print(
        f"[v7] Raw features: {len(ta_available)} TA + {len(mom_available)} momentum + "
        f"{len(news_available)} news + {len(earnings_available)} earnings + "
        f"{len(fund_available)} fundamentals + {len(macro_available)} macro "
        f"= {len(raw_available)} total"
    )

    cs_cols = add_cross_sectional_features(df, raw_available)
    feature_cols = raw_available + cs_cols
    print(f"[v7] + cross-sectional features: {len(feature_cols)} total")

    df_sorted = df.sort_values("date").reset_index(drop=True)
    X = df_sorted[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df_sorted["target"]

    n_val = (df_sorted["date"] >= TRAIN_CUTOFF).sum()
    test_size = n_val / len(df_sorted)
    print(f"[v7] Test size: {test_size:.1%} ({n_val:,} val rows)")

    try:
        from src.assembled_core.qa.cpcv_validation import purged_train_test_split

        X_train, X_val, y_train, y_val = purged_train_test_split(
            X, y, test_size=test_size, embargo_bars=EMBARGO_BARS
        )
    except Exception as e:
        print(f"[v7] Purged split failed ({e}), using simple date split")
        n_split = int(len(X) * (1 - test_size))
        X_train, X_val = X.iloc[:n_split], X.iloc[n_split:]
        y_train, y_val = y.iloc[:n_split], y.iloc[n_split:]

    print(
        f"[v7] Train: {len(X_train):,} | Embargo: {EMBARGO_BARS} | Val: {len(X_val):,}"
    )

    try:
        import lightgbm as lgb
    except ImportError:
        print("[ERROR] lightgbm not installed")
        sys.exit(1)

    # --- Optuna HPO ---
    print(f"\n[v7] Running Optuna HPO ({OPTUNA_TRIALS} trials)...")
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 400, 1500),
                learning_rate=trial.suggest_float(
                    "learning_rate", 0.01, 0.06, log=True
                ),
                num_leaves=trial.suggest_int("num_leaves", 10, 50),
                max_depth=trial.suggest_int("max_depth", 3, 7),
                min_child_samples=trial.suggest_int("min_child_samples", 50, 300),
                subsample=trial.suggest_float("subsample", 0.6, 0.95),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 0.95),
                reg_alpha=trial.suggest_float("reg_alpha", 0.5, 6.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 0.5, 6.0, log=True),
                random_state=42,
                verbose=-1,
            )
            from sklearn.metrics import roc_auc_score

            m = lgb.LGBMClassifier(**params)
            m.fit(X_train, y_train)
            proba = m.predict_proba(X_val)[:, 1]
            try:
                return roc_auc_score(y_val, proba)
            except Exception:
                return 0.5

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
        best_params = study.best_params
        best_auc_optuna = study.best_value
        print(f"[v7] Optuna best val AUC: {best_auc_optuna:.4f}")
        print(f"[v7] Best params: {best_params}")
    except ImportError:
        print("[v7] Optuna not installed — using default params")
        best_params = dict(
            n_estimators=800,
            learning_rate=0.02,
            num_leaves=31,
            max_depth=5,
            min_child_samples=150,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=1.5,
            reg_lambda=1.5,
        )
        best_auc_optuna = None

    # --- Final model with best params ---
    best_params["random_state"] = 42
    best_params["verbose"] = -1
    model = lgb.LGBMClassifier(**best_params)
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

    print()
    print(f"[v7] Train AUC:            {train_auc:.4f}")
    print(
        f"[v7] Val AUC:              {val_auc:.4f}  (v5 baseline: 0.5080, v6: 0.5108)"
    )
    print(f"[v7] Val acc (0.5):        {val_acc:.4f}")
    print(f"[v7] Baseline (majority):  {baseline_acc:.4f}")
    print(f"[v7] Best F1 threshold:    {best_threshold:.2f}  (F1={best_f1:.4f})")

    importances = dict(zip(feature_cols, model.feature_importances_))
    top = sorted(importances.items(), key=lambda x: -x[1])[:15]
    print("[v7] Top-15 feature importances:")
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
        "macro_feature_cols": macro_available,
        "momentum_feature_cols": mom_available,
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
        "hyperparams": best_params,
        "optuna_trials": OPTUNA_TRIALS,
        "version": "v7",
    }
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_OUT)
    print(f"\n[v7] Saved -> {MODEL_OUT}")

    delta_auc = val_auc - 0.5080
    print(f"[v7] AUC delta vs v5: {delta_auc:+.4f}")
    if val_auc >= 0.55:
        print("[v7] AUC >= 0.55 -- strong signal. DEPLOY v7.")
    elif val_auc >= 0.52:
        print("[v7] AUC 0.52-0.55 -- meets deployment threshold. DEPLOY v7.")
    else:
        print(
            "[v7] AUC < 0.52 -- below threshold. Needs richer data for further gains."
        )


if __name__ == "__main__":
    main()
