#!/usr/bin/env python3
"""train_meta_model_v6.py — Insider + Macro + Extended Fundamentals meta-model.

Extends v5 with:
- insider_trading: net buy pressure (90-day rolling, lagged by filing_date+1)
- macro: VIX level, yield curve spread (10y-3m), VIX/20d-MA ratio (date join)
- fundamentals: roe, roa, profit_margins, debt_to_equity (forward-fill, lag+1d)
- earnings: revenue_surprise_pct (from revenue_actual vs revenue_estimate, lag+1d)

All features PIT-safe: merged with ≥1-day lag from disclosure/filing date.
Target: cross-sectional rank > 0.5 within each date (same as v4/v5).
Hyperparameters: slightly relaxed regularization to accommodate richer feature set.

Usage:
    python scripts/training/train_meta_model_v6.py
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
INSIDER_FILE = ROOT / "output" / "insider_trading.parquet"
MACRO_FILE = ROOT / "output" / "macro.parquet"
MODEL_OUT = ROOT / "models" / "meta_model_lgbm_v6.joblib"

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
INSIDER_FEATURE_COLS = [
    "insider_net_buy_90d",
    "insider_buy_ratio_90d",
    "insider_txn_count_90d",
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


def _normalize_date(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col]).dt.tz_localize(None).dt.normalize()
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
# Feature loaders
# ---------------------------------------------------------------------------


def _load_news_features(panel_dates, panel_symbols) -> pd.DataFrame:
    if not NEWS_FILE.exists():
        print("[v6] News file not found — skipped")
        return pd.DataFrame()
    news = pd.read_parquet(NEWS_FILE).rename(columns={"timestamp": "date"})
    news = _normalize_date(news, "date")
    n_dates = news["date"].nunique() if "date" in news.columns else 0
    if n_dates < 60:
        print(f"[v6] News: {n_dates} unique dates — too sparse, skipped")
        return pd.DataFrame()
    news["merge_date"] = news["date"] + pd.Timedelta(days=1)
    news = news.sort_values(["symbol", "date"])
    news["sentiment_3d_avg"] = news.groupby("symbol")["sentiment_score"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    news["sentiment_count_3d"] = news.groupby("symbol")["count"].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )
    result = news.rename(columns={"sentiment_score": "sentiment_score_lag1"})[
        [
            "merge_date",
            "symbol",
            "sentiment_score_lag1",
            "sentiment_3d_avg",
            "sentiment_count_3d",
        ]
    ].rename(columns={"merge_date": "date"})
    print(f"[v6] News: {len(result)} rows, {result['symbol'].nunique()} symbols")
    return result


def _load_earnings_features(panel_dates, panel_symbols) -> pd.DataFrame:
    if not EARNINGS_FILE.exists():
        print("[v6] Earnings file not found — skipped")
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

    # EPS surprise
    earn = earn.dropna(subset=["eps_surprise_pct"]).copy()
    earn["eps_surprise_pct_lag1"] = earn["eps_surprise_pct"].clip(-200, 200)

    # Revenue surprise
    rev_cols = {"revenue_actual", "revenue_estimate"} & set(earn.columns)
    if len(rev_cols) == 2:
        rev_actual = pd.to_numeric(earn["revenue_actual"], errors="coerce")
        rev_est = pd.to_numeric(earn["revenue_estimate"], errors="coerce")
        denom = rev_est.abs().clip(lower=1e-9)
        earn["revenue_surprise_pct_lag1"] = ((rev_actual - rev_est) / denom).clip(-5, 5)
    else:
        earn["revenue_surprise_pct_lag1"] = np.nan

    earn_features = earn[
        ["merge_date", "symbol", "eps_surprise_pct_lag1", "revenue_surprise_pct_lag1"]
    ].copy()

    # Days since last earnings per symbol-date
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
    print(
        f"[v6] Earnings: {len(result)} rows, {result['symbol'].nunique()} symbols, "
        f"revenue_surprise available: {result['revenue_surprise_pct_lag1'].notna().sum()}"
    )
    return result


def _load_fundamentals_features() -> pd.DataFrame:
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

    # Forward-fill per symbol then shift 1 day for PIT safety
    result_parts = []
    for sym, grp in fund.groupby("symbol"):
        grp = grp.sort_values("date").copy()
        for col in keep:
            grp[col] = grp[col].ffill()
        # Lag: shift all feature values 1 day forward in calendar time
        grp["date"] = grp["date"] + pd.Timedelta(days=1)
        result_parts.append(grp[["date", "symbol"] + keep])

    if not result_parts:
        return pd.DataFrame()
    result = pd.concat(result_parts, ignore_index=True)
    print(f"[v6] Fundamentals: {len(result)} rows, features: {keep}")
    return result


def _load_insider_features(panel_dates, panel_symbols) -> pd.DataFrame:
    """Build 90-day rolling net insider buy pressure per symbol-date."""
    if not INSIDER_FILE.exists():
        print("[v6] Insider file not found — skipped")
        return pd.DataFrame()

    ins = pd.read_parquet(INSIDER_FILE)

    # Normalise date column
    date_col = "filing_date" if "filing_date" in ins.columns else "timestamp"
    ins = ins.rename(columns={date_col: "filing_date"})
    ins = _normalize_date(ins, "filing_date")

    if "transaction_type" not in ins.columns or "shares" not in ins.columns:
        print("[v6] Insider: missing required columns — skipped")
        return pd.DataFrame()

    ins["shares"] = pd.to_numeric(ins["shares"], errors="coerce").fillna(0.0)
    ins["txn_upper"] = ins["transaction_type"].str.upper().fillna("")

    # Classify: buys positive, sales negative, other zero
    ins["signed_shares"] = np.where(
        ins["txn_upper"].str.contains("BUY|PURCHASE|P -|^P$", regex=True),
        ins["shares"],
        np.where(
            ins["txn_upper"].str.contains("SELL|SALE|S -|^S$|DISPOSE", regex=True),
            -ins["shares"],
            0.0,
        ),
    )

    if "symbol" not in ins.columns:
        print("[v6] Insider: no symbol column — skipped")
        return pd.DataFrame()

    # Vectorised: compute rolling 90-day net buy per symbol using merge_asof approach.
    # Strategy: for each panel (date, symbol), find all insider txns in (date-91d, date-1d].
    lookback_days = 90
    lag_days = 1

    # Build a lookup frame: panel dates × symbol merged with insider data
    panel_keys = (
        pd.DataFrame({"date": panel_dates.values, "symbol": panel_symbols.values})
        .drop_duplicates()
        .sort_values(["symbol", "date"])
    )

    ins = ins.sort_values(["symbol", "filing_date"])

    result_parts = []
    for sym, grp in ins.groupby("symbol"):
        sym_panel = panel_keys[panel_keys["symbol"] == sym]
        if sym_panel.empty:
            continue
        grp = grp[["filing_date", "signed_shares", "shares"]].copy()

        rows_s = []
        for d in sym_panel["date"]:
            cutoff_end = d - pd.Timedelta(days=lag_days)
            cutoff_start = cutoff_end - pd.Timedelta(days=lookback_days)
            mask = (grp["filing_date"] >= cutoff_start) & (
                grp["filing_date"] <= cutoff_end
            )
            w = grp[mask]
            if w.empty:
                continue
            net_buy = float(w["signed_shares"].sum())
            total_abs = float(w["shares"].abs().sum())
            rows_s.append(
                {
                    "date": d,
                    "symbol": sym,
                    "insider_net_buy_90d": net_buy,
                    "insider_buy_ratio_90d": net_buy / max(total_abs, 1.0),
                    "insider_txn_count_90d": float(len(w)),
                }
            )
        if rows_s:
            result_parts.append(pd.DataFrame(rows_s))

    if not result_parts:
        print("[v6] Insider: no matching symbol data — skipped")
        return pd.DataFrame()

    result = pd.concat(result_parts, ignore_index=True)
    # Clip extreme values
    result["insider_net_buy_90d"] = result["insider_net_buy_90d"].clip(-1e7, 1e7)
    result["insider_buy_ratio_90d"] = result["insider_buy_ratio_90d"].clip(-1, 1)
    result["insider_txn_count_90d"] = result["insider_txn_count_90d"].clip(0, 200)
    print(
        f"[v6] Insider: {len(result)} rows, {result['symbol'].nunique()} symbols, "
        f"buy_events: {(result['insider_net_buy_90d'] > 0).sum()}"
    )
    return result


def _load_macro_features(panel_dates) -> pd.DataFrame:
    """Build macro conditioning features for each panel date."""
    if not MACRO_FILE.exists():
        print("[v6] Macro file not found — skipped")
        return pd.DataFrame()

    macro = pd.read_parquet(MACRO_FILE)
    date_col = next((c for c in ["timestamp", "date"] if c in macro.columns), None)
    if date_col is None:
        print("[v6] Macro: no date column — skipped")
        return pd.DataFrame()
    macro = macro.rename(columns={date_col: "date"})
    macro = _normalize_date(macro, "date")

    # Aggregate to daily if needed (e.g. if hourly data)
    agg_dict = {}
    for c in macro.columns:
        if c != "date" and pd.api.types.is_numeric_dtype(macro[c]):
            agg_dict[c] = "last"
    if not agg_dict:
        return pd.DataFrame()
    macro = macro.groupby("date").agg(agg_dict).reset_index()
    macro = macro.sort_values("date")

    # VIX level (use 'vix' or 'vix_close')
    vix_col = next((c for c in ["vix", "vix_close"] if c in macro.columns), None)
    # Yield curve: 10y - 3m
    y10_col = next(
        (c for c in ["tnx_10y_yield", "treasury_10y"] if c in macro.columns), None
    )
    y3m_col = next(
        (c for c in ["irx_3m_yield", "treasury_2y"] if c in macro.columns), None
    )

    if vix_col is None:
        print("[v6] Macro: no VIX column found — skipped")
        return pd.DataFrame()

    macro["vix_level"] = macro[vix_col].clip(5, 100)
    macro["vix_ma20"] = macro["vix_level"].rolling(20, min_periods=1).mean()
    macro["vix_vs_ma20"] = (
        macro["vix_level"] / macro["vix_ma20"].clip(lower=1e-6)
    ).clip(0.3, 5)

    if y10_col and y3m_col:
        macro["yield_curve_spread"] = (macro[y10_col] - macro[y3m_col]).clip(-5, 5)
    else:
        macro["yield_curve_spread"] = 0.0

    # Composite macro risk score: high VIX + inverted curve = risk-off
    macro["macro_risk_score"] = (
        (macro["vix_level"] - 20) / 20  # normalised VIX excess
        - macro["yield_curve_spread"] / 2  # inverted curve adds risk
    ).clip(-3, 3)

    keep = ["date"] + [c for c in MACRO_FEATURE_COLS if c in macro.columns]
    result = macro[keep].copy()

    # Lag 1 day for PIT safety
    result["date"] = result["date"] + pd.Timedelta(days=1)
    print(
        f"[v6] Macro: {len(result)} rows, "
        f"{result['date'].min()} to {result['date'].max()}, "
        f"features: {[c for c in MACRO_FEATURE_COLS if c in result.columns]}"
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"[v6] Loading panel: {PANEL_FILE}")
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
        f"[v6] After peer filter: {len(df):,} rows, target rate={df['target'].mean():.3f}"
    )

    # --- Merge features ---
    print("\n[v6] Loading alt-data features...")

    # News (sparse — likely skipped)
    news_df = _load_news_features(df["date"], df["symbol"])
    if not news_df.empty:
        df = df.merge(news_df, on=["date", "symbol"], how="left")

    # Earnings + revenue surprise
    earn_df = _load_earnings_features(df["date"], df["symbol"])
    if not earn_df.empty:
        df = df.merge(earn_df, on=["date", "symbol"], how="left")

    # Fundamentals (forward-filled per symbol)
    fund_df = _load_fundamentals_features()
    if not fund_df.empty:
        df = df.merge(fund_df, on=["date", "symbol"], how="left")
        # Forward-fill per symbol (fundamentals reported infrequently)
        fund_feats = [c for c in FUNDAMENTALS_FEATURE_COLS if c in df.columns]
        if fund_feats:
            df = df.sort_values(["symbol", "date"])
            df[fund_feats] = df.groupby("symbol")[fund_feats].ffill()

    # Insider trading
    ins_df = _load_insider_features(df["date"], df["symbol"])
    if not ins_df.empty:
        df = df.merge(ins_df, on=["date", "symbol"], how="left")

    # Macro (date-level, no symbol)
    macro_df = _load_macro_features(df["date"])
    if not macro_df.empty:
        df = df.merge(macro_df, on="date", how="left")
        # Forward-fill macro (weekend gaps)
        macro_feats = [c for c in MACRO_FEATURE_COLS if c in df.columns]
        if macro_feats:
            df = df.sort_values("date")
            df[macro_feats] = df[macro_feats].ffill()

    # --- Build feature set ---
    print()
    ta_available = [c for c in TA_FEATURE_COLS if c in df.columns]
    news_available = [c for c in NEWS_FEATURE_COLS if c in df.columns]
    earnings_available = [c for c in EARNINGS_FEATURE_COLS if c in df.columns]
    fund_available = [c for c in FUNDAMENTALS_FEATURE_COLS if c in df.columns]
    insider_available = [c for c in INSIDER_FEATURE_COLS if c in df.columns]
    macro_available = [c for c in MACRO_FEATURE_COLS if c in df.columns]

    raw_available = (
        ta_available
        + news_available
        + earnings_available
        + fund_available
        + insider_available
        + macro_available
    )
    print(
        f"[v6] Raw features: {len(ta_available)} TA + {len(news_available)} news + "
        f"{len(earnings_available)} earnings + {len(fund_available)} fundamentals + "
        f"{len(insider_available)} insider + {len(macro_available)} macro "
        f"= {len(raw_available)} total"
    )

    cs_cols = add_cross_sectional_features(df, raw_available)
    feature_cols = raw_available + cs_cols
    print(f"[v6] + cross-sectional features: {len(feature_cols)} total")

    df_sorted = df.sort_values("date").reset_index(drop=True)
    X = df_sorted[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df_sorted["target"]

    # Train/val split
    n_val = (df_sorted["date"] >= TRAIN_CUTOFF).sum()
    test_size = n_val / len(df_sorted)
    print(f"[v6] Test size: {test_size:.1%} ({n_val:,} val rows)")

    try:
        from src.assembled_core.qa.cpcv_validation import purged_train_test_split

        X_train, X_val, y_train, y_val = purged_train_test_split(
            X, y, test_size=test_size, embargo_bars=EMBARGO_BARS
        )
    except Exception as e:
        print(f"[v6] Purged split failed ({e}), using simple date split")
        n_split = int(len(X) * (1 - test_size))
        X_train, X_val = X.iloc[:n_split], X.iloc[n_split:]
        y_train, y_val = y.iloc[:n_split], y.iloc[n_split:]

    print(
        f"[v6] Train: {len(X_train):,} | Embargo: {EMBARGO_BARS} | Val: {len(X_val):,}"
    )

    try:
        import lightgbm as lgb
    except ImportError:
        print("[ERROR] lightgbm not installed — pip install lightgbm")
        sys.exit(1)

    # Slightly relaxed regularization for the richer feature set
    lgb_params = dict(
        n_estimators=800,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=5,
        min_child_samples=150,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=1.5,
        reg_lambda=1.5,
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

    print()
    print(f"[v6] Train AUC:            {train_auc:.4f}")
    print(f"[v6] Val AUC:              {val_auc:.4f}  (v5 baseline: 0.5080)")
    print(f"[v6] Val acc (0.5):        {val_acc:.4f}")
    print(f"[v6] Baseline (majority):  {baseline_acc:.4f}")
    print(f"[v6] Best F1 threshold:    {best_threshold:.2f}  (F1={best_f1:.4f})")

    importances = dict(zip(feature_cols, model.feature_importances_))
    top = sorted(importances.items(), key=lambda x: -x[1])[:15]
    print("[v6] Top-15 feature importances:")
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
        "insider_feature_cols": insider_available,
        "macro_feature_cols": macro_available,
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
        "version": "v6",
    }
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_OUT)
    print(f"\n[v6] Saved -> {MODEL_OUT}")

    delta_auc = val_auc - 0.5080  # v5 baseline
    print(f"[v6] AUC delta vs v5: {delta_auc:+.4f}")
    if val_auc >= 0.55:
        print("[v6] ✓ AUC ≥ 0.55 — strong signal from alt-data. DEPLOY v6.")
    elif val_auc >= 0.52:
        print("[v6] ✓ AUC 0.52-0.55 — meets deployment threshold. DEPLOY v6.")
    else:
        print("[v6] ✗ AUC < 0.52 — below deployment threshold. v5 remains active.")


if __name__ == "__main__":
    main()
