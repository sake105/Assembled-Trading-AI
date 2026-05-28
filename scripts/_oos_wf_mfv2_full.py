"""One-shot OOS Walk-Forward for multifactor_v2 — full altdata stack.

Activates factors that are available with local files vs. the TA-only baseline:
  F18  sector_rotation_bias      — configs/security_meta.csv + sector ETFs (yfinance)
  F19  earnings_surprise_z       — output/events_earnings.parquet (via altdata_loader)
  F23  macro_growth_momentum     — output/macro.parquet (via altdata_loader)
  F24  macro_inflation_surprise  — output/macro.parquet (via altdata_loader)
  F25  intermarket_bond_equity   — TLT/SPY (ETF panel → gpr column in price slice)
  F26  intermarket_credit_spread — HYG/IEF (ETF panel → column in price slice)
  F27  intermarket_yield_curve   — output/macro.parquet yield_curve_spread column
  F31  geo_risk_composite        — output/macro_gpr.parquet gpr_index column

Still 0.0 (expected — no data or live-only):
  F20/F32 insider_*              — insider_trading.parquet all 'unknown'
  F21/F22 news_*                 — data only from 2025-12-22
  F28/F29/F35 options/vix_*     — live CBOE fetch only
  F30  congress_activity         — no data files
  F33  pead_sue_score            — depends on earnings data quality
  F34  buyback_drift_score       — no data files

Usage:
    python scripts/_oos_wf_mfv2_full.py

Outputs:
    docs/results/2026_05_mfv2_factor_activation_log.md
    docs/results/2026_05_mfv2_full_stack_real_oos.md

KEINE Änderungen an strategy, policy.yaml oder anderen Produktionsdateien.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("oos_mfv2_full")
log.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PERIOD_START = pd.Timestamp("2018-01-01", tz="UTC")
PERIOD_END = pd.Timestamp("2025-12-31", tz="UTC")
TRAIN_WINDOW_DAYS = 252
TEST_WINDOW_DAYS = 252
STEP_SIZE_DAYS = 252
WARMUP_BARS = 250
COMMISSION_BPS = 10.0
INITIAL_CAPITAL = 100_000.0
WATCHLIST = ROOT / "watchlist.txt"
PRICE_CACHE = ROOT / "output" / "oos_alpaca_prices_cache.parquet"
ETF_CACHE = ROOT / "output" / "oos_etf_prices_cache.parquet"
OUT_AUDIT_MD = ROOT / "docs" / "results" / "2026_05_mfv2_factor_activation_log.md"
OUT_OOS_MD = ROOT / "docs" / "results" / "2026_05_mfv2_full_stack_real_oos.md"

AUDIT_DATE = pd.Timestamp("2023-06-30", tz="UTC")

# Sector ETFs needed for F18
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLU", "XLP", "XLY"]
# Intermarket ETFs for F25/F26
INTERMARKET_ETFS = ["TLT", "HYG", "IEF"]
ALL_ETF_SYMBOLS = SECTOR_ETFS + INTERMARKET_ETFS

# Factors that MUST be non-zero after wiring (triggers STOPP if still 0.0)
STOPP_FACTORS = {
    "macro_growth_momentum_z": "F23 — output/macro.parquet must be present",
    "macro_inflation_surprise_z": "F24 — output/macro.parquet must be present",
    # intermarket_bond_equity (F25) excluded: binary flag (TLT>0 & SPY<0) is
    # legitimately 0.0 in bull-market regimes (e.g. 2023). Wiring is verified
    # via intermarket_credit_spread sharing the same enrichment path.
    "intermarket_credit_spread": "F26 — HYG/IEF ETF prices must be available",
    "intermarket_yield_curve": "F27 — output/macro.parquet yield_curve_spread must exist",
    "geo_risk_composite": "F31 — output/macro_gpr.parquet gpr_index must exist",
}

SECURITY_META_PATH = str(ROOT / "configs" / "security_meta.csv")
STRATEGY_CFG = {"security_meta_path": SECURITY_META_PATH}


# ---------------------------------------------------------------------------
# 1 — Price loading (Alpaca or cache)
# ---------------------------------------------------------------------------
def _load_env():
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    ak = os.environ.get("ALPACA_API_KEY", "")
    sk = os.environ.get("ALPACA_API_SECRET", "")
    if not ak or not sk:
        raise EnvironmentError("ALPACA_API_KEY / ALPACA_API_SECRET not set")
    return ak, sk


def _fetch_alpaca(
    symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    from alpaca.data import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    ak, sk = _load_env()
    client = StockHistoricalDataClient(api_key=ak, secret_key=sk)
    log.info("Fetching %d symbols from Alpaca…", len(symbols))
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        adjustment="split",
    )
    bars = client.get_stock_bars(req)
    df = bars.df.reset_index()
    if "timestamp" not in df.columns:
        df = df.rename(columns={df.columns[0]: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.rename(columns=str.lower)
    keep = [
        c
        for c in ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        if c in df.columns
    ]
    return df[keep].sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _get_prices(all_symbols: list[str]) -> pd.DataFrame:
    fetch_start = PERIOD_START - pd.Timedelta(days=400)
    if PRICE_CACHE.exists():
        log.info("Loading cached Alpaca prices from %s", PRICE_CACHE)
        prices = pd.read_parquet(PRICE_CACHE)
        prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
        missing = [s for s in all_symbols if s not in prices["symbol"].unique()]
        if not missing:
            return prices
        log.info("Cache missing %d symbols, re-fetching", len(missing))
    prices = _fetch_alpaca(all_symbols, start=fetch_start, end=PERIOD_END)
    PRICE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(PRICE_CACHE, index=False)
    return prices


# ---------------------------------------------------------------------------
# 2 — ETF prices (yfinance, one-time, cached)
# ---------------------------------------------------------------------------
def _fetch_etf_yfinance(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch daily close prices for ETF symbols via yfinance."""
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        log.warning("[ETF] yfinance not installed — ETF fetch skipped")
        return pd.DataFrame()

    log.info("Fetching %d ETFs via yfinance (%s → %s)…", len(symbols), start, end)
    raw = yf.download(symbols, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        closes = (
            raw["Close"]
            if "Close" in raw.columns.get_level_values(0)
            else raw.xs("Close", axis=1, level=0)
        )
    else:
        closes = raw

    closes.index = pd.to_datetime(closes.index, utc=True)
    rows = []
    for sym in closes.columns:
        s = closes[sym].dropna()
        if s.empty:
            continue
        tmp = pd.DataFrame({"timestamp": s.index, "symbol": sym, "close": s.values})
        rows.append(tmp)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["symbol", "timestamp"])


def _get_etf_prices() -> pd.DataFrame:
    fetch_start = (PERIOD_START - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    fetch_end = PERIOD_END.strftime("%Y-%m-%d")
    if ETF_CACHE.exists():
        log.info("Loading cached ETF prices from %s", ETF_CACHE)
        etf = pd.read_parquet(ETF_CACHE)
        etf["timestamp"] = pd.to_datetime(etf["timestamp"], utc=True)
        missing = [s for s in ALL_ETF_SYMBOLS if s not in etf["symbol"].unique()]
        if not missing:
            return etf
        log.info("ETF cache missing %d symbols, re-fetching", len(missing))
    etf = _fetch_etf_yfinance(ALL_ETF_SYMBOLS, start=fetch_start, end=fetch_end)
    if not etf.empty:
        ETF_CACHE.parent.mkdir(parents=True, exist_ok=True)
        etf.to_parquet(ETF_CACHE, index=False)
        log.info("ETF prices cached to %s", ETF_CACHE)
    return etf


# ---------------------------------------------------------------------------
# 3 — Macro panel (daily, pre-computed)
# ---------------------------------------------------------------------------
def _build_macro_panel(prices: pd.DataFrame, etf_prices: pd.DataFrame) -> pd.DataFrame:
    """Build daily-indexed DataFrame with 4 intermarket + GPR columns.

    Columns produced:
        gpr_index                 — Caldara-Iacoviello GPR index (monthly → daily ffill)
        yield_curve_slope         — 10y-2y spread from macro.parquet (monthly → daily ffill)
        bond_equity_divergence_flag — 1 when TLT 20d return > 0 AND SPY 20d return < 0
        credit_spread_change_5d   — HYG/IEF ratio 5-day pct change
    """
    panel_parts = []

    # --- GPR index ---
    gpr_path = ROOT / "output" / "macro_gpr.parquet"
    if gpr_path.exists():
        gpr = pd.read_parquet(gpr_path)
        gpr["_d"] = pd.to_datetime(gpr["timestamp"]).dt.tz_convert(None).dt.normalize()
        gpr = gpr[gpr["gpr_index"].notna()][["_d", "gpr_index"]].set_index("_d")
        panel_parts.append(gpr)
        log.info("[Macro] GPR panel: %d monthly rows", len(gpr))
    else:
        log.warning("[Macro] macro_gpr.parquet not found — gpr_index will be 0.0")

    # --- Yield curve slope ---
    macro_path = ROOT / "output" / "macro.parquet"
    if macro_path.exists():
        macro = pd.read_parquet(macro_path)
        macro["_d"] = (
            pd.to_datetime(macro["timestamp"]).dt.tz_convert(None).dt.normalize()
        )
        macro = macro[macro["yield_curve_spread"].notna()][
            ["_d", "yield_curve_spread"]
        ].copy()
        macro = macro.rename(
            columns={"yield_curve_spread": "yield_curve_slope"}
        ).set_index("_d")
        panel_parts.append(macro)
        log.info("[Macro] Yield curve panel: %d monthly rows", len(macro))
    else:
        log.warning("[Macro] macro.parquet not found — yield_curve_slope will be 0.0")

    # --- Merge all parts on date index ---
    if panel_parts:
        panel = panel_parts[0]
        for part in panel_parts[1:]:
            panel = panel.join(part, how="outer")
        panel = panel.sort_index()
    else:
        panel = pd.DataFrame(index=pd.DatetimeIndex([]))

    # Resample to daily calendar (tz-naive)
    fetch_start_naive = (PERIOD_START - pd.Timedelta(days=400)).tz_convert(None)
    fetch_end_naive = PERIOD_END.tz_convert(None)
    daily_idx = pd.date_range(
        start=fetch_start_naive
        if panel.empty
        else max(panel.index.min(), fetch_start_naive),
        end=fetch_end_naive,
        freq="D",
    )
    panel = panel.reindex(daily_idx).ffill()

    # Ensure required columns exist
    for col in ["gpr_index", "yield_curve_slope"]:
        if col not in panel.columns:
            panel[col] = 0.0

    # --- Bond-equity divergence flag (TLT/SPY 20d returns) ---
    spy_s = prices[prices["symbol"] == "SPY"].copy()
    spy_s = spy_s.sort_values("timestamp")
    spy_s["_d"] = pd.to_datetime(spy_s["timestamp"]).dt.tz_convert(None).dt.normalize()
    spy_daily = spy_s.set_index("_d")["close"].resample("D").last().ffill()

    if not etf_prices.empty and "TLT" in etf_prices["symbol"].unique():
        tlt_s = etf_prices[etf_prices["symbol"] == "TLT"].sort_values("timestamp")
        tlt_s["_d"] = (
            pd.to_datetime(tlt_s["timestamp"]).dt.tz_convert(None).dt.normalize()
        )
        tlt_daily = tlt_s.set_index("_d")["close"].resample("D").last().ffill()

        tlt_ret20 = tlt_daily.pct_change(20)
        spy_ret20 = spy_daily.pct_change(20)
        flag = (tlt_ret20 > 0) & (spy_ret20 < 0)
        panel["bond_equity_divergence_flag"] = (
            flag.reindex(panel.index, method="ffill").fillna(0.0).astype(float)
        )
    else:
        log.warning(
            "[Macro] TLT not in ETF cache — bond_equity_divergence_flag will be 0.0"
        )
        panel["bond_equity_divergence_flag"] = 0.0

    # --- Credit spread change 5d (HYG/IEF) ---
    if (
        not etf_prices.empty
        and "HYG" in etf_prices["symbol"].unique()
        and "IEF" in etf_prices["symbol"].unique()
    ):

        def _etf_series(sym):
            s = etf_prices[etf_prices["symbol"] == sym].sort_values("timestamp")
            s["_d"] = pd.to_datetime(s["timestamp"]).dt.tz_convert(None).dt.normalize()
            return s.set_index("_d")["close"].resample("D").last().ffill()

        hyg_daily = _etf_series("HYG")
        ief_daily = _etf_series("IEF")
        ratio = hyg_daily / ief_daily.replace(0.0, float("nan"))
        credit_5d = ratio.pct_change(5)
        panel["credit_spread_change_5d"] = credit_5d.reindex(
            panel.index, method="ffill"
        ).fillna(0.0)
    else:
        log.warning(
            "[Macro] HYG/IEF not in ETF cache — credit_spread_change_5d will be 0.0"
        )
        panel["credit_spread_change_5d"] = 0.0

    log.info(
        "[Macro] Panel built: %d daily rows, cols=%s", len(panel), list(panel.columns)
    )
    return panel


# ---------------------------------------------------------------------------
# 4 — Price slice enrichment
# ---------------------------------------------------------------------------
_MACRO_COLS = [
    "gpr_index",
    "yield_curve_slope",
    "bond_equity_divergence_flag",
    "credit_spread_change_5d",
]


def _enrich_price_slice(
    price_slice: pd.DataFrame,
    etf_prices: pd.DataFrame,
    macro_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Merge macro columns (broadcast per date) and append sector ETF rows."""
    ps = price_slice.copy()

    # Merge macro columns on date (tz-naive key)
    _ts = pd.to_datetime(ps["timestamp"])
    ps["_mk"] = (
        _ts.dt.tz_convert(None).dt.normalize()
        if _ts.dt.tz is not None
        else _ts.dt.normalize()
    )
    mp = macro_panel[_MACRO_COLS].reset_index()
    mp.rename(columns={mp.columns[0]: "_mk"}, inplace=True)
    mp["_mk"] = pd.to_datetime(mp["_mk"])

    ps = ps.merge(mp, on="_mk", how="left", suffixes=("", "_mp"))
    for col in _MACRO_COLS:
        if col in ps.columns:
            ps[col] = ps[col].ffill().fillna(0.0)
    ps = ps.drop(columns=["_mk"])

    # Add sector ETF rows (needed for _compute_sector_rotation_bias).
    # yfinance timestamps (00:00 UTC) differ from Alpaca (05:00 UTC): match on date,
    # then remap ETF timestamps to Alpaca-format so tail(1) groupby works correctly.
    if not etf_prices.empty:
        _ps_ts = pd.to_datetime(ps["timestamp"])
        _ps_norm = (
            _ps_ts.dt.tz_convert(None).dt.normalize()
            if _ps_ts.dt.tz is not None
            else _ps_ts.dt.normalize()
        )
        _date_to_alpaca_ts = dict(zip(_ps_norm, ps["timestamp"]))
        slice_dates = set(_ps_norm.dt.date)

        sector_sym_set = set(SECTOR_ETFS)
        etf_sub = etf_prices[etf_prices["symbol"].isin(sector_sym_set)].copy()
        _etf_ts = pd.to_datetime(etf_sub["timestamp"])
        etf_sub["_norm"] = (
            _etf_ts.dt.tz_convert(None).dt.normalize()
            if _etf_ts.dt.tz is not None
            else _etf_ts.dt.normalize()
        )
        etf_sub = etf_sub[etf_sub["_norm"].dt.date.isin(slice_dates)]
        if not etf_sub.empty:
            etf_sub["timestamp"] = etf_sub["_norm"].map(_date_to_alpaca_ts)
            etf_sub = etf_sub.dropna(subset=["timestamp"]).drop(columns=["_norm"])
            ps = pd.concat([ps, etf_sub], ignore_index=True)

    return ps


# ---------------------------------------------------------------------------
# 5 — Factor audit at AUDIT_DATE
# ---------------------------------------------------------------------------
def _run_factor_audit(
    prices: pd.DataFrame,
    etf_prices: pd.DataFrame,
    macro_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Compute all factor sub-functions at AUDIT_DATE; return per-factor summary."""
    from src.assembled_core.features.ta_features import add_all_features
    from src.assembled_core.strategies.multifactor_v2 import (  # noqa: PLC0415
        _compute_earnings_insider_factors,
        _compute_geo_risk_composite,
        _compute_intermarket_factors,
        _compute_news_macro_factors,
        _compute_sector_rotation_bias,
    )

    # PIT-safety: _compute_options_factors does live CBOE fetch; disable for historical audit.
    _compute_options_factors = lambda syms, latest: {}  # noqa: E731

    warmup_start = AUDIT_DATE - pd.Timedelta(days=400)
    raw_slice = prices[
        (prices["timestamp"] >= warmup_start) & (prices["timestamp"] <= AUDIT_DATE)
    ].copy()

    if raw_slice.empty:
        raise ValueError(
            f"No prices available for audit window ending {AUDIT_DATE.date()}"
        )

    enriched = add_all_features(raw_slice, use_namespace=True)
    enriched = _enrich_price_slice(enriched, etf_prices, macro_panel)

    # Guard: alert if macro enrichment silently produced all-zero columns (F-senior-4).
    for _mcol in ["credit_spread_change_5d", "yield_curve_slope", "gpr_index"]:
        if _mcol in enriched.columns and enriched[_mcol].abs().sum() == 0:
            log.warning(
                "[Audit] Macro column '%s' is all-zero — _enrich_price_slice may have silently failed",
                _mcol,
            )

    latest = (
        enriched.sort_values("timestamp")
        .groupby("symbol", group_keys=False)
        .tail(1)
        .copy()
    )
    # Exclude sector ETFs from the symbol list used for signal generation
    strategy_latest = latest[~latest["symbol"].isin(set(SECTOR_ETFS))].copy()
    latest_symbols = strategy_latest["symbol"].tolist()
    _bar_as_of = pd.Timestamp(strategy_latest["timestamp"].max())

    log.info(
        "[Audit] Bar as-of: %s, symbols: %d", _bar_as_of.date(), len(latest_symbols)
    )

    factor_results: dict[str, pd.Series] = {}

    # --- Altdata sub-functions ---
    earn_z, insider_z = _compute_earnings_insider_factors(
        latest_symbols, STRATEGY_CFG, as_of=_bar_as_of
    )
    factor_results["earnings_surprise_z"] = (
        earn_z if not earn_z.empty else pd.Series(0.0, index=latest_symbols)
    )
    factor_results["insider_activity_score"] = (
        insider_z if not insider_z.empty else pd.Series(0.0, index=latest_symbols)
    )

    news_macro = _compute_news_macro_factors(
        latest_symbols, STRATEGY_CFG, as_of=_bar_as_of
    )
    for key in [
        "news_sentiment_7d_z",
        "news_volume_spike_z",
        "macro_growth_momentum_z",
        "macro_inflation_surprise_z",
    ]:
        factor_results[key] = news_macro.get(key, pd.Series(0.0, index=latest_symbols))

    intermarket = _compute_intermarket_factors(latest_symbols, strategy_latest)
    for key in [
        "intermarket_bond_equity",
        "intermarket_credit_spread",
        "intermarket_yield_curve",
    ]:
        factor_results[key] = intermarket.get(key, pd.Series(0.0, index=latest_symbols))

    options = _compute_options_factors(latest_symbols, strategy_latest)
    for key in ["options_put_call_extreme", "vix_regime_score"]:
        factor_results[key] = options.get(key, pd.Series(0.0, index=latest_symbols))

    geo = _compute_geo_risk_composite(latest_symbols, strategy_latest, as_of=_bar_as_of)
    factor_results["geo_risk_composite"] = geo.get(
        "geo_risk_composite", pd.Series(0.0, index=latest_symbols)
    )

    sector = _compute_sector_rotation_bias(enriched, latest_symbols, STRATEGY_CFG)
    factor_results["sector_rotation_bias"] = (
        sector
        if isinstance(sector, pd.Series)
        else pd.Series(0.0, index=latest_symbols)
    )

    # --- TA factors: inspect ta_* columns in strategy_latest ---
    ta_col_map = {
        "trend_ema_spread": "ta_ema_spread_v1",
        "trend_ma200_position": "ta_ma_200_v1",
        "trend_adx_strength": "ta_adx_v1",
        "trend_macd_hist": "ta_macd_hist_v1",
        "mom_rsi_centered": "ta_rsi_v1",
        "mom_volume_weighted": "ta_vol_weighted_mom_20d_v1",
        "mom_obv_trend": "ta_obv_trend_v1",
        "mr_bollinger_pctb": "ta_bollinger_pctb_v1",
        "mr_stoch_oversold": "ta_stoch_k_v1",
        "vol_abnormal": "ta_volume_ratio_v1",
        "vola_regime_score": "ta_atr_ratio_v1",
    }
    for factor_name, col in ta_col_map.items():
        if col in strategy_latest.columns:
            vals = pd.to_numeric(strategy_latest[col], errors="coerce").fillna(0.0)
            vals.index = strategy_latest["symbol"].values
            factor_results[factor_name] = vals
        else:
            factor_results[factor_name] = pd.Series(0.0, index=latest_symbols)

    # --- Compile audit rows ---
    rows = []
    for fname, vals in factor_results.items():
        if isinstance(vals, pd.Series) and len(vals) > 0:
            numeric = pd.to_numeric(vals, errors="coerce").fillna(0.0)
            mean_val = float(numeric.mean())
            pct_nonzero = float((numeric.abs() > 1e-10).mean())
        else:
            mean_val = 0.0
            pct_nonzero = 0.0

        is_stopp = fname in STOPP_FACTORS
        if is_stopp and pct_nonzero < 1e-10:
            status = "STOPP"
        elif pct_nonzero < 1e-10:
            status = "ZERO"
        else:
            status = "ACTIVE"

        rows.append(
            {
                "factor": fname,
                "mean_value": mean_val,
                "pct_nonzero": pct_nonzero,
                "status": status,
                "is_stopp_factor": is_stopp,
            }
        )

    return pd.DataFrame(rows)


def _check_stopp(audit_df: pd.DataFrame) -> list[str]:
    """Return list of STOPP violations (should-be-active factors still 0.0)."""
    return audit_df[audit_df["status"] == "STOPP"]["factor"].tolist()


# ---------------------------------------------------------------------------
# 6 — Build walk-forward backtest function
# ---------------------------------------------------------------------------
def _make_backtest_fn(
    prices: pd.DataFrame,
    etf_prices: pd.DataFrame,
    macro_panel: pd.DataFrame,
) -> object:
    from src.assembled_core.features.ta_features import add_all_features
    import src.assembled_core.strategies.multifactor_v2 as _mfv2_mod  # noqa: PLC0415
    from src.assembled_core.strategies.multifactor_v2 import (
        compute_signals as mfv2_compute_signals,
    )
    from src.assembled_core.strategies.multifactor_v2 import (
        compute_target_positions as mfv2_compute_targets,
    )

    # PIT-safety: _compute_options_factors does a live CBOE fetch (no historical path).
    # Injecting today's VIX into historical folds is look-ahead. Disable for this script.
    _mfv2_mod._compute_options_factors = lambda syms, latest: {}

    def backtest_fn(
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
    ) -> dict:
        warmup_dates = np.sort(prices["timestamp"].unique())
        warmup_dates = warmup_dates[warmup_dates < test_start]
        warmup_start = (
            warmup_dates[-WARMUP_BARS]
            if len(warmup_dates) >= WARMUP_BARS
            else test_start
        )

        window_prices = prices[
            (prices["timestamp"] >= warmup_start) & (prices["timestamp"] < test_end)
        ].copy()

        if window_prices.empty or window_prices["symbol"].nunique() < 5:
            raise ValueError(
                f"Insufficient price data for {test_start.date()}–{test_end.date()}"
            )

        enriched = add_all_features(window_prices, use_namespace=True)

        test_ts_sorted = sorted(
            enriched[enriched["timestamp"] >= test_start]["timestamp"].unique()
        )
        monthly_dates: list = []
        last_month = None
        for ts in test_ts_sorted:
            m = pd.Timestamp(ts).to_period("M")
            if m != last_month:
                monthly_dates.append(ts)
                last_month = m

        def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
            all_sigs = []
            for rebal_ts in monthly_dates:
                price_slice = enriched[enriched["timestamp"] <= rebal_ts].copy()
                if price_slice["symbol"].nunique() < 3:
                    continue
                try:
                    enriched_slice = _enrich_price_slice(
                        price_slice, etf_prices, macro_panel
                    )
                    sigs = mfv2_compute_signals(
                        enriched_slice, strategy_cfg=STRATEGY_CFG
                    )
                    if sigs.empty or "direction" not in sigs.columns:
                        continue
                    long_sigs = sigs[sigs["direction"] == "LONG"][
                        ["timestamp", "symbol", "direction", "score"]
                    ].copy()
                    long_sigs["timestamp"] = rebal_ts
                    all_sigs.append(long_sigs)
                except Exception as exc:
                    log.warning(
                        "[MFV2-FULL] compute_signals skip %s: %s",
                        pd.Timestamp(rebal_ts).date(),
                        exc,
                    )
            if not all_sigs:
                return pd.DataFrame(
                    columns=["timestamp", "symbol", "direction", "score"]
                )
            return pd.concat(all_sigs, ignore_index=True)

        def position_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
            if signals.empty:
                return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])
            return mfv2_compute_targets(signals, capital)

        from src.assembled_core.qa.backtest_engine import run_portfolio_backtest

        result = run_portfolio_backtest(
            prices=window_prices,
            signal_fn=signal_fn,
            position_sizing_fn=position_fn,
            start_capital=INITIAL_CAPITAL,
            commission_bps=COMMISSION_BPS,
            spread_w=0.25,
            impact_w=0.5,
            include_costs=True,
            rebalance_freq="1d",
        )
        m = result.metrics
        equity = result.equity
        eq_test = (
            equity[equity["timestamp"] >= test_start].copy()
            if not equity.empty and "timestamp" in equity.columns
            else equity.copy()
        )
        if len(eq_test) >= 2 and "equity" in eq_test.columns:
            start_eq = eq_test["equity"].iloc[0]
            end_eq = eq_test["equity"].iloc[-1]
            total_ret = end_eq / start_eq - 1 if start_eq > 0 else float("nan")
            n_years = len(eq_test) / 252
            cagr = (
                (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
                if not np.isnan(total_ret)
                else float("nan")
            )
            peak = eq_test["equity"].cummax()
            max_dd = float(((eq_test["equity"] - peak) / (peak + 1e-10)).min())
        else:
            cagr = float("nan")
            max_dd = float("nan")

        return {
            "test_sharpe": m.get("sharpe", float("nan")),
            "test_cagr": cagr,
            "test_max_dd": max_dd,
            "test_trades": int(m.get("trades", 0)),
        }

    return backtest_fn


# ---------------------------------------------------------------------------
# 7 — SPY buy-and-hold baseline
# ---------------------------------------------------------------------------
def _spy_buyhold(
    spy_prices: pd.DataFrame, test_start: pd.Timestamp, test_end: pd.Timestamp
) -> dict:
    df = spy_prices[
        (spy_prices["timestamp"] >= test_start) & (spy_prices["timestamp"] < test_end)
    ].sort_values("timestamp")
    if len(df) < 5:
        return {"bh_cagr": float("nan"), "bh_sharpe": float("nan")}
    rets = df["close"].pct_change().dropna()
    n_years = len(df) / 252
    total_ret = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    sharpe = (rets.mean() / (rets.std() + 1e-10)) * np.sqrt(252)
    return {"bh_cagr": cagr, "bh_sharpe": sharpe}


# ---------------------------------------------------------------------------
# 8 — Report helpers
# ---------------------------------------------------------------------------
def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v * 100:.1f}%"


def _fmt_f(v, d=2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{d}f}"


def _write_audit_report(audit_df: pd.DataFrame, stopp_violations: list[str]):
    active = audit_df[audit_df["status"] == "ACTIVE"]
    zero = audit_df[audit_df["status"] == "ZERO"]
    stopp = audit_df[audit_df["status"] == "STOPP"]

    lines = [
        "# multifactor_v2 — Faktor-Aktivierungs-Log (Paket 3c.2)",
        "",
        f"**Erstellt:** {pd.Timestamp.utcnow().date()}  ",
        f"**Audit-Datum:** {AUDIT_DATE.date()}  ",
        "**Branch:** main  ",
        "**Zweck:** Paket 3c.2 — Verifikation der Faktor-Aktivierung vor Walk-Forward.",
        "",
        "---",
        "",
        "## Zusammenfassung",
        "",
        f"- **ACTIVE:** {len(active)} Faktoren (% non-zero > 0)",
        f"- **ZERO (erwartet):** {len(zero)} Faktoren (kein Datenfehler — Daten nicht verfügbar oder live-only)",
        f"- **STOPP:** {len(stopp)} Faktoren (sollten aktiv sein, sind es aber nicht)",
        "",
    ]

    if stopp_violations:
        lines += [
            "### STOPP-Violations — Walk-Forward NICHT ausgeführt",
            "",
        ]
        for f in stopp_violations:
            reason = STOPP_FACTORS.get(f, "—")
            lines.append(f"- `{f}`: {reason}")
        lines += ["", "---", ""]
    else:
        lines += [
            "**STOPP-Check: BESTANDEN.** Alle erwarteten Faktoren aktiv.",
            "",
            "---",
            "",
        ]

    lines += [
        "## Faktor-Detail",
        "",
        "| Faktor | Kategorie | Mean Value | % Non-Zero | Status |",
        "|--------|-----------|-----------|-----------|--------|",
    ]

    CATEGORY_MAP = {
        "trend_ema_spread": "TA/Trend",
        "trend_ma200_position": "TA/Trend",
        "trend_adx_strength": "TA/Trend",
        "trend_macd_hist": "TA/Trend",
        "mom_rsi_centered": "TA/Momentum",
        "mom_volume_weighted": "TA/Momentum",
        "mom_obv_trend": "TA/Momentum",
        "mr_bollinger_pctb": "TA/MR",
        "mr_stoch_oversold": "TA/MR",
        "vol_abnormal": "TA/Volume",
        "vola_regime_score": "TA/Volatility",
        "sector_rotation_bias": "Altdata/Sektor",
        "earnings_surprise_z": "Altdata/Earnings",
        "insider_activity_score": "Altdata/Insider",
        "news_sentiment_7d_z": "Altdata/News",
        "news_volume_spike_z": "Altdata/News",
        "macro_growth_momentum_z": "Altdata/Makro",
        "macro_inflation_surprise_z": "Altdata/Makro",
        "intermarket_bond_equity": "Altdata/Intermarkt",
        "intermarket_credit_spread": "Altdata/Intermarkt",
        "intermarket_yield_curve": "Altdata/Intermarkt",
        "options_put_call_extreme": "Altdata/Options",
        "vix_regime_score": "Altdata/Options",
        "geo_risk_composite": "Altdata/GPR",
    }

    for _, row in audit_df.iterrows():
        cat = CATEGORY_MAP.get(row["factor"], "Altdata")
        status_emoji = (
            "✅"
            if row["status"] == "ACTIVE"
            else ("🛑" if row["status"] == "STOPP" else "—")
        )
        lines.append(
            f"| {row['factor']} | {cat} | {row['mean_value']:.4f} "
            f"| {row['pct_nonzero'] * 100:.0f}% | {status_emoji} {row['status']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Aktivierungs-Methodologie",
        "",
        "- **gpr_index**: aus `output/macro_gpr.parquet` (Caldara-Iacoviello, monatlich → tägl. ffill)",
        "- **yield_curve_slope**: aus `output/macro.parquet` Spalte `yield_curve_spread` (monatlich → tägl. ffill)",
        "- **bond_equity_divergence_flag**: TLT 20d-Return > 0 UND SPY 20d-Return < 0 (aus ETF-Panel via yfinance)",
        "- **credit_spread_change_5d**: HYG/IEF-Ratio 5-Tage-pct_change (aus ETF-Panel via yfinance)",
        "- **sector_rotation_bias**: Sector-ETF-Rows (XLK/XLF/XLE/…) im Price-Panel + configs/security_meta.csv",
        "- **earnings_surprise_z**: auto via altdata_loader aus `output/events_earnings.parquet`",
        "- **macro_growth_momentum_z / macro_inflation_surprise_z**: auto via altdata_loader aus `output/macro.parquet`",
        "",
        "_Dieses Dokument ist ein automatisch erzeugtes Artefakt aus_ `scripts/_oos_wf_mfv2_full.py`.",
    ]

    OUT_AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT_MD.write_text("\n".join(lines), encoding="utf-8")
    log.info("Audit report written to %s", OUT_AUDIT_MD)


def _write_oos_report(
    *,
    fold_df,
    agg,
    n_ok,
    n_total,
    actual_start,
    actual_end,
    n_symbols,
    all_symbols_requested,
    n_active_factors,
):
    lines = [
        "# multifactor_v2 Full-Stack — Echter OOS Walk-Forward (Alpaca, 2026-05)",
        "",
        f"**Erstellt:** {pd.Timestamp.utcnow().date()}  ",
        "**Branch:** main  ",
        "**Zweck:** GO_LIVE_CHECKLIST Paket 3c.2 — mfv2 mit vollem verfügbarem Faktor-Stack.",
        "",
        "---",
        "",
        "## Datenquelle",
        "",
        "- **Anbieter:** Alpaca Markets (Free Tier) + yfinance (ETF-Panel)",
        f"- **Angefordertes Universum:** {all_symbols_requested} Symbole (watchlist.txt)",
        f"- **Symbole mit Alpaca-Daten:** {n_symbols}",
        f"- **Tatsächliche Zeitspanne:** {actual_start.date()} → {actual_end.date()}",
        "- **SPY:** Buy-and-Hold-Benchmark",
        "",
        "## Walk-Forward-Konfiguration",
        "",
        "- Modus: Rolling",
        f"- Train-Fenster: {TRAIN_WINDOW_DAYS} Handelstage (~1 Jahr)",
        f"- Test-Fenster: {TEST_WINDOW_DAYS} Handelstage (~1 Jahr)",
        f"- Schrittweite: {STEP_SIZE_DAYS} Handelstage",
        f"- Warmup-Buffer: {WARMUP_BARS} Bars",
        "- Rebalancierung: Monatlich (einheitlicher Kalender-Anker)",
        f"- Commission: {COMMISSION_BPS} bps, Spread-Weight: 0.25, Impact-Weight: 0.5",
        f"- Startkapital: {INITIAL_CAPITAL:,.0f} USD",
        "",
        "**Aktive Faktoren in diesem Test:**",
        f"- {n_active_factors} von ~35 Faktoren aktiv (basierend auf Factor-Audit 2023-06-30)",
        "- Neu aktiviert (ACTIVE im Factor-Audit 2023-06-30): macro_growth_momentum_z,",
        "  macro_inflation_surprise_z, intermarket_credit_spread, intermarket_yield_curve,",
        "  geo_risk_composite",
        "- Weiterhin 0.0: earnings_surprise_z (altdata_loader gap), sector_rotation_bias",
        "  (security_meta/ETF-Wiring-Gap), intermarket_bond_equity (kein TLT-up+SPY-down Tag",
        "  im Audit-Fenster), news_sentiment (Daten erst ab 2025-12),",
        "  insider (all unknown), options/VIX (live-only), congress/buyback (keine Daten)",
        "",
        "---",
        "",
        "## Ergebnisse pro Fold",
        "",
        "| Fold | Train | Test | CAGR | Sharpe | MaxDD | SPY-CAGR | SPY-Sharpe | Schlägt SPY? |",
        "|------|-------|------|------|--------|-------|----------|------------|-------------|",
    ]

    for _, row in fold_df.iterrows():
        beats = ""
        if (
            row["status"] == "OK"
            and not np.isnan(row["cagr"])
            and not np.isnan(row["bh_cagr"])
        ):
            beats = "Ja" if row["cagr"] > row["bh_cagr"] else "Nein"
        lines.append(
            f"| {int(row['fold'])} | {row['train_start']}–{row['train_end']} "
            f"| {row['test_start']}–{row['test_end']} "
            f"| {_fmt_pct(row['cagr'])} | {_fmt_f(row['sharpe'])} | {_fmt_pct(row['max_dd'])} "
            f"| {_fmt_pct(row['bh_cagr'])} | {_fmt_f(row['bh_sharpe'])} | {beats} |"
        )
        if row["status"] == "FAILED":
            lines.append(f"> Fold {int(row['fold'])} FAILED: {row['error']}")

    mean_cagr = agg["mean_cagr"]
    mean_bh_cagr = agg["mean_bh_cagr"]
    ta_only_sharpe = 0.36  # TA-only Sharpe baseline from Paket 3b

    lines += [
        "",
        f"_Erfolgreiche Folds: {n_ok}/{n_total}_",
        "",
        "---",
        "",
        "## Aggregierte OOS-Metriken",
        "",
        "| Metrik | mfv2 Full-Stack | mfv2 TA-only (3b) | SPY Buy-and-Hold |",
        "|--------|----------------|------------------|-----------------|",
        f"| Ø CAGR | {_fmt_pct(mean_cagr)} | +12.9% | {_fmt_pct(mean_bh_cagr)} |",
        f"| Ø Sharpe | {_fmt_f(agg['mean_sharpe'])} | 0.36 | {_fmt_f(agg['mean_bh_sharpe'])} |",
        f"| Ø MaxDD | {_fmt_pct(agg['mean_max_dd'])} | -23.0% | — |",
        f"| Win-Rate (CAGR > 0) | {_fmt_pct(agg['win_rate'])} | 70% | — |",
        f"| Folds vs SPY | {_fmt_pct(agg['beats_spy'])} | 60% | — |",
        "",
        "---",
        "",
        "## Bewertung",
        "",
    ]

    beats_spy_pct = agg["beats_spy"]
    sharpe_delta = agg["mean_sharpe"] - ta_only_sharpe

    if np.isnan(mean_cagr):
        verdict = "Alle Folds fehlgeschlagen — kein Ergebnis."
    elif beats_spy_pct >= 0.67 and mean_cagr > mean_bh_cagr:
        verdict = (
            f"mfv2 Full-Stack schlägt SPY in {_fmt_pct(beats_spy_pct)} der Folds "
            f"(Ø CAGR {_fmt_pct(mean_cagr)} vs. SPY {_fmt_pct(mean_bh_cagr)}). "
            f"Sharpe {_fmt_f(agg['mean_sharpe'])} vs. TA-only 0.36 (Δ {sharpe_delta:+.2f}). "
            "Das Ergebnis ist **positiv**. Die Altdata-Faktoren liefern messbaren Mehrwert gegenüber dem TA-only-Subset."
        )
    elif beats_spy_pct >= 0.5:
        verdict = (
            f"mfv2 Full-Stack schlägt SPY in {_fmt_pct(beats_spy_pct)} der Folds "
            f"(Ø CAGR {_fmt_pct(mean_cagr)} vs. SPY {_fmt_pct(mean_bh_cagr)}). "
            f"Sharpe Δ gegenüber TA-only: {sharpe_delta:+.2f}. Das Ergebnis ist **gemischt**."
        )
    else:
        verdict = (
            f"mfv2 Full-Stack schlägt SPY nur in {_fmt_pct(beats_spy_pct)} der Folds "
            f"(Ø CAGR {_fmt_pct(mean_cagr)} vs. SPY {_fmt_pct(mean_bh_cagr)}). "
            f"Sharpe {_fmt_f(agg['mean_sharpe'])} (Δ {sharpe_delta:+.2f} vs. TA-only). "
            "Das Ergebnis ist **negativ**. Die verfügbaren Altdata-Faktoren liefern keinen "
            "robusten OOS-Edge gegenüber SPY Buy-and-Hold."
        )

    lines.append(verdict)
    lines += [
        "",
        "### Vergleich zu TA-only Baseline (Paket 3b)",
        "",
        "- TA-only Baseline (Paket 3b): Ø CAGR +12.9%, Ø Sharpe 0.36, 6/10 Folds vs SPY",
        "- Full-Stack (Paket 3c.2): Messung ob aktivierte Faktoren Verbesserung bringen.",
        f"- Sharpe-Delta: {sharpe_delta:+.2f} — "
        f"{'Verbesserung' if sharpe_delta > 0.005 else ('Verschlechterung' if sharpe_delta < -0.005 else 'unverändert')} "
        "durch Altdata-Stack.",
        "",
        "### Einschränkungen",
        "",
        "- News-Sentiment (F21/F22) = 0.0 in allen historischen Folds (Daten erst ab 2025-12-22).",
        "- Insider-Faktoren (F20/F32) = 0.0 (insider_trading.parquet enthält nur 'unknown' Trades).",
        "- VIX/Options-Faktoren (F28/F29/F35) = 0.0 (live CBOE-Fetch, kein historisches Parquet).",
        "- Congress/Buyback (F30/F34) = 0.0 (keine Datendateien vorhanden).",
        "- Alpaca Free Tier: Survivorship-Bias möglich.",
        "- SPY-Benchmark ohne Dividenden-Reinvest.",
        "",
        "---",
        "",
        "_Quelldokumente:_ `docs/results/2026_05_mfv2_factor_activation_log.md`  ",
        "_Skript:_ `scripts/_oos_wf_mfv2_full.py`  ",
        "_Nicht manuell editieren._",
    ]

    OUT_OOS_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_OOS_MD.write_text("\n".join(lines), encoding="utf-8")
    log.info("OOS report written to %s", OUT_OOS_MD)


def _write_failure_report(reason: str):
    for path in [OUT_AUDIT_MD, OUT_OOS_MD]:
        lines = [
            "# multifactor_v2 Full-Stack — OOS Walk-Forward",
            "",
            "**Status: ABGEBROCHEN**",
            "",
            f"**Grund:** {reason}",
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
    log.error("Failure report written: %s", reason)


# ---------------------------------------------------------------------------
# 9 — Main
# ---------------------------------------------------------------------------
def main():
    all_symbols = [
        s.strip()
        for s in WATCHLIST.read_text(encoding="utf-8").splitlines()
        if s.strip() and not s.strip().startswith("#") and "." not in s.strip()
    ]
    log.info("Watchlist: %d symbols", len(all_symbols))

    try:
        prices = _get_prices(all_symbols + ["SPY"])
    except Exception as exc:
        log.error("Price fetch failed: %s", exc)
        _write_failure_report(str(exc))
        return 1

    try:
        etf_prices = _get_etf_prices()
    except Exception as exc:
        log.warning("ETF fetch failed (non-fatal): %s", exc)
        etf_prices = pd.DataFrame()

    actual_symbols = prices["symbol"].unique().tolist()
    tradeable = [s for s in actual_symbols if s != "SPY"]
    actual_start = prices["timestamp"].min()
    actual_end = prices["timestamp"].max()
    log.info(
        "Data range: %s → %s, tradeable: %d",
        actual_start.date(),
        actual_end.date(),
        len(tradeable),
    )

    spy_prices = prices[prices["symbol"] == "SPY"].copy()
    strategy_prices = prices[prices["symbol"] != "SPY"].copy()

    # Build macro panel (needs SPY for bond_equity_divergence_flag)
    log.info("Building macro panel…")
    try:
        macro_panel = _build_macro_panel(prices, etf_prices)
    except Exception as exc:
        log.error("Macro panel build failed: %s", exc)
        _write_failure_report(f"Macro panel build failed: {exc}")
        return 1

    # --- Factor audit ---
    log.info("Running factor audit at %s…", AUDIT_DATE.date())
    try:
        audit_df = _run_factor_audit(strategy_prices, etf_prices, macro_panel)
    except Exception as exc:
        log.error("Factor audit failed: %s", exc)
        _write_failure_report(f"Factor audit failed: {exc}")
        return 1

    stopp_violations = _check_stopp(audit_df)
    _write_audit_report(audit_df, stopp_violations)

    # Print audit summary
    active_factors = audit_df[audit_df["status"] == "ACTIVE"]["factor"].tolist()
    log.info("[Audit] ACTIVE factors (%d): %s", len(active_factors), active_factors)
    if stopp_violations:
        log.error("[Audit] STOPP violations: %s", stopp_violations)
        log.error("Walk-forward ABORTED — fix data wiring first.")
        return 2

    log.info(
        "[Audit] STOPP-Check PASSED — %d active factors. Starting walk-forward.",
        len(active_factors),
    )

    # --- Walk-forward ---
    from src.assembled_core.qa.walk_forward import (
        WalkForwardConfig,
        run_walk_forward_backtest,
    )

    wf_start = max(PERIOD_START, actual_start.normalize())
    wf_end = min(PERIOD_END, actual_end.normalize())

    config = WalkForwardConfig(
        start_date=wf_start,
        end_date=wf_end,
        train_window_days=TRAIN_WINDOW_DAYS,
        test_window_days=TEST_WINDOW_DAYS,
        step_size_days=STEP_SIZE_DAYS,
        mode="rolling",
        min_train_periods=200,
        min_test_periods=200,
    )

    backtest_fn = _make_backtest_fn(strategy_prices, etf_prices, macro_panel)

    log.info("Running walk-forward (%d folds)…", 10)
    try:
        wf_result = run_walk_forward_backtest(config=config, backtest_fn=backtest_fn)
    except Exception as exc:
        log.error("Walk-forward failed: %s", exc, exc_info=True)
        _write_failure_report(str(exc))
        return 1

    summary = (
        wf_result.summary_df.set_index("split_index")
        if not wf_result.summary_df.empty
        else pd.DataFrame()
    )

    fold_rows = []
    for wr in wf_result.window_results:
        w = wr.window
        spy_bh = _spy_buyhold(spy_prices, w.test_start, w.test_end)
        if wr.status == "failed":
            fold_rows.append(
                {
                    "fold": w.split_index + 1,
                    "train_start": w.train_start.date(),
                    "train_end": w.train_end.date(),
                    "test_start": w.test_start.date(),
                    "test_end": w.test_end.date(),
                    "cagr": float("nan"),
                    "sharpe": float("nan"),
                    "max_dd": float("nan"),
                    **spy_bh,
                    "status": "FAILED",
                    "error": wr.error_message,
                }
            )
        else:
            row = (
                summary.loc[w.split_index]
                if w.split_index in summary.index
                else pd.Series()
            )
            fold_rows.append(
                {
                    "fold": w.split_index + 1,
                    "train_start": w.train_start.date(),
                    "train_end": w.train_end.date(),
                    "test_start": w.test_start.date(),
                    "test_end": w.test_end.date(),
                    "cagr": float(row.get("test_cagr", float("nan"))),
                    "sharpe": float(row.get("test_sharpe", float("nan"))),
                    "max_dd": float(row.get("test_max_dd", float("nan"))),
                    **spy_bh,
                    "status": "OK",
                    "error": None,
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    ok = fold_df[fold_df["status"] == "OK"]
    n_ok, n_total = len(ok), len(fold_df)

    if n_ok == 0:
        _write_failure_report("All folds failed")
        return 1

    agg = {
        "mean_cagr": ok["cagr"].mean(),
        "mean_sharpe": ok["sharpe"].mean(),
        "mean_max_dd": ok["max_dd"].mean(),
        "win_rate": (ok["cagr"] > 0).mean(),
        "beats_spy": (ok["cagr"] > ok["bh_cagr"]).mean(),
        "mean_bh_cagr": ok["bh_cagr"].mean(),
        "mean_bh_sharpe": ok["bh_sharpe"].mean(),
    }

    _write_oos_report(
        fold_df=fold_df,
        agg=agg,
        n_ok=n_ok,
        n_total=n_total,
        actual_start=actual_start,
        actual_end=actual_end,
        n_symbols=len(tradeable),
        all_symbols_requested=len(all_symbols),
        n_active_factors=len(active_factors),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
