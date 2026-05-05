"""Intermarket Cross-Asset Factors.

Builds macro factors from cross-asset ETF price data (bonds, gold, USD,
credit spreads) and yield curve data from FRED. These are universal factors
— the same value applies to all symbols on a given date.

ETF proxies used (via yfinance — no additional API key required):
    TLT  — 20+ Year Treasury Bond ETF (long duration bonds)
    IEF  — 7-10 Year Treasury Bond ETF (intermediate bonds)
    GLD  — SPDR Gold Shares
    UUP  — Invesco DB USD Index Bullish Fund (DXY proxy)
    HYG  — iShares iBoxx High Yield Corporate Bond ETF
    LQD  — iShares Investment Grade Corporate Bond ETF
    SPY  — S&P 500 (equity benchmark)

Factors produced:
    bond_equity_ratio_20d        — TLT/SPY relative strength (20d MA ratio)
    dollar_trend_20d             — UUP 20-day momentum (USD strength)
    dollar_trend_60d             — UUP 60-day momentum (USD trend)
    credit_spread_change_5d      — HYG/LQD spread proxy, 5-day change
    credit_spread_change_20d     — HYG/LQD spread proxy, 20-day change
    gold_equity_divergence       — GLD return minus SPY return (20d), risk-off signal
    yield_curve_slope            — 10yr - 2yr US Treasury yield from FRED
    hy_ig_ratio                  — HYG/LQD price ratio (credit risk proxy)
    bond_equity_divergence_flag  — 1 when bonds rising while equities falling (defensive regime)

Main entry point:
    build_intermarket_factors(start_date, end_date) -> pd.DataFrame
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TIMESTAMP = "timestamp"

# ETF symbols for cross-asset factors
_ETF_SYMBOLS = ["TLT", "IEF", "GLD", "UUP", "HYG", "LQD", "SPY"]

# yfinance tickers for yield curve (replaces FRED / pandas_datareader)
# ^TNX = CBOE 10-Year Treasury Note yield (in %, e.g. 4.5 = 4.5%)
# 2YY=F = CME 2-Year Treasury Note yield futures (in %)
_YF_10Y = "^TNX"
_YF_2Y = "2YY=F"


def _fetch_etf_prices(
    symbols: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch daily close prices for ETF symbols via yfinance."""
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        logger.warning("[Intermarket] yfinance not installed — ETF fetch skipped")
        return pd.DataFrame()

    try:
        raw = yf.download(
            symbols,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            return pd.DataFrame()

        # Flatten multi-level columns: (Close, SPY) → SPY
        if isinstance(raw.columns, pd.MultiIndex):
            closes = (
                raw["Close"]
                if "Close" in raw.columns.get_level_values(0)
                else raw.xs("Close", axis=1, level=0)
            )
        else:
            closes = raw[["Close"]] if "Close" in raw.columns else raw

        closes.index = pd.to_datetime(closes.index)
        closes = closes.reset_index().rename(
            columns={"Date": "timestamp", "Datetime": "timestamp", "index": "timestamp"}
        )
        if closes.columns[0] != "timestamp":
            closes = closes.rename(columns={closes.columns[0]: "timestamp"})
        closes["timestamp"] = pd.to_datetime(closes["timestamp"])
        return closes.sort_values("timestamp").reset_index(drop=True)
    except Exception as exc:
        logger.warning("[Intermarket] ETF download failed: %s", exc)
        return pd.DataFrame()


def _fetch_yield_curve(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch 10Y and 2Y US Treasury yields via yfinance (^TNX, 2YY=F)."""
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        logger.warning("[Intermarket] yfinance not installed — yield curve skipped")
        return pd.DataFrame()

    try:
        raw = yf.download(
            [_YF_10Y, _YF_2Y],
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            logger.warning("[Intermarket] Yield curve download returned empty DataFrame")
            return pd.DataFrame()

        closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        closes = closes.reset_index()
        col0 = closes.columns[0]
        if str(col0).lower() in ("date", "datetime", "index"):
            closes = closes.rename(columns={col0: "timestamp"})
        else:
            closes.insert(0, "timestamp", closes.pop(col0))
        closes["timestamp"] = pd.to_datetime(closes["timestamp"])

        rename_map: dict = {}
        for col in closes.columns:
            s = str(col)
            if _YF_10Y in s or "TNX" in s.upper():
                rename_map[col] = "yield_10y"
            elif _YF_2Y in s or "2YY" in s.upper():
                rename_map[col] = "yield_2y"
        closes = closes.rename(columns=rename_map)

        if "yield_10y" not in closes.columns or "yield_2y" not in closes.columns:
            logger.warning("[Intermarket] Yield curve columns missing after download")
            return pd.DataFrame()

        df = closes[["timestamp", "yield_10y", "yield_2y"]].dropna(
            subset=["yield_10y", "yield_2y"]
        )
        df["yield_curve_slope"] = df["yield_10y"] - df["yield_2y"]
        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.info(
            "[Intermarket] Fetched yield curve: %d rows (%s to %s)",
            len(df),
            start_date,
            end_date,
        )
        return df[["timestamp", "yield_10y", "yield_2y", "yield_curve_slope"]]
    except Exception as exc:
        logger.warning("[Intermarket] Yield curve fetch failed: %s", exc)
        return pd.DataFrame()


def build_intermarket_factors(
    start_date: str = "2015-01-01",
    end_date: Optional[str] = None,
    timestamp_col: str = _TIMESTAMP,
) -> pd.DataFrame:
    """Compute intermarket cross-asset factors.

    Fetches ETF prices and yield curve data, then computes momentum,
    spread, and relative-strength factors for macro regime conditioning.

    Args:
        start_date: Start date for data fetch (default: "2015-01-01").
        end_date: End date (default: today).
        timestamp_col: Timestamp column name in output.

    Returns:
        DataFrame with one row per trading date and intermarket factor columns.
        Universal factors (same value per date across all symbols).
    """
    from datetime import datetime

    end = end_date or datetime.today().strftime("%Y-%m-%d")

    etf_df = _fetch_etf_prices(_ETF_SYMBOLS, start_date, end)
    yc_df = _fetch_yield_curve(start_date, end)

    if etf_df.empty and yc_df.empty:
        logger.warning("[Intermarket] All data sources empty — returning empty factors")
        return pd.DataFrame(columns=[timestamp_col])

    result = pd.DataFrame()

    if not etf_df.empty:
        ts_col = "timestamp" if "timestamp" in etf_df.columns else etf_df.columns[0]
        result[timestamp_col] = etf_df[ts_col]

        def _ret(sym: str, window: int) -> pd.Series:
            if sym not in etf_df.columns:
                return pd.Series(np.nan, index=etf_df.index)
            return etf_df[sym].pct_change(window, fill_method=None)

        def _ma(sym: str, window: int) -> pd.Series:
            if sym not in etf_df.columns:
                return pd.Series(np.nan, index=etf_df.index)
            return etf_df[sym].rolling(window, min_periods=window // 2).mean()

        # Bond/equity relative strength: TLT MA / SPY MA
        if "TLT" in etf_df.columns and "SPY" in etf_df.columns:
            tlt_ma = _ma("TLT", 20)
            spy_ma = _ma("SPY", 20)
            result["bond_equity_ratio_20d"] = (
                tlt_ma / spy_ma.replace(0, np.nan)
            ).fillna(np.nan)

            # Divergence flag: bonds up, equities down over 20d
            tlt_ret = _ret("TLT", 20)
            spy_ret = _ret("SPY", 20)
            result["bond_equity_divergence_flag"] = np.where(
                (tlt_ret > 0) & (spy_ret < 0),
                1.0,
                np.where((tlt_ret < 0) & (spy_ret > 0), -1.0, 0.0),
            )

        # USD trend (DXY proxy via UUP)
        if "UUP" in etf_df.columns:
            result["dollar_trend_20d"] = _ret("UUP", 20)
            result["dollar_trend_60d"] = _ret("UUP", 60)

        # Credit spread proxy: HYG/LQD spread (negative relationship — HY tightens in risk-on)
        if "HYG" in etf_df.columns and "LQD" in etf_df.columns:
            hy_ig = etf_df["HYG"] / etf_df["LQD"].replace(0, np.nan)
            result["hy_ig_ratio"] = hy_ig.values
            result["credit_spread_change_5d"] = hy_ig.pct_change(
                5, fill_method=None
            ).values
            result["credit_spread_change_20d"] = hy_ig.pct_change(
                20, fill_method=None
            ).values

        # Gold / equity divergence: GLD minus SPY return (20d) — risk-off signal
        if "GLD" in etf_df.columns and "SPY" in etf_df.columns:
            gld_ret = _ret("GLD", 20)
            spy_ret = _ret("SPY", 20)
            result["gold_equity_divergence"] = (gld_ret - spy_ret).values

        result = result.sort_values(timestamp_col).reset_index(drop=True)

    # Merge yield curve
    if not yc_df.empty:
        yc = yc_df.copy()
        yc[timestamp_col] = pd.to_datetime(yc["timestamp"])
        if result.empty:
            result = yc.rename(columns={"timestamp": timestamp_col})
        else:
            result = pd.merge_asof(
                result.sort_values(timestamp_col),
                yc[[timestamp_col, "yield_curve_slope", "yield_10y", "yield_2y"]],
                on=timestamp_col,
                direction="backward",
                tolerance=pd.Timedelta("5D"),
            )

    result = result.ffill(limit=5)
    logger.info(
        "[Intermarket] Built %d rows with factors: %s",
        len(result),
        [c for c in result.columns if c != timestamp_col],
    )
    return result


def align_intermarket_factors_to_panel(
    price_panel: pd.DataFrame,
    intermarket_factors: pd.DataFrame,
    symbol_col: str = "symbol",
    timestamp_col: str = _TIMESTAMP,
) -> pd.DataFrame:
    """Merge intermarket factors onto a daily price panel (PIT-safe).

    Args:
        price_panel: Daily OHLCV panel with symbol and timestamp.
        intermarket_factors: Output of build_intermarket_factors().
        symbol_col: Symbol column name.
        timestamp_col: Timestamp column name.

    Returns:
        price_panel with intermarket factor columns appended.
    """
    if intermarket_factors.empty or price_panel.empty:
        return price_panel.copy()

    panel = price_panel.copy()
    panel[timestamp_col] = pd.to_datetime(panel[timestamp_col])
    im = intermarket_factors.copy()
    im[timestamp_col] = pd.to_datetime(im[timestamp_col])
    im = im.sort_values(timestamp_col)

    feature_cols = [c for c in im.columns if c != timestamp_col]

    merged = pd.merge_asof(
        panel.sort_values(timestamp_col),
        im[[timestamp_col] + feature_cols],
        on=timestamp_col,
        direction="backward",
    )
    return merged.sort_values([symbol_col, timestamp_col]).reset_index(drop=True)


def get_intermarket_factor_names() -> list[str]:
    """Return list of all intermarket factor column names."""
    return [
        "bond_equity_ratio_20d",
        "dollar_trend_20d",
        "dollar_trend_60d",
        "credit_spread_change_5d",
        "credit_spread_change_20d",
        "gold_equity_divergence",
        "yield_curve_slope",
        "yield_10y",
        "yield_2y",
        "hy_ig_ratio",
        "bond_equity_divergence_flag",
    ]
