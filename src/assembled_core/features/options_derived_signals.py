"""Options-Derived Regime Signals — VIX Term Structure and Put/Call Ratio.

Builds macro regime conditioning factors from freely available options-market
data. These factors enrich the macro overlay and regime detection pipeline.

Factors produced:
    vix_level          — current VIX value
    vix_change_5d      — 5-day change in VIX
    vix_change_20d     — 20-day change in VIX
    vix_term_slope     — VIX3M - VIX (positive = contango / calm, negative = backwardation / fear)
    vix_regime         — categorical: "low" (<15), "normal" (15–25), "high" (25–35), "extreme" (>35)
    put_call_ratio_raw — raw daily equity put/call ratio
    put_call_ratio_ma_20d — 20-day moving average of put/call ratio
    equity_put_call_extreme — 1 when put/call ratio > 1.2 (excessive fear), -1 when < 0.7 (complacency)
    vix_zscore_252d    — z-score of VIX relative to 252-day window (standardised fear level)

All values are universal (one value per date, same for all symbols) and are
designed to be merged onto a daily price panel.

Main entry point:
    build_options_regime_factors(cboe_df) -> pd.DataFrame
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TIMESTAMP = "timestamp"

# VIX regime thresholds
_VIX_LOW = 15.0
_VIX_NORMAL_HIGH = 25.0
_VIX_HIGH_MAX = 35.0

# Put/Call extremes
_PCR_FEAR_THRESHOLD = 1.2
_PCR_COMPLACENCY_THRESHOLD = 0.7


def build_options_regime_factors(
    cboe_df: pd.DataFrame,
    timestamp_col: str = _TIMESTAMP,
    vix_col: str = "vix",
    vix3m_col: str = "vix3m",
    pcr_col: str = "put_call_ratio",
) -> pd.DataFrame:
    """Compute options-derived regime factors from CBOE data.

    Args:
        cboe_df: DataFrame from CBOESource.fetch_options_regime_data() or
                 compatible format with columns: timestamp, vix, vix3m,
                 put_call_ratio. Missing columns are handled gracefully.
        timestamp_col: Timestamp column name (default: "timestamp")
        vix_col: VIX column name (default: "vix")
        vix3m_col: VIX3M column name (default: "vix3m")
        pcr_col: Put/Call ratio column name (default: "put_call_ratio")

    Returns:
        DataFrame with timestamp + options regime factor columns. One row
        per trading date. Suitable for merging onto a daily price panel via
        pd.merge_asof or a join on timestamp.
    """
    if cboe_df.empty:
        logger.warning("[Options] Input cboe_df is empty — returning empty factors")
        return pd.DataFrame(columns=[timestamp_col])

    df = cboe_df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    # VIX factors
    if vix_col in df.columns:
        df["vix_level"] = df[vix_col]
        df["vix_change_5d"] = df[vix_col].diff(5)
        df["vix_change_20d"] = df[vix_col].diff(20)

        # Z-score of VIX relative to 252-day rolling window
        rolling_mean = df[vix_col].rolling(252, min_periods=63).mean()
        rolling_std = df[vix_col].rolling(252, min_periods=63).std()
        df["vix_zscore_252d"] = (df[vix_col] - rolling_mean) / rolling_std.replace(
            0, np.nan
        )

        # Categorical regime
        vix = df[vix_col]
        conditions = [
            vix < _VIX_LOW,
            (vix >= _VIX_LOW) & (vix < _VIX_NORMAL_HIGH),
            (vix >= _VIX_NORMAL_HIGH) & (vix < _VIX_HIGH_MAX),
            vix >= _VIX_HIGH_MAX,
        ]
        choices = ["low", "normal", "high", "extreme"]
        df["vix_regime"] = np.select(conditions, choices, default="normal")
    else:
        logger.warning("[Options] vix column not found — VIX factors skipped")

    # VIX term slope: VIX3M - VIX
    if vix3m_col in df.columns and vix_col in df.columns:
        df["vix_term_slope"] = df[vix3m_col] - df[vix_col]
    else:
        logger.warning("[Options] vix3m or vix column missing — term slope skipped")

    # Put/Call Ratio factors
    if pcr_col in df.columns:
        df["put_call_ratio_raw"] = df[pcr_col]
        df["put_call_ratio_ma_20d"] = df[pcr_col].rolling(20, min_periods=10).mean()

        pcr = df[pcr_col]
        pcr_extreme = np.where(
            pcr > _PCR_FEAR_THRESHOLD,
            1.0,
            np.where(pcr < _PCR_COMPLACENCY_THRESHOLD, -1.0, 0.0),
        )
        df["equity_put_call_extreme"] = pcr_extreme
    else:
        logger.warning(
            "[Options] put_call_ratio column not found — PCR factors skipped"
        )

    # Keep only timestamp + output factor columns
    factor_cols = [
        "vix_level",
        "vix_change_5d",
        "vix_change_20d",
        "vix_term_slope",
        "vix_regime",
        "put_call_ratio_raw",
        "put_call_ratio_ma_20d",
        "equity_put_call_extreme",
        "vix_zscore_252d",
    ]
    output_cols = [timestamp_col] + [c for c in factor_cols if c in df.columns]
    result = df[output_cols].copy()

    logger.info(
        "[Options] Built %d options regime factor rows with columns: %s",
        len(result),
        [c for c in factor_cols if c in df.columns],
    )
    return result


def align_options_factors_to_panel(
    price_panel: pd.DataFrame,
    options_factors: pd.DataFrame,
    symbol_col: str = "symbol",
    timestamp_col: str = _TIMESTAMP,
) -> pd.DataFrame:
    """Merge options regime factors onto a daily price panel.

    Uses backward merge_asof so each daily bar gets the most recent
    available options factor value (PIT-safe).

    Args:
        price_panel: Daily OHLCV panel with symbol and timestamp columns.
        options_factors: Output of build_options_regime_factors().
        symbol_col: Symbol column name in price_panel.
        timestamp_col: Timestamp column name.

    Returns:
        price_panel with options factor columns appended.
    """
    if options_factors.empty or price_panel.empty:
        return price_panel.copy()

    panel = price_panel.copy()
    panel[timestamp_col] = pd.to_datetime(panel[timestamp_col])
    opts = options_factors.copy()
    opts[timestamp_col] = pd.to_datetime(opts[timestamp_col])
    opts = opts.sort_values(timestamp_col)

    feature_cols = [c for c in opts.columns if c != timestamp_col]

    # options factors are universal (same for all symbols) — merge once
    panel_sorted = panel.sort_values(timestamp_col)
    merged = pd.merge_asof(
        panel_sorted,
        opts[[timestamp_col] + feature_cols],
        on=timestamp_col,
        direction="backward",
    )
    return merged.sort_values([symbol_col, timestamp_col]).reset_index(drop=True)


def get_options_factor_names() -> list[str]:
    """Return list of all options regime factor column names."""
    return [
        "vix_level",
        "vix_change_5d",
        "vix_change_20d",
        "vix_term_slope",
        "vix_regime",
        "put_call_ratio_raw",
        "put_call_ratio_ma_20d",
        "equity_put_call_extreme",
        "vix_zscore_252d",
    ]


# ---------------------------------------------------------------------------
# Options Skew and IV Features (Plan 3.6)
# ---------------------------------------------------------------------------


def compute_vix_term_structure(vix: float, vix3m: float) -> float:
    """VIX term structure: VIX3M/VIX. <1 = backwardation (acute fear)."""
    if vix < 0.1:
        return 1.0
    return vix3m / vix


def compute_implied_vs_realized_spread(vix: float, realized_vol_20d: float) -> float:
    """Spread between implied and realized vol. Positive = fear > reality."""
    return vix / 100 - realized_vol_20d


def compute_skew_vix_divergence(skew: float, vix: float) -> float:
    """SKEW/VIX ratio. Rises when tail risk increases without panic."""
    if vix < 1.0:
        return 0.0
    return skew / vix
