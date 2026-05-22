"""Alt-Data Feature Builder: BLS Labor Market Features.

Transforms Bureau of Labor Statistics (BLS) time-series data into
ML-ready macro labor-market features for use in factor models.

BLS source columns (from bls_source.fetch_bls_series):
    timestamp (UTC, first day of period), series_id, value (float),
    period (raw BLS period code), year (int).

Expected series IDs used by this builder:
    LNS14000000  — US Unemployment Rate (monthly, %)
    CEU0000000001 — Total Nonfarm Payroll Employment (monthly, thousands)

Audit: C2-059
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "build_bls_labor_features",
    "UNEMPLOYMENT_SERIES_ID",
    "NONFARM_PAYROLL_SERIES_ID",
]

UNEMPLOYMENT_SERIES_ID = "LNS14000000"
NONFARM_PAYROLL_SERIES_ID = "CEU0000000001"

# Regime thresholds
_HAWKISH_UNEMP_THRESHOLD = 4.0  # below → tight labor market (hawkish)
_DOVISH_UNEMP_THRESHOLD = 6.0  # above → loose labor market (dovish)

_OUTPUT_COLS = [
    "timestamp",
    "unemployment_rate",
    "unemployment_3m_change",
    "nonfarm_payroll_mom",
    "labor_market_regime",
]


def build_bls_labor_features(
    bls_df: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Build BLS-based labor market features.

    PIT-safe: only uses observations with timestamp <= as_of.

    Args:
        bls_df: Output of bls_source.fetch_bls_series().  Expected columns:
            ``timestamp``, ``series_id``, ``value``.  Extra columns are ignored.
        as_of: Point-in-time cutoff.

    Returns:
        DataFrame with columns:
            timestamp, unemployment_rate, unemployment_3m_change,
            nonfarm_payroll_mom, labor_market_regime (str: hawkish/neutral/dovish).
        Returns empty DataFrame with correct columns when input is empty
        or no data satisfies the PIT filter.
    """
    empty = pd.DataFrame(columns=_OUTPUT_COLS)

    if bls_df is None or bls_df.empty:
        return empty.copy()

    required = {"timestamp", "series_id", "value"}
    if not required.issubset(bls_df.columns):
        logger.warning(
            "[BLS] Missing columns %s — returning empty features.",
            required - set(bls_df.columns),
        )
        return empty.copy()

    df = bls_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("UTC")
    else:
        as_of_ts = as_of_ts.tz_convert("UTC")

    # PIT filter
    df = df[df["timestamp"] <= as_of_ts].copy()

    if df.empty:
        return empty.copy()

    # --- Extract unemployment series ---
    unemp_df = df[df["series_id"] == UNEMPLOYMENT_SERIES_ID].copy()
    unemp_df = unemp_df.sort_values("timestamp")
    unemp_series = unemp_df.set_index("timestamp")["value"].rename("unemployment_rate")

    # --- Extract nonfarm payroll series ---
    payroll_df = df[df["series_id"] == NONFARM_PAYROLL_SERIES_ID].copy()
    payroll_df = payroll_df.sort_values("timestamp")
    payroll_series = payroll_df.set_index("timestamp")["value"].rename(
        "nonfarm_payroll"
    )

    if unemp_series.empty and payroll_series.empty:
        return empty.copy()

    # Build aligned result frame on union of timestamps
    all_ts = unemp_series.index.union(payroll_series.index).sort_values()
    result = pd.DataFrame(index=all_ts)
    result.index.name = "timestamp"

    if not unemp_series.empty:
        result = result.join(unemp_series, how="left")
        # 3-month trailing change in unemployment rate
        result["unemployment_3m_change"] = result["unemployment_rate"].diff(3)
    else:
        result["unemployment_rate"] = np.nan
        result["unemployment_3m_change"] = np.nan

    if not payroll_series.empty:
        result = result.join(payroll_series, how="left")
        # Month-over-month % change in nonfarm payrolls
        result["nonfarm_payroll_mom"] = result["nonfarm_payroll"].pct_change(1)
        result = result.drop(columns=["nonfarm_payroll"])
    else:
        result["nonfarm_payroll_mom"] = np.nan

    # --- Labor market regime ---
    def _classify_regime(unemp: float) -> str:
        if np.isnan(unemp):
            return "neutral"
        if unemp < _HAWKISH_UNEMP_THRESHOLD:
            return "hawkish"
        if unemp > _DOVISH_UNEMP_THRESHOLD:
            return "dovish"
        return "neutral"

    result["labor_market_regime"] = result["unemployment_rate"].apply(_classify_regime)

    result = result.reset_index()
    # Ensure all expected columns exist
    for col in _OUTPUT_COLS:
        if col not in result.columns:
            result[col] = np.nan if col != "labor_market_regime" else "neutral"

    return result[_OUTPUT_COLS].copy()
