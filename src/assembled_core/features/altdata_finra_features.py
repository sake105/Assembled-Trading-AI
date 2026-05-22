"""Alt-Data Feature Builder: FINRA Short Interest Features.

Transforms FINRA short interest data (from finra_source) into
per-symbol ML-ready features for short-squeeze and sentiment analysis.

FINRA source provides records with keys (from batch_short_interest_features /
short_interest_features):
    si_qty             — short interest shares quantity
    si_pct_float       — short interest as % of float
    days_to_cover      — short interest / avg daily volume
    si_change_pct      — period-over-period change in short interest quantity

When consumed via raw EquityShortInterest API records the columns are:
    symbolCode, shortInterestQty, shortInterestSharesPct,
    avgDailyVol, daysToClose, settlementDate

This builder accepts a DataFrame in the pre-computed per-symbol format
produced by batch_short_interest_features (index = ticker, columns as above).
It also accepts a tidy long-format DataFrame with columns:
    symbol, timestamp, si_qty, si_pct_float, days_to_cover, si_change_pct

Audit: C2-059
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "build_finra_short_interest_features",
]

# Percentile thresholds for si_regime
_HIGH_SI_PERCENTILE = 0.75
_LOW_SI_PERCENTILE = 0.25

_OUTPUT_COLS = [
    "symbol",
    "short_interest_ratio",
    "short_interest_pct_float",
    "short_squeeze_score",
    "si_regime",
]


def build_finra_short_interest_features(
    finra_df: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Build FINRA short-interest features per symbol.

    PIT-safe: drops any rows with timestamp > as_of when a ``timestamp``
    column is present.  When the input is the wide per-symbol format produced
    by ``batch_short_interest_features`` (index = symbol, no timestamp), the
    whole frame is treated as current (caller's responsibility to pass PIT-safe
    data).

    Args:
        finra_df: Either:
            - Wide format: index = symbol, columns include any of
              ``si_qty``, ``si_pct_float``, ``days_to_cover``, ``si_change_pct``.
            - Long format: columns include ``symbol``, ``timestamp``, and the
              feature columns above.
        as_of: Point-in-time cutoff.

    Returns:
        DataFrame with columns:
            symbol, short_interest_ratio (days-to-cover),
            short_interest_pct_float, short_squeeze_score,
            si_regime (str: high/medium/low).
        Empty DataFrame with correct columns when input is empty.
    """
    empty = pd.DataFrame(columns=_OUTPUT_COLS)

    if finra_df is None or finra_df.empty:
        return empty.copy()

    as_of_ts = pd.Timestamp(as_of)

    # --- Normalise to long tidy format ---
    if "symbol" not in finra_df.columns:
        # Wide format: index = symbol (may be named "ticker", "index", etc.)
        df = finra_df.reset_index()
        for _cand in ("ticker", "Ticker", "Symbol", "index"):
            if _cand in df.columns and "symbol" not in df.columns:
                df = df.rename(columns={_cand: "symbol"})
    else:
        df = finra_df.copy()

    # PIT filter when timestamp column is available
    if "timestamp" not in df.columns:
        logger.warning(
            "[FINRA] No 'timestamp' column — PIT filter skipped, "
            "caller is responsible for data currency."
        )
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if as_of_ts.tzinfo is None:
            as_of_ts_utc = as_of_ts.tz_localize("UTC")
        else:
            as_of_ts_utc = as_of_ts.tz_convert("UTC")
        df = df[df["timestamp"] <= as_of_ts_utc].copy()

    if df.empty:
        return empty.copy()

    if "symbol" not in df.columns:
        logger.warning("[FINRA] No 'symbol' column after normalisation.")
        return empty.copy()

    # For each symbol, take the most recent row (or single row if no timestamp)
    if "timestamp" in df.columns:
        df = (
            df.sort_values("timestamp")
            .groupby("symbol", sort=False)
            .last()
            .reset_index()
        )

    # --- Map columns ---
    # days_to_cover → short_interest_ratio
    if "days_to_cover" in df.columns:
        df["short_interest_ratio"] = pd.to_numeric(df["days_to_cover"], errors="coerce")
    elif "daysToClose" in df.columns:
        df["short_interest_ratio"] = pd.to_numeric(df["daysToClose"], errors="coerce")
    else:
        df["short_interest_ratio"] = np.nan

    # si_pct_float
    if "si_pct_float" in df.columns:
        df["short_interest_pct_float"] = pd.to_numeric(
            df["si_pct_float"], errors="coerce"
        )
    elif "shortInterestSharesPct" in df.columns:
        df["short_interest_pct_float"] = pd.to_numeric(
            df["shortInterestSharesPct"], errors="coerce"
        )
    else:
        df["short_interest_pct_float"] = np.nan

    # short_squeeze_score: based on 3-month change in short interest
    # High si_change_pct → high squeeze potential (short side crowded and increasing)
    # We cap at ±5 for stability
    if "si_change_pct" in df.columns:
        df["short_squeeze_score"] = pd.to_numeric(
            df["si_change_pct"], errors="coerce"
        ).clip(-5.0, 5.0)
    else:
        df["short_squeeze_score"] = np.nan

    # --- si_regime: cross-sectional percentile of short_interest_ratio ---
    sir = df["short_interest_ratio"]
    sir_nonnan = sir.dropna()
    if len(sir_nonnan) >= 3:
        high_thresh = sir_nonnan.quantile(_HIGH_SI_PERCENTILE)
        low_thresh = sir_nonnan.quantile(_LOW_SI_PERCENTILE)

        def _classify(v: float) -> str:
            if np.isnan(v):
                return "medium"
            if v >= high_thresh:
                return "high"
            if v <= low_thresh:
                return "low"
            return "medium"

        df["si_regime"] = sir.apply(_classify)
    else:
        df["si_regime"] = "medium"

    result = df[["symbol"] + [c for c in _OUTPUT_COLS if c != "symbol"]].copy()
    # Ensure all output columns exist
    for col in _OUTPUT_COLS:
        if col not in result.columns:
            result[col] = np.nan if col != "si_regime" else "medium"

    return result[_OUTPUT_COLS].reset_index(drop=True)
