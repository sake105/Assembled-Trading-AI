"""Cross-sectional Earnings Surprise and Insider Activity factors (B2.3).

This module provides PIT-safe, cross-sectional factor computation for a single
``as_of_date`` and a symbol set. It is deliberately independent from the
time-series builders in :mod:`altdata_earnings_insider_factors` — which return
one row per (symbol, timestamp) — because the multifactor dispatch consumes a
single snapshot per rebalance date.

Factors implemented (plan section 4.1, factors 19-20):

- ``earnings_surprise_z``: last EPS surprise percentage, gated on
  ``filing_date <= as_of_date``, linearly decayed between 90 and 120 days,
  then cross-sectionally z-scored and clipped to +/- 3.
- ``insider_activity_score``: signed insider USD flow over the trailing 60
  calendar days, gated on ``filing_date <= as_of_date``, optionally
  normalized by market cap, then cross-sectionally z-scored and clipped.

PIT-safety guarantees:

- Both factors key exclusively on ``filing_date`` (disclosure date). Any row
  with ``filing_date > as_of_date`` is dropped unconditionally before any
  aggregation.
- Decay and window boundaries are computed from ``filing_date`` only.
- No price or forward-looking information is used.

This module is feature-only — it is NOT wired into multifactor dispatch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARNINGS_REQUIRED_COLS = ("symbol", "filing_date", "eps_actual", "eps_estimate")
INSIDER_REQUIRED_COLS = ("symbol", "filing_date", "transaction_type", "value_usd")

EARNINGS_DECAY_START_DAYS = 90
EARNINGS_DECAY_END_DAYS = 120
INSIDER_WINDOW_DAYS = 60
CLIP_BOUND = 3.0
SAFE_DIVIDE_EPS = 1e-6


def _validate_columns(df: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}. "
            f"Got: {list(df.columns)}"
        )


def _zscore_clip(values: pd.Series) -> pd.Series:
    """Cross-sectional z-score, clipped to +/- CLIP_BOUND.

    Single-valid-observation case returns NaN for that observation (can't
    z-score one point). All-NaN input returns all-NaN.
    """
    valid = values.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=values.index, dtype=float)
    mean = valid.mean()
    std = valid.std(ddof=0)
    if std < SAFE_DIVIDE_EPS:
        # Degenerate (all identical) -> all zeros for valid entries.
        out = pd.Series(np.nan, index=values.index, dtype=float)
        out.loc[valid.index] = 0.0
        return out
    z = (values - mean) / std
    return z.clip(lower=-CLIP_BOUND, upper=CLIP_BOUND)


def _earnings_surprise_raw(
    as_of_date: pd.Timestamp,
    symbols: list[str],
    earnings_df: pd.DataFrame,
) -> pd.Series:
    """Compute raw (pre-zscore) earnings surprise per symbol.

    Returns NaN where no PIT-visible filing exists, where estimate is too
    small (safe-divide), or where the filing is older than the decay window.
    """
    out = pd.Series(np.nan, index=symbols, dtype=float, name="earnings_surprise_raw")
    if earnings_df.empty:
        return out

    df = earnings_df.copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    # PIT gate: drop anything not yet disclosed at as_of_date.
    df = df[df["filing_date"] <= as_of_date]
    if df.empty:
        return out

    df = df[df["symbol"].isin(symbols)]
    if df.empty:
        return out

    # Take the most recent filing per symbol.
    df = df.sort_values("filing_date").groupby("symbol", as_index=False).tail(1)

    for _, row in df.iterrows():
        sym = row["symbol"]
        est = row["eps_estimate"]
        act = row["eps_actual"]
        if pd.isna(est) or pd.isna(act):
            continue
        if abs(est) < SAFE_DIVIDE_EPS:
            # Safe-divide: undefined surprise, keep as NaN.
            continue
        surprise = (act - est) / abs(est)

        days_old = (as_of_date - row["filing_date"]).days
        if days_old <= EARNINGS_DECAY_START_DAYS:
            scale = 1.0
        elif days_old >= EARNINGS_DECAY_END_DAYS:
            scale = 0.0
        else:
            # Linear decay from 1.0 at 90d to 0.0 at 120d.
            scale = max(
                0.0,
                1.0 - (days_old - EARNINGS_DECAY_START_DAYS)
                / (EARNINGS_DECAY_END_DAYS - EARNINGS_DECAY_START_DAYS),
            )
        out.loc[sym] = surprise * scale

    return out


def _insider_activity_raw(
    as_of_date: pd.Timestamp,
    symbols: list[str],
    insider_df: pd.DataFrame,
    market_cap_df: pd.DataFrame | None,
) -> pd.Series:
    """Compute raw (pre-zscore) insider activity per symbol.

    Signed signed-value sum in the trailing INSIDER_WINDOW_DAYS calendar
    days (purchases positive, sales negative), normalized by market cap if
    available else by the absolute signed sum across the section (fallback
    scale so z-score is meaningful).
    """
    out = pd.Series(np.nan, index=symbols, dtype=float, name="insider_activity_raw")
    if insider_df.empty:
        return out

    df = insider_df.copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    window_start = as_of_date - pd.Timedelta(days=INSIDER_WINDOW_DAYS)

    # PIT gate + window gate.
    df = df[(df["filing_date"] <= as_of_date) & (df["filing_date"] > window_start)]
    if df.empty:
        return out

    df = df[df["symbol"].isin(symbols)].copy()
    if df.empty:
        return out

    # Signed value: purchases positive, sales negative.
    txn_type = df["transaction_type"].astype(str).str.upper().str.strip()
    sign = np.where(txn_type == "P", 1.0, np.where(txn_type == "S", -1.0, 0.0))
    df["_signed"] = pd.to_numeric(df["value_usd"], errors="coerce") * sign
    df = df.dropna(subset=["_signed"])

    summed = df.groupby("symbol")["_signed"].sum()

    # Normalization.
    if market_cap_df is not None and not market_cap_df.empty:
        mcap = market_cap_df.set_index("symbol")["market_cap"]
        for sym in summed.index:
            cap = mcap.get(sym, np.nan)
            if pd.notna(cap) and cap > 0:
                out.loc[sym] = summed.loc[sym] / cap
            else:
                # Missing/invalid market cap -> fallback scale below.
                out.loc[sym] = summed.loc[sym]
    else:
        # Fallback scale: divide by section-wide mean absolute signed flow so
        # z-scoring is meaningful even in unit-less flow terms. If everything
        # is zero, the z-score stage will handle the degenerate case.
        denom = summed.abs().mean()
        if denom and denom > 0:
            for sym in summed.index:
                out.loc[sym] = summed.loc[sym] / denom
        else:
            for sym in summed.index:
                out.loc[sym] = 0.0

    return out


def compute_earnings_insider_factors(
    as_of_date: pd.Timestamp,
    symbols: list[str],
    earnings_df: pd.DataFrame,
    insider_df: pd.DataFrame,
    market_cap_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute cross-sectional earnings surprise and insider activity factors.

    Args:
        as_of_date: Point-in-time cutoff. Must be a pandas Timestamp.
        symbols: Universe for the cross-section. Output is indexed by these.
        earnings_df: DataFrame with columns
            ``symbol, filing_date, eps_actual, eps_estimate``.
        insider_df: DataFrame with columns
            ``symbol, filing_date, transaction_type, value_usd`` where
            ``transaction_type`` is ``"P"`` (purchase) or ``"S"`` (sale).
        market_cap_df: Optional DataFrame with columns ``symbol, market_cap``
            used for insider flow normalization. If missing or None, a
            cross-sectional fallback normalization is used.

    Returns:
        DataFrame indexed by ``symbols`` with columns
        ``["earnings_surprise_z", "insider_activity_score"]``. Symbols with
        no PIT-visible data receive NaN (not zero).

    Raises:
        ValueError: If ``as_of_date`` is not a ``pd.Timestamp`` or any
            required column is missing.
    """
    if not isinstance(as_of_date, pd.Timestamp):
        raise ValueError(
            f"as_of_date must be a pandas Timestamp, got {type(as_of_date).__name__}"
        )

    _validate_columns(earnings_df, EARNINGS_REQUIRED_COLS, "earnings_df")
    _validate_columns(insider_df, INSIDER_REQUIRED_COLS, "insider_df")

    symbols = list(symbols)

    raw_earnings = _earnings_surprise_raw(as_of_date, symbols, earnings_df)
    raw_insider = _insider_activity_raw(
        as_of_date, symbols, insider_df, market_cap_df
    )

    earnings_z = _zscore_clip(raw_earnings)
    earnings_z.name = "earnings_surprise_z"
    insider_z = _zscore_clip(raw_insider)
    insider_z.name = "insider_activity_score"

    out = pd.DataFrame(
        {
            "earnings_surprise_z": earnings_z,
            "insider_activity_score": insider_z,
        },
        index=symbols,
    )
    return out
