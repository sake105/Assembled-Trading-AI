"""Signal decay and alpha half-life analysis (V6).

Provides tools to measure how quickly a signal's predictive power decays:
- IC half-life: How many days until IC drops to half its peak.
- Forward return half-life: How many days of forward returns a signal predicts.
- Signal autocorrelation: Persistence/mean-reversion of factor values.
- Rank stability: How stable are the top-N stock rankings over time.

These metrics are critical for determining optimal rebalancing frequency
and for detecting stale signals that should not be traded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


@dataclass
class SignalDecayProfile:
    """Decay characteristics of a single signal/factor."""

    factor_name: str
    ic_half_life_days: float | None  # Days until IC decays to 50%
    ic_mean: float  # Average IC across all timestamps
    ic_ir: float  # IC Information Ratio = mean(IC) / std(IC)
    forward_return_half_life_days: float | None  # Days of predictive power
    autocorrelation_1d: float  # 1-day autocorrelation of factor values
    autocorrelation_5d: float  # 5-day autocorrelation
    rank_stability_5d: float  # Rank correlation between t and t+5
    rank_stability_20d: float  # Rank correlation between t and t+20
    is_stale: bool  # True if signal is likely stale (low IC, high autocorr)


def compute_ic_series(
    factor_panel: pd.DataFrame,
    factor_col: str,
    forward_return_col: str = "fwd_return_1m",
) -> pd.Series:
    """Compute cross-sectional IC (Spearman rank correlation) per timestamp.

    Args:
        factor_panel: Panel with timestamp, symbol, factor_col, forward_return_col.
        factor_col: Column name of the factor.
        forward_return_col: Column name of forward returns.

    Returns:
        Series indexed by timestamp with IC values.
    """
    if factor_col not in factor_panel.columns or forward_return_col not in factor_panel.columns:
        return pd.Series(dtype=float)

    ic = (
        factor_panel.groupby("timestamp")
        .apply(
            lambda g: g[factor_col].corr(g[forward_return_col], method="spearman"),
            include_groups=False,
        )
        .dropna()
    )
    return ic


def compute_ic_half_life(ic_series: pd.Series) -> float | None:
    """Estimate IC half-life via exponential decay fit.

    Fits IC(lag) = IC(0) * exp(-lambda * lag) and returns
    half_life = ln(2) / lambda.

    Args:
        ic_series: IC values indexed by timestamp.

    Returns:
        Half-life in number of periods (days), or None if fit fails.
    """
    if len(ic_series) < 10:
        return None

    # Compute autocorrelation of IC series at various lags
    max_lag = min(60, len(ic_series) // 3)
    if max_lag < 5:
        return None

    lags = np.arange(1, max_lag + 1)
    autocorrs = []
    for lag in lags:
        ac = ic_series.autocorr(lag=lag)
        if ac is not None and not np.isnan(ac):
            autocorrs.append(ac)
        else:
            autocorrs.append(0.0)

    autocorrs = np.array(autocorrs)

    # Fit exponential decay: log(|autocorr|) = -lambda * lag
    # Only use positive autocorrelations for the fit
    valid = autocorrs > 0.01
    if valid.sum() < 3:
        return None

    log_ac = np.log(autocorrs[valid])
    lag_valid = lags[valid]

    try:
        # Simple linear regression: log_ac = a - lambda * lag
        coeffs = np.polyfit(lag_valid, log_ac, 1)
        decay_rate = -coeffs[0]  # lambda

        if decay_rate <= 0:
            return None

        half_life = np.log(2) / decay_rate
        return float(np.clip(half_life, 0.5, 500))
    except Exception as exc:
        # Previously returned None silently on any fit failure. Downstream
        # SignalDecayProfile then showed ic_half_life_days=None which is
        # indistinguishable from "insufficient data" — a genuinely
        # undecayed signal and a numpy fit bomb looked the same. Emit a
        # WARN so the distinction is observable in logs.
        import logging
        logging.getLogger(__name__).warning(
            "[SignalDecay] ic_half_life polyfit failed: %s — returning None",
            exc,
        )
        return None


def compute_forward_return_half_life(
    factor_panel: pd.DataFrame,
    factor_col: str,
    max_horizon: int = 60,
    horizons: list[int] | None = None,
) -> float | None:
    """Estimate how many days forward the signal predicts returns.

    Computes IC at multiple forward horizons and finds where IC drops to 50%.

    Args:
        factor_panel: Panel with timestamp, symbol, factor_col, close.
        factor_col: Factor column name.
        max_horizon: Maximum forward horizon in days.
        horizons: Specific horizons to test.

    Returns:
        Forward return half-life in days, or None.
    """
    if factor_col not in factor_panel.columns or "close" not in factor_panel.columns:
        return None

    if horizons is None:
        horizons = [1, 2, 3, 5, 10, 20, 40, 60]
        horizons = [h for h in horizons if h <= max_horizon]

    # Compute forward returns at each horizon
    df = factor_panel.sort_values(["symbol", "timestamp"]).copy()
    horizon_ics: dict[int, float] = {}

    for h in horizons:
        fwd_col = f"_fwd_{h}d"
        df[fwd_col] = df.groupby("symbol")["close"].transform(
            lambda s: s.shift(-h) / s - 1.0
        )
        ic = (
            df.groupby("timestamp")
            .apply(
                lambda g: g[factor_col].corr(g[fwd_col], method="spearman"),
                include_groups=False,
            )
            .dropna()
        )
        if len(ic) > 5:
            horizon_ics[h] = float(ic.abs().mean())

    if len(horizon_ics) < 3:
        return None

    # Find where IC drops to 50% of peak
    peak_ic = max(horizon_ics.values())
    if peak_ic < 0.01:
        return None

    threshold = peak_ic * 0.5
    sorted_horizons = sorted(horizon_ics.keys())

    for i, h in enumerate(sorted_horizons):
        if horizon_ics[h] < threshold:
            # Interpolate between previous and current horizon
            if i > 0:
                h_prev = sorted_horizons[i - 1]
                ic_prev = horizon_ics[h_prev]
                ic_curr = horizon_ics[h]
                frac = (ic_prev - threshold) / max(ic_prev - ic_curr, 1e-10)
                return float(h_prev + frac * (h - h_prev))
            return float(h)

    # IC never drops below 50% — return max horizon
    return float(sorted_horizons[-1])


def compute_signal_autocorrelation(
    factor_panel: pd.DataFrame,
    factor_col: str,
    lags: list[int] | None = None,
) -> dict[int, float]:
    """Compute autocorrelation of factor values at various lags.

    High autocorrelation = persistent signal (slow turnover needed).
    Low autocorrelation = fast-decaying signal (frequent rebalancing needed).

    Args:
        factor_panel: Panel with timestamp, symbol, factor_col.
        factor_col: Factor column name.
        lags: Lags to compute (default: [1, 5, 10, 20]).

    Returns:
        Dict mapping lag -> autocorrelation.
    """
    if factor_col not in factor_panel.columns:
        return {}

    if lags is None:
        lags = [1, 5, 10, 20]

    # Compute per-symbol autocorrelation, then average
    result: dict[int, float] = {}
    for lag in lags:
        per_sym = (
            factor_panel.sort_values(["symbol", "timestamp"])
            .groupby("symbol")[factor_col]
            .apply(lambda s: s.autocorr(lag=lag) if len(s) > lag + 5 else np.nan)
            .dropna()
        )
        result[lag] = float(per_sym.mean()) if len(per_sym) > 0 else 0.0

    return result


def compute_rank_stability(
    factor_panel: pd.DataFrame,
    factor_col: str,
    top_n: int = 20,
    lag_days: int = 5,
) -> float:
    """Compute stability of top-N ranked symbols over time.

    Measures what fraction of top-N symbols at time t are still in top-N at t+lag.

    Args:
        factor_panel: Panel with timestamp, symbol, factor_col.
        factor_col: Factor column name.
        top_n: Number of top symbols to track.
        lag_days: Number of days to measure stability.

    Returns:
        Average overlap fraction (0-1). Higher = more stable rankings.
    """
    if factor_col not in factor_panel.columns or "timestamp" not in factor_panel.columns:
        return 0.0

    timestamps = sorted(factor_panel["timestamp"].unique())
    if len(timestamps) < lag_days + 1:
        return 0.0

    overlaps = []
    for i, ts in enumerate(timestamps[:-lag_days]):
        ts_future = timestamps[min(i + lag_days, len(timestamps) - 1)]

        df_now = factor_panel[factor_panel["timestamp"] == ts]
        df_later = factor_panel[factor_panel["timestamp"] == ts_future]

        if len(df_now) < top_n or len(df_later) < top_n:
            continue

        top_now = set(df_now.nlargest(top_n, factor_col)["symbol"])
        top_later = set(df_later.nlargest(top_n, factor_col)["symbol"])

        overlap = len(top_now & top_later) / top_n
        overlaps.append(overlap)

    return float(np.mean(overlaps)) if overlaps else 0.0


def analyze_signal_decay(
    factor_panel: pd.DataFrame,
    factor_col: str,
    forward_return_col: str = "fwd_return_1m",
    stale_threshold_ic: float = 0.02,
    stale_threshold_autocorr: float = 0.95,
) -> SignalDecayProfile:
    """Run full signal decay analysis for a single factor.

    Args:
        factor_panel: Panel with timestamp, symbol, factor_col, forward_return_col, close.
        factor_col: Factor column name.
        forward_return_col: Forward return column.
        stale_threshold_ic: IC below this = stale.
        stale_threshold_autocorr: Autocorrelation above this = stale (not updating).

    Returns:
        SignalDecayProfile with all decay metrics.
    """
    ic_series = compute_ic_series(factor_panel, factor_col, forward_return_col)
    ic_mean = float(ic_series.abs().mean()) if len(ic_series) > 0 else 0.0
    ic_std = float(ic_series.std()) if len(ic_series) > 1 else 1.0
    ic_ir = ic_mean / max(ic_std, 1e-10)

    ic_hl = compute_ic_half_life(ic_series)
    fwd_hl = compute_forward_return_half_life(factor_panel, factor_col)

    autocorrs = compute_signal_autocorrelation(factor_panel, factor_col, [1, 5])
    ac_1d = autocorrs.get(1, 0.0)
    ac_5d = autocorrs.get(5, 0.0)

    rank_5d = compute_rank_stability(factor_panel, factor_col, top_n=20, lag_days=5)
    rank_20d = compute_rank_stability(factor_panel, factor_col, top_n=20, lag_days=20)

    is_stale = (ic_mean < stale_threshold_ic) or (ac_1d > stale_threshold_autocorr)

    profile = SignalDecayProfile(
        factor_name=factor_col,
        ic_half_life_days=ic_hl,
        ic_mean=round(ic_mean, 4),
        ic_ir=round(ic_ir, 4),
        forward_return_half_life_days=fwd_hl,
        autocorrelation_1d=round(ac_1d, 4),
        autocorrelation_5d=round(ac_5d, 4),
        rank_stability_5d=round(rank_5d, 4),
        rank_stability_20d=round(rank_20d, 4),
        is_stale=is_stale,
    )

    _log.info(
        "Signal decay [%s]: IC=%.3f, IC_IR=%.2f, IC_HL=%.1f days, "
        "FWD_HL=%.1f days, AC1d=%.2f, rank_stab_5d=%.2f, stale=%s",
        factor_col, ic_mean, ic_ir,
        ic_hl or -1, fwd_hl or -1,
        ac_1d, rank_5d, is_stale,
    )
    return profile


def analyze_all_signals(
    factor_panel: pd.DataFrame,
    factor_cols: list[str],
    forward_return_col: str = "fwd_return_1m",
) -> list[SignalDecayProfile]:
    """Analyze decay for all factors in a panel.

    Args:
        factor_panel: Full factor panel.
        factor_cols: List of factor column names.
        forward_return_col: Forward return column.

    Returns:
        List of SignalDecayProfile, sorted by IC descending.
    """
    profiles = []
    for col in factor_cols:
        if col in factor_panel.columns:
            profile = analyze_signal_decay(factor_panel, col, forward_return_col)
            profiles.append(profile)

    # Sort by IC descending
    profiles.sort(key=lambda p: p.ic_mean, reverse=True)
    return profiles


__all__ = [
    "SignalDecayProfile",
    "compute_ic_series",
    "compute_ic_half_life",
    "compute_forward_return_half_life",
    "compute_signal_autocorrelation",
    "compute_rank_stability",
    "analyze_signal_decay",
    "analyze_all_signals",
]
