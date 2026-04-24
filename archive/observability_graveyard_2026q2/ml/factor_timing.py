"""Factor Timing — Dynamic Factor Weight Adjustment (M25).

Adjusts factor weights dynamically based on:
  1. Factor momentum: Overweight factors with strong recent performance
  2. Factor crowding: Underweight factors that are overcrowded
  3. Factor mean-reversion: Fade factors after extreme performance
  4. Regime conditioning: Blend timing signals with regime-based weights

The goal is to exploit time-varying factor premia — factor performance
is not constant, and timing can add 50-200 bps annually.

Reference:
    Arnott, R., Beck, N., Kalesnik, V. (2016).
    "Timing 'Smart Beta' Strategies? Of Course! Buy Low, Sell High!"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FactorTimingConfig:
    """Configuration for factor timing signals.

    Attributes:
        momentum_lookback: Lookback window for factor momentum (periods).
        momentum_weight: Weight of momentum signal in timing blend.
        mean_reversion_lookback: Lookback for factor mean reversion.
        mean_reversion_weight: Weight of mean reversion signal.
        crowding_weight: Weight of crowding signal.
        max_tilt_pct: Maximum percentage tilt from base weight (e.g., 0.50 = 50%).
        smoothing_halflife: Exponential smoothing half-life for signals.
    """

    momentum_lookback: int = 12
    momentum_weight: float = 0.4
    mean_reversion_lookback: int = 60
    mean_reversion_weight: float = 0.3
    crowding_weight: float = 0.3
    max_tilt_pct: float = 0.50
    smoothing_halflife: int = 3


@dataclass
class FactorTimingResult:
    """Result of factor timing analysis.

    Attributes:
        adjusted_weights: Final factor weights after timing adjustment.
        base_weights: Original weights before adjustment.
        momentum_scores: Factor momentum z-scores.
        crowding_scores: Factor crowding z-scores (higher = more crowded).
        mean_reversion_scores: Factor mean reversion signals.
        tilt_applied: Percentage tilt applied to each factor.
    """

    adjusted_weights: dict[str, float]
    base_weights: dict[str, float]
    momentum_scores: dict[str, float]
    crowding_scores: dict[str, float]
    mean_reversion_scores: dict[str, float]
    tilt_applied: dict[str, float]


def compute_factor_momentum(
    factor_returns: pd.DataFrame,
    lookback: int = 12,
) -> dict[str, float]:
    """Compute factor momentum: recent cumulative return z-scored.

    Factors with strong recent performance get positive momentum scores.

    Args:
        factor_returns: DataFrame with factor names as columns, periods as rows.
            Values are factor returns per period.
        lookback: Number of recent periods to use.

    Returns:
        Dict of factor_name -> momentum z-score.
    """
    if factor_returns.empty or len(factor_returns) < 3:
        return {col: 0.0 for col in factor_returns.columns}

    recent = factor_returns.tail(lookback)
    cum_returns = recent.sum()

    mean = cum_returns.mean()
    std = cum_returns.std()
    if std < 1e-10:
        return {col: 0.0 for col in factor_returns.columns}

    z_scores = (cum_returns - mean) / std
    return {col: round(float(z_scores[col]), 4) for col in factor_returns.columns}


def compute_factor_crowding(
    factor_exposures: pd.DataFrame,
    lookback: int = 20,
) -> dict[str, float]:
    """Estimate factor crowding from exposure concentration.

    A factor is "crowded" when many assets have extreme exposure to it,
    measured by the cross-sectional kurtosis of exposures.

    Args:
        factor_exposures: DataFrame with factors as columns, assets as rows.
            Values are factor exposures (z-scored).
        lookback: Not used directly, included for API consistency.

    Returns:
        Dict of factor_name -> crowding score (higher = more crowded).
    """
    if factor_exposures.empty:
        return {}

    crowding = {}
    for col in factor_exposures.columns:
        vals = factor_exposures[col].dropna()
        if len(vals) < 5:
            crowding[col] = 0.0
            continue

        # Kurtosis as crowding proxy (high kurtosis = concentrated bets)
        mean = vals.mean()
        std = vals.std()
        if std < 1e-10:
            crowding[col] = 0.0
            continue

        z = (vals - mean) / std
        kurtosis = float((z ** 4).mean() - 3.0)  # excess kurtosis
        crowding[col] = round(max(kurtosis, 0.0), 4)

    # Z-score the crowding scores
    vals = np.array(list(crowding.values()))
    if len(vals) > 1 and vals.std() > 1e-10:
        z = (vals - vals.mean()) / vals.std()
        for i, key in enumerate(crowding):
            crowding[key] = round(float(z[i]), 4)

    return crowding


def compute_factor_mean_reversion(
    factor_returns: pd.DataFrame,
    short_lookback: int = 12,
    long_lookback: int = 60,
) -> dict[str, float]:
    """Compute factor mean-reversion signal.

    Factors that outperformed over the long term but underperformed recently
    may be poised for a rebound (and vice versa). Signal =
    long_term_z - short_term_z.

    Args:
        factor_returns: Factor returns DataFrame.
        short_lookback: Recent window.
        long_lookback: Long-term window.

    Returns:
        Dict of factor_name -> mean reversion score (positive = expect rebound).
    """
    if factor_returns.empty or len(factor_returns) < long_lookback:
        return {col: 0.0 for col in factor_returns.columns}

    short_cum = factor_returns.tail(short_lookback).sum()
    long_cum = factor_returns.tail(long_lookback).sum()

    short_mean = short_cum.mean()
    short_std = short_cum.std()
    long_mean = long_cum.mean()
    long_std = long_cum.std()

    result = {}
    for col in factor_returns.columns:
        short_z = (short_cum[col] - short_mean) / short_std if short_std > 1e-10 else 0.0
        long_z = (long_cum[col] - long_mean) / long_std if long_std > 1e-10 else 0.0
        # Mean reversion: long-term winners that recently underperformed
        result[col] = round(float(long_z - short_z), 4)

    return result


def adjust_factor_weights(
    base_weights: dict[str, float],
    factor_returns: pd.DataFrame | None = None,
    factor_exposures: pd.DataFrame | None = None,
    config: FactorTimingConfig | None = None,
) -> FactorTimingResult:
    """Adjust factor weights using timing signals.

    Combines momentum, crowding, and mean-reversion signals to tilt
    base weights. The adjustment is bounded by max_tilt_pct.

    Args:
        base_weights: Original factor weights (should sum to ~1.0).
        factor_returns: Historical factor returns (periods x factors).
        factor_exposures: Current cross-sectional factor exposures (assets x factors).
        config: Timing configuration.

    Returns:
        FactorTimingResult with adjusted weights and diagnostics.
    """
    cfg = config or FactorTimingConfig()
    factors = list(base_weights.keys())

    # Compute signals
    mom_scores: dict[str, float] = {}
    crowd_scores: dict[str, float] = {}
    mr_scores: dict[str, float] = {}

    if factor_returns is not None and not factor_returns.empty:
        available = [f for f in factors if f in factor_returns.columns]
        if available:
            mom_scores = compute_factor_momentum(
                factor_returns[available], cfg.momentum_lookback,
            )
            mr_scores = compute_factor_mean_reversion(
                factor_returns[available],
                cfg.momentum_lookback,
                cfg.mean_reversion_lookback,
            )

    if factor_exposures is not None and not factor_exposures.empty:
        available = [f for f in factors if f in factor_exposures.columns]
        if available:
            crowd_scores = compute_factor_crowding(factor_exposures[available])

    # Blend signals into a single tilt per factor
    tilt = {}
    for f in factors:
        mom = mom_scores.get(f, 0.0) * cfg.momentum_weight
        crowd = -crowd_scores.get(f, 0.0) * cfg.crowding_weight  # negative: underweight crowded
        mr = mr_scores.get(f, 0.0) * cfg.mean_reversion_weight
        raw_tilt = mom + crowd + mr

        # Clip to max tilt
        clipped_tilt = np.clip(raw_tilt, -cfg.max_tilt_pct, cfg.max_tilt_pct)
        tilt[f] = round(float(clipped_tilt), 4)

    # Apply tilt to base weights
    adjusted = {}
    for f in factors:
        base = base_weights[f]
        adjusted[f] = max(0.0, base * (1.0 + tilt[f]))

    # Renormalize to sum to 1.0
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {f: round(w / total, 6) for f, w in adjusted.items()}
    else:
        adjusted = dict(base_weights)

    logger.info(
        "[FactorTiming] Adjusted %d factors, max tilt=%.1f%%, "
        "top tilt: %s, bottom tilt: %s",
        len(factors), cfg.max_tilt_pct * 100,
        max(tilt.items(), key=lambda x: x[1]) if tilt else ("N/A", 0),
        min(tilt.items(), key=lambda x: x[1]) if tilt else ("N/A", 0),
    )

    return FactorTimingResult(
        adjusted_weights=adjusted,
        base_weights=dict(base_weights),
        momentum_scores={f: mom_scores.get(f, 0.0) for f in factors},
        crowding_scores={f: crowd_scores.get(f, 0.0) for f in factors},
        mean_reversion_scores={f: mr_scores.get(f, 0.0) for f in factors},
        tilt_applied=tilt,
    )


def compute_value_spread(
    factor_returns: pd.DataFrame,
    factor_valuations: dict[str, pd.Series] | None = None,
    lookback: int = 252,
) -> dict[str, float]:
    """Compute value spread for factor timing (M37 Task 37.2).

    Value spread = long-leg valuation - short-leg valuation.
    Wide spread → factor is cheap → overweight.

    Args:
        factor_returns: Factor return streams.
        factor_valuations: {factor: valuation_ratio} (e.g., B/M spread).
            If None, uses cumulative return deviation as proxy.
        lookback: Window for z-scoring.

    Returns:
        {factor: value_spread_z} dictionary.
    """
    spreads = {}
    for factor in factor_returns.columns:
        if factor_valuations and factor in factor_valuations:
            val = factor_valuations[factor]
        else:
            # Proxy: cumulative return deviation from trend
            cum_ret = factor_returns[factor].cumsum()
            trend = cum_ret.rolling(lookback, min_periods=60).mean()
            val = cum_ret - trend

        if len(val.dropna()) < 60:
            spreads[factor] = 0.0
            continue

        # Z-score of current value spread
        current = val.iloc[-1]
        mean = val.rolling(lookback, min_periods=60).mean().iloc[-1]
        std = val.rolling(lookback, min_periods=60).std().iloc[-1]
        if std > 1e-8:
            z = float((current - mean) / std)
        else:
            z = 0.0

        spreads[factor] = round(np.clip(z, -3, 3), 4)

    logger.info("[FactorTiming] Value spreads: %s",
                {k: v for k, v in sorted(spreads.items(), key=lambda x: -abs(x[1]))[:3]})

    return spreads


def compute_macro_conditional_timing(
    factor_returns: pd.DataFrame,
    macro_indicators: dict[str, pd.Series],
    lookback: int = 252,
) -> dict[str, float]:
    """Macro-conditional factor timing (M37 Task 37.3).

    Adjusts factor weights based on macroeconomic regime:
    - Expansion: overweight momentum, growth
    - Contraction: overweight value, quality, low-vol
    - Rising rates: underweight duration-sensitive factors
    - Risk-off: overweight defensive factors

    Args:
        factor_returns: Factor return streams.
        macro_indicators: {name: series} e.g., {"pmi": ..., "yield_curve": ..., "vix": ...}.
        lookback: Window for regime classification.

    Returns:
        {factor: macro_tilt} dictionary.
    """
    # Default macro regime sensitivities
    # Positive = benefits from expansion/risk-on
    MACRO_SENSITIVITY = {
        "momentum": 0.5,
        "value": -0.3,
        "size": 0.2,
        "quality": -0.4,
        "low_vol": -0.6,
        "growth": 0.4,
        "profitability": -0.2,
        "investment": -0.1,
    }

    # Compute composite macro signal
    macro_z_scores = {}
    for name, series in macro_indicators.items():
        if len(series.dropna()) < 60:
            continue
        current = series.iloc[-1]
        mean = series.rolling(lookback, min_periods=60).mean().iloc[-1]
        std = series.rolling(lookback, min_periods=60).std().iloc[-1]
        if std > 1e-8:
            macro_z_scores[name] = float((current - mean) / std)

    if not macro_z_scores:
        return {f: 0.0 for f in factor_returns.columns}

    # Composite: average of available macro z-scores
    # PMI, ISM positive → expansion; VIX positive → contraction (invert)
    composite = 0.0
    n = 0
    for name, z in macro_z_scores.items():
        if "vix" in name.lower() or "volatility" in name.lower():
            composite -= z  # Invert fear gauges
        else:
            composite += z
        n += 1

    composite = composite / max(n, 1)
    composite = np.clip(composite, -2, 2)

    # Apply sensitivities
    tilts = {}
    for factor in factor_returns.columns:
        factor_lower = factor.lower().replace(" ", "_")
        sensitivity = MACRO_SENSITIVITY.get(factor_lower, 0.0)
        tilts[factor] = round(float(composite * sensitivity * 0.5), 4)

    logger.info("[FactorTiming] Macro composite=%.2f, tilts: %s", composite,
                {k: v for k, v in sorted(tilts.items(), key=lambda x: -abs(x[1]))[:3]})

    return tilts


__all__ = [
    "FactorTimingConfig",
    "FactorTimingResult",
    "compute_factor_momentum",
    "compute_factor_crowding",
    "compute_factor_mean_reversion",
    "compute_value_spread",
    "compute_macro_conditional_timing",
    "adjust_factor_weights",
]
