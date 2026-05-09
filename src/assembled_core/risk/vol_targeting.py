"""Portfolio-level volatility targeting: scale exposure to hit target annualized vol.

Computes realized vol from an equity curve or return series, then derives a
multiplier that brings expected portfolio vol toward a target level.

Combines multiplicatively with other overlays (GeoRisk, profit_lock):
    final_multiplier = geo * profit_lock * vol_scale_factor

M6-T01/T02: define target vol and realized vol calc, implement exposure scaling.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def compute_realized_vol(
    returns: pd.Series,
    lookback_days: int = 20,
    annualize_factor: float = 252.0,
    min_observations: int = 5,
) -> float:
    """Compute realized annualized volatility from a returns series.

    Args:
        returns: Period returns (e.g. daily pct_change). NaN values are dropped.
        lookback_days: Number of recent observations to use.
        annualize_factor: Multiplier to annualize (252 for daily, 52 for weekly).
        min_observations: Minimum non-NaN observations required; returns NaN if fewer.

    Returns:
        Annualized realized volatility (float). Returns float('nan') if insufficient
        data.
    """
    if returns is None or not isinstance(returns, pd.Series) or returns.empty:
        return float("nan")

    tail = returns.dropna().tail(lookback_days)
    if len(tail) < min_observations:
        return float("nan")

    std = float(tail.std(ddof=1))
    return std * (annualize_factor**0.5)


def compute_vol_scale_factor(
    realized_vol: float,
    target_vol: float,
    min_scale: float = 0.0,
    max_scale: float = 1.5,
) -> float:
    """Compute exposure scaling factor to hit target_vol.

    scale_factor = target_vol / realized_vol, clamped to [min_scale, max_scale].

    Args:
        realized_vol: Current realized annualized vol.
        target_vol: Target annualized vol.
        min_scale: Minimum allowed scale (default 0.0 = fully unwind).
        max_scale: Maximum allowed scale (default 1.5 = allow mild leverage-like scaling).

    Returns:
        Scale factor in [min_scale, max_scale]. Returns 1.0 if inputs are invalid
        (nan, zero, or negative).
    """
    if realized_vol != realized_vol:  # NaN check (nan != nan is True)
        return 1.0
    if target_vol is None or target_vol <= 0.0:
        return 1.0
    if realized_vol is None or realized_vol <= 0.0:
        return 1.0

    raw = target_vol / realized_vol
    return float(max(min_scale, min(max_scale, raw)))


def apply_vol_targeting_to_weights(
    target_weights: dict[str, float],
    scale_factor: float,
) -> dict[str, float]:
    """Scale a symbol→weight dict by the vol-targeting scale factor.

    Args:
        target_weights: symbol -> weight mapping.
        scale_factor: Multiplier from compute_vol_scale_factor.

    Returns:
        New dict with every weight scaled by scale_factor.
    """
    if not target_weights:
        return {}
    return {sym: w * scale_factor for sym, w in target_weights.items()}


def compute_vol_targeting_result(
    equity_curve: pd.Series,
    policy: dict[str, Any],
    now_idx: int = -1,
) -> tuple[float, float, float]:
    """Main entry: derive scale factor, realized vol, and target vol from policy.

    Args:
        equity_curve: Series of equity *values* (not returns). Converted internally.
        policy: dict with vol_targeting section:
            - enabled (bool)
            - target_vol_annual (float, e.g. 0.20)
            - lookback_days (int, default 20)
            - min_scale (float, default 0.0)
            - max_scale (float, default 1.5)
            - annualize_factor (float, default 252.0)
            - min_observations (int, default 5)
        now_idx: Index into equity_curve to treat as "now". -1 = last element.

    Returns:
        (scale_factor, realized_vol, target_vol).
        scale_factor = 1.0 if disabled or insufficient data.
        realized_vol / target_vol = float('nan') if disabled or data missing.
    """
    vt = (policy or {}).get("vol_targeting") or {}
    if not vt.get("enabled", False):
        return 1.0, float("nan"), float("nan")

    target_vol = float(vt.get("target_vol_annual", 0.20) or 0.20)
    lookback_days = int(vt.get("lookback_days", 20) or 20)
    min_scale = float(vt.get("min_scale", 0.0) or 0.0)
    leverage_allowed = bool(
        (policy or {}).get("scope", {}).get("leverage_allowed", True)
    )
    default_max = 1.5 if leverage_allowed else 1.0
    max_scale = float(vt.get("max_scale", default_max) or default_max)
    if not leverage_allowed and max_scale > 1.0:
        max_scale = 1.0
    annualize_factor = float(vt.get("annualize_factor", 252.0) or 252.0)
    min_observations = int(vt.get("min_observations", 5) or 5)

    if (
        equity_curve is None
        or not isinstance(equity_curve, pd.Series)
        or equity_curve.empty
    ):
        return 1.0, float("nan"), target_vol

    # Slice to now_idx (allow negative indexing like -1 for last)
    if now_idx == -1 or now_idx >= len(equity_curve):
        curve = equity_curve
    elif now_idx >= 0:
        curve = equity_curve.iloc[: now_idx + 1]
    else:
        curve = equity_curve

    if len(curve) < 2:
        return 1.0, float("nan"), target_vol

    returns = curve.pct_change(fill_method=None)
    realized_vol = compute_realized_vol(
        returns,
        lookback_days=lookback_days,
        annualize_factor=annualize_factor,
        min_observations=min_observations,
    )

    scale_factor = compute_vol_scale_factor(
        realized_vol,
        target_vol,
        min_scale=min_scale,
        max_scale=max_scale,
    )
    return scale_factor, realized_vol, target_vol


__all__ = [
    "compute_realized_vol",
    "compute_vol_scale_factor",
    "apply_vol_targeting_to_weights",
    "compute_vol_targeting_result",
]
