"""EWMA-based vol-targeting on main (audit C2-066).

The existing :mod:`risk.vol_targeting` uses a *backward-looking* rolling
standard-deviation estimate. The audit notes this is biased high after
shocks (slow to come down) and biased low going into shocks (slow to
ramp up). The classical fix is **Exponentially Weighted Moving Average
(EWMA)** volatility (JP Morgan RiskMetrics 1996, eq. 5.6):

.. math::

    \\sigma_t^2 = \\lambda \\sigma_{t-1}^2 + (1 - \\lambda) r_t^2

with :math:`\\lambda \\approx 0.94` for daily data. EWMA's one-step-
ahead forecast is itself :math:`\\sigma_t` — exactly what
vol-targeting should be scaling against.

This helper does **not** displace :mod:`risk.vol_targeting`. It is a
forward-looking variant. Callers opt in via
``policy.vol_targeting.method: "ewma"``. Where ``method == "realized"``
or absent, the existing simple-realized-vol path is used unchanged.

We keep dependency footprint at numpy/pandas — no arch.GARCH. ERWEITERUNG
has a fuller GARCH path; the audit's cherry-pick gate (§8.8) blocks
bringing it to main until the OOS re-run validates it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_LAMBDA = 0.94
"""RiskMetrics-default decay for daily data (Boudt et al. 2008 reaffirm)."""


@dataclass(frozen=True)
class EWMAVolEstimate:
    """One EWMA-vol estimate plus its history-array for diagnostics."""

    forecast_vol_annual: float
    forecast_vol_period: float
    last_observation_count: int
    lambda_used: float


def ewma_vol_forecast(
    returns: pd.Series,
    *,
    lambda_: float = DEFAULT_LAMBDA,
    annualize_factor: float = 252.0,
    min_observations: int = 30,
) -> EWMAVolEstimate:
    """Compute the one-step-ahead EWMA volatility forecast.

    Args:
        returns: per-period returns (e.g. daily pct_change). NaN values
            are dropped. **Must be in time-ascending order.**
        lambda_: smoothing constant; 0.94 = RiskMetrics-daily default.
            Larger values give heavier weight to past observations.
        annualize_factor: 252 for daily, 52 for weekly, etc.
        min_observations: refuse to forecast if fewer non-NaN samples.

    Returns:
        :class:`EWMAVolEstimate` with the annualized forecast volatility,
        the un-annualized per-period forecast, and diagnostic counts.
        ``forecast_vol_*`` are ``nan`` when too few observations.

    Raises:
        ValueError: if ``lambda_`` outside (0, 1).
    """
    if not (0.0 < lambda_ < 1.0):
        raise ValueError(f"lambda_ must be in (0, 1), got {lambda_}")
    if returns is None or not isinstance(returns, pd.Series):
        return EWMAVolEstimate(float("nan"), float("nan"), 0, lambda_)
    clean = returns.dropna().to_numpy(dtype=float)
    n = clean.size
    if n < min_observations:
        return EWMAVolEstimate(float("nan"), float("nan"), n, lambda_)

    # Standard recursive EWMA — initialize with the in-sample variance.
    var_t = float(np.var(clean, ddof=1))
    for r in clean:
        var_t = lambda_ * var_t + (1.0 - lambda_) * r * r
    sigma_t = float(np.sqrt(max(var_t, 0.0)))
    return EWMAVolEstimate(
        forecast_vol_annual=sigma_t * np.sqrt(annualize_factor),
        forecast_vol_period=sigma_t,
        last_observation_count=n,
        lambda_used=lambda_,
    )


def compute_ewma_scale_factor(
    forecast_vol_annual: float,
    target_vol_annual: float,
    *,
    min_scale: float = 0.0,
    max_scale: float = 1.5,
) -> float:
    """Same arithmetic as :func:`compute_vol_scale_factor` but forecast-driven."""
    if not np.isfinite(forecast_vol_annual) or forecast_vol_annual <= 0.0:
        return 1.0
    if target_vol_annual is None or target_vol_annual <= 0.0:
        return 1.0
    raw = target_vol_annual / forecast_vol_annual
    return float(max(min_scale, min(max_scale, raw)))


__all__ = [
    "DEFAULT_LAMBDA",
    "EWMAVolEstimate",
    "ewma_vol_forecast",
    "compute_ewma_scale_factor",
]
