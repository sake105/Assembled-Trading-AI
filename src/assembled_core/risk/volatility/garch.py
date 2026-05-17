"""GARCH(1,1) volatility model — thin wrapper around `arch` library.

PIT-safe: only uses returns up to (but not including) the forecast timestamp.
No look-ahead. Forecasts are point-estimates of next-period conditional
volatility, NOT a calibration to future data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

try:
    from arch import arch_model
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'arch' package is required for GARCH volatility modelling. "
        "Install it with: pip install arch==8.0.0"
    ) from exc

# Trading days per year — used for annualisation.
_ANNUALISE = math.sqrt(252)


@dataclass
class GarchForecast:
    """Result of a GARCH forecast call."""

    next_period_volatility: float
    """Annualised volatility forecast for the next period (e.g. 0.18 → 18 % p.a.)."""

    conditional_volatility: pd.Series
    """In-sample fitted conditional volatility (same index as the input returns)."""

    params: dict[str, float]
    """Fit parameters: keys include 'omega', 'alpha[1]', 'beta[1]' (and 'mu' if
    mean='constant')."""

    convergence: bool
    """True when the underlying optimiser reported a successful convergence."""

    log_likelihood: float
    """Log-likelihood of the fitted model."""


def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    mean: Literal["constant", "zero"] = "constant",
    rescale: bool = True,
) -> GarchForecast:
    """Fit a GARCH(p, q) model to *returns* and forecast next-period volatility.

    Args:
        returns: Series of period returns (NOT log-prices).  Index must be a
            DatetimeIndex for PIT safety; UTC-aware preferred.
        p: GARCH lag order (default 1 → GARCH(1,1)).
        q: ARCH lag order (default 1).
        mean: Mean model — ``'constant'`` (fits a mean term μ) or ``'zero'``
            (constrains μ = 0).
        rescale: If ``True``, rescales returns to ~1 % scale internally for
            numerical stability; output is rescaled back automatically.

    Returns:
        :class:`GarchForecast` with annualised next-period vol forecast,
        in-sample conditional vol series (same index as *returns*), fit params,
        and convergence flag.

    Raises:
        ValueError: If *returns* is empty, has fewer than 50 observations, or
            contains NaN / ±inf values.
    """
    # ------------------------------------------------------------------ guards
    if len(returns) == 0:
        raise ValueError("returns is empty — need at least 50 observations.")
    if len(returns) < 50:
        raise ValueError(
            f"returns has only {len(returns)} observations; at least 50 required "
            "for a reliable GARCH fit."
        )
    if returns.isna().any():
        raise ValueError(
            "returns contains NaN values — drop or fill them before fitting."
        )
    if not np.isfinite(returns.values).all():
        raise ValueError(
            "returns contains ±inf values — clean the series before fitting."
        )

    # ------------------------------------------------------------------ fit
    # arch expects returns expressed in percent-like scale (e.g. 0.01 → 1 %)
    # when rescale=True the library handles the internal scaling itself.
    model = arch_model(
        returns,
        vol="GARCH",
        p=p,
        q=q,
        mean=mean,
        rescale=rescale,
    )

    result = model.fit(disp="off", show_warning=False)

    # ------------------------------------------------------------------ convergence
    # ARCHModelResult wraps scipy OptimizeResult — access via .optimization_result
    opt_res = getattr(result, "optimization_result", None)
    if opt_res is not None:
        convergence: bool = bool(getattr(opt_res, "success", False))
    else:
        # Fallback: treat as converged if loglikelihood is finite
        convergence = math.isfinite(result.loglikelihood)

    # ------------------------------------------------------------------ params
    params_series: pd.Series = result.params
    params: dict[str, float] = {str(k): float(v) for k, v in params_series.items()}

    # ------------------------------------------------------------------ conditional vol
    # When rescale=True, arch scales returns by model.scale internally.
    # result.conditional_volatility is expressed in *scaled* units, so we
    # divide back to restore original-return units.
    _scale: float = float(getattr(model, "scale", 1.0))
    raw_cond_vol = result.conditional_volatility.copy()
    cond_vol: pd.Series = raw_cond_vol / _scale
    cond_vol.index = returns.index  # ensure index alignment

    # ------------------------------------------------------------------ forecast
    # horizon=1 → one-step-ahead forecast; .variance gives variance values in
    # *scaled* units² (i.e. scaled-return variance).
    forecast = result.forecast(horizon=1, reindex=False)
    # forecast.variance is a DataFrame; last row, first column is h.1
    next_var_scaled = float(forecast.variance.iloc[-1, 0])
    # Convert back to original-return units: vol = sqrt(var_scaled) / scale
    # Then annualise by sqrt(252).
    next_period_vol = (math.sqrt(next_var_scaled) / _scale) * _ANNUALISE

    return GarchForecast(
        next_period_volatility=next_period_vol,
        conditional_volatility=cond_vol,
        params=params,
        convergence=convergence,
        log_likelihood=float(result.loglikelihood),
    )
