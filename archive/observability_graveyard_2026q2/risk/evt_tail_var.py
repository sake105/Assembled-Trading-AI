"""EVT Peaks-Over-Threshold tail VaR (C9 — diagnostic sidecar).

This module provides a small, self-contained implementation of classic
Extreme Value Theory tail risk estimators, used as a *diagnostic* overlay
next to :mod:`assembled_core.risk.var_methods`.

The Pickands-Balkema-de Haan theorem states that exceedances over a
sufficiently high threshold ``u`` of a loss random variable ``L`` are
asymptotically distributed as a Generalized Pareto Distribution (GPD)
with shape ``xi`` and scale ``beta``. Given such a fit one can
extrapolate the tail further than the empirical sample alone allows,
which is useful at confidence levels like 99% or 99.9% where the
empirical quantile is supported by only a handful of observations.

Conventions
-----------
All functions in this module operate on **losses** — positive numbers
represent a bad outcome. This matches
:class:`assembled_core.risk.var_methods.PortfolioVaR`. VaR and ES values
returned here are also positive loss magnitudes.

Note: the public API does **not** flip the sign of its input. Callers
that have raw return series must pass ``-returns`` (or an equivalent
loss transform) explicitly. Feeding raw returns directly is still
numerically valid — the functions treat whatever they receive as the
"loss" axis — but then the result describes the tail of that axis.

Implementation notes
--------------------
The GPD fit uses the method-of-moments estimator, which requires only
numpy. This module deliberately does **not** hard-depend on scipy so
that it can be imported everywhere in the codebase. A scipy-based MLE
implementation already exists in
:mod:`assembled_core.ml.evt_models`; this sidecar is intentionally
independent and covers the case where scipy is unavailable or where a
simpler, faster point estimate is preferred.

This module is strictly additive and must not be wired into the live
``risk_metrics`` path without a separate, reviewed change.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "evt_expected_shortfall",
    "evt_var",
    "fit_pot_gpd",
]


_XI_ZERO_TOL = 1e-8
_MIN_EXCEEDANCES = 20


def _as_losses(losses: np.ndarray) -> np.ndarray:
    """Coerce input to a finite 1-D float array without changing sign."""
    arr = np.asarray(losses, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    return arr


def fit_pot_gpd(
    losses: np.ndarray,
    threshold_pct: float = 0.90,
) -> dict:
    """Fit a Generalized Pareto Distribution to loss exceedances.

    Uses the Peaks-Over-Threshold approach with a method-of-moments
    estimator so that the function has no scipy dependency.

    Parameters
    ----------
    losses : np.ndarray
        1-D array of losses (positive = bad). Input is **not** sign-flipped.
    threshold_pct : float, default 0.90
        Quantile used to pick the GPD threshold ``u``. Must be in
        ``(0, 1)``.

    Returns
    -------
    dict
        ``{"threshold": u, "shape": xi, "scale": beta,
        "n_exceedances": int, "n_total": int}``.

    Raises
    ------
    ValueError
        If ``threshold_pct`` is not in ``(0, 1)``, if fewer than 20
        exceedances are available, or if the exceedance sample has
        non-positive variance (in which case method-of-moments is
        undefined).
    """
    if not (0.0 < threshold_pct < 1.0):
        raise ValueError(
            f"threshold_pct must be in (0, 1), got {threshold_pct}"
        )

    arr = _as_losses(losses)
    n_total = int(arr.size)
    if n_total == 0:
        raise ValueError("losses array is empty")

    u = float(np.quantile(arr, threshold_pct))
    exceed = arr[arr > u] - u
    n_exc = int(exceed.size)

    if n_exc < _MIN_EXCEEDANCES:
        raise ValueError(
            f"insufficient exceedances: {n_exc} < {_MIN_EXCEEDANCES}"
        )

    mean_y = float(np.mean(exceed))
    var_y = float(np.var(exceed, ddof=1))
    if var_y <= 0.0 or not np.isfinite(var_y):
        raise ValueError("insufficient exceedances: non-positive variance")

    ratio = (mean_y * mean_y) / var_y
    xi = 0.5 * (1.0 - ratio)
    beta = 0.5 * mean_y * (ratio + 1.0)

    if not np.isfinite(xi) or not np.isfinite(beta) or beta <= 0.0:
        raise ValueError(
            "GPD fit produced non-finite or non-positive parameters"
        )

    return {
        "threshold": u,
        "shape": float(xi),
        "scale": float(beta),
        "n_exceedances": n_exc,
        "n_total": n_total,
    }


def _validate_alpha(alpha: float, threshold_pct: float) -> None:
    if not (0.0 < threshold_pct < 1.0):
        raise ValueError(
            f"threshold_pct must be in (0, 1), got {threshold_pct}"
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if alpha <= threshold_pct:
        raise ValueError(
            "alpha must be strictly greater than threshold_pct "
            f"(alpha={alpha}, threshold_pct={threshold_pct}); EVT tail is "
            "only defined beyond the fitting threshold"
        )


def evt_var(
    losses: np.ndarray,
    alpha: float = 0.99,
    threshold_pct: float = 0.90,
) -> float:
    """POT-GPD Value-at-Risk at confidence level ``alpha``.

    Formula (see e.g. McNeil, Frey, Embrechts, *Quantitative Risk
    Management*):

        VaR_alpha = u + (beta / xi) * ((n/Nu * (1 - alpha))**(-xi) - 1)

    for ``xi != 0`` and

        VaR_alpha = u + beta * ln((n/Nu) / (1 - alpha))

    in the exponential-tail limit ``xi -> 0``.

    Parameters
    ----------
    losses : np.ndarray
        Positive-loss convention, not sign-flipped by this function.
    alpha : float, default 0.99
        Confidence level. Must be strictly greater than ``threshold_pct``.
    threshold_pct : float, default 0.90
        Threshold quantile used for the GPD fit.

    Returns
    -------
    float
        Positive loss magnitude at level ``alpha``.
    """
    _validate_alpha(alpha, threshold_pct)
    fit = fit_pot_gpd(losses, threshold_pct=threshold_pct)

    u = fit["threshold"]
    xi = fit["shape"]
    beta = fit["scale"]
    n_total = fit["n_total"]
    n_exc = fit["n_exceedances"]

    # n / Nu = 1 / p_exceed, so n/Nu * (1 - alpha) = (1 - alpha) / p_exceed.
    p_exceed = n_exc / n_total
    tail_ratio = (1.0 - alpha) / p_exceed  # = (n/Nu * (1-alpha))

    if abs(xi) < _XI_ZERO_TOL:
        var = u + beta * np.log(1.0 / tail_ratio)
    else:
        var = u + (beta / xi) * (tail_ratio ** (-xi) - 1.0)

    return float(var)


def evt_expected_shortfall(
    losses: np.ndarray,
    alpha: float = 0.99,
    threshold_pct: float = 0.90,
) -> float:
    """POT-GPD Expected Shortfall at confidence level ``alpha``.

    Closed-form for the GPD:

        ES_alpha = (VaR_alpha + beta - xi * u) / (1 - xi)

    valid for ``xi < 1``. For ``xi >= 1`` the GPD has infinite mean and
    the expected shortfall is undefined, which is reported as a
    ``ValueError``.
    """
    _validate_alpha(alpha, threshold_pct)
    fit = fit_pot_gpd(losses, threshold_pct=threshold_pct)

    u = fit["threshold"]
    xi = fit["shape"]
    beta = fit["scale"]

    if xi >= 1.0:
        raise ValueError(
            "infinite ES — shape parameter too heavy "
            f"(xi={xi:.4f} >= 1)"
        )

    var = evt_var(losses, alpha=alpha, threshold_pct=threshold_pct)
    es = (var + beta - xi * u) / (1.0 - xi)
    return float(es)
