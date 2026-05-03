"""Portfolio Value-at-Risk methods (C5a + C5b).

This module provides a small, testable ``PortfolioVaR`` class offering several
VaR methodologies used in institutional risk management.

Methods:
    - historical_var      : empirical percentile of the portfolio return series
    - parametric_var      : Normal (Gaussian) VaR using mean and std
    - cornish_fisher_var  : Cornish-Fisher expansion correcting z_alpha for
                            sample skewness and (excess) kurtosis
    - expected_shortfall  : historical average of returns below the VaR threshold
                            (also known as CVaR)
    - monte_carlo_var     : Gaussian MC simulation from empirical cov matrix
    - component_var       : Euler decomposition of portfolio VaR by asset

Conventions
-----------
All VaR outputs are returned as **positive numbers** representing the magnitude
of the loss at the given confidence level. This is the standard industry
convention: a VaR of 0.025 means "the portfolio is expected to lose at most
2.5% over the given horizon at the chosen confidence level".

Horizon scaling uses the square-root-of-time rule ``VaR_h = VaR_1 * sqrt(h)``.
This is only strictly valid for i.i.d. returns; it is an approximation in the
presence of autocorrelation or heteroskedasticity.

The module depends only on numpy and pandas — no scipy.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Hand-table of z_alpha (one-sided, standard normal inverse CDF).
# Keys are the confidence level alpha in (0, 1).
_Z_TABLE: dict[float, float] = {
    0.50: 0.0000,
    0.80: 0.8416,
    0.90: 1.2816,
    0.95: 1.6449,
    0.975: 1.9600,
    0.99: 2.3263,
    0.995: 2.5758,
    0.999: 3.0902,
}


def _z_from_alpha(alpha: float) -> float:
    """Return z_alpha for a one-sided normal confidence level.

    Uses a small hand-table for common values and a monotone linear
    interpolation between table points for intermediate alphas. This is
    sufficient for the Cornish-Fisher and parametric pathways without
    requiring scipy.

    Parameters
    ----------
    alpha : float
        Confidence level in the open interval (0, 1).

    Returns
    -------
    float
        The standard-normal quantile at ``alpha``.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if alpha in _Z_TABLE:
        return _Z_TABLE[alpha]

    # Linear interpolation between nearest table neighbours.
    keys = sorted(_Z_TABLE.keys())
    if alpha < keys[0]:
        # Mirror around 0.5 for tiny alphas.
        return -_z_from_alpha(1.0 - alpha) if alpha < 0.5 else _Z_TABLE[keys[0]]
    if alpha > keys[-1]:
        return _Z_TABLE[keys[-1]]

    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= alpha <= hi:
            frac = (alpha - lo) / (hi - lo)
            return _Z_TABLE[lo] + frac * (_Z_TABLE[hi] - _Z_TABLE[lo])

    # Fallback (should not be reached).
    return _Z_TABLE[0.95]


class PortfolioVaR:
    """Compute portfolio Value-at-Risk via several methodologies.

    Parameters
    ----------
    returns : pandas.DataFrame
        Panel of per-symbol arithmetic returns indexed by date (rows) and
        symbol (columns). Must be non-empty.
    weights : pandas.Series
        Per-symbol portfolio weights. Not normalized; negative weights
        (short positions) are allowed. Symbols missing from ``returns`` are
        dropped; symbols missing from ``weights`` contribute zero.

    Notes
    -----
    All VaR methods return **positive** loss magnitudes. A value of ``0.03``
    means "the portfolio is expected to lose at most 3% over the horizon at
    the chosen confidence level".

    Horizon scaling uses the square-root-of-time rule, which is only valid
    for i.i.d. returns.
    """

    def __init__(self, returns: pd.DataFrame, weights: pd.Series) -> None:
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame")
        if not isinstance(weights, pd.Series):
            raise TypeError("weights must be a pandas Series")
        if returns.empty:
            raise ValueError("returns DataFrame is empty")
        if len(weights) == 0:
            raise ValueError("weights Series is empty")

        # Align weights to columns; missing symbols → 0 weight.
        aligned = weights.reindex(returns.columns).fillna(0.0).astype(float)

        self._returns = returns.astype(float)
        self._weights = aligned

        # Pre-compute the portfolio return series (dot product row-wise).
        pr = self._returns.mul(self._weights, axis=1).sum(axis=1)
        pr = pr.dropna()
        if pr.empty:
            raise ValueError("portfolio return series is empty after alignment")
        self._portfolio_returns = pr

    # ------------------------------------------------------------------ #
    # Core properties
    # ------------------------------------------------------------------ #

    @property
    def portfolio_returns(self) -> pd.Series:
        """Return the aligned portfolio return series (row-wise dot product)."""
        return self._portfolio_returns

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_alpha(alpha: float) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    @staticmethod
    def _validate_horizon(horizon: int) -> None:
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError(f"horizon must be a positive integer, got {horizon}")

    # ------------------------------------------------------------------ #
    # VaR methodologies
    # ------------------------------------------------------------------ #

    def historical_var(self, alpha: float = 0.95, horizon: int = 1) -> float:
        """Empirical (historical) VaR.

        Returns the magnitude of the ``(1 - alpha)`` quantile of the
        portfolio-return distribution, scaled to ``horizon`` days via
        the square-root-of-time rule.
        """
        self._validate_alpha(alpha)
        self._validate_horizon(horizon)

        q = float(np.quantile(self._portfolio_returns.to_numpy(), 1.0 - alpha))
        var_1d = -q  # loss magnitude
        return float(var_1d * math.sqrt(horizon))

    def parametric_var(self, alpha: float = 0.95, horizon: int = 1) -> float:
        """Gaussian (parametric) VaR.

        Computes ``VaR = -(mu - z_alpha * sigma) * sqrt(horizon)``.
        """
        self._validate_alpha(alpha)
        self._validate_horizon(horizon)

        arr = self._portfolio_returns.to_numpy()
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        z = _z_from_alpha(alpha)

        var_1d = -(mu - z * sigma)
        return float(var_1d * math.sqrt(horizon))

    def cornish_fisher_var(self, alpha: float = 0.95, horizon: int = 1) -> float:
        """Cornish-Fisher adjusted VaR.

        Adjusts the Gaussian quantile ``z_alpha`` for sample skewness S and
        excess kurtosis K:

            z_CF = z + (z^2 - 1)/6 * S
                     + (z^3 - 3z)/24 * K
                     - (2 z^3 - 5 z)/36 * S^2

        Then ``VaR = -(mu - z_CF * sigma) * sqrt(horizon)``.
        """
        self._validate_alpha(alpha)
        self._validate_horizon(horizon)

        arr = self._portfolio_returns.to_numpy()
        n = len(arr)
        mu = float(np.mean(arr))
        if n < 2:
            return float(-mu * math.sqrt(horizon))

        sigma = float(np.std(arr, ddof=1))
        if sigma == 0.0:
            return float(-mu * math.sqrt(horizon))

        # Sample skewness and excess kurtosis (bias-uncorrected moment form).
        centered = arr - mu
        m2 = float(np.mean(centered**2))
        m3 = float(np.mean(centered**3))
        m4 = float(np.mean(centered**4))
        skew = m3 / (m2**1.5) if m2 > 0 else 0.0
        kurt_excess = (m4 / (m2**2) - 3.0) if m2 > 0 else 0.0

        z = _z_from_alpha(alpha)
        z_cf = (
            z
            + (z**2 - 1.0) / 6.0 * skew
            + (z**3 - 3.0 * z) / 24.0 * kurt_excess
            - (2.0 * z**3 - 5.0 * z) / 36.0 * (skew**2)
        )

        var_1d = -(mu - z_cf * sigma)
        return float(var_1d * math.sqrt(horizon))

    def expected_shortfall(self, alpha: float = 0.95, horizon: int = 1) -> float:
        """Historical Expected Shortfall (a.k.a. CVaR).

        Returns the magnitude of the average portfolio return conditional on
        the return being at or below the ``(1 - alpha)`` empirical quantile,
        scaled by ``sqrt(horizon)``.
        """
        self._validate_alpha(alpha)
        self._validate_horizon(horizon)

        arr = self._portfolio_returns.to_numpy()
        threshold = float(np.quantile(arr, 1.0 - alpha))
        tail = arr[arr <= threshold]
        if tail.size == 0:
            # Fallback: use the worst observation.
            tail = np.array([float(np.min(arr))])

        es_1d = -float(np.mean(tail))
        return float(es_1d * math.sqrt(horizon))

    # ------------------------------------------------------------------
    # Monte-Carlo VaR (C5b)
    # ------------------------------------------------------------------

    def monte_carlo_var(
        self,
        alpha: float = 0.95,
        horizon: int = 1,
        n_sims: int = 10_000,
        seed: int | None = 42,
    ) -> float:
        """Monte-Carlo VaR via multivariate Normal simulation.

        Simulates ``n_sims`` joint return scenarios from the empirical mean
        vector and covariance matrix of the asset returns, then computes the
        portfolio return distribution and reads off the VaR quantile.

        This is a Gaussian MC (no copula). For fat-tail MC, combine with
        the EVT module (:mod:`evt_tail_var`).

        Returns positive loss magnitude at ``alpha`` confidence.
        """
        self._validate_alpha(alpha)
        self._validate_horizon(horizon)

        rng = np.random.default_rng(seed)

        # Empirical moments
        mean_vec = self._returns.mean().to_numpy()
        cov_mat = self._returns.cov().to_numpy().copy()
        # Tikhonov regularisation: ensure PSD even with collinear assets
        cov_mat += np.eye(cov_mat.shape[0]) * 1e-8

        # Simulate joint returns
        sim_returns = rng.multivariate_normal(mean_vec, cov_mat, size=n_sims)
        w = self._weights.reindex(self._returns.columns, fill_value=0.0).to_numpy()
        portfolio_sims = sim_returns @ w

        var_1d = -float(np.quantile(portfolio_sims, 1.0 - alpha))
        return float(var_1d * math.sqrt(horizon))

    # ------------------------------------------------------------------
    # Component VaR (C5b)
    # ------------------------------------------------------------------

    def component_var(self, alpha: float = 0.95) -> pd.Series:
        """Component VaR: contribution of each asset to portfolio VaR.

        Uses the marginal VaR approach:
            CVaR_i = w_i * (Sigma @ w)_i / sigma_p * VaR_portfolio

        The sum of component VaRs equals the portfolio VaR (Euler
        decomposition property).

        Returns a Series indexed by asset symbols.
        """
        self._validate_alpha(alpha)

        w = self._weights.reindex(self._returns.columns, fill_value=0.0)
        cov_mat = self._returns.cov()
        sigma_w = cov_mat.values @ w.values  # Sigma @ w
        sigma_p = float(np.sqrt(w.values @ sigma_w))  # portfolio vol

        if sigma_p < 1e-12:
            return pd.Series(0.0, index=self._returns.columns, dtype=float)

        portfolio_var = self.parametric_var(alpha, horizon=1)
        marginal_var = sigma_w / sigma_p  # dVaR/dw_i (proportional)
        component = w.values * marginal_var * (portfolio_var / sigma_p)

        # Normalize so components sum exactly to portfolio_var
        raw_sum = float(np.sum(component))
        if abs(raw_sum) > 1e-12:
            component = component * (portfolio_var / raw_sum)

        return pd.Series(component, index=self._returns.columns, dtype=float)
