"""Black-Litterman Portfolio Optimizer.

Combines market equilibrium (reverse-optimized implied returns) with
user/model views to produce posterior expected returns, then optimizes
a portfolio under constraints.

This is the institutional-standard approach used at major asset managers:
rather than using raw factor scores directly as expected returns, BL
combines the stable market equilibrium with your model's views weighted
by your confidence in each view.

Usage:
    from src.assembled_core.portfolio.black_litterman import BlackLittermanOptimizer

    bl = BlackLittermanOptimizer(risk_aversion=2.5)
    weights = bl.optimize(
        market_weights=market_weights,   # current market cap weights
        sigma=cov_matrix,                # covariance matrix (annualized)
        views={"AAPL": 0.05, "MSFT": -0.02},  # symbol → expected excess return
        confidence={"AAPL": 0.7, "MSFT": 0.5},
    )
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    minimize = None


class BlackLittermanOptimizer:
    """Black-Litterman portfolio optimizer.

    Implements the full BL pipeline:
    1. Compute implied equilibrium returns: Pi = delta * Sigma * w_mkt
    2. Build view matrix P and view uncertainty Omega from model signals
    3. Compute BL posterior returns: mu_BL = [(tau*Sigma)^-1 + P'Omega^-1 P]^-1
                                              [(tau*Sigma)^-1 Pi + P'Omega^-1 Q]
    4. Maximize Sharpe ratio subject to constraints via scipy.optimize

    Attributes:
        risk_aversion: Market risk aversion coefficient delta (default: 2.5).
            Typical institutional range: 1.5–4.0.
        tau: Scaling factor for prior uncertainty (default: 0.05).
            Controls how much the prior is trusted vs. views.
        max_position: Maximum weight for any single position (default: 0.05 = 5%).
        min_position: Minimum weight (default: 0.0 = long-only).
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        max_position: float = 0.05,
        min_position: float = 0.0,
    ) -> None:
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.max_position = max_position
        self.min_position = min_position

    def compute_implied_returns(
        self,
        market_weights: pd.Series,
        sigma: pd.DataFrame,
    ) -> pd.Series:
        """Reverse-optimize equilibrium returns from market weights.

        Pi = delta * Sigma * w_mkt

        Args:
            market_weights: Market cap weights (index = symbols, values sum ~1.0).
            sigma: Annualized covariance matrix (symbols × symbols).

        Returns:
            Implied equilibrium returns as Series (index = symbols).
        """
        symbols = list(market_weights.index)
        w = market_weights.reindex(symbols).fillna(0).values
        S = sigma.reindex(index=symbols, columns=symbols).fillna(0).values
        pi = self.risk_aversion * S @ w
        return pd.Series(pi, index=symbols, name="implied_returns")

    def compute_posterior_returns(
        self,
        pi: pd.Series,
        sigma: pd.DataFrame,
        views: dict[str, float],
        confidence: Optional[dict[str, float]] = None,
    ) -> pd.Series:
        """Compute Black-Litterman posterior expected returns.

        Args:
            pi: Implied equilibrium returns from compute_implied_returns().
            sigma: Annualized covariance matrix.
            views: Dict mapping symbol → absolute view on expected return.
                   E.g. {"AAPL": 0.08} means AAPL expected to return 8% p.a.
            confidence: Dict mapping symbol → confidence in view [0, 1].
                        Default: 0.5 for all views.

        Returns:
            Posterior expected returns as Series (same index as pi).
        """
        symbols = list(pi.index)
        n = len(symbols)
        S = sigma.reindex(index=symbols, columns=symbols).fillna(0).values
        pi_arr = pi.reindex(symbols).fillna(0).values

        view_symbols = [s for s in views if s in symbols]
        if not view_symbols:
            logger.warning(
                "[BL] No valid view symbols found — returning equilibrium returns"
            )
            return pi.copy()

        k = len(view_symbols)
        symbol_idx = {s: i for i, s in enumerate(symbols)}

        # Build P matrix (k × n): one row per view, 1 for viewed symbol
        P = np.zeros((k, n))
        Q = np.zeros(k)
        for row, sym in enumerate(view_symbols):
            P[row, symbol_idx[sym]] = 1.0
            Q[row] = views[sym]

        # Build Omega: diagonal view uncertainty matrix
        # Omega_ii = (1/confidence_i - 1) * (P_i @ tau*Sigma @ P_i')
        # Default confidence = 0.5
        conf = confidence or {}
        omega_diag = np.zeros(k)
        tauS = self.tau * S
        for row, sym in enumerate(view_symbols):
            c = float(conf.get(sym, 0.5))
            c = max(1e-3, min(1.0 - 1e-3, c))
            p_i = P[row]
            variance_scale = float(p_i @ tauS @ p_i)
            omega_diag[row] = (1.0 / c - 1.0) * max(variance_scale, 1e-8)

        Omega = np.diag(omega_diag)

        try:
            Omega_inv = np.linalg.inv(Omega)
            tauS_inv = np.linalg.inv(tauS + np.eye(n) * 1e-8)
            M_inv = tauS_inv + P.T @ Omega_inv @ P
            M = np.linalg.inv(M_inv + np.eye(n) * 1e-8)
            mu_bl = M @ (tauS_inv @ pi_arr + P.T @ Omega_inv @ Q)
        except np.linalg.LinAlgError as exc:
            logger.warning(
                "[BL] Matrix inversion failed (%s) — returning equilibrium", exc
            )
            return pi.copy()

        return pd.Series(mu_bl, index=symbols, name="bl_expected_returns")

    def optimize(
        self,
        market_weights: pd.Series,
        sigma: pd.DataFrame,
        views: dict[str, float],
        confidence: Optional[dict[str, float]] = None,
        current_weights: Optional[dict[str, float]] = None,
        turnover_penalty: float = 0.0,
    ) -> pd.Series:
        """Full Black-Litterman optimize: compute posterior returns then maximize Sharpe.

        Args:
            market_weights: Market cap weights (symbols → weight).
            sigma: Annualized covariance matrix.
            views: Symbol → absolute expected annual return view.
            confidence: Symbol → confidence in view [0, 1].
            current_weights: Current portfolio weights (for turnover penalty).
            turnover_penalty: Lambda_tc for L1 turnover penalty (0 = disabled).

        Returns:
            Optimal portfolio weights as Series (index = symbols).
        """
        pi = self.compute_implied_returns(market_weights, sigma)
        mu_bl = self.compute_posterior_returns(pi, sigma, views, confidence)

        symbols = list(mu_bl.index)
        S = sigma.reindex(index=symbols, columns=symbols).fillna(0).values
        mu = mu_bl.values

        if not SCIPY_AVAILABLE:
            # Parity with the SLSQP fallback path (lines 246-254): tag the
            # Series so downstream callers can distinguish an equal-weight
            # return due to missing-scipy from a genuinely converged BL
            # solve. Previously this path returned an untagged Series and
            # the caller had no way to detect the degraded result.
            logger.warning(
                "[BL] scipy not available — returning equal weights (flagged)"
            )
            n = len(symbols)
            weights = pd.Series(
                np.ones(n) / n, index=symbols, name="bl_weights_equal_fallback"
            )
            weights.attrs["bl_converged"] = False
            weights.attrs["bl_fallback_reason"] = "scipy_unavailable"
            return weights

        n = len(symbols)
        w0 = np.ones(n) / n
        w_old = np.array([(current_weights or {}).get(s, 1.0 / n) for s in symbols])

        def neg_sharpe(w: np.ndarray) -> float:
            port_ret = float(mu @ w)
            port_vol = float(np.sqrt(w @ S @ w + 1e-10))
            obj = -port_ret / port_vol
            if turnover_penalty > 0:
                obj += turnover_penalty * float(np.sum(np.abs(w - w_old)))
            return obj

        def neg_sharpe_grad(w: np.ndarray) -> np.ndarray:
            port_vol = float(np.sqrt(w @ S @ w + 1e-10))
            grad_ret = mu
            grad_vol = (S @ w) / port_vol
            sharpe = float(mu @ w) / port_vol
            grad: np.ndarray = -(grad_ret / port_vol - sharpe * grad_vol / port_vol)
            if turnover_penalty > 0:
                grad += turnover_penalty * np.sign(w - w_old)
            return grad

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(self.min_position, self.max_position)] * n

        fallback_reason: str | None = None
        try:
            result = minimize(
                neg_sharpe,
                w0,
                jac=neg_sharpe_grad,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-9},
            )
            if result.success:
                w_opt = np.maximum(result.x, 0.0)
                if np.any(np.isnan(w_opt)):
                    fallback_reason = "nan_in_optimizer_result"
                    logger.warning(
                        "[BL] NaN in optimizer result — returning equal weights"
                    )
                    w_opt = w0
                else:
                    w_opt /= w_opt.sum() if w_opt.sum() > 1e-8 else 1.0
            else:
                fallback_reason = f"non_convergence:{result.message}"
                logger.warning(
                    "[BL] Optimization did not converge: %s — returning equal weights (flagged)",
                    result.message,
                )
                w_opt = w0
        except Exception as exc:
            fallback_reason = f"exception:{exc}"
            logger.warning(
                "[BL] Optimization error: %s — returning equal weights (flagged)", exc
            )
            w_opt = w0

        name = "bl_weights_equal_fallback" if fallback_reason else "bl_weights"
        weights = pd.Series(w_opt, index=symbols, name=name)
        if fallback_reason is not None:
            # Tag the diagnostic on .attrs so downstream callers can gate on it
            # instead of treating an equal-weight fallback as an optimised result.
            weights.attrs["bl_converged"] = False
            weights.attrs["bl_fallback_reason"] = fallback_reason
        else:
            weights.attrs["bl_converged"] = True
        logger.info(
            "[BL] Optimized weights: top 3 = %s",
            weights.nlargest(3).to_dict(),
        )
        return weights

    def optimize_from_scores(
        self,
        scores: pd.Series,
        sigma: pd.DataFrame,
        confidence: float = 0.5,
        return_scale: float = 0.10,
    ) -> pd.Series:
        """Convenience method: convert factor scores to views and optimize.

        Normalizes factor scores to return views: top-ranked symbols get
        positive views, bottom-ranked get negative views.

        Args:
            scores: Symbol → factor score (e.g. composite alpha score).
            sigma: Annualized covariance matrix.
            confidence: Uniform confidence in all views (default: 0.5).
            return_scale: Scale factor — max view magnitude in decimal
                          (default: 0.10 = 10% p.a. max view).

        Returns:
            Optimal portfolio weights.
        """
        symbols = [s for s in scores.index if s in sigma.index]
        if not symbols:
            logger.warning("[BL] No symbol overlap between scores and sigma")
            n = len(sigma)
            return pd.Series(np.ones(n) / n, index=sigma.index)

        s = scores.reindex(symbols)
        std_s = s.std()
        denom = float(std_s) if np.isfinite(std_s) and std_s > 1e-8 else 1.0
        normalized = (s - s.mean()) / denom
        max_norm = normalized.abs().max()
        views = (normalized / max(float(max_norm), 1e-8) * return_scale).to_dict()

        n_syms = len(symbols)
        market_weights = pd.Series(np.ones(n_syms) / n_syms, index=symbols)
        conf_dict = {sym: confidence for sym in symbols}

        return self.optimize(market_weights, sigma, views, conf_dict)


# ---------------------------------------------------------------------------
# Intel → BL Views Bridge (Plan 5.5)
# ---------------------------------------------------------------------------


def intel_to_bl_views(
    sector_impacts: dict[str, float],
    bayesian_confidence: dict[str, float] | None = None,
    regime_multiplier: float = 1.0,
    return_scale: float = 0.05,
) -> tuple[dict[str, float], dict[str, float]]:
    """Convert intel pipeline sector impacts to Black-Litterman views.

    Positive intel signal → positive view (sector will outperform).
    Negative signal → negative view.

    Args:
        sector_impacts: Symbol/ETF → impact score (from IntelSignalAdapter).
            Positive = beneficiary, negative = loser.
        bayesian_confidence: Symbol → Bayesian posterior confidence [0, 1].
            If None, uses abs(impact) as proxy.
        regime_multiplier: Scale views by regime factor (crisis=0.5, bull=1.5).
        return_scale: Maximum view magnitude in decimal (default 5% p.a.).

    Returns:
        Tuple of (views_dict, confidence_dict) ready for BL optimizer.
    """
    if not sector_impacts:
        return {}, {}

    # Normalize impacts to [-1, 1]
    max_abs = max(abs(v) for v in sector_impacts.values()) or 1.0
    views: dict[str, float] = {}
    confidence: dict[str, float] = {}

    for sym, impact in sector_impacts.items():
        normalized = impact / max_abs
        view_return = normalized * return_scale * regime_multiplier
        views[sym] = round(view_return, 6)

        if bayesian_confidence and sym in bayesian_confidence:
            confidence[sym] = min(1.0, max(0.05, bayesian_confidence[sym]))
        else:
            confidence[sym] = min(1.0, max(0.1, abs(normalized) * 0.8))

    return views, confidence


# ---------------------------------------------------------------------------
# 5.7  Robust BL (Uncertainty-Set around mu)
# ---------------------------------------------------------------------------


def robust_bl_shrinkage(
    mu_bl: np.ndarray,
    sigma: np.ndarray,
    n_obs: int = 252,
    kappa_scale: float = 1.0,
) -> np.ndarray:
    """Shrink BL expected returns toward zero by estimation uncertainty.

    Applies ellipsoid uncertainty: mu_robust = mu_bl * shrinkage_factor
    where shrinkage_factor = 1 - kappa / ||mu_bl||, clamped to [0, 1].

    Args:
        mu_bl: BL posterior expected returns (N,).
        sigma: Covariance matrix (N×N).
        n_obs: Number of observations used for estimation.
        kappa_scale: Multiplier for uncertainty radius.

    Returns:
        Shrunk expected returns.
    """
    n = len(mu_bl)
    # Estimation uncertainty proportional to sqrt(1/n_obs)
    kappa = kappa_scale * np.sqrt(n / max(n_obs, 1))
    mu_norm = np.sqrt(mu_bl @ mu_bl)

    if mu_norm < 1e-10:
        return mu_bl.copy()

    shrinkage = max(0.0, 1.0 - kappa / mu_norm)
    shrunk: np.ndarray = mu_bl * shrinkage
    return shrunk
