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
    from scipy.optimize import minimize  # type: ignore
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    minimize = None  # type: ignore


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
            logger.warning("[BL] No valid view symbols found — returning equilibrium returns")
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
            logger.warning("[BL] Matrix inversion failed (%s) — returning equilibrium", exc)
            return pi.copy()

        return pd.Series(mu_bl, index=symbols, name="bl_expected_returns")

    def optimize(
        self,
        market_weights: pd.Series,
        sigma: pd.DataFrame,
        views: dict[str, float],
        confidence: Optional[dict[str, float]] = None,
    ) -> pd.Series:
        """Full Black-Litterman optimize: compute posterior returns then maximize Sharpe.

        Args:
            market_weights: Market cap weights (symbols → weight).
            sigma: Annualized covariance matrix.
            views: Symbol → absolute expected annual return view.
            confidence: Symbol → confidence in view [0, 1].

        Returns:
            Optimal portfolio weights as Series (index = symbols).
        """
        pi = self.compute_implied_returns(market_weights, sigma)
        mu_bl = self.compute_posterior_returns(pi, sigma, views, confidence)

        symbols = list(mu_bl.index)
        S = sigma.reindex(index=symbols, columns=symbols).fillna(0).values
        mu = mu_bl.values

        if not SCIPY_AVAILABLE:
            logger.warning("[BL] scipy not available — returning equal weights")
            n = len(symbols)
            return pd.Series(np.ones(n) / n, index=symbols)

        n = len(symbols)
        w0 = np.ones(n) / n

        def neg_sharpe(w: np.ndarray) -> float:
            port_ret = float(mu @ w)
            port_vol = float(np.sqrt(w @ S @ w + 1e-10))
            return -port_ret / port_vol

        def neg_sharpe_grad(w: np.ndarray) -> np.ndarray:
            port_vol = float(np.sqrt(w @ S @ w + 1e-10))
            grad_ret = mu
            grad_vol = (S @ w) / port_vol
            sharpe = float(mu @ w) / port_vol
            return -(grad_ret / port_vol - sharpe * grad_vol / port_vol)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(self.min_position, self.max_position)] * n

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
                w_opt /= w_opt.sum() if w_opt.sum() > 1e-8 else 1.0
            else:
                logger.warning("[BL] Optimization did not converge: %s", result.message)
                w_opt = w0
        except Exception as exc:
            logger.warning("[BL] Optimization error: %s — returning equal weights", exc)
            w_opt = w0

        weights = pd.Series(w_opt, index=symbols, name="bl_weights")
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
        normalized = (s - s.mean()) / (s.std() + 1e-8)
        views = (normalized / normalized.abs().max() * return_scale).to_dict()

        n_syms = len(symbols)
        market_weights = pd.Series(np.ones(n_syms) / n_syms, index=symbols)
        conf_dict = {sym: confidence for sym in symbols}

        return self.optimize(market_weights, sigma, views, conf_dict)
