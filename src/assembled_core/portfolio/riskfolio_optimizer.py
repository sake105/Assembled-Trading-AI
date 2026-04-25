"""Portfolio Optimization via Riskfolio-Lib and skfolio.

From 11_FREE_MODELLE.md §11.12.
24 risk measures (CVaR, EVaR, GMD, MAD) + Ledoit covariance.

Install: pip install riskfolio-lib==7.2.1 cvxpy==1.8.2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    risk_measure: str = "CVaR"
    objective: str = "Sharpe"
    method_mu: str = "hist"
    method_cov: str = "ledoit"
    alpha: float = 0.05
    max_weight: float = 0.15
    min_weight: float = 0.0


def _try_riskfolio():
    try:
        import riskfolio as rp
        return rp
    except ImportError:
        logger.warning("riskfolio-lib not installed — pip install riskfolio-lib==7.2.1")
        return None


def optimize_portfolio(
    returns: pd.DataFrame,
    config: OptimizationConfig | None = None,
) -> pd.Series | None:
    """Compute optimal portfolio weights via Riskfolio-Lib.

    Args:
        returns: Daily returns DataFrame — index=date, columns=tickers.
        config: Optimization config (defaults to CVaR-Sharpe with Ledoit cov).

    Returns:
        Series of weights indexed by ticker, or None on failure.
    """
    rp = _try_riskfolio()
    if rp is None:
        return None

    cfg = config or OptimizationConfig()

    try:
        port = rp.Portfolio(returns=returns)
        port.assets_stats(method_mu=cfg.method_mu, method_cov=cfg.method_cov)

        # Apply weight constraints
        port.upperlng = cfg.max_weight
        port.lowerlng = cfg.min_weight

        weights = port.optimization(
            model="Classic",
            rm=cfg.risk_measure,
            obj=cfg.objective,
            rf=0.0,
            l=0,
            hist=True,
        )
        if weights is None or weights.empty:
            return None

        return weights["weights"].squeeze()
    except Exception as exc:
        logger.debug("Riskfolio optimization failed: %s", exc)
        return None


def hrp_weights(returns: pd.DataFrame) -> pd.Series | None:
    """Hierarchical Risk Parity weights via Riskfolio-Lib.

    Args:
        returns: Daily returns DataFrame.

    Returns:
        Series of HRP weights, or None on failure.
    """
    rp = _try_riskfolio()
    if rp is None:
        return None

    try:
        port = rp.HCPortfolio(returns=returns)
        weights = port.optimization(model="HRP", rm="MV", rf=0.0, linkage="ward")
        if weights is None or weights.empty:
            return None
        return weights["weights"].squeeze()
    except Exception as exc:
        logger.debug("HRP optimization failed: %s", exc)
        return None


def cvar_budget(
    returns: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.Series | None:
    """Compute equal CVaR-contribution budget weights.

    Args:
        returns: Daily returns DataFrame.
        alpha: CVaR confidence level (default 5%).

    Returns:
        Series of weights summing to 1.0, or None on failure.
    """
    rp = _try_riskfolio()
    if rp is None:
        return None

    try:
        port = rp.Portfolio(returns=returns)
        port.assets_stats(method_mu="hist", method_cov="ledoit")
        weights = port.optimization(model="Classic", rm="CVaR", obj="ERC", rf=0.0, hist=True)
        if weights is None or weights.empty:
            return None
        return weights["weights"].squeeze()
    except Exception as exc:
        logger.debug("CVaR-budget optimization failed: %s", exc)
        return None


def equal_weight_fallback(tickers: list[str]) -> pd.Series:
    """Return equal-weight portfolio when optimization fails."""
    n = len(tickers)
    if n == 0:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / n, index=tickers, name="weights")


__all__ = [
    "OptimizationConfig",
    "optimize_portfolio",
    "hrp_weights",
    "cvar_budget",
    "equal_weight_fallback",
]
