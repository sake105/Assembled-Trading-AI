"""Hierarchical Risk Parity (HRP) — Lopez de Prado 2016.

HRP constructs portfolios without matrix inversion, making it robust to
ill-conditioned or singular covariance matrices that break traditional
Mean-Variance Optimization.

Algorithm:
1. Correlation matrix → distance matrix: ``d(i,j) = sqrt(0.5*(1-rho(i,j)))``
2. Single-linkage hierarchical clustering → dendrogram
3. Quasi-diagonalization (seriation) of covariance matrix
4. Recursive bisection: each split weighted by inverse cluster variance

Empirically outperforms MVO out-of-sample (Lopez de Prado 2016, Raffinot 2017).

Requires: ``scipy`` for hierarchical clustering.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def _correlation_distance(corr: np.ndarray) -> np.ndarray:
    """Convert correlation matrix to distance matrix."""
    # d(i,j) = sqrt(0.5 * (1 - rho(i,j)))
    dist: np.ndarray = np.sqrt(0.5 * (1.0 - corr))
    return dist


def _quasi_diagonalize(link: np.ndarray, n: int) -> list[int]:
    """Reorder assets by dendrogram leaf ordering (seriation)."""
    return list(leaves_list(link))


def _get_cluster_var(cov: np.ndarray, items: list[int]) -> float:
    """Compute inverse-variance portfolio variance for a subset of assets."""
    sub_cov = cov[np.ix_(items, items)]
    # Inverse-variance weights within sub-cluster
    diag = np.diag(sub_cov)
    diag = np.maximum(diag, 1e-10)
    ivp = 1.0 / diag
    ivp /= ivp.sum()
    return float(ivp @ sub_cov @ ivp)


def compute_hrp_weights(
    returns: pd.DataFrame,
    *,
    cov_method: str = "sample",
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> dict[str, float]:
    """Compute Hierarchical Risk Parity portfolio weights.

    Args:
        returns: Wide-format DataFrame (dates × symbols) of daily returns.
        cov_method: Covariance estimation method. Currently only ``"sample"``
            is supported.
        min_weight: Minimum weight per asset (floor).
        max_weight: Maximum weight per asset (cap).

    Returns:
        Dict mapping symbol → weight (sums to 1.0).
        Empty dict if scipy is unavailable or returns has < 2 assets.
    """
    if not SCIPY_AVAILABLE:
        logger.warning("[HRP] scipy not installed — cannot compute HRP weights")
        return {}

    symbols = list(returns.columns)
    n = len(symbols)

    if n < 2:
        if n == 1:
            return {symbols[0]: 1.0}
        return {}

    # Drop rows with any NaN
    clean = returns.dropna()
    if len(clean) < 30:
        logger.warning("[HRP] insufficient data (%d rows after dropna)", len(clean))
        return {}

    # Drop symbols with degenerate variance (constant price, halted, etc.).
    # Silently rewriting their NaN correlations to 0.0 would assign them a
    # real weight as if they were simply uncorrelated — a dead symbol should
    # not receive allocation.
    stds = clean.std()
    degenerate = stds[(stds <= 1e-12) | ~np.isfinite(stds)].index.tolist()
    if degenerate:
        logger.warning(
            "[HRP] dropping %d degenerate-variance symbols: %s",
            len(degenerate),
            degenerate[:20],
        )
        clean = clean.drop(columns=degenerate)
        symbols = list(clean.columns)
        n = len(symbols)
        if n < 2:
            if n == 1:
                return {symbols[0]: 1.0}
            return {}

    # Step 1: Covariance and correlation
    cov = clean.cov().values
    corr = clean.corr().values

    # Any remaining NaN correlation is a genuine numerical artifact
    # (not a dead symbol) — replacing with 0 is acceptable here.
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    # Step 2: Distance matrix → hierarchical clustering
    dist = _correlation_distance(corr)
    np.fill_diagonal(dist, 0.0)

    # Convert to condensed form for linkage
    try:
        condensed_dist = squareform(dist, checks=False)
    except Exception:
        # Fallback: manual condensed form
        condensed_dist = dist[np.triu_indices(n, k=1)]

    link = linkage(condensed_dist, method="single")

    # Step 3: Quasi-diagonalize
    sort_ix = _quasi_diagonalize(link, n)

    # Step 4: Recursive bisection
    weights: npt.NDArray[np.float64] = np.ones(n)

    def _recursive_bisect(items: list[int]) -> None:
        if len(items) <= 1:
            return
        mid = len(items) // 2
        left = items[:mid]
        right = items[mid:]

        # Cluster variance for left and right
        var_left = _get_cluster_var(cov, left)
        var_right = _get_cluster_var(cov, right)

        # Allocation: inverse variance proportion
        total_var = var_left + var_right
        if total_var < 1e-15:
            alpha = 0.5
        else:
            alpha = 1.0 - var_left / total_var  # lower var → higher weight

        # Scale weights
        for i in left:
            weights[i] *= alpha
        for i in right:
            weights[i] *= 1.0 - alpha

        # Recurse
        _recursive_bisect(left)
        _recursive_bisect(right)

    _recursive_bisect(sort_ix)

    # Normalize
    total = weights.sum()
    if total > 0:
        weights /= total

    # Apply constraints
    weights = np.clip(weights, min_weight, max_weight)
    # Re-normalize after clipping
    total = weights.sum()
    if total > 0:
        weights /= total

    result = {symbols[i]: round(float(weights[i]), 6) for i in range(n)}

    logger.debug(
        "[HRP] Computed weights for %d assets: max=%.4f, min=%.4f, HHI=%.4f",
        n,
        max(result.values()),
        min(result.values()),
        sum(w**2 for w in result.values()),
    )

    return result


def hrp_vs_equal_weight_comparison(
    returns: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Compare HRP vs equal-weight portfolio metrics.

    Useful for validating that HRP actually improves risk-adjusted returns
    in the specific universe being traded.

    Returns:
        Dict with ``hrp`` and ``equal_weight`` sub-dicts containing
        ``sharpe``, ``volatility``, ``max_drawdown``.
    """
    if not SCIPY_AVAILABLE or returns.empty:
        return {}

    symbols = list(returns.columns)
    n = len(symbols)
    if n < 2:
        return {}

    # HRP weights
    hrp_w = compute_hrp_weights(returns)
    if not hrp_w:
        return {}

    # Equal weights
    ew_w = {s: 1.0 / n for s in symbols}

    def _portfolio_stats(w: dict[str, float]) -> dict[str, float]:
        weights_arr = np.array([w.get(s, 0) for s in symbols])
        port_ret = returns.values @ weights_arr
        cumret = np.cumprod(1 + port_ret)
        peak = np.maximum.accumulate(cumret)
        peak = np.maximum(peak, 1e-10)
        dd = (cumret - peak) / peak

        ann_ret = float(np.mean(port_ret)) * 252
        ann_vol = float(np.std(port_ret)) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        return {
            "sharpe": round(sharpe, 4),
            "volatility": round(ann_vol, 6),
            "max_drawdown": round(float(dd.min()), 6),
        }

    return {
        "hrp": _portfolio_stats(hrp_w),
        "equal_weight": _portfolio_stats(ew_w),
    }


def hrp_with_turnover_control(
    returns: pd.DataFrame,
    current_weights: dict[str, float],
    max_turnover: float = 0.30,
    blend_speed: float = 0.5,
    **kwargs,
) -> dict[str, float]:
    """HRP with post-optimization turnover overlay.

    Blends HRP target weights with current weights to control turnover.
    If full rebalance turnover exceeds max_turnover, partially adjust
    using blend_speed.

    Args:
        returns: Wide-format DataFrame (dates × symbols) of daily returns.
        current_weights: Current portfolio weights.
        max_turnover: Maximum one-way turnover allowed.
        blend_speed: How fast to converge to HRP target (0-1).
            0 = keep current, 1 = full rebalance.
        **kwargs: Passed to compute_hrp_weights.

    Returns:
        Dict mapping symbol → weight with turnover control.
    """
    hrp_target = compute_hrp_weights(returns, **kwargs)
    if not hrp_target:
        return current_weights

    symbols = sorted(set(hrp_target) | set(current_weights))

    w_target = np.array([hrp_target.get(s, 0.0) for s in symbols])
    w_curr = np.array([current_weights.get(s, 0.0) for s in symbols])

    # Full turnover
    full_turnover = float(np.sum(np.abs(w_target - w_curr)))

    if full_turnover <= max_turnover:
        # Within budget — use HRP target directly
        return hrp_target

    # Partial adjustment: blend current and target
    effective_speed = min(blend_speed, max_turnover / (full_turnover + 1e-10))
    w_final = w_curr + effective_speed * (w_target - w_curr)
    w_final = np.maximum(w_final, 0.0)
    total = w_final.sum()
    if total > 1e-8:
        w_final /= total

    logger.info(
        "[HRP] Turnover control: full=%.4f, max=%.4f, effective_speed=%.4f",
        full_turnover,
        max_turnover,
        effective_speed,
    )

    return {s: round(float(w_final[i]), 6) for i, s in enumerate(symbols)}


__all__ = [
    "compute_hrp_weights",
    "hrp_vs_equal_weight_comparison",
    "hrp_with_turnover_control",
]
