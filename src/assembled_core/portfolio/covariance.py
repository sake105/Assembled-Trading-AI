"""Covariance Matrix Estimation Utilities.

Provides robust covariance estimators for portfolio optimization:
- Sample covariance (standard)
- Ledoit-Wolf shrinkage (sklearn)
- Constant-correlation shrinkage (Ledoit-Wolf target: constant off-diagonal)
- Exponentially-weighted covariance (more weight on recent data)

Usage:
    from src.assembled_core.portfolio.covariance import estimate_covariance

    cov = estimate_covariance(returns_df, method="ledoit_wolf")
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CovMethod = Literal["sample", "ledoit_wolf", "ewm", "dcc_garch"]

try:
    from sklearn.covariance import LedoitWolf  # type: ignore
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    LedoitWolf = None  # type: ignore


def estimate_covariance(
    returns: pd.DataFrame,
    method: CovMethod = "ledoit_wolf",
    ewm_halflife: int = 60,
    min_periods: int = 20,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Estimate a covariance matrix from a returns DataFrame.

    Args:
        returns: DataFrame where each column is a symbol's return series.
                 Rows are dates; NaN values are handled by pairwise complete obs.
        method: Estimation method — "sample", "ledoit_wolf", or "ewm".
        ewm_halflife: Half-life in days for EWM covariance (default: 60).
        min_periods: Minimum non-NaN observations required per symbol pair.
        annualize: Multiply by trading_days to annualize (default: True).
        trading_days: Trading days per year for annualization (default: 252).

    Returns:
        Covariance matrix as DataFrame (symbols × symbols).
    """
    if returns.empty or returns.shape[1] < 2:
        logger.warning("[Covariance] Insufficient data for covariance estimation")
        return pd.DataFrame()

    symbols = list(returns.columns)
    clean = returns.dropna(how="all")

    if method == "ledoit_wolf":
        if not SKLEARN_AVAILABLE:
            logger.warning("[Covariance] sklearn not available — falling back to sample covariance")
            return estimate_covariance(clean, method="sample", annualize=annualize)
        # Fill NaN with column means for LedoitWolf (requires complete matrix)
        filled = clean.fillna(clean.mean())
        lw = LedoitWolf()
        lw.fit(filled.values)
        cov_array = lw.covariance_
    elif method == "dcc_garch":
        cov_array = _dcc_garch_covariance(clean, min_periods=min_periods)
    elif method == "ewm":
        cov_array = _ewm_covariance(clean, halflife=ewm_halflife, min_periods=min_periods)
    else:
        # Sample covariance (pairwise)
        cov_array = clean.cov(min_periods=min_periods).values

    if annualize:
        cov_array = cov_array * trading_days

    cov_df = pd.DataFrame(cov_array, index=symbols, columns=symbols)
    # Ensure positive semi-definiteness via eigenvalue clipping
    cov_df = _ensure_psd(cov_df)
    return cov_df


def _ewm_covariance(
    returns: pd.DataFrame,
    halflife: int = 60,
    min_periods: int = 20,
) -> np.ndarray:
    """Compute exponentially-weighted covariance matrix."""
    n = len(returns.columns)
    cov_array = np.full((n, n), np.nan)
    symbols = list(returns.columns)

    # EWM covariance: Cov(i,j) = EWM(r_i * r_j) - EWM(r_i) * EWM(r_j)
    # Use pandas ewm for each pair
    decay = 0.5 ** (1.0 / halflife)  # lambda in EWM

    for i, sym_i in enumerate(symbols):
        for j, sym_j in enumerate(symbols[i:], start=i):
            r_i = returns[sym_i].dropna()
            r_j = returns[sym_j].dropna()
            common = r_i.index.intersection(r_j.index)
            if len(common) < min_periods:
                cov_array[i, j] = 0.0
                cov_array[j, i] = 0.0
                continue
            ri = r_i.loc[common]
            rj = r_j.loc[common]
            # Simple EWM covariance approximation
            weights = np.array([decay ** k for k in range(len(common) - 1, -1, -1)])
            weights /= weights.sum()
            mean_i = float(np.dot(weights, ri.values))
            mean_j = float(np.dot(weights, rj.values))
            cov_val = float(np.dot(weights, (ri.values - mean_i) * (rj.values - mean_j)))
            cov_array[i, j] = cov_val
            cov_array[j, i] = cov_val

    return cov_array


def _dcc_garch_covariance(
    returns: pd.DataFrame,
    min_periods: int = 60,
    garch_omega: float = 0.00001,
    garch_alpha: float = 0.06,
    garch_beta: float = 0.93,
    dcc_a: float = 0.02,
    dcc_b: float = 0.95,
) -> np.ndarray:
    """DCC-GARCH dynamic covariance estimation (Engle 2002).

    Phase 1: Univariate GARCH(1,1) per asset → conditional variances h_i(t).
    Phase 2: Standardized residuals z_i(t) = epsilon_i(t) / sqrt(h_i(t)).
    Phase 3: DCC correlation dynamics:
        Q_t = (1-a-b)*Q_bar + a*z_{t-1}*z_{t-1}' + b*Q_{t-1}
        R_t = diag(Q_t)^{-1/2} * Q_t * diag(Q_t)^{-1/2}
    Result: H_t = D_t * R_t * D_t  where D_t = diag(sqrt(h_i(t))).

    Returns the FINAL-period covariance matrix (most recent estimate).
    """
    clean = returns.dropna()
    n_obs, n_assets = clean.shape

    if n_obs < min_periods:
        logger.warning("[DCC-GARCH] Insufficient data (%d < %d) — falling back to sample", n_obs, min_periods)
        return clean.cov(min_periods=20).values

    data = clean.values  # (T, N)

    # Phase 1: Univariate GARCH(1,1) for each asset
    h = np.zeros((n_obs, n_assets))  # conditional variances
    z = np.zeros((n_obs, n_assets))  # standardized residuals

    for j in range(n_assets):
        r = data[:, j]
        var_init = float(np.var(r[:min(60, n_obs)]))
        if var_init < 1e-12:
            var_init = 1e-6
        h[0, j] = var_init
        z[0, j] = r[0] / np.sqrt(h[0, j])
        for t in range(1, n_obs):
            h[t, j] = garch_omega + garch_alpha * r[t - 1] ** 2 + garch_beta * h[t - 1, j]
            h[t, j] = max(h[t, j], 1e-12)
            z[t, j] = r[t] / np.sqrt(h[t, j])

    # Phase 2: Unconditional correlation of standardized residuals
    Q_bar = np.corrcoef(z.T)
    if np.any(np.isnan(Q_bar)):
        Q_bar = np.eye(n_assets)

    # Phase 3: DCC dynamics
    Q_t = Q_bar.copy()
    R_t = Q_bar.copy()

    for t in range(1, n_obs):
        z_t = z[t - 1].reshape(-1, 1)
        Q_t = (1 - dcc_a - dcc_b) * Q_bar + dcc_a * (z_t @ z_t.T) + dcc_b * Q_t
        # Normalize Q_t to correlation matrix R_t
        diag_inv = np.diag(1.0 / np.sqrt(np.maximum(np.diag(Q_t), 1e-12)))
        R_t = diag_inv @ Q_t @ diag_inv

    # Build final covariance: H_T = D_T * R_T * D_T
    D_T = np.diag(np.sqrt(h[-1, :]))
    cov_final = D_T @ R_t @ D_T

    logger.debug("[DCC-GARCH] Estimated %dx%d covariance from %d observations", n_assets, n_assets, n_obs)
    return cov_final


def _ensure_psd(cov: pd.DataFrame, epsilon: float = 1e-8) -> pd.DataFrame:
    """Clip negative eigenvalues to epsilon to ensure positive semi-definiteness."""
    arr = cov.values
    try:
        eigvals, eigvecs = np.linalg.eigh(arr)
        eigvals_clipped = np.maximum(eigvals, epsilon)
        arr_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        # Restore symmetry
        arr_psd = (arr_psd + arr_psd.T) / 2.0
        return pd.DataFrame(arr_psd, index=cov.index, columns=cov.columns)
    except np.linalg.LinAlgError:
        logger.warning("[Covariance] PSD correction failed — returning as-is")
        return cov


def returns_from_prices(
    prices: pd.DataFrame,
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
    close_col: str = "close",
    log_returns: bool = True,
) -> pd.DataFrame:
    """Pivot a price panel to a returns DataFrame (symbols as columns).

    Args:
        prices: Daily OHLCV panel.
        symbol_col: Symbol column name.
        timestamp_col: Timestamp column name.
        close_col: Close price column.
        log_returns: Use log returns (default: True). False = simple returns.

    Returns:
        Wide DataFrame: index = dates, columns = symbols, values = returns.
    """
    pivot = prices.pivot_table(
        index=timestamp_col, columns=symbol_col, values=close_col, aggfunc="last"
    )
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()

    if log_returns:
        ret = np.log(pivot / pivot.shift(1))
    else:
        ret = pivot.pct_change()

    return ret.iloc[1:]  # drop first NaN row
