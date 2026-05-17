"""Diebold-Yilmaz (2012) Spillover Index for directional volatility/return spillovers.

Audit C4-079 (KNOWN_ISSUES §8.13) closure: builds the connectedness framework
from scratch. The Diebold-Yilmaz (2012) approach quantifies cross-asset
spillovers via the forecast-error variance decomposition (FEVD) of a VAR
model — answering "how much of asset i's H-step forecast error variance is
explained by shocks to asset j?".

Pesaran-Shin (1998) generalized FEVD is used by default — order-independent
(no Cholesky ordering choice) and the canonical form in modern DY literature.

Outputs:

  - **Total Spillover Index (TSI)**: scalar in [0%, 100%], the share of
    forecast-error variance attributable to cross-asset spillovers.
  - **Directional spillovers**: ``to_others_i`` (variance variable i exports),
    ``from_others_i`` (variance variable i imports), ``net_i = to − from``
    (positive = net transmitter, negative = net receiver).
  - **Pairwise net**: matrix of net pairwise spillovers (used for network plots).

Window/Lag sensitivity (audit emphasis): both VAR lag order ``p`` and FEVD
horizon ``h`` are exposed as parameters. The audit specifically asked these
to be documented + parametrised; both are.

References:
- Diebold, F. X., Yilmaz, K. (2012). *Better to Give than to Receive:
  Predictive Directional Measurement of Volatility Spillovers*. IJF 28(1).
- Pesaran, H. H., Shin, Y. (1998). *Generalized Impulse Response Analysis
  in Linear Multivariate Models*. Economics Letters 58(1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SpilloverResult:
    """Result of a Diebold-Yilmaz spillover computation.

    Attributes:
        total_spillover_index_pct: TSI in [0, 100]. Higher = more connectedness.
        fevd_matrix: H-step generalized FEVD as a (N, N) DataFrame with row/col
            labels = variable names. Element (i, j) = % of i's H-step forecast
            variance explained by shocks to j. Rows sum to 100 by construction.
        to_others: Series (length N) of "spillover TO others" per variable
            (sum of column j minus diagonal, sum of off-diagonal in column j).
        from_others: Series of "spillover FROM others" per variable
            (sum of row i minus diagonal).
        net: Series of net spillovers = to_others − from_others.
        lag: VAR lag order used.
        horizon: FEVD forecast horizon used.
        n_obs: Number of observations after VAR fitting.
    """

    total_spillover_index_pct: float
    fevd_matrix: pd.DataFrame
    to_others: pd.Series
    from_others: pd.Series
    net: pd.Series
    lag: int
    horizon: int
    n_obs: int


def _generalized_fevd(var_result, horizon: int) -> np.ndarray:
    """Pesaran-Shin (1998) generalized FEVD at the given horizon.

    For each variable i and shock source j:
        gFEVD_ij(H) = (σ_jj^{-1} · Σ_{l=0}^{H-1} (e_i' · A_l · Σ · e_j)²)
                     / Σ_{l=0}^{H-1} (e_i' · A_l · Σ · A_l' · e_i)

    where Σ is the residual covariance, A_l are the MA-coefficient matrices
    from the VAR(p), and e_k is the k-th unit vector.

    Returns:
        (N, N) array where row i sums to 1.0 after row-normalisation (so
        the rows can be interpreted as proper variance shares).
    """
    sigma = np.asarray(var_result.sigma_u)  # residual covariance (N, N)
    n_vars = sigma.shape[0]
    # MA coefficient matrices A_0, A_1, ..., A_{H-1}. statsmodels exposes via
    # var_result.ma_rep(maxn=horizon-1) returning shape (horizon, N, N).
    ma = np.asarray(var_result.ma_rep(maxn=horizon - 1))
    if ma.shape != (horizon, n_vars, n_vars):
        raise ValueError(
            f"ma_rep shape mismatch: expected ({horizon},{n_vars},{n_vars}), got {ma.shape}"
        )

    fevd = np.zeros((n_vars, n_vars))
    sigma_diag = np.diag(sigma)  # σ_jj for each j

    for i in range(n_vars):
        denominator = 0.0
        for h in range(horizon):
            A_l = ma[h]
            # e_i' · A_l · Σ · A_l' · e_i = (A_l · Σ · A_l')[i, i]
            denominator += float((A_l @ sigma @ A_l.T)[i, i])
        for j in range(n_vars):
            numerator = 0.0
            inv_sigma_jj = 1.0 / sigma_diag[j] if sigma_diag[j] > 0 else 0.0
            for h in range(horizon):
                A_l = ma[h]
                # e_i' · A_l · Σ · e_j = (A_l · Σ)[i, j]
                cross = float((A_l @ sigma)[i, j])
                numerator += cross**2
            fevd[i, j] = (
                inv_sigma_jj * numerator / denominator if denominator > 0 else 0.0
            )

    # Row-normalise so each row sums to 1.0 (generalized FEVD does NOT sum to
    # 1.0 by construction because shocks are correlated; DY-2012 §3 normalises
    # by row sum to recover share interpretation).
    row_sums = fevd.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return fevd / row_sums


def compute_spillover_index(
    returns: pd.DataFrame,
    lag: int = 4,
    horizon: int = 10,
) -> SpilloverResult:
    """Compute Diebold-Yilmaz (2012) Total Spillover Index + directional spillovers.

    Pipeline:
        1. Fit VAR(``lag``) on the input return panel.
        2. Compute the Pesaran-Shin generalized FEVD at horizon ``horizon``.
        3. Row-normalise to get variance shares (each row sums to 100%).
        4. TSI = (sum of off-diagonal) / (sum of all) · 100.
        5. Decompose into to_others / from_others / net per variable.

    Args:
        returns: (T, N) DataFrame of asset returns. Columns = variable names;
            rows = time. Must be stationary (returns, not prices).
        lag: VAR lag order ``p`` (default 4 per DY-2012 §4 daily-data choice).
        horizon: FEVD forecast horizon in periods (default 10 per DY-2012 §4).

    Returns:
        SpilloverResult with TSI, fevd matrix, and directional decomposition.

    Raises:
        ValueError: If returns has <3 columns or fewer than ``50 + lag``
            observations.
        ImportError: If statsmodels is not installed.
    """
    if returns.shape[1] < 2:
        raise ValueError(
            f"compute_spillover_index: need ≥2 variables for spillover, got {returns.shape[1]}"
        )
    if returns.shape[0] < 50 + lag:
        raise ValueError(
            f"compute_spillover_index: need ≥{50 + lag} obs, got {returns.shape[0]}"
        )
    if lag < 1:
        raise ValueError(f"lag must be ≥1, got {lag}")
    if horizon < 1:
        raise ValueError(f"horizon must be ≥1, got {horizon}")

    clean = returns.dropna()
    if clean.shape[0] < 50 + lag:
        raise ValueError(
            f"compute_spillover_index: ≥{50 + lag} non-NaN obs required, got {clean.shape[0]}"
        )

    from statsmodels.tsa.vector_ar.var_model import VAR

    model = VAR(clean.to_numpy())
    var_result = model.fit(maxlags=lag, ic=None)

    fevd_arr = _generalized_fevd(var_result, horizon=horizon)
    fevd_pct = fevd_arr * 100.0

    var_names = list(clean.columns)
    fevd_df = pd.DataFrame(fevd_pct, index=var_names, columns=var_names)

    # Total Spillover Index: sum of off-diagonal / sum of all
    diag = np.diag(fevd_pct)
    off_diag_sum = float(fevd_pct.sum() - diag.sum())
    total_sum = float(fevd_pct.sum())
    tsi = (off_diag_sum / total_sum) * 100.0 if total_sum > 0 else 0.0
    # TSI in [0, 100]; equivalently = mean off-diagonal share since rows sum
    # to ~100. The DY-2012 §3 formula multiplies by 100 again because shares
    # are already in [0, 100] — net result is same range.

    # Directional spillovers
    # to_others[j] = sum of column j MINUS diag (i.e. variance variable j
    # transmits to all OTHERS), normalised to %
    fevd_no_diag = fevd_pct.copy()
    np.fill_diagonal(fevd_no_diag, 0.0)
    to_others = pd.Series(
        fevd_no_diag.sum(axis=0), index=var_names, name="to_others_pct"
    )
    from_others = pd.Series(
        fevd_no_diag.sum(axis=1), index=var_names, name="from_others_pct"
    )
    net = (to_others - from_others).rename("net_pct")

    return SpilloverResult(
        total_spillover_index_pct=float(tsi),
        fevd_matrix=fevd_df,
        to_others=to_others,
        from_others=from_others,
        net=net,
        lag=lag,
        horizon=horizon,
        n_obs=int(clean.shape[0]),
    )


def rolling_spillover_index(
    returns: pd.DataFrame,
    window: int = 200,
    step: int = 5,
    lag: int = 4,
    horizon: int = 10,
) -> pd.DataFrame:
    """Rolling-window TSI time series — the canonical DY-2012 figure.

    Args:
        returns: (T, N) return panel.
        window: Rolling window length (default 200 trading days per DY §4).
        step: Step between window evaluations (default 5 — weekly cadence).
        lag: VAR lag for each window.
        horizon: FEVD horizon for each window.

    Returns:
        DataFrame with columns ``end_timestamp`` (right edge of window) and
        ``tsi_pct`` (Total Spillover Index for that window). Rows correspond
        to each successful window fit; failed windows are silently skipped
        (logged at DEBUG).

    Raises:
        ValueError: If window < 50 + lag or step < 1.
    """
    if window < 50 + lag:
        raise ValueError(
            f"rolling_spillover_index: window must be ≥{50 + lag}, got {window}"
        )
    if step < 1:
        raise ValueError(f"step must be ≥1, got {step}")

    rows = []
    t = returns.index if isinstance(returns.index, pd.DatetimeIndex) else None
    for end_idx in range(window, len(returns) + 1, step):
        chunk = returns.iloc[end_idx - window : end_idx]
        try:
            result = compute_spillover_index(chunk, lag=lag, horizon=horizon)
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.debug(
                "rolling_spillover_index: window ending at row %d failed: %s",
                end_idx,
                exc,
            )
            continue
        rows.append(
            {
                "end_timestamp": t[end_idx - 1] if t is not None else end_idx - 1,
                "tsi_pct": result.total_spillover_index_pct,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "SpilloverResult",
    "compute_spillover_index",
    "rolling_spillover_index",
]
