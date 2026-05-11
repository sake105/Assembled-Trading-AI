"""Two-Scale Realized Variance (Zhang/Mykland/Aït-Sahalia 2005).

Problem mit naiver Realized Variance
------------------------------------
RV_naïve = Σ r_i² (sum of squared intraday returns).
**Problem**: bei sehr hoher Frequenz dominiert Microstructure-Noise → RV ist
**biased upward** und proportional zur Sampling-Frequenz.

Lösung Two-Scale
----------------
Split returns into two scales (fine + sparse):
    RV_fine = Σ r_i² (z. B. 1-min)
    RV_sparse = Σ r_j² (z. B. 30-min, J = T/K subsamples)

Estimator:
    TS_RV = RV_sparse_avg − (n_fine / n_sparse) × E[noise²]

Mit E[noise²] aus RV_fine - true_RV. Optimaler Bias-Variance-Tradeoff.

Reference
---------
Zhang, L., Mykland, P. & Aït-Sahalia, Y. (2005). A Tale of Two Time Scales.
*JASA* 100.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TSRVEstimate:
    rv_sparse: float
    rv_fine: float
    noise_variance: float
    tsrv: float
    n_fine: int
    n_sparse: int


def two_scale_realized_variance(
    intraday_prices: pd.Series,
    sparse_step: int = 30,
) -> TSRVEstimate:
    """Compute TS-RV from a single-day intraday price-series.

    Args:
        intraday_prices: pd.Series of intraday prices (e.g. 1-min bars).
        sparse_step: subsample-step (e.g. 30 means every 30th observation).

    Returns:
        TSRVEstimate.
    """
    p = pd.Series(intraday_prices).dropna()
    n = len(p)
    if n < sparse_step * 4:
        raise ValueError(f"need >= {sparse_step * 4} observations")
    log_p = np.log(p.values)
    fine_returns = np.diff(log_p)
    n_fine = len(fine_returns)
    rv_fine = float(np.sum(fine_returns**2))

    # Average over sparse subsamples (K = sparse_step)
    sparse_rvs = []
    for offset in range(sparse_step):
        sub = log_p[offset::sparse_step]
        if len(sub) < 2:
            continue
        sub_returns = np.diff(sub)
        sparse_rvs.append(float(np.sum(sub_returns**2)))
    if not sparse_rvs:
        raise ValueError("no sparse subsamples")
    rv_sparse = float(np.mean(sparse_rvs))
    n_sparse = int(
        np.mean([len(log_p[off::sparse_step]) - 1 for off in range(sparse_step)])
    )

    # Noise variance estimate: E[noise²] ≈ RV_fine / (2 × n_fine) (Zhang formulation)
    noise_var = rv_fine / (2 * n_fine) if n_fine > 0 else 0.0
    # Bias correction
    tsrv = rv_sparse - (2 * n_fine / sparse_step) * noise_var
    tsrv = max(tsrv, 0.0)

    return TSRVEstimate(
        rv_sparse=rv_sparse,
        rv_fine=rv_fine,
        noise_variance=noise_var,
        tsrv=tsrv,
        n_fine=n_fine,
        n_sparse=n_sparse,
    )


def realized_kernel_variance(intraday_returns: np.ndarray, bandwidth: int = 5) -> float:
    """Barndorff-Nielsen/Hansen/Lunde/Shephard Realized-Kernel-Estimator.

    Uses Parzen-kernel weights to mitigate microstructure noise.

    Args:
        intraday_returns: array of high-freq returns.
        bandwidth: kernel-bandwidth (rule-of-thumb ~5-10).

    Returns:
        Realized kernel variance.
    """
    r = np.asarray(intraday_returns, dtype=float)
    if len(r) == 0:
        return float("nan")
    rv = float(np.sum(r**2))
    # Autocovariance-corrections via Parzen-kernel
    n = len(r)
    H = min(bandwidth, n - 1)
    for h in range(1, H + 1):
        x = h / H
        if x <= 0.5:
            k = 1 - 6 * x * x + 6 * x**3
        else:
            k = 2 * (1 - x) ** 3
        gamma_h = float(np.sum(r[h:] * r[:-h]))
        rv += 2 * k * gamma_h
    return max(rv, 0.0)


def rolling_intraday_volatility_panel(
    intraday_prices: pd.DataFrame,
    sparse_step: int = 30,
) -> pd.Series:
    """Apply TS-RV per day across a multi-day intraday panel.

    Args:
        intraday_prices: DataFrame indexed by datetime, single column = price.

    Returns:
        Series of daily TS-RV indexed by date.
    """
    s = (
        intraday_prices.iloc[:, 0]
        if isinstance(intraday_prices, pd.DataFrame)
        else intraday_prices
    )
    s = pd.Series(s).dropna()
    s.index = pd.to_datetime(s.index)
    out = []
    for d, group in s.groupby(s.index.normalize()):
        if len(group) < sparse_step * 4:
            continue
        try:
            ts = two_scale_realized_variance(group, sparse_step=sparse_step)
            out.append({"date": d, "tsrv": ts.tsrv})
        except ValueError:
            continue
    df = pd.DataFrame(out)
    return df.set_index("date")["tsrv"] if not df.empty else pd.Series(dtype=float)


__all__ = [
    "TSRVEstimate",
    "two_scale_realized_variance",
    "realized_kernel_variance",
    "rolling_intraday_volatility_panel",
]
