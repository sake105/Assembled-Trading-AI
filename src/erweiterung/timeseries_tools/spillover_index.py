"""Diebold-Yilmaz Spillover Index (2012).

Theorie
-------
Forecast-Error-Variance-Decomposition (FEVD) eines VAR-Modells liefert je
Asset eine Decomposition: Wie viel der h-Schritt-Forecast-Varianz von Asset i
wird durch Schocks in Asset j (j ≠ i) erklärt?

Spillover-Indices:
- **Total Spillover**: Σ_{i≠j} θ^g_ij(h) / Σ_all θ^g_ij(h) × 100%
- **Net Pairwise**: Anteil i→j minus j→i
- **Directional From/To**: Inflow vs Outflow je Asset

Reference
---------
- Diebold, F. & Yilmaz, K. (2012). Better to give than to receive: Predictive
  directional measurement of volatility spillovers. *Int. J. Forecasting* 28.

Anwendung
---------
- Welche Märkte sind aktuell Spillover-Quellen vs. -Senken?
- Crisis-Detection: Total-Spillover-Spike = system-weite Krise.
- Lead-Lag-Asset-Selection: "From-Spillover" hoch = Lead-Asset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.timeseries_tools.var_model import fit_var


def generalized_fevd(var_fit, horizon: int = 10) -> np.ndarray:
    """Generalized Forecast-Error-Variance-Decomposition (Pesaran/Shin 1998).

    Im Unterschied zu orthogonalized FEVD: ordnungs-invariant.

    Args:
        var_fit: VARFit aus timeseries_tools.var_model.
        horizon: forecast-horizon.

    Returns:
        Array (n_vars, n_vars) — θ^g_ij(h) = share of i's H-step FEV explained by j.
    """
    n = var_fit.n_vars
    Sigma = var_fit.sigma  # residual covariance
    p = var_fit.p
    # Companion form A
    A = np.zeros((n * p, n * p))
    A[:n, :] = var_fit.coefficients[:, 1:]
    if p > 1:
        A[n:, :-n] = np.eye(n * (p - 1))
    # Phi_h = J A^h J', J = [I_n, 0, ..., 0]
    J = np.zeros((n, n * p))
    J[:, :n] = np.eye(n)

    sigma_diag = np.diag(Sigma)
    theta = np.zeros((n, n))
    cum_num = np.zeros((n, n))
    cum_den = np.zeros(n)

    A_power = np.eye(n * p)
    for _h in range(horizon):
        Phi = J @ A_power @ J.T  # (n, n)
        prod = Phi @ Sigma
        # Numerator: (e_i' Phi Sigma e_j)^2 / sigma_jj
        for i in range(n):
            for j in range(n):
                num = (prod[i, j]) ** 2
                if sigma_diag[j] > 0:
                    cum_num[i, j] += num / sigma_diag[j]
            cum_den[i] += float(Phi[i] @ Sigma @ Phi[i].T)
        A_power = A_power @ A

    for i in range(n):
        if cum_den[i] > 0:
            theta[i] = cum_num[i] / cum_den[i]
    # Row-normalize so each row sums to 1
    row_sums = theta.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return theta / row_sums


def spillover_indices(returns: pd.DataFrame, p: int = 1, horizon: int = 10) -> dict:
    """Compute Diebold-Yilmaz Spillover indices.

    Args:
        returns: DataFrame (T, N).
        p: VAR lag order.
        horizon: forecast horizon for FEVD.

    Returns:
        dict mit:
        - ``total_spillover``: scalar %.
        - ``from_spillover``: Series je Asset (% from others to this).
        - ``to_spillover``: Series je Asset (% from this to others).
        - ``net_spillover``: to - from (positive = net transmitter).
        - ``pairwise_matrix``: DataFrame N×N FEVD-Shares.
    """
    fit = fit_var(returns, p=p)
    theta = generalized_fevd(fit, horizon=horizon)
    n = theta.shape[0]

    pairs_df = pd.DataFrame(theta, index=returns.columns, columns=returns.columns)
    # Total Spillover Index = (Σ_{i≠j} θ_ij) / Σ θ × 100
    off_diag = theta - np.diag(np.diag(theta))
    total = float(off_diag.sum() / theta.sum() * 100)

    from_spillover = pd.Series(
        off_diag.sum(axis=1) / theta.sum(axis=1) * 100,
        index=returns.columns,
    )
    to_spillover = pd.Series(
        off_diag.sum(axis=0) / theta.sum(axis=1).sum() * n * 100,
        index=returns.columns,
    )
    net = to_spillover - from_spillover

    return {
        "total_spillover_pct": total,
        "from_spillover": from_spillover,
        "to_spillover": to_spillover,
        "net_spillover": net,
        "pairwise_matrix": pairs_df,
    }


def rolling_total_spillover(
    returns: pd.DataFrame,
    window: int = 100,
    p: int = 1,
    horizon: int = 10,
    step: int = 5,
) -> pd.Series:
    """Rolling Total Spillover Index — Crisis-Detection-Indikator."""
    out = []
    indices = []
    for end in range(window, len(returns) + 1, step):
        sub = returns.iloc[end - window : end]
        try:
            si = spillover_indices(sub, p=p, horizon=horizon)
            out.append(si["total_spillover_pct"])
            indices.append(returns.index[end - 1])
        except Exception:  # noqa: BLE001
            continue
    return pd.Series(out, index=indices, name="total_spillover_pct")


__all__ = ["generalized_fevd", "spillover_indices", "rolling_total_spillover"]
