"""Vector-Autoregression (VAR) — multivariate Time-Series-Modell.

Theorie
-------
VAR(p): y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + ε_t,
mit y_t ∈ R^n und ε_t ~ N(0, Σ).

Anwendungen
-----------
- Lag-Selection für Lead-Lag-Strategien
- Impulse-Response-Functions: Wie beeinflusst Schock in Asset A Asset B über h Perioden?
- Granger-Causality (multivariat)
- Forecasting Multi-Asset-Returns

Reference
---------
Hamilton, J. (1994). *Time Series Analysis*, Chapter 11. Princeton.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VARFit:
    coefficients: np.ndarray  # shape (n_vars, n_vars * p + 1)
    sigma: np.ndarray  # residual covariance (n_vars, n_vars)
    n_vars: int
    p: int
    aic: float
    bic: float
    n_obs: int


def fit_var(data: pd.DataFrame, p: int = 1) -> VARFit:
    """Fit VAR(p) by stacked OLS.

    Args:
        data: DataFrame (T, n_vars).
        p: lag order.

    Returns:
        VARFit.
    """
    Y = data.dropna(how="any").values
    T, n = Y.shape
    if T <= p + 5:
        raise ValueError(f"need T > {p + 5}")

    # Build lagged matrix
    X = np.column_stack([np.ones(T - p)] + [Y[p - i - 1 : T - i - 1] for i in range(p)])
    Y_target = Y[p:]
    # OLS
    XtX = X.T @ X
    coef = np.linalg.solve(XtX, X.T @ Y_target).T  # (n_vars, 1 + p*n_vars)
    resid = Y_target - X @ coef.T
    sigma = (resid.T @ resid) / (T - p - X.shape[1])

    # AIC/BIC
    sign, logdet = np.linalg.slogdet(sigma)
    if sign <= 0:
        log_det_sigma = -np.inf
    else:
        log_det_sigma = float(logdet)
    k_params = n * X.shape[1]
    aic = float(log_det_sigma + 2 * k_params / T)
    bic = float(log_det_sigma + np.log(T) * k_params / T)

    return VARFit(
        coefficients=coef,
        sigma=sigma,
        n_vars=n,
        p=p,
        aic=aic,
        bic=bic,
        n_obs=T - p,
    )


def select_lag_order(data: pd.DataFrame, max_p: int = 8) -> dict:
    """Select lag order via BIC.

    Returns dict with bic-table, best_p.
    """
    results = []
    for p in range(1, max_p + 1):
        try:
            fit = fit_var(data, p=p)
            results.append({"p": p, "aic": fit.aic, "bic": fit.bic})
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not results:
        return {"error": "no valid fits"}
    df = pd.DataFrame(results)
    return {"table": df, "best_p_bic": int(df.loc[df["bic"].idxmin(), "p"])}


def granger_causality_var(
    data: pd.DataFrame,
    cause: str,
    effect: str,
    p: int = 1,
) -> dict:
    """F-Test für Granger-Causality: does `cause` Granger-cause `effect`?

    Restricted Model: effect ~ lags(effect)
    Unrestricted:    effect ~ lags(effect) + lags(cause)
    """
    if cause not in data.columns or effect not in data.columns:
        return {"error": "columns missing"}

    Y = data[[effect]].values.flatten()
    T = len(Y)
    if T <= p + 5:
        return {"error": "too few obs"}

    # Restricted (no `cause`)
    X_r = np.column_stack(
        [np.ones(T - p)] + [Y[p - i - 1 : T - i - 1] for i in range(p)]
    )
    y_target = Y[p:]
    coef_r, *_ = np.linalg.lstsq(X_r, y_target, rcond=None)
    rss_r = float(((y_target - X_r @ coef_r) ** 2).sum())

    # Unrestricted (with `cause`)
    C = data[cause].values
    X_u = np.column_stack(
        [np.ones(T - p)]
        + [Y[p - i - 1 : T - i - 1] for i in range(p)]
        + [C[p - i - 1 : T - i - 1] for i in range(p)]
    )
    coef_u, *_ = np.linalg.lstsq(X_u, y_target, rcond=None)
    rss_u = float(((y_target - X_u @ coef_u) ** 2).sum())

    if rss_u <= 0:
        return {"error": "rss_u zero"}
    df_num = p
    df_den = (T - p) - (2 * p + 1)
    if df_den <= 0:
        return {"error": "too few residual dof"}
    F = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
    try:
        from scipy.stats import f as f_dist  # type: ignore

        p_val = 1 - f_dist.cdf(F, df_num, df_den)
    except ImportError:
        p_val = float("nan")
    return {
        "F": float(F),
        "p_value": float(p_val) if p_val is not None else None,
        "df_num": df_num,
        "df_den": df_den,
    }


def impulse_response(fit: VARFit, horizon: int = 10) -> np.ndarray:
    """Orthogonalized impulse-response-function (Cholesky-Identification).

    Returns:
        Array (horizon+1, n_vars, n_vars) — IRF[h, j, i] = response of variable j to shock in i at horizon h.
    """
    n = fit.n_vars
    # Stack VAR(p) -> VAR(1) companion form
    p = fit.p
    A = np.zeros((n * p, n * p))
    A[:n, :] = fit.coefficients[:, 1:]  # skip intercept
    if p > 1:
        A[n:, :-n] = np.eye(n * (p - 1))
    # Cholesky of Σ
    try:
        L = np.linalg.cholesky(fit.sigma)
    except np.linalg.LinAlgError:
        L = np.eye(n)

    irf = np.zeros((horizon + 1, n, n))
    irf[0] = L
    A_power = np.eye(n * p)
    for h in range(1, horizon + 1):
        A_power = A_power @ A
        # impact_h = A_power[:n, :n] @ L
        irf[h] = A_power[:n, :n] @ L
    return irf


__all__ = [
    "VARFit",
    "fit_var",
    "select_lag_order",
    "granger_causality_var",
    "impulse_response",
]
