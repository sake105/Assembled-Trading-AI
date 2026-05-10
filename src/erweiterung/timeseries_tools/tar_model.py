"""Threshold-Autoregression (TAR) — Self-Exciting (SETAR) Model.

Reference
---------
Tong, H. (1990). *Non-Linear Time Series: A Dynamical System Approach*. Oxford.

Theorie
-------
Linear-AR-Modelle nehmen konstante Parameter an. **SETAR(p; d, r)** lässt diese
abhängig von einem Threshold r in der eigenen Lag-d-Variable variieren:

    y_t = φ_0^{(1)} + φ_1^{(1)} y_{t-1} + ... + φ_p^{(1)} y_{t-p} + ε_t^{(1)},  if y_{t-d} ≤ r
        = φ_0^{(2)} + φ_1^{(2)} y_{t-1} + ... + φ_p^{(2)} y_{t-p} + ε_t^{(2)},  if y_{t-d} > r

Anwendung
---------
- Asymmetrische Reaktion auf positive vs. negative Returns
- Bull/Bear-Regime-Switching
- Volatility-Asymmetrie ohne explicit MS-GARCH
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SETARFit:
    threshold: float
    delay: int
    p: int
    phi_regime1: np.ndarray  # below threshold
    phi_regime2: np.ndarray  # above threshold
    rss: float
    n_obs_r1: int
    n_obs_r2: int


def fit_setar(
    series: pd.Series,
    p: int = 1,
    delay: int = 1,
    threshold_grid: np.ndarray | None = None,
) -> SETARFit:
    """Fit SETAR(p; delay, r) via grid-search over threshold r.

    Args:
        series: 1-D series.
        p: AR-order.
        delay: lag used for threshold variable.
        threshold_grid: candidates for r. Default = quantiles of series.

    Returns:
        SETARFit.
    """
    s = pd.Series(series).dropna().values.astype(float)
    n = len(s)
    if n < max(p, delay) + 30:
        raise ValueError("not enough obs")

    y = s[max(p, delay) :]
    X = np.column_stack(
        [np.ones(len(y))] + [s[max(p, delay) - i - 1 : n - i - 1] for i in range(p)]
    )
    thresh_var = s[max(p, delay) - delay : n - delay]

    if threshold_grid is None:
        threshold_grid = np.quantile(thresh_var, [0.1, 0.25, 0.5, 0.75, 0.9])

    best_rss = np.inf
    best_fit = None

    for r in threshold_grid:
        mask = thresh_var <= r
        if mask.sum() < p + 3 or (~mask).sum() < p + 3:
            continue
        X1, y1 = X[mask], y[mask]
        X2, y2 = X[~mask], y[~mask]
        phi1, *_ = np.linalg.lstsq(X1, y1, rcond=None)
        phi2, *_ = np.linalg.lstsq(X2, y2, rcond=None)
        rss = float(((y1 - X1 @ phi1) ** 2).sum() + ((y2 - X2 @ phi2) ** 2).sum())
        if rss < best_rss:
            best_rss = rss
            best_fit = SETARFit(
                threshold=float(r),
                delay=delay,
                p=p,
                phi_regime1=phi1,
                phi_regime2=phi2,
                rss=rss,
                n_obs_r1=int(mask.sum()),
                n_obs_r2=int((~mask).sum()),
            )
    if best_fit is None:
        raise RuntimeError("SETAR fit failed")
    return best_fit


def setar_forecast(fit: SETARFit, history: pd.Series) -> float:
    """One-step-ahead forecast based on current regime."""
    s = pd.Series(history).dropna().values
    if len(s) < max(fit.p, fit.delay):
        return float("nan")
    threshold_var = s[-fit.delay]
    if threshold_var <= fit.threshold:
        phi = fit.phi_regime1
    else:
        phi = fit.phi_regime2
    lags = [1.0] + [s[-i - 1] for i in range(fit.p)]
    return float(np.array(lags) @ phi)


def linearity_test_tsay(series: pd.Series, p: int = 1, delay: int = 1) -> dict:
    """Tsay 1989 test for linearity vs SETAR.

    Returns F-stat and p-value (rough; large F => reject linearity).
    """
    s = pd.Series(series).dropna().values.astype(float)
    n = len(s)
    if n < max(p, delay) + 30:
        return {"error": "too few obs"}
    y = s[max(p, delay) :]
    X = np.column_stack(
        [np.ones(len(y))] + [s[max(p, delay) - i - 1 : n - i - 1] for i in range(p)]
    )
    # Sort by threshold variable
    thresh_var = s[max(p, delay) - delay : n - delay]
    order = np.argsort(thresh_var)
    X_o = X[order]
    y_o = y[order]
    # Rolling residuals via cumulative OLS
    n_obs = len(y_o)
    # Linear model RSS
    beta_lin, *_ = np.linalg.lstsq(X_o, y_o, rcond=None)
    rss_lin = float(((y_o - X_o @ beta_lin) ** 2).sum())
    # SETAR fit
    try:
        setar = fit_setar(series, p=p, delay=delay)
        rss_setar = setar.rss
    except RuntimeError:
        return {"error": "setar fit failed"}
    df_num = p + 1
    df_den = n_obs - 2 * (p + 1)
    if df_den <= 0 or rss_setar <= 0:
        return {"error": "bad dof"}
    F = ((rss_lin - rss_setar) / df_num) / (rss_setar / df_den)
    try:
        from scipy.stats import f as f_dist  # type: ignore

        p_val = 1 - f_dist.cdf(F, df_num, df_den)
    except ImportError:
        p_val = float("nan")
    return {"F": float(F), "p_value": float(p_val) if p_val is not None else None}


__all__ = ["SETARFit", "fit_setar", "setar_forecast", "linearity_test_tsay"]
