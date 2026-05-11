"""Nelson-Siegel Yield-Curve-Faktoren (Nelson/Siegel 1987; Diebold/Li 2006).

Theorie
-------
Yield-Curve y(τ) (Yield vs Maturity τ in years) wird approximiert durch 3
latente Faktoren:

    y(τ) = β_0 + β_1 · (1 − exp(−λτ)) / (λτ)
         + β_2 · [(1 − exp(−λτ)) / (λτ) − exp(−λτ)]

mit
- β_0 = **Level** (long-run)
- β_1 = **Slope** (short-rate spread vs long)
- β_2 = **Curvature** (mid-maturity hump)
- λ = decay-Parameter (üblich 0.0609 für τ in years; defines hump position).

Anwendung
---------
- Faktor-Modell für Bond-Portfolios.
- Forecasting: Diebold/Li (2006) zeigen AR(1) auf NS-Faktoren prognostiziert Yields.
- Spread-Trading: Slope-Anomaly-Signals.
- Recession-Prediction: Inverter Slope = Recession-Indikator.

Reference
---------
- Nelson, C. & Siegel, A. (1987). Parsimonious Modeling of Yield Curves. *JoB* 60.
- Diebold, F. & Li, C. (2006). Forecasting the term structure of government
  bond yields. *J. Econometrics* 130.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def nelson_siegel_basis(tau: np.ndarray, lam: float = 0.0609) -> np.ndarray:
    """Return basis-matrix B for NS-decomposition.

    Args:
        tau: maturity array (years).
        lam: decay parameter.

    Returns:
        Array (n_maturities, 3) — columns [level, slope, curvature].
    """
    t = np.asarray(tau, dtype=float)
    lam_t = lam * t
    # Avoid division by zero at tau=0
    eps = 1e-9
    safe_lam_t = np.where(lam_t == 0, eps, lam_t)
    slope = (1 - np.exp(-safe_lam_t)) / safe_lam_t
    curv = slope - np.exp(-safe_lam_t)
    level = np.ones_like(t)
    return np.column_stack([level, slope, curv])


def fit_nelson_siegel(yields: np.ndarray, tau: np.ndarray, lam: float = 0.0609) -> dict:
    """OLS fit of NS to a single yield-curve.

    Args:
        yields: array of yields (same length as tau).
        tau: maturities.
        lam: decay parameter.

    Returns:
        dict mit β_0, β_1, β_2, fit-residuals, R².
    """
    y = np.asarray(yields, dtype=float)
    if len(y) < 3:
        raise ValueError("need at least 3 maturity points")
    X = nelson_siegel_basis(tau, lam)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "level": float(beta[0]),
        "slope": float(beta[1]),  # Negativ = upward-sloping (short < long)
        "curvature": float(beta[2]),
        "residuals": (y - pred).tolist(),
        "r_squared": r2,
    }


def fit_panel_ns(
    yield_curves: pd.DataFrame, tau: np.ndarray, lam: float = 0.0609
) -> pd.DataFrame:
    """Fit NS-factors for each row of a yield-curve panel.

    Args:
        yield_curves: DataFrame indexed by date, columns = maturities (same length as tau).
        tau: array of maturities.
        lam: decay.

    Returns:
        DataFrame indexed by date with columns [level, slope, curvature, r_squared].
    """
    out_rows = []
    for d, row in yield_curves.iterrows():
        y = row.values.astype(float)
        if np.isnan(y).any():
            continue
        try:
            fit = fit_nelson_siegel(y, tau, lam)
            out_rows.append(
                {
                    "date": d,
                    "level": fit["level"],
                    "slope": fit["slope"],
                    "curvature": fit["curvature"],
                    "r_squared": fit["r_squared"],
                }
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
    return pd.DataFrame(out_rows).set_index("date")


def yield_curve_inversion_signal(
    ns_factors: pd.DataFrame, slope_col: str = "slope"
) -> pd.Series:
    """Inversion-Indicator: slope < 0 = inverted curve = recession signal.

    Reference: Estrella/Mishkin (1998).
    """
    return (ns_factors[slope_col] < 0).astype(int)


def forecast_yields_ar1(
    ns_factors: pd.DataFrame, tau: np.ndarray, h: int = 1, lam: float = 0.0609
) -> pd.DataFrame:
    """Diebold-Li-Style AR(1)-Forecast auf NS-Faktoren + Rekonstruktion.

    Args:
        ns_factors: from fit_panel_ns.
        tau: target maturities.
        h: forecast horizon (in periods).
        lam: decay.

    Returns:
        DataFrame mit forecasted yields per maturity.
    """
    if ns_factors.empty:
        return pd.DataFrame()
    forecasts = {}
    for col in ("level", "slope", "curvature"):
        s = ns_factors[col].dropna()
        if len(s) < 30:
            forecasts[col] = float(s.mean()) if not s.empty else 0.0
            continue
        # Fit AR(1): x_t = c + φ x_{t-1} + ε
        x = s.values[1:]
        x_lag = s.values[:-1]
        X = np.column_stack([np.ones(len(x_lag)), x_lag])
        coef, *_ = np.linalg.lstsq(X, x, rcond=None)
        c, phi = float(coef[0]), float(coef[1])
        last = float(s.iloc[-1])
        f = last
        for _ in range(h):
            f = c + phi * f
        forecasts[col] = f
    # Reconstruct yields at target tau
    B = nelson_siegel_basis(tau, lam)
    factors = np.array([forecasts["level"], forecasts["slope"], forecasts["curvature"]])
    pred_yields = B @ factors
    return pd.DataFrame({"tau": tau, "forecast_yield": pred_yields})


__all__ = [
    "nelson_siegel_basis",
    "fit_nelson_siegel",
    "fit_panel_ns",
    "yield_curve_inversion_signal",
    "forecast_yields_ar1",
]
