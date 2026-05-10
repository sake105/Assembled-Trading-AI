"""Log-Periodic Power-Law (LPPL) Bubble-Detector (Sornette).

Theorie
-------
Sornette und Mitarbeiter argumentieren: speculative bubbles haben charakteristische
Log-Periodic-Power-Law-Signatur:

    log(p_t) = A + B (t_c - t)^β + C (t_c - t)^β cos(ω log(t_c - t) - φ)

mit
- t_c = critical time (Crash-Datum)
- β ∈ (0, 1) = power-law exponent
- ω = log-periodic frequency
- B < 0 für bubble (price ↑)

Anwendung
---------
- "Late-Stage Bubble" Detection: wenn LPPL gut fittet auf recent prices, ist
  Markt nahe am singularity-Punkt t_c.
- Risk-Reduction-Trigger bei diagnostiziertem Bubble.

Reference
---------
- Sornette, D. (2003). *Why Stock Markets Crash*. Princeton.
- Johansen, A., Ledoit, O. & Sornette, D. (2000). Crashes as Critical Points.
  *Int. J. Theor. Appl. Finance* 3.

Implementation
--------------
Vereinfachter Fit: Grid-Search über t_c, dann OLS auf nicht-linearisierte Form.
Voll-MLE würde Levenberg-Marquardt benötigen — hier didaktisch kompakt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LPPLFit:
    t_c: float
    beta: float
    omega: float
    phi: float
    A: float
    B: float
    C: float
    rmse: float
    days_to_critical: int  # t_c relative to last observation


def _lppl_log_price(
    t: np.ndarray,
    t_c: float,
    A: float,
    B: float,
    C: float,
    beta: float,
    omega: float,
    phi: float,
) -> np.ndarray:
    tau = t_c - t
    tau = np.where(tau > 0, tau, 1e-9)
    return A + B * tau**beta + C * tau**beta * np.cos(omega * np.log(tau) - phi)


def fit_lppl(
    log_prices: pd.Series,
    t_c_grid: np.ndarray | None = None,
    omega_grid: np.ndarray | None = None,
    beta_grid: np.ndarray | None = None,
) -> LPPLFit:
    """Fit LPPL via Grid-Search.

    Args:
        log_prices: Series of log-prices.
        t_c_grid, omega_grid, beta_grid: Such-Grids. Default = sinnvoller Range.

    Returns:
        LPPLFit mit besten Parametern.
    """
    s = pd.Series(log_prices).dropna()
    n = len(s)
    if n < 50:
        raise ValueError("need >= 50 obs")
    t = np.arange(n, dtype=float)
    y = s.values

    if t_c_grid is None:
        # Search t_c after last obs (i.e., critical point in future)
        t_c_grid = np.linspace(n + 1, n + n // 2, 20)
    if omega_grid is None:
        omega_grid = np.linspace(5, 15, 11)
    if beta_grid is None:
        beta_grid = np.linspace(0.1, 0.9, 9)

    best_rmse = np.inf
    best_params = None

    for t_c in t_c_grid:
        tau = t_c - t
        if (tau <= 0).any():
            continue
        for beta in beta_grid:
            f1_beta = tau**beta
            for omega in omega_grid:
                log_tau = np.log(tau)
                f_cos = f1_beta * np.cos(omega * log_tau)
                f_sin = f1_beta * np.sin(omega * log_tau)
                # Linear regression: y = A + B·f1_beta + C1·f_cos + C2·f_sin
                X = np.column_stack([np.ones(n), f1_beta, f_cos, f_sin])
                try:
                    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                pred = X @ coef
                rmse = float(np.sqrt(((y - pred) ** 2).mean()))
                if rmse < best_rmse:
                    A, B, C1, C2 = coef
                    C_amp = float(np.sqrt(C1**2 + C2**2))
                    phi = float(np.arctan2(C2, C1))
                    best_rmse = rmse
                    best_params = (t_c, beta, omega, phi, float(A), float(B), C_amp)

    if best_params is None:
        raise RuntimeError("LPPL fit failed")
    t_c, beta, omega, phi, A, B, C = best_params
    return LPPLFit(
        t_c=t_c,
        beta=beta,
        omega=omega,
        phi=phi,
        A=A,
        B=B,
        C=C,
        rmse=best_rmse,
        days_to_critical=int(round(t_c - n)),
    )


def bubble_likelihood_score(
    log_prices: pd.Series,
    benchmark_rmse_quantile: float = 0.5,
) -> dict:
    """Bubble-Likelihood-Score (heuristisch).

    Hoch wenn:
    - B < 0 (positive trend, super-exponential growth)
    - β ∈ (0.1, 0.9)
    - RMSE niedrig vs. random-walk benchmark

    Returns:
        dict mit score, fit-params, evidence.
    """
    try:
        fit = fit_lppl(log_prices)
    except RuntimeError:
        return {"score": 0.0, "evidence": "no_fit"}

    # Score = combination of indicators
    score_components = []
    # 1. negative B (super-exp growth)
    if fit.B < 0:
        score_components.append(1.0)
    else:
        score_components.append(0.0)
    # 2. beta in valid range
    if 0.1 < fit.beta < 0.9:
        score_components.append(1.0)
    else:
        score_components.append(0.3)
    # 3. days_to_critical reasonable (<100 days = imminent)
    if 5 < fit.days_to_critical < 200:
        score_components.append(0.8)
    else:
        score_components.append(0.3)
    # 4. RMSE — compare to log-price std
    relative_rmse = fit.rmse / float(np.std(log_prices.dropna()))
    if relative_rmse < 0.1:
        score_components.append(1.0)
    else:
        score_components.append(max(0.0, 1 - relative_rmse * 2))

    score = float(np.mean(score_components))
    return {
        "score": score,
        "fit": fit,
        "components": score_components,
        "relative_rmse": float(relative_rmse),
    }


__all__ = ["LPPLFit", "fit_lppl", "bubble_likelihood_score"]
