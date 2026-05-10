"""CVaR Portfolio-Optimization (Rockafellar/Uryasev 2000).

Theorie
-------
Conditional Value-at-Risk (CVaR_α, auch ES) ist der erwartete Verlust **bedingt
darauf, dass VaR_α überschritten wurde**:
    CVaR_α(L) = E[L | L >= VaR_α(L)]

Properties:
- Coherent risk measure (im Gegensatz zu VaR).
- Konvex => mit LP lösbar.
- Erfasst Tail-Loss explicit.

LP-Formulierung (Rockafellar-Uryasev)
--------------------------------------
Mit Loss-Szenarien L_s (s = 1..S) und Gewichten w:
    min  α + (1 / ((1-confidence) S)) Σ_s u_s
    s.t. u_s >= L_s(w) - α   ∀ s
         u_s >= 0
         + Portfolio-Constraints

Implementation
--------------
Wir verwenden ``scipy.optimize.linprog`` (oder cvxpy falls vorhanden).
Falls keines verfügbar: fallback auf simulated-annealing-Heuristik.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def cvar_optimal_weights(
    return_scenarios: pd.DataFrame,
    confidence: float = 0.95,
    target_return: Optional[float] = None,
    long_only: bool = True,
    max_weight: float = 0.20,
) -> tuple[pd.Series, dict]:
    """Solve CVaR-Optimization via LP.

    Args:
        return_scenarios: DataFrame (S × N) mit Return-Szenarien (z. B. historisch
            oder Monte-Carlo).
        confidence: 0.95 = 95% CVaR (entspricht 5% worst tail).
        target_return: Wenn gesetzt, Constraint w·μ >= target_return.
        long_only: Wenn True, w >= 0.
        max_weight: Cap pro Asset.

    Returns:
        (weights_series, metrics_dict).

    metrics_dict enthält
    --------------------
    - cvar: optimaler CVaR-Wert (positiv = Verlust)
    - var: zugehöriger VaR
    - expected_return: erwarteter Portfolio-Return
    - status: solver status string
    """
    try:
        from scipy.optimize import linprog  # type: ignore
    except ImportError:
        return _cvar_fallback_simulated_annealing(
            return_scenarios, confidence, target_return, long_only, max_weight
        )

    R = return_scenarios.values  # (S, N) Returns
    S, N = R.shape
    L = -R  # Loss = -Return (S, N)
    mu = R.mean(axis=0)

    # Variables: w (N), alpha (1), u (S). Total dim = N + 1 + S.
    n_var = N + 1 + S
    # Objective: min alpha + (1/((1-confidence)*S)) sum(u_s)
    c = np.zeros(n_var)
    c[N] = 1.0  # alpha
    inv = 1.0 / ((1 - confidence) * S)
    c[N + 1 :] = inv

    # Inequality constraints:
    # u_s >= L_s @ w - alpha  =>  L_s @ w - alpha - u_s <= 0
    # In linprog form: A_ub @ x <= b_ub
    A_ub = np.zeros((S, n_var))
    A_ub[:, :N] = L  # L_s @ w
    A_ub[:, N] = -1.0  # -alpha
    A_ub[:, N + 1 :] = -np.eye(S)  # -u_s
    b_ub = np.zeros(S)

    # Sum constraint: sum(w) = 1
    A_eq = np.zeros((1, n_var))
    A_eq[0, :N] = 1.0
    b_eq = np.array([1.0])

    # Bounds
    if long_only:
        bounds_w = [(0.0, max_weight)] * N
    else:
        bounds_w = [(-max_weight, max_weight)] * N
    bounds = bounds_w + [(None, None)] + [(0, None)] * S

    # Optional return constraint
    if target_return is not None:
        # mu @ w >= target  =>  -mu @ w <= -target
        ret_row = np.zeros(n_var)
        ret_row[:N] = -mu
        A_ub = np.vstack([A_ub, ret_row])
        b_ub = np.concatenate([b_ub, [-target_return]])

    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        return _cvar_fallback_simulated_annealing(
            return_scenarios, confidence, target_return, long_only, max_weight
        )
    w = res.x[:N]
    alpha = float(res.x[N])
    u = res.x[N + 1 :]
    cvar_val = float(alpha + inv * u.sum())
    var_val = alpha
    exp_ret = float(mu @ w)

    weights = pd.Series(w, index=return_scenarios.columns)
    return weights, {
        "cvar": cvar_val,
        "var": var_val,
        "expected_return": exp_ret,
        "status": "optimal",
    }


def _cvar_fallback_simulated_annealing(
    return_scenarios: pd.DataFrame,
    confidence: float,
    target_return: Optional[float],
    long_only: bool,
    max_weight: float,
) -> tuple[pd.Series, dict]:
    """SA-Fallback wenn scipy nicht verfügbar."""
    R = return_scenarios.values
    N = R.shape[1]
    rng = np.random.default_rng(42)

    def cvar(w: np.ndarray) -> float:
        port_returns = R @ w
        losses = -port_returns
        var = np.quantile(losses, confidence)
        tail = losses[losses >= var]
        return float(tail.mean()) if len(tail) > 0 else float(var)

    def feasible(w: np.ndarray) -> bool:
        if long_only and (w < 0).any():
            return False
        if (w > max_weight).any():
            return False
        if abs(w.sum() - 1.0) > 1e-6:
            return False
        return True

    w = np.ones(N) / N
    best_w = w.copy()
    best_cvar = cvar(w)
    T = 1.0
    for _ in range(2000):
        w_new = w + rng.normal(0, 0.05, N)
        w_new = w_new.clip(0 if long_only else -max_weight, max_weight)
        if w_new.sum() <= 0:
            continue
        w_new = w_new / w_new.sum()
        c_new = cvar(w_new)
        if c_new < best_cvar or rng.random() < np.exp(-(c_new - best_cvar) / T):
            w = w_new
            if c_new < best_cvar:
                best_cvar = c_new
                best_w = w.copy()
        T *= 0.999
    weights = pd.Series(best_w, index=return_scenarios.columns)
    return weights, {
        "cvar": best_cvar,
        "var": np.quantile(-(R @ best_w), confidence),
        "expected_return": float((R.mean(axis=0)) @ best_w),
        "status": "sa_heuristic",
    }


__all__ = ["cvar_optimal_weights"]
