"""Distributionally Robust Optimization (DRO) portfolio construction.

Implements two DRO portfolio variants — audit items C2-036 and C2-037 — using
only numpy and scipy so the module works without cvxpy or MOSEK.

**C2-036 — Wasserstein DRO (Esfahani & Kuhn 2018 / Rockafellar-Uryasev CVaR)**
Reference: Esfahani P.M., Kuhn D. (2018). *Data-driven distributionally robust
optimization using the Wasserstein metric: performance guarantees and tractable
reformulations*. Mathematical Programming 171(1-2), 115-166.

The Wasserstein-DRO worst-case expected loss problem for a linear portfolio
loss ℓ(z, w) = −z·w with a type-1 Wasserstein ball of radius ε (using
L∞-norm in scenario space) yields, via strong LP duality (Proposition 3.5
of Esfahani & Kuhn 2018), the CVaR-equivalent LP formulation of Rockafellar &
Uryasev (2000):

    max_{w ∈ Δ, ζ ∈ R, u ∈ R^T}
        (1/T) Σ_t r_t·w  −  γ·[ ζ + (1/T) Σ_t u_t ]
    s.t.  u_t ≥ −r_t·w − ζ   ∀ t = 1…T
          u_t ≥ 0
          Σ w_i = 1,  w_i ≥ 0  (long-only)

where γ = 1/(1-α) with confidence level α ∈ (0,1) (default α=0.95, γ=20),
and ε ↔ α via the Proposition 3.5 equivalence.
This is a **bounded, feasible LP** with O(T + n) variables and O(T + 2)
constraints, solvable by ``scipy.optimize.linprog``.

The ``epsilon`` parameter is mapped to α via α = max(0, 1 − 1/γ) with
γ = 1 + epsilon (so larger ε → higher CVaR penalty → more diversification).

**C2-037 — KL-Divergence DRO (Ben-Tal et al. 2013)**
Reference: Ben-Tal A., den Hertog D., De Waegenaere A., Melenberg B.,
Rennen G. (2013). *Robust solutions of optimization problems affected by
uncertain probabilities*. Management Science 59(2), 341-357.

The KL-DRO worst-case expected loss problem with a KL ball of radius ρ around
the uniform distribution p = (1/T, …, 1/T) has a dual:

    min_w  min_{η>0}  η · ρ + η · log( (1/T) Σ_t exp(−r_t · w / η) )
    s.t.   Σ w_i = 1,  w_i ≥ 0

This is a jointly convex problem in (w, η) solved via scipy SLSQP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: scipy.optimize (linprog / minimize)
# ---------------------------------------------------------------------------
# scipy is imported lazily (inside the solver functions) so that *importing*
# this module never fails when scipy is absent. The helpful ImportError is
# raised only when a DRO solver is actually called without scipy installed.
_SCIPY_AVAILABLE: bool
try:  # pragma: no cover — trivial availability probe
    import scipy.optimize as _scipy_optimize  # noqa: F401

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised only when scipy is absent
    _SCIPY_AVAILABLE = False


def _require_scipy_optimize() -> Any:
    """Return ``scipy.optimize`` or raise a clear, informative ImportError.

    Called by the DRO solvers right before they need ``linprog`` / ``minimize``.
    Keeps module import scipy-free while giving a precise error at call time.
    """
    try:
        import scipy.optimize as scipy_optimize
    except ImportError as exc:  # pragma: no cover — only without scipy
        raise ImportError(
            "DRO portfolio solvers require scipy.optimize "
            "(linprog / minimize), which is not installed. "
            "Install scipy to use wasserstein_dro_portfolio / kl_dro_portfolio."
        ) from exc
    return scipy_optimize


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_EPSILON: float = 0.01  # Wasserstein epsilon (maps to CVaR level)
_DEFAULT_KL_RADIUS: float = 0.1  # KL ball radius
_DEFAULT_RISK_AVERSION: float = 1.0

# Minimum CVaR penalty multiplier γ (must be ≥ 1)
_GAMMA_MIN: float = 1.0
# KL optimisation
_KL_ETA_INIT: float = 0.1  # initial η for SLSQP
_KL_ETA_MIN: float = 1e-6  # lower bound for η
_SLSQP_MAX_ITER: int = 500
_SLSQP_FTOL: float = 1e-9

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DROResult:
    """Output of any DRO portfolio solver in this module.

    Attributes:
        weights: 1-D ndarray of portfolio weights, length = n_assets.
            Long-only by construction (w_i ≥ 0, Σ w_i = 1).
        expected_return: In-sample mean portfolio return ``(1/T) Σ_t r_t·w``.
        worst_case_return: Distributionally robust worst-case expected return
            over the DRO ambiguity set.  ≤ expected_return by construction.
        epsilon: The ambiguity-set radius used (Wasserstein ε or KL ρ).
        solver: Name of the underlying scipy solver used.
        converged: True when the solver reported success.
    """

    weights: np.ndarray
    expected_return: float
    worst_case_return: float
    epsilon: float
    solver: str
    converged: bool


# ---------------------------------------------------------------------------
# Wasserstein DRO — C2-036
# ---------------------------------------------------------------------------


def wasserstein_dro_portfolio(
    returns: np.ndarray,
    epsilon: float = _DEFAULT_EPSILON,
    risk_aversion: float = _DEFAULT_RISK_AVERSION,
) -> DROResult:
    """Wasserstein DRO portfolio (audit C2-036, Esfahani & Kuhn 2018).

    Solves a Wasserstein-robust mean–CVaR portfolio problem via
    ``scipy.optimize.linprog`` (no cvxpy / MOSEK required).

    By Proposition 3.5 of Esfahani & Kuhn (2018) the type-1 Wasserstein
    worst-case expected-loss problem for a linear portfolio loss is equivalent
    to the CVaR-LP of Rockafellar & Uryasev (2000):

        max_{w ∈ Δ, ζ ∈ R, u ∈ R^T≥0}
            (1/T) Σ_t r_t·w  −  γ·[ ζ + (1/T) Σ_t u_t ]
        s.t.  u_t ≥ −r_t·w − ζ   ∀ t = 1…T
              u_t ≥ 0
              Σ w_i = 1,  w_i ≥ 0

    The CVaR penalty weight γ = 1 + epsilon · risk_aversion controls the
    trade-off between expected return and tail-risk robustness:

    - γ = 1 (ε = 0)  → pure mean-return maximisation
    - γ → ∞          → minimise worst-case CVaR (fully robust)
    - Larger ε        → higher γ → more conservative → more diversified

    The LP has n + 1 + T variables and T + 2 constraints, making it tractable
    for typical research scenarios (T ≤ 5000, n ≤ 200).

    Args:
        returns: Array of shape (T, n) — T historical return scenarios for
            n assets.  Must be 2-D with T ≥ 2 and n ≥ 1.
        epsilon: Controls the CVaR penalty as γ = 1 + ε · risk_aversion.
            Must be ≥ 0.  Larger ε → more robust → more diversified.
            Default 0.01.
        risk_aversion: Multiplies ε in the penalty weight (γ = 1 + ε · γ_ra).
            Default 1.0.

    Returns:
        :class:`DROResult` with ``solver="scipy_linprog"``.
        ``worst_case_return`` is the CVaR-penalised objective value.

    Raises:
        ValueError: if inputs are malformed.
    """
    R, T, n = _validate_returns(returns)
    if epsilon < 0:
        raise ValueError(f"epsilon must be ≥ 0, got {epsilon}")
    if risk_aversion <= 0:
        raise ValueError(f"risk_aversion must be > 0, got {risk_aversion}")

    # γ = 1 + ε·γ_ra: the CVaR-penalty weight (≥ 1, so problem is always bounded)
    gamma = max(_GAMMA_MIN, 1.0 + float(epsilon * risk_aversion))

    # ------------------------------------------------------------------
    # Decision vector: x = [w_1, …, w_n,  ζ,  u_1, …, u_T]
    # Indices:         w → 0:n,  ζ → n,  u → n+1 : n+1+T
    # Total variables: n + 1 + T
    # ------------------------------------------------------------------
    n_vars = n + 1 + T

    # Objective: minimise negative robust expected return
    # min  −(1/T) Σ_t r_t·w  +  γ·ζ  +  γ·(1/T) Σ_t u_t
    c = np.zeros(n_vars)
    c[:n] = -np.mean(R, axis=0)  # −mean(r_t·w) w.r.t. w
    c[n] = gamma  # γ on ζ
    c[n + 1 :] = gamma / T  # γ/T on each u_t

    # Inequality constraints:  −r_t·w − ζ − u_t ≤ 0  ⟺  u_t ≥ −r_t·w − ζ
    A_ub = np.zeros((T, n_vars))
    for t in range(T):
        A_ub[t, :n] = -R[t, :]  # −r_{ti} coefficients on w_i
        A_ub[t, n] = -1.0  # −ζ
        A_ub[t, n + 1 + t] = -1.0  # −u_t
    b_ub = np.zeros(T)

    # Equality constraint: Σ w_i = 1
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    # Bounds: w_i ∈ [0,1], ζ free (VaR threshold), u_t ≥ 0 (excess loss)
    bounds = (
        [(0.0, 1.0)] * n  # weights long-only
        + [(None, None)]  # ζ (Value-at-Risk threshold) free
        + [(0.0, None)] * T  # u_t ≥ 0 (excess loss above ζ)
    )

    linprog = _require_scipy_optimize().linprog
    result_lp = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    converged = bool(result_lp.success)
    if not converged:
        logger.warning(
            "wasserstein_dro_portfolio: linprog did not converge — "
            "message: %s. Falling back to equal weights.",
            result_lp.message,
        )
        w = np.full(n, 1.0 / n)
    else:
        w = np.clip(result_lp.x[:n], 0.0, None)
        w_sum = w.sum()
        if w_sum < 1e-12:
            w = np.full(n, 1.0 / n)
        else:
            w /= w_sum

    exp_ret = float(np.mean(R @ w))

    if converged:
        zeta = float(result_lp.x[n])
        u_vals = result_lp.x[n + 1 :]
        cvar = float(zeta + np.mean(u_vals))
        # Worst-case return = expected return minus CVaR penalty
        wc_ret = exp_ret - gamma * cvar
    else:
        wc_ret = float("-inf")

    return DROResult(
        weights=w,
        expected_return=exp_ret,
        worst_case_return=min(wc_ret, exp_ret),
        epsilon=float(epsilon),
        solver="scipy_linprog",
        converged=converged,
    )


# ---------------------------------------------------------------------------
# KL-Divergence DRO — C2-037
# ---------------------------------------------------------------------------


def kl_dro_portfolio(
    returns: np.ndarray,
    kl_radius: float = _DEFAULT_KL_RADIUS,
    risk_aversion: float = _DEFAULT_RISK_AVERSION,
) -> DROResult:
    """KL-Divergence DRO portfolio (audit C2-037, Ben-Tal et al. 2013).

    Solves the KL-robust portfolio problem via ``scipy.optimize.minimize``
    (SLSQP).  The dual objective is jointly convex in (w, η):

        min_{w ∈ Δ, η > 0}  η · ρ_adj + η · log( (1/T) Σ_t exp(−r_t·w / η) )

    where ρ_adj = ρ · risk_aversion.  This is the cumulant generating function
    scaled by η; the inner quantity is a log-sum-exp which is smooth and
    convex, making SLSQP reliable.

    For numerical stability the log-sum-exp is computed with the standard
    max-shift trick to avoid overflow / underflow.

    Args:
        returns: Array of shape (T, n) — T historical return scenarios for
            n assets.  Must be 2-D with T ≥ 2 and n ≥ 1.
        kl_radius: KL ball radius ρ (≥ 0).  Larger ρ → more robust →
            more diversified.  Default 0.1.
        risk_aversion: Scales the effective KL radius (ρ_adj = ρ · γ).
            Default 1.0.

    Returns:
        :class:`DROResult` with ``solver="scipy_slsqp"``.

    Raises:
        ValueError: if inputs are malformed.
    """
    R, T, n = _validate_returns(returns)
    if kl_radius < 0:
        raise ValueError(f"kl_radius must be ≥ 0, got {kl_radius}")
    if risk_aversion <= 0:
        raise ValueError(f"risk_aversion must be > 0, got {risk_aversion}")

    rho_adj = float(kl_radius * risk_aversion)

    # ------------------------------------------------------------------
    # Decision vector: x = [w_1, …, w_n,  η]
    # ------------------------------------------------------------------
    def objective(x: np.ndarray) -> float:
        """KL dual objective: η·ρ + η·log((1/T)Σ_t exp(-r_t·w/η))."""
        w = x[:n]
        eta = x[n]
        if eta < _KL_ETA_MIN:
            return 1e12  # infeasible region
        scores = -(R @ w) / eta  # shape (T,)
        # Numerically stable log-sum-exp
        s_max = np.max(scores)
        lse = s_max + np.log(np.mean(np.exp(scores - s_max)))
        return float(eta * rho_adj + eta * lse)

    def grad_objective(x: np.ndarray) -> np.ndarray:
        """Analytic gradient for faster SLSQP convergence."""
        w = x[:n]
        eta = x[n]
        if eta < _KL_ETA_MIN:
            return np.zeros(n + 1)
        scores = -(R @ w) / eta
        s_max = np.max(scores)
        exp_shifted = np.exp(scores - s_max)
        softmax = exp_shifted / exp_shifted.sum()  # shape (T,)

        # ∂/∂w: η · (1/η) · Σ_t softmax_t · (-r_t) = -R' softmax
        grad_w = -R.T @ softmax  # shape (n,)

        # ∂/∂η: ρ + log((1/T) Σ_t exp(score_t)) + η·(1/T)·Σ_t exp·(score_t/η)
        #      = ρ + lse + Σ_t softmax_t · score_t
        lse = s_max + np.log(np.mean(exp_shifted))
        grad_eta = rho_adj + lse + float(softmax @ scores)
        return np.append(grad_w, grad_eta)

    # Initial point: equal weights, moderate η
    w0 = np.full(n, 1.0 / n)
    x0 = np.append(w0, _KL_ETA_INIT)

    # Constraints: Σ w_i = 1, η ≥ η_min (via bounds)
    constraints = {"type": "eq", "fun": lambda x: x[:n].sum() - 1.0}
    bounds_slsqp = [(0.0, 1.0)] * n + [(_KL_ETA_MIN, None)]

    minimize = _require_scipy_optimize().minimize
    res = minimize(
        objective,
        x0,
        jac=grad_objective,
        method="SLSQP",
        bounds=bounds_slsqp,
        constraints=constraints,
        options={"maxiter": _SLSQP_MAX_ITER, "ftol": _SLSQP_FTOL},
    )

    converged = bool(res.success)
    if not converged:
        logger.warning(
            "kl_dro_portfolio: SLSQP did not converge — message: %s. "
            "Falling back to equal weights.",
            res.message,
        )
        w = np.full(n, 1.0 / n)
        eta_star = _KL_ETA_INIT
    else:
        w = np.clip(res.x[:n], 0.0, None)
        w_sum = w.sum()
        if w_sum < 1e-12:
            w = np.full(n, 1.0 / n)
        else:
            w /= w_sum
        eta_star = float(res.x[n])

    exp_ret = float(np.mean(R @ w))

    # Worst-case expected return: −(dual objective value) + η·ρ cancellation
    # wc = −(η·ρ + η·lse)  → worst-case expected return under KL ambiguity
    scores = -(R @ w) / eta_star
    s_max = np.max(scores)
    lse = s_max + np.log(np.mean(np.exp(scores - s_max)))
    wc_ret = float(-(eta_star * rho_adj + eta_star * lse))

    return DROResult(
        weights=w,
        expected_return=exp_ret,
        worst_case_return=min(wc_ret, exp_ret),
        epsilon=float(kl_radius),
        solver="scipy_slsqp",
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------


def dro_portfolio(
    returns: np.ndarray,
    method: Literal["wasserstein", "kl"] = "wasserstein",
    **kwargs,
) -> DROResult:
    """Dispatch to a DRO portfolio solver by name.

    Args:
        returns: Array of shape (T, n) — T historical return scenarios for
            n assets.
        method: ``"wasserstein"`` (C2-036, default) or ``"kl"`` (C2-037).
        **kwargs: Forwarded to the selected solver
            (:func:`wasserstein_dro_portfolio` or :func:`kl_dro_portfolio`).

    Returns:
        :class:`DROResult` from the selected solver.

    Raises:
        ValueError: if ``method`` is not a recognised solver name.
    """
    if method == "wasserstein":
        return wasserstein_dro_portfolio(returns, **kwargs)
    if method == "kl":
        return kl_dro_portfolio(returns, **kwargs)
    raise ValueError(f"Unknown DRO method {method!r}. Choose 'wasserstein' or 'kl'.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_returns(returns: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Validate and coerce the returns array.

    Returns:
        (R, T, n) where R is the coerced float64 array, T = n_scenarios,
        n = n_assets.

    Raises:
        ValueError: on shape, finiteness or size violations.
    """
    R = np.asarray(returns, dtype=float)
    if R.ndim == 1:
        # Treat as single-asset, T scenarios
        R = R.reshape(-1, 1)
    if R.ndim != 2:
        raise ValueError(
            f"returns must be a 2-D array (T × n), got shape {returns.shape}"
        )
    T, n = R.shape
    if T < 2:
        raise ValueError(f"need at least 2 return scenarios, got T={T}")
    if n < 1:
        raise ValueError(f"need at least 1 asset, got n={n}")
    if not np.all(np.isfinite(R)):
        raise ValueError("returns contains NaN or Inf values")
    return R, T, n


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "DROResult",
    "wasserstein_dro_portfolio",
    "kl_dro_portfolio",
    "dro_portfolio",
]
