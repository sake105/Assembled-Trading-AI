"""Pure-numpy multivariate portfolio optimizers (textbook reference + fallback).

KNOWN_ISSUES §6.5.1 closure: complements the existing 22-module ``portfolio/``
ecosystem with **dependency-light** implementations of the three canonical
quant primitives that were either covered by external-dep wrappers or only
by simplified variants:

- **Mean-Variance (Markowitz 1952)** — efficient frontier, min-variance,
  max-Sharpe (tangency portfolio). Closed-form unconstrained + scipy-based
  constrained solver.
- **Equal Risk Contribution (Maillard et al. 2010)** — *true* risk parity
  using the covariance matrix, NOT the inverse-vol simplification in
  ``position_sizing.compute_risk_parity_weights``. Each asset contributes
  equal marginal risk to total portfolio variance.
- **Multivariate Fractional Kelly (Thorp 2006)** — ``w = k · Σ⁻¹ · (μ − r_f)``
  with half-Kelly default per industry practice.

**Why a new module** when ``portfolio/`` already has 22 files including
``riskfolio_optimizer.py`` (riskfolio wrapper), ``hrp_sizing.py``,
``kelly_robust.py``, ``black_litterman.py``, ``market_neutral_optimizer.py``,
etc.?

1. **Dependency-light reference.** ``riskfolio_optimizer`` depends on the
   ``riskfolio`` package which may not be available in all environments.
   This module needs only numpy + scipy.
2. **Textbook clarity.** Plain formulas one can audit against a graduate
   textbook (Markowitz 1952 §III, Maillard 2010 eq. 11, Thorp 2006 §4).
   Useful as the reference any other optimizer can be validated against.
3. **Closes the cov→weights pipeline.** Now that ``risk/dcc_garch`` (C4-072)
   produces a proper conditional covariance Σ_T, this module is the natural
   consumer: ``current_covariance(dcc_result) → max_sharpe_weights(μ, Σ)``.

**Does NOT replace**:
- ``position_sizing.compute_risk_parity_weights`` — that's inverse-vol (1/σ),
  fine for single-portfolio sizing without covariance.
- ``kelly_robust.robust_kelly_fraction`` — single-asset with uncertainty-aware
  Kelly fraction (parameter-bayes), our ``multivariate_kelly`` is multi-asset
  with point estimate of (μ, Σ). Complementary.
- ``hrp_sizing`` — Lopez de Prado HRP, different philosophy (recursive
  bisection on hierarchical clusters).
- ``black_litterman`` — prior+views→posterior, this module accepts μ directly.
- ``riskfolio_optimizer`` — when ``riskfolio`` is available it offers more
  objectives (CVaR, omega, MAD); this module is the dependency-light path.

References:
- Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance 7(1).
- Maillard, S., Roncalli, T., Teïletche, J. (2010). *On the Properties of
  Equally-Weighted Risk Contribution Portfolios*. JPM 36(4).
- Thorp, E. O. (2006). *The Kelly Criterion in Blackjack, Sports Betting, and
  the Stock Market*. Ch. 9 in Zenios & Ziemba (eds), Handbook of Asset and
  Liability Management.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: scipy.optimize.minimize
# ---------------------------------------------------------------------------
# scipy is imported lazily (inside the constrained-solver functions) so that
# *importing* this module never fails when scipy is absent. The closed-form
# unconstrained paths (e.g. analytic min-variance / max-Sharpe) work without
# scipy; the helpful ImportError is raised only when a constrained solver that
# truly needs scipy.optimize.minimize is actually called.
_SCIPY_AVAILABLE: bool
try:  # pragma: no cover — trivial availability probe
    import scipy.optimize as _scipy_optimize  # noqa: F401

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised only when scipy is absent
    _SCIPY_AVAILABLE = False


def _require_minimize():
    """Return ``scipy.optimize.minimize`` or raise a clear ImportError.

    Called by the constrained optimisers right before they invoke ``minimize``.
    Keeps module import scipy-free while giving a precise error at call time.
    """
    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover — only without scipy
        raise ImportError(
            "This constrained optimizer requires scipy.optimize.minimize, "
            "which is not installed. Install scipy, or use an unconstrained "
            "closed-form path where available."
        ) from exc
    return minimize


@dataclass
class OptimizerResult:
    """Common return type for all optimizers in this module.

    Attributes:
        weights: pd.Series of portfolio weights (index = asset names).
            Sum to 1.0 by construction for long-only / fully invested.
        expected_return: ``w' · μ`` (NaN if μ not provided to the optimizer).
        expected_volatility: ``√(w' · Σ · w)`` — portfolio std deviation.
        sharpe_ratio: ``(expected_return − r_f) / expected_volatility``
            (NaN if μ not provided).
        n_assets: Number of assets with non-zero weight.
        method: Which optimizer produced this result.
        converged: Whether the underlying solver reported success. For the
            closed-form unconstrained paths, always True.
    """

    weights: pd.Series
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    n_assets: int
    method: str
    converged: bool


def _validate_covariance(
    covariance: pd.DataFrame, check_psd: bool = True
) -> np.ndarray:
    """Validate covariance: square, symmetric, finite, (PSD). Returns array.

    Risk-Zone hardening (F-stage1-portopt-5): default ``check_psd=True``
    catches a non-PSD Σ (e.g. numerically broken EWMA or failed shrinkage)
    before it silently produces ``σ_p = 0`` via the
    ``max(w'Σw, 0)`` paper-over in ``_portfolio_stats``. Callers in tight
    loops may pass ``check_psd=False`` after their own pre-check.
    """
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"covariance must be square, got shape {covariance.shape}")
    if covariance.shape[0] < 2:
        raise ValueError(f"need ≥2 assets, got {covariance.shape[0]}")
    # F-postcommit-5: caller may build a DataFrame where index != columns
    # (e.g. typo). The numerics would proceed silently with weights labeled
    # by columns; Risk-Zone primitives must not accept that mislabeling.
    if list(covariance.index) != list(covariance.columns):
        raise ValueError("covariance.index and covariance.columns must match exactly")
    arr = covariance.to_numpy()
    if not np.all(np.isfinite(arr)):
        raise ValueError("covariance contains NaN/inf")
    asym = float(np.max(np.abs(arr - arr.T)))
    if asym > 1e-8:
        raise ValueError(
            f"covariance not symmetric (max |Σ − Σ'| = {asym:.2e}); "
            "pre-symmetrise via 0.5 * (Σ + Σ') before calling"
        )
    sym: np.ndarray = 0.5 * (arr + arr.T)
    if check_psd:
        eigvals = np.linalg.eigvalsh(sym)
        min_eig = float(eigvals.min())
        if min_eig < -1e-8:
            raise ValueError(f"covariance not PSD: min eigenvalue = {min_eig:.2e}")
        if min_eig < 1e-12:
            logger.warning("covariance near-singular: min eigenvalue = %.2e", min_eig)
    return sym


def _portfolio_stats(
    weights: np.ndarray,
    cov: np.ndarray,
    expected_returns: np.ndarray | None,
    risk_free: float,
) -> tuple[float, float, float]:
    """Compute (μ_p, σ_p, Sharpe) for the given weights."""
    sigma_p = float(np.sqrt(max(weights @ cov @ weights, 0.0)))
    if expected_returns is None:
        return float("nan"), sigma_p, float("nan")
    mu_p = float(weights @ expected_returns)
    sharpe = (mu_p - risk_free) / sigma_p if sigma_p > 0 else float("nan")
    return mu_p, sigma_p, sharpe


def min_variance_weights(
    covariance: pd.DataFrame,
    long_only: bool = True,
    weight_bounds: tuple[float, float] | None = None,
) -> OptimizerResult:
    """Global Minimum-Variance Portfolio.

    Unconstrained closed form: ``w_mv = Σ⁻¹ · 1 / (1' Σ⁻¹ 1)``.
    With long-only or weight_bounds constraints, solves ``min w'Σw`` s.t.
    ``sum(w) = 1`` via scipy SLSQP.

    Caller responsibility (F-postcommit-4): ``covariance`` must be PIT-safe.

    Args:
        covariance: (N, N) DataFrame, symmetric PSD covariance matrix.
        long_only: If True, enforce ``w_i ≥ 0`` (most common use case).
        weight_bounds: ``(min_w, max_w)`` per-asset bounds. Default None
            uses (0, 1) for long_only or (-1, 1) otherwise.

    Returns:
        OptimizerResult with weights, expected_volatility, n_assets, method,
        converged. expected_return and sharpe_ratio are NaN (no μ provided).

    Raises:
        ValueError: If covariance is malformed (not square, asymmetric, NaN).
    """
    cov = _validate_covariance(covariance)
    n = cov.shape[0]
    names = list(covariance.columns)

    if not long_only and weight_bounds is None:
        # Unconstrained closed form
        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return _solve_min_var_constrained(cov, names, long_only, weight_bounds)
        ones = np.ones(n)
        w = inv @ ones
        w = w / float(w @ ones)
        _, sigma_p, _ = _portfolio_stats(w, cov, None, 0.0)
        return OptimizerResult(
            weights=pd.Series(w, index=names),
            expected_return=float("nan"),
            expected_volatility=sigma_p,
            sharpe_ratio=float("nan"),
            n_assets=int(np.sum(np.abs(w) > 1e-8)),
            method="min_variance_closed_form",
            converged=True,
        )

    return _solve_min_var_constrained(cov, names, long_only, weight_bounds)


def _solve_min_var_constrained(
    cov: np.ndarray,
    names: list[str],
    long_only: bool,
    weight_bounds: tuple[float, float] | None,
) -> OptimizerResult:
    """Constrained min-variance via scipy SLSQP."""
    n = cov.shape[0]
    if weight_bounds is None:
        bounds = [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n
    else:
        bounds = [weight_bounds] * n

    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    x0 = np.full(n, 1.0 / n)

    minimize = _require_minimize()
    result = minimize(
        lambda w: float(w @ cov @ w),
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    w = result.x
    _, sigma_p, _ = _portfolio_stats(w, cov, None, 0.0)
    return OptimizerResult(
        weights=pd.Series(w, index=names),
        expected_return=float("nan"),
        expected_volatility=sigma_p,
        sharpe_ratio=float("nan"),
        n_assets=int(np.sum(np.abs(w) > 1e-8)),
        method="min_variance_slsqp",
        converged=bool(result.success),
    )


def max_sharpe_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.0,
    long_only: bool = True,
    weight_bounds: tuple[float, float] | None = None,
) -> OptimizerResult:
    """Tangency (Maximum Sharpe) Portfolio.

    Unconstrained closed form:
        ``w_t = Σ⁻¹ · (μ − r_f · 1) / (1' · Σ⁻¹ · (μ − r_f · 1))``.

    Long-only / bounded: maximises ``(w'(μ−r_f))/√(w'Σw)`` via scipy SLSQP.

    Caller responsibility (F-postcommit-4): ``expected_returns`` and
    ``covariance`` must be PIT-safe.

    Args:
        expected_returns: Per-asset expected returns μ (annualized typical).
        covariance: (N, N) covariance matrix Σ matching expected_returns index.
        risk_free_rate: r_f for excess-return computation.
        long_only: Enforce w_i ≥ 0.
        weight_bounds: Per-asset bounds override.

    Returns:
        OptimizerResult with all fields populated.

    Raises:
        ValueError: Index mismatch or malformed inputs.
    """
    if list(expected_returns.index) != list(covariance.columns):
        raise ValueError(
            "expected_returns.index and covariance.columns must match exactly"
        )
    cov = _validate_covariance(covariance)
    mu = expected_returns.to_numpy(dtype=float)
    if not np.all(np.isfinite(mu)):
        raise ValueError("expected_returns contains NaN/inf")

    names = list(covariance.columns)

    if not long_only and weight_bounds is None:
        # Unconstrained closed form (tangency portfolio)
        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return _solve_max_sharpe_constrained(
                mu, cov, risk_free_rate, names, long_only, weight_bounds
            )
        excess = mu - risk_free_rate
        denom = float(np.ones(len(mu)) @ inv @ excess)
        if abs(denom) < 1e-12:
            # Excess returns sum to zero in inverse-covariance metric — fallback
            return _solve_max_sharpe_constrained(
                mu, cov, risk_free_rate, names, long_only, weight_bounds
            )
        if denom < 0:
            # 1' Σ⁻¹ (μ-r_f) < 0 — closed-form would land on the inefficient
            # branch (min-Sharpe instead of max-Sharpe). Typical in defensive
            # regimes where all μ_i < r_f. Fallback to SLSQP which correctly
            # maximises Sharpe by minimising −Sharpe (F-stage1-portopt-3).
            logger.warning(
                "max_sharpe: 1' Σ⁻¹ (μ-r_f) = %.3e < 0 — closed-form is on "
                "the inefficient branch; falling back to SLSQP.",
                denom,
            )
            return _solve_max_sharpe_constrained(
                mu, cov, risk_free_rate, names, long_only, weight_bounds
            )
        w = (inv @ excess) / denom
        mu_p, sigma_p, sharpe = _portfolio_stats(w, cov, mu, risk_free_rate)
        return OptimizerResult(
            weights=pd.Series(w, index=names),
            expected_return=mu_p,
            expected_volatility=sigma_p,
            sharpe_ratio=sharpe,
            n_assets=int(np.sum(np.abs(w) > 1e-8)),
            method="max_sharpe_closed_form",
            converged=True,
        )

    return _solve_max_sharpe_constrained(
        mu, cov, risk_free_rate, names, long_only, weight_bounds
    )


def _solve_max_sharpe_constrained(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free: float,
    names: list[str],
    long_only: bool,
    weight_bounds: tuple[float, float] | None,
) -> OptimizerResult:
    """Constrained max-Sharpe via scipy SLSQP (minimise negative Sharpe)."""
    n = cov.shape[0]
    if weight_bounds is None:
        bounds = [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n
    else:
        bounds = [weight_bounds] * n

    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    x0 = np.full(n, 1.0 / n)
    excess = mu - risk_free

    def neg_sharpe(w: np.ndarray) -> float:
        sigma_p = float(np.sqrt(max(w @ cov @ w, 1e-12)))
        return -float(w @ excess) / sigma_p

    minimize = _require_minimize()
    result = minimize(
        neg_sharpe,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )
    w = result.x
    mu_p, sigma_p, sharpe = _portfolio_stats(w, cov, mu, risk_free)
    return OptimizerResult(
        weights=pd.Series(w, index=names),
        expected_return=mu_p,
        expected_volatility=sigma_p,
        sharpe_ratio=sharpe,
        n_assets=int(np.sum(np.abs(w) > 1e-8)),
        method="max_sharpe_slsqp",
        converged=bool(result.success),
    )


def mean_variance_efficient_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    n_points: int = 20,
    long_only: bool = True,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Trace ``n_points`` along the mean-variance efficient frontier.

    For each target return between min(μ) and max(μ), solve
    ``min w'Σw`` s.t. ``w'μ = target_return`` and ``sum(w)=1``.

    Caller responsibility (F-postcommit-4): ``expected_returns`` and
    ``covariance`` must be PIT-safe.

    Args:
        expected_returns: μ per asset.
        covariance: Σ.
        n_points: Number of grid points along the frontier.
        long_only: Enforce w_i ≥ 0.
        risk_free_rate: For Sharpe column.

    Returns:
        DataFrame with ``n_points`` rows (one per target_return). Columns:
        - ``target_return``: target μ_p
        - ``volatility``: realised σ_p (NaN if solver did not converge)
        - ``sharpe``: (μ_p − r_f) / σ_p (NaN if solver did not converge)
        - ``weights``: dict of {asset → weight} per row (None on failure)
        - ``converged``: bool — whether SLSQP reported success for this row

        Non-converged rows are KEPT with NaN volatility/sharpe and a False
        ``converged`` flag so the caller can distinguish "convergence failed"
        from "target_return outside feasible range" (F-stage1-portopt-1).
    """
    if list(expected_returns.index) != list(covariance.columns):
        raise ValueError("expected_returns.index must match covariance.columns")

    cov = _validate_covariance(covariance)
    mu = expected_returns.to_numpy(dtype=float)
    names = list(covariance.columns)
    n = cov.shape[0]

    target_returns = np.linspace(float(mu.min()), float(mu.max()), n_points)
    bounds = [(0.0, 1.0)] * n if long_only else [(-1.0, 1.0)] * n
    x0 = np.full(n, 1.0 / n)

    rows: list[dict[str, object]] = []
    n_failed = 0
    minimize = _require_minimize()
    for target in target_returns:
        constraints = [
            {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
            {"type": "eq", "fun": lambda w, t=target: float(w @ mu - t)},
        ]
        result = minimize(
            lambda w: float(w @ cov @ w),
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-10},
        )
        if not result.success:
            n_failed += 1
            rows.append(
                {
                    "target_return": float(target),
                    "volatility": float("nan"),
                    "sharpe": float("nan"),
                    "weights": None,
                    "converged": False,
                }
            )
            continue
        w = result.x
        sigma_p = float(np.sqrt(max(w @ cov @ w, 0.0)))
        mu_p = float(w @ mu)
        sharpe = (mu_p - risk_free_rate) / sigma_p if sigma_p > 0 else float("nan")
        rows.append(
            {
                "target_return": float(target),
                "volatility": sigma_p,
                "sharpe": sharpe,
                "weights": dict(zip(names, w)),
                "converged": True,
            }
        )

    if n_failed:
        logger.warning(
            "mean_variance_efficient_frontier: %d/%d target_returns did not "
            "converge (rows kept with converged=False).",
            n_failed,
            n_points,
        )

    return pd.DataFrame(rows)


def equal_risk_contribution_weights(
    covariance: pd.DataFrame,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> OptimizerResult:
    """Equal Risk Contribution (Maillard et al. 2010) — true risk parity.

    Each asset's contribution to total portfolio variance is equalised:
        ``w_i · (Σ w)_i = const ∀ i``.

    Solved by Newton-like iterative descent on
    ``∑_i (w_i · (Σw)_i − σ_p² / N)²``. Converges quickly for well-
    conditioned covariances.

    Differs from inverse-vol weighting (``1/σ_i``) because ERC uses the
    full covariance: a high-σ asset uncorrelated with others gets MORE
    weight than 1/σ would suggest; correlated diversifiers get less.

    Caller responsibility (F-postcommit-4): ``covariance`` must be PIT-safe.

    Args:
        covariance: Σ.
        max_iter: Maximum iterations for the fixed-point loop.
        tol: Convergence tolerance on the L∞ change in weights.

    Returns:
        OptimizerResult with weights summing to 1.0, expected_volatility,
        and converged flag.
    """
    cov = _validate_covariance(covariance)
    n = cov.shape[0]
    names = list(covariance.columns)

    # Maillard 2010 §3.2: minimise the sum of squared deviations of risk
    # contributions from their mean. SLSQP with sum(w)=1, w>0 constraints
    # converges robustly even for ill-conditioned covariances; the simple
    # multiplicative fixed-point did not (pre-commit local test caught it).
    def objective(w: np.ndarray) -> float:
        sigma_w = cov @ w
        rc = w * sigma_w  # per-asset risk contribution
        # rc.mean() == σ_p² / N by definition of risk contribution (F-senior-portopt-6)
        mean_rc = float(rc.mean())
        return float(np.sum((rc - mean_rc) ** 2))

    # Initial guess: inverse-vol weights (close to ERC for diagonal Σ)
    vols = np.sqrt(np.diag(cov))
    x0 = 1.0 / np.maximum(vols, 1e-12)
    x0 = x0 / x0.sum()

    minimize = _require_minimize()
    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(1e-8, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
        options={"maxiter": max_iter, "ftol": tol},
    )
    w = result.x / result.x.sum()  # re-normalise against tiny numerical drift

    _, sigma_p, _ = _portfolio_stats(w, cov, None, 0.0)
    return OptimizerResult(
        weights=pd.Series(w, index=names),
        expected_return=float("nan"),
        expected_volatility=sigma_p,
        sharpe_ratio=float("nan"),
        n_assets=int(np.sum(np.abs(w) > 1e-8)),
        method="equal_risk_contribution",
        converged=bool(result.success),
    )


def multivariate_kelly_weights(
    expected_excess_returns: pd.Series,
    covariance: pd.DataFrame,
    kelly_fraction: float = 0.5,
    max_leverage: float = 1.0,
    long_only: bool = False,
    renormalize_to_unity: bool = False,
) -> OptimizerResult:
    """Multivariate Fractional Kelly (Thorp 2006).

    Full Kelly: ``w* = Σ⁻¹ · (μ − r_f)``.
    Fractional Kelly: ``w = kelly_fraction · w*`` (default 0.5 = half-Kelly,
    the practical default per Thorp 2006 — captures most growth-rate
    benefit at much lower volatility cost).

    **Order of operations and contract** (F-stage1-portopt-2, F-postcommit-1):

    1. Compute ``w = kelly_fraction · Σ⁻¹ · (μ − r_f)``.
    2. If ``long_only=True``: clip ``w_i ← max(w_i, 0)``.
    3. If ``renormalize_to_unity=True``: rescale to ``sum(w) = 1`` (full
       invest, no implicit cash). Off by default — Kelly is a *growth-rate*
       optimiser, NOT a full-invest portfolio. The natural Kelly answer is
       often partial investment.
    4. Apply leverage cap LAST: if ``sum(|w|) > max_leverage``, rescale to
       ``sum(|w|) = max_leverage``. The cap is the hard invariant — applied
       after renormalize so shorts cannot inflate gross exposure past the
       cap.

    Caller responsibility: ``μ`` and ``Σ`` must be PIT-safe.

    Args:
        expected_excess_returns: μ − r_f per asset (excess returns).
        covariance: Σ.
        kelly_fraction: Fractional Kelly multiplier in (0, 1]. Default 0.5.
        max_leverage: Cap on ``sum(|w|)``. Default 1.0 (no leverage).
            Set to 0 or negative to disable the cap entirely.
        long_only: Project negative weights to 0 (rare for Kelly — typically
            Kelly produces short positions when μ < r_f for some asset).
            When True, ``sum(|w|) = sum(w)`` and the cap acts on the gross
            (= net) exposure. With ``renormalize_to_unity=False`` (default)
            the result may have ``sum(w) < 1``, implying implicit cash of
            ``1 - sum(w)``.
        renormalize_to_unity: If True, rescale final weights to sum to 1.0.
            Default False preserves Kelly's growth-rate semantics. Setting
            True turns the Kelly direction into a fully-invested portfolio
            weighted by Kelly's relative recommendations.

    Returns:
        OptimizerResult.

    Raises:
        ValueError: kelly_fraction outside (0, 1] or index mismatch.
    """
    if not (0.0 < kelly_fraction <= 1.0):
        raise ValueError(f"kelly_fraction must be in (0, 1], got {kelly_fraction}")
    # F-postcommit-2: NaN/inf max_leverage would silently bypass the cap
    # (any NaN comparison is False). Catch this explicitly — it is almost
    # certainly an upstream config bug, never a valid "opt-out".
    if not np.isfinite(max_leverage):
        raise ValueError(
            f"max_leverage must be finite, got {max_leverage} "
            "(use max_leverage <= 0 to disable the cap)"
        )
    if list(expected_excess_returns.index) != list(covariance.columns):
        raise ValueError("expected_excess_returns.index must match covariance.columns")
    cov = _validate_covariance(covariance)
    excess = expected_excess_returns.to_numpy(dtype=float)
    if not np.all(np.isfinite(excess)):
        raise ValueError("expected_excess_returns contains NaN/inf")
    names = list(covariance.columns)

    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # Fall back to pseudo-inverse for singular covariance
        inv = np.linalg.pinv(cov)

    w_full_kelly = inv @ excess
    w = kelly_fraction * w_full_kelly

    if long_only:
        w = np.maximum(w, 0.0)

    # F-postcommit-1: leverage cap must be the LAST step. Previously the cap
    # ran before renormalize_to_unity, so with shorts present renormalize
    # could re-inflate sum(|w|) far above max_leverage (3x observed). New
    # order: (a) optional renormalize_to_unity, (b) leverage cap. The cap is
    # the hard invariant; renormalize is an optional reshaping.
    if renormalize_to_unity:
        s = float(np.sum(w))
        if abs(s) > 1e-12:
            w = w / s
        else:
            # F-senior-portopt-1: market-neutral Kelly direction sums to ~0;
            # renormalize_to_unity has no meaningful target. Surface this
            # honestly rather than silently leaving w unnormalized.
            logger.warning(
                "multivariate_kelly_weights: renormalize_to_unity=True but "
                "sum(w) = %.3e is near zero (market-neutral direction); "
                "leaving weights unnormalized.",
                s,
            )

    # Apply leverage cap on sum of absolute weights AFTER any renormalisation.
    # `max_leverage > 0` guard: pass max_leverage<=0 to disable the cap.
    abs_sum = float(np.sum(np.abs(w)))
    if max_leverage > 0 and abs_sum > max_leverage:
        w = w * (max_leverage / abs_sum)

    mu_p, sigma_p, _ = _portfolio_stats(w, cov, excess, 0.0)
    # Sharpe here is on excess return / vol
    sharpe = mu_p / sigma_p if sigma_p > 0 else float("nan")
    return OptimizerResult(
        weights=pd.Series(w, index=names),
        expected_return=mu_p,
        expected_volatility=sigma_p,
        sharpe_ratio=sharpe,
        n_assets=int(np.sum(np.abs(w) > 1e-8)),
        method=f"multivariate_kelly_k{kelly_fraction:.2f}",
        converged=True,
    )


__all__ = [
    "OptimizerResult",
    "min_variance_weights",
    "max_sharpe_weights",
    "mean_variance_efficient_frontier",
    "equal_risk_contribution_weights",
    "multivariate_kelly_weights",
]
