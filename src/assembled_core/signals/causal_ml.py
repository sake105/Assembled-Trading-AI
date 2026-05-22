"""Causal ML: Double-ML PLR and Causal Forest implementations.

Audit references
----------------
* C2-025 — Double-ML Partially Linear Regression (PLR / Robinson 1988)
* C2-026 — Causal Forest for heterogeneous treatment-effect estimation

Overview
--------
Both estimators follow a graceful-degradation pattern:

PLR (``fit_plr``)
    1. Try ``doubleml`` library first (``DoubleMLPLR``).
    2. Fall back to a pure ``scipy`` / ``sklearn`` implementation of Robinson's
       double-residualisation with cross-fitting.

Causal Forest (``fit_causal_forest``)
    1. Try ``econml.dml.CausalForestDML`` first.
    2. Fall back to an honest-random-forest approximation using
       ``sklearn.ensemble.RandomForestRegressor``.
    3. If sklearn is also unavailable return a placeholder with
       ``converged=False``.

No bare ``assert`` statements appear in this module.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from doubleml import DoubleMLData, DoubleMLPLR

    HAS_DOUBLEML = True
except Exception as _e:  # ImportError or internal errors in older builds
    logger.debug("[causal_ml] doubleml not available: %s", _e)
    HAS_DOUBLEML = False

try:
    from econml.dml import CausalForestDML

    HAS_ECONML = True
except Exception:
    HAS_ECONML = False

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold

    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    from scipy import stats as _scipy_stats

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PLRResult:
    """Result from :func:`fit_plr`.

    Attributes
    ----------
    theta:
        Estimated causal effect of treatment *D* on outcome *Y*.
    se:
        Standard error of *theta*.
    t_stat:
        t-statistic (theta / se).
    pvalue:
        Two-sided p-value for the null hypothesis ``theta == 0``.
    n_obs:
        Number of observations used.
    n_folds:
        Number of cross-fitting folds.
    method:
        Description of the estimator used (``'plr_robinson'`` for the
        fallback, ``'doubleml_plr'`` when the doubleml library is available).
    """

    theta: float
    se: float
    t_stat: float
    pvalue: float
    n_obs: int
    n_folds: int
    method: str = "plr_robinson"


@dataclass
class CausalForestResult:
    """Result from :func:`fit_causal_forest`.

    Attributes
    ----------
    cate:
        Conditional Average Treatment Effect estimates, shape ``(n_obs,)``.
    ate:
        Average Treatment Effect (mean of *cate*).
    ate_se:
        Standard error of *ate* (std of CATE / sqrt(n_obs)).
    n_obs:
        Number of observations.
    method:
        Estimator used (``'causal_forest_econml'``, ``'honest_rf_approx'``, or
        ``'placeholder_no_sklearn'``).
    converged:
        ``True`` when a real estimator ran successfully.
    """

    cate: np.ndarray
    ate: float
    ate_se: float
    n_obs: int
    method: str
    converged: bool


# ---------------------------------------------------------------------------
# PLR — Robinson (1988) double-residualisation with cross-fitting
# ---------------------------------------------------------------------------


def _plr_fallback(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    n_folds: int,
) -> PLRResult:
    """Pure-sklearn Robinson PLR with K-fold cross-fitting.

    Parameters
    ----------
    Y:
        Outcome vector, shape ``(n,)``.
    D:
        Binary or continuous treatment vector, shape ``(n,)``.
    X:
        Covariate matrix, shape ``(n, p)``.
    n_folds:
        Number of cross-fitting folds (>= 2).

    Returns
    -------
    PLRResult
    """
    n = len(Y)
    Y_res = np.zeros(n, dtype=np.float64)
    D_res = np.zeros(n, dtype=np.float64)

    kf = KFold(n_splits=n_folds, shuffle=False)
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        Y_tr, Y_val = Y[train_idx], Y[val_idx]
        D_tr, D_val = D[train_idx], D[val_idx]

        # Nuisance model g: X → Y
        g_hat = LinearRegression().fit(X_tr, Y_tr)
        Y_res[val_idx] = Y_val - g_hat.predict(X_val)

        # Nuisance model m: X → D
        m_hat = LinearRegression().fit(X_tr, D_tr)
        D_res[val_idx] = D_val - m_hat.predict(X_val)

    # Robinson moment condition: OLS of Ỹ on D̃ (no intercept)
    # θ = (D̃ᵀ D̃)⁻¹ D̃ᵀ Ỹ
    theta_arr, _, _, _ = np.linalg.lstsq(D_res.reshape(-1, 1), Y_res, rcond=None)
    theta = float(theta_arr[0])

    # Heteroskedasticity-robust SE (sandwich)
    e = Y_res - D_res * theta
    d2 = float(np.dot(D_res, D_res))
    if d2 == 0.0:
        logger.warning(
            "[PLR] D_res is zero (constant treatment after cross-fitting) — "
            "se/t/pvalue are NaN. theta estimate is unreliable."
        )
        se = float("nan")
    else:
        # HC0-style: se² = (D̃ᵀD̃)⁻² * Σ(D̃ᵢ²·eᵢ²)
        numerator = float(np.dot(D_res**2, e**2))
        se = float(np.sqrt(numerator) / d2)

    if se == 0.0 or np.isnan(se):
        t_stat = float("nan")
        pvalue = float("nan")
    else:
        t_stat = theta / se
        if HAS_SCIPY:
            pvalue = float(2 * _scipy_stats.t.sf(abs(t_stat), df=n - 2))
        else:
            # Normal approximation when scipy is absent
            pvalue = float(2 * (1.0 - _norm_cdf(abs(t_stat))))

    return PLRResult(
        theta=theta,
        se=se,
        t_stat=t_stat,
        pvalue=pvalue,
        n_obs=n,
        n_folds=n_folds,
        method="plr_robinson",
    )


def _norm_cdf(x: float) -> float:
    """Rough standard-normal CDF via erfc (no scipy required)."""
    import math

    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def fit_plr(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    n_folds: int = 5,
) -> PLRResult:
    """Estimate the causal effect of treatment *D* on outcome *Y* via PLR.

    The Partially Linear Regression model (Robinson 1988) is:

        Y = θ·D + g(X) + ε
        D = m(X) + v

    where ``g`` and ``m`` are nuisance functions estimated non-parametrically
    via cross-fitting.  The method is doubly-robust in the sense that
    consistent estimation of either nuisance function suffices.

    Parameters
    ----------
    Y:
        Outcome, shape ``(n,)``.
    D:
        Treatment (binary or continuous), shape ``(n,)``.
    X:
        Pre-treatment covariates, shape ``(n, p)``.
    n_folds:
        Number of cross-fitting folds (>= 2).  Default ``5``.

    Returns
    -------
    PLRResult
        Named result with ``theta``, ``se``, ``t_stat``, ``pvalue``,
        ``n_obs``, ``n_folds``, ``method``.

    Notes
    -----
    When ``doubleml`` is installed it is used directly.  Otherwise a pure
    ``sklearn`` / ``scipy`` fallback is applied.
    """
    Y = np.asarray(Y, dtype=np.float64).ravel()
    D = np.asarray(D, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if HAS_DOUBLEML and HAS_SKLEARN:
        try:
            import pandas as pd

            col_names = [f"x{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=col_names)
            df["y"] = Y
            df["d"] = D
            dml_data = DoubleMLData(df, y_col="y", d_cols="d", x_cols=col_names)
            learner_g = LinearRegression()
            learner_m = LinearRegression()
            obj = DoubleMLPLR(
                dml_data,
                ml_l=learner_g,
                ml_m=learner_m,
                n_folds=n_folds,
            )
            obj.fit()
            coef = float(obj.coef[0])
            se_val = float(obj.se[0])
            t_val = coef / se_val if se_val != 0 else float("nan")
            pval = float(obj.pval[0])
            return PLRResult(
                theta=coef,
                se=se_val,
                t_stat=t_val,
                pvalue=pval,
                n_obs=len(Y),
                n_folds=n_folds,
                method="doubleml_plr",
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("doubleml.DoubleMLPLR failed (%s); falling back.", exc)

    if not HAS_SKLEARN:
        raise RuntimeError(
            "fit_plr requires sklearn (or doubleml).  "
            "Install with: pip install scikit-learn"
        )

    return _plr_fallback(Y, D, X, n_folds)


# ---------------------------------------------------------------------------
# Causal Forest
# ---------------------------------------------------------------------------


def fit_causal_forest(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 42,
) -> CausalForestResult:
    """Estimate heterogeneous treatment effects via a Causal Forest.

    Parameters
    ----------
    Y:
        Outcome, shape ``(n,)``.
    D:
        Treatment (binary or continuous), shape ``(n,)``.
    X:
        Pre-treatment covariates, shape ``(n, p)``.
    n_estimators:
        Number of trees in the forest.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    CausalForestResult
        Named result with ``cate`` (length-n array), ``ate``, ``ate_se``,
        ``n_obs``, ``method``, ``converged``.

    Notes
    -----
    **econml available (preferred):**
    Delegates to ``CausalForestDML`` from the ``econml`` package.

    **sklearn only (fallback — documented approximation):**
    Implements a simple "honest" forest:

    1. Split data into train (50 %) and honest (50 %) halves.
    2. Fit a ``RandomForestRegressor`` on the train half to predict *Y*.
    3. On the honest half compute per-sample residuals relative to a
       reference treatment effect  ``θ_ols * D``, where ``θ_ols`` is the
       OLS slope from a univariate regression of *Y* on *D*.
    4. CATE ≈ residual / std(D_honest) — crude approximation only.

    This approximation has no formal guarantees; it is labelled
    ``method='honest_rf_approx'`` to distinguish it from the true estimator.

    **Neither available:**
    Returns a placeholder result with ``converged=False``.
    """
    Y = np.asarray(Y, dtype=np.float64).ravel()
    D = np.asarray(D, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = len(Y)

    # --- econml path ---
    if HAS_ECONML and HAS_SKLEARN:
        try:
            est = CausalForestDML(
                n_estimators=n_estimators,
                random_state=random_state,
                discrete_treatment=False,
            )
            est.fit(Y, D, X=X)
            cate = est.effect(X).ravel()
            ate = float(np.mean(cate))
            ate_se = float(np.std(cate, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
            return CausalForestResult(
                cate=cate,
                ate=ate,
                ate_se=ate_se,
                n_obs=n,
                method="causal_forest_econml",
                converged=True,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("CausalForestDML failed (%s); falling back.", exc)

    # --- honest RF approximation ---
    #
    # Strategy: use Robinson-style double-residualisation with random-forest
    # nuisance models to produce an approximately consistent ATE estimate, then
    # build per-sample CATE via a second honest forest on the Robinson residuals.
    #
    # Step 1 (cross-fitting): obtain out-of-fold residuals Ỹ = Y - Ê[Y|X] and
    #   D̃ = D - Ê[D|X] using two-fold cross-fitting with RF nuisance models.
    # Step 2 (global theta): OLS of Ỹ on D̃  → scalar ATE estimate theta_hat.
    # Step 3 (CATE via honest forest): split data in half; on the honest half,
    #   use the local residuals Ỹᵢ / D̃ᵢ (when |D̃ᵢ| is non-negligible) as
    #   per-sample pseudo-outcomes; predict these from X using an RF trained on
    #   the training half.  This is a well-known CATE proxy (Nie & Wager 2021
    #   R-learner, simplified).
    if HAS_SKLEARN:
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(n)
        split = max(n // 2, 1)
        train_idx = idx[:split]
        honest_idx = idx[split:] if split < n else idx[:split]

        # -- Cross-fitting for Robinson residuals (2 folds) --
        Y_res = np.zeros(n, dtype=np.float64)
        D_res = np.zeros(n, dtype=np.float64)

        kf_idx = [(train_idx, honest_idx), (honest_idx, train_idx)]
        for tr_i, val_i in kf_idx:
            if len(tr_i) == 0 or len(val_i) == 0:
                continue
            rf_y = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=1,
            )
            rf_y.fit(X[tr_i], Y[tr_i])
            Y_res[val_i] = Y[val_i] - rf_y.predict(X[val_i])

            rf_d = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state + 1,
                n_jobs=1,
            )
            rf_d.fit(X[tr_i], D[tr_i])
            D_res[val_i] = D[val_i] - rf_d.predict(X[val_i])

        # -- Global ATE via OLS on Robinson residuals --
        d2 = float(np.dot(D_res, D_res))
        if d2 == 0.0:
            theta_hat = 0.0
        else:
            theta_hat = float(np.dot(D_res, Y_res) / d2)

        # -- Per-sample pseudo-outcomes for R-learner CATE --
        # pseudo_i = (Ỹᵢ - theta_hat * D̃ᵢ) / D̃ᵢ + theta_hat
        #           ≈ CATE_i  when |D̃ᵢ| is non-negligible
        d_res_std = float(np.std(D_res, ddof=1)) if n > 1 else 1.0
        min_d = max(0.1 * d_res_std, 1e-8)  # trimming threshold

        keep = np.abs(D_res) >= min_d
        if keep.sum() < max(5, n // 10):
            # Fallback: constant CATE = theta_hat everywhere
            cate_full = np.full(n, theta_hat)
        else:
            pseudo = np.where(
                keep,
                (Y_res - theta_hat * D_res) / np.where(keep, D_res, 1.0) + theta_hat,
                theta_hat,
            )

            # Honest RF: train on train_idx, predict on all X
            rf_cate = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state + 2,
                n_jobs=1,
            )
            rf_cate.fit(X[train_idx], pseudo[train_idx])
            cate_full = rf_cate.predict(X)

        ate = float(np.mean(cate_full))
        ate_se = (
            float(np.std(cate_full, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        )
        return CausalForestResult(
            cate=cate_full,
            ate=ate,
            ate_se=ate_se,
            n_obs=n,
            method="honest_rf_approx",
            converged=True,
        )

    # --- No estimator available ---
    warnings.warn(
        "Neither econml nor sklearn is available. "
        "fit_causal_forest returns a placeholder result.",
        ImportWarning,
        stacklevel=2,
    )
    return CausalForestResult(
        cate=np.full(n, float("nan")),
        ate=float("nan"),
        ate_se=float("nan"),
        n_obs=n,
        method="placeholder_no_sklearn",
        converged=False,
    )


__all__ = [
    "CausalForestResult",
    "PLRResult",
    "fit_causal_forest",
    "fit_plr",
]
