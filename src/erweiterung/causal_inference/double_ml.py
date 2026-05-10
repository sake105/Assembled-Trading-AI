"""DoubleML — Chernozhukov et al. (2018) Double/Debiased Machine Learning.

Idee
----
Im klassischen OLS: Y = β·T + α·X + ε. Wenn man β (Treatment-Effekt von T)
unverfälscht schätzen will, muss X korrekt kontrolliert werden — sonst leakt
Confounding-Bias durch X in β̂.

DoubleML löst das via:
1. Cross-Fit ML-Modelle: ŷ(X), t̂(X) auf Hälfte der Daten.
2. Residuen: y - ŷ(X), t - t̂(X).
3. OLS auf Residuen → unbiased β̂.

Vorteil: ML-Modell darf beliebig komplex sein (Random-Forest, Boost), und
β̂ ist trotzdem √n-konsistent + asymptotisch normal.

Anwendung in Trading
--------------------
- "Hat Faktor X **kausalen** Effekt auf Forward-Return, oder nur Korrelation?"
- A/B-Test-style Auswertung von Trading-Regeln über confound-bekannten Daten.
- Quantify "alpha" eines Signals unter Kontrolle anderer Signale.

Reference
---------
Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W.
& Robins, J. (2018). Double/debiased machine learning for treatment and
structural parameters. *Econometrics Journal* 21.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class DMLResult:
    treatment_effect: float
    standard_error: float
    confidence_interval: tuple[float, float]
    t_stat: float
    p_value: float
    n_obs: int


def _default_regressor(X_train, y_train, X_test):
    """Default = Ridge-Regression (NumPy-only, robust)."""
    Xb = np.column_stack([np.ones(len(X_train)), X_train])
    n, p = Xb.shape
    A = Xb.T @ Xb + 1e-2 * np.eye(p)
    beta = np.linalg.solve(A, Xb.T @ y_train)
    Xeb = np.column_stack([np.ones(len(X_test)), X_test])
    return Xeb @ beta


def double_ml(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
    fit_y: Callable | None = None,
    fit_t: Callable | None = None,
    n_folds: int = 5,
    seed: int = 42,
) -> DMLResult:
    """DoubleML für Treatment-Effekt von T auf Y unter Kontrolle X.

    Args:
        Y: Outcome (n,).
        T: Treatment (n,) — continuous oder binary.
        X: Confounder-Features (n, p).
        fit_y: Callable(X_tr, y_tr, X_te) → predictions. Default Ridge.
        fit_t: dito für Treatment.
        n_folds: K-Fold-Splits.
        seed: RNG.

    Returns:
        DMLResult mit Effect-Estimate + SE + p-value.
    """
    Y = np.asarray(Y, dtype=float)
    T = np.asarray(T, dtype=float)
    X = np.asarray(X, dtype=float)
    n = len(Y)
    if n < 100:
        raise ValueError("need >= 100 samples")

    fit_y = fit_y or _default_regressor
    fit_t = fit_t or _default_regressor

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_size = n // n_folds
    Y_resid = np.zeros(n)
    T_resid = np.zeros(n)

    for k in range(n_folds):
        start = k * fold_size
        end = (k + 1) * fold_size if k < n_folds - 1 else n
        val_idx = perm[start:end]
        train_idx = np.concatenate([perm[:start], perm[end:]])
        # Predict on val
        y_pred = fit_y(X[train_idx], Y[train_idx], X[val_idx])
        t_pred = fit_t(X[train_idx], T[train_idx], X[val_idx])
        Y_resid[val_idx] = Y[val_idx] - y_pred
        T_resid[val_idx] = T[val_idx] - t_pred

    # Final OLS of Y_resid on T_resid
    if T_resid.var() <= 1e-12:
        raise ValueError("treatment residuals have no variation")
    beta = float(np.cov(T_resid, Y_resid, ddof=0)[0, 1] / T_resid.var())
    # Standard error (heteroskedasticity-robust, simple version)
    psi = (Y_resid - beta * T_resid) * T_resid
    var_beta = float(np.mean(psi**2) / (np.mean(T_resid**2) ** 2)) / n
    se = float(np.sqrt(var_beta))
    if se == 0:
        return DMLResult(beta, 0.0, (beta, beta), float("inf"), 0.0, n)
    t_stat = beta / se
    # Two-sided p-value via normal approximation
    from math import erf, sqrt as msqrt

    p_value = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / msqrt(2))))
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se

    return DMLResult(
        treatment_effect=beta,
        standard_error=se,
        confidence_interval=(ci_low, ci_high),
        t_stat=t_stat,
        p_value=p_value,
        n_obs=n,
    )


def propensity_score_matching(
    treatment: pd.Series,
    outcome: pd.Series,
    covariates: pd.DataFrame,
    n_matches: int = 1,
) -> dict:
    """Naive Propensity-Score-Matching (Rosenbaum/Rubin 1983) — nur binary T.

    1. Fit logistic regression e(X) = P(T=1|X).
    2. For each treated unit, find n_matches control units with closest e(X).
    3. ATE = mean(Y_treated - Y_matched_controls).

    Args:
        treatment: 0/1 Series.
        outcome: continuous Series.
        covariates: DataFrame.
        n_matches: nearest neighbors per treated.

    Returns:
        dict mit ate, ate_se, n_treated, n_matched.
    """
    df = pd.concat(
        [treatment.rename("T"), outcome.rename("Y"), covariates],
        axis=1,
    ).dropna()
    if df["T"].nunique() < 2:
        return {"error": "no variation in treatment"}

    T_vals = df["T"].values
    Y_vals = df["Y"].values
    X_vals = df[[c for c in covariates.columns]].values
    # Standardize covariates
    X_std = (X_vals - X_vals.mean(axis=0)) / (X_vals.std(axis=0) + 1e-9)

    # Fit logistic regression
    Xb = np.column_stack([np.ones(len(X_std)), X_std])
    # Newton-Raphson for binary logistic
    beta = np.zeros(Xb.shape[1])
    for _ in range(30):
        eta = Xb @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        W = p * (1 - p)
        XWX = Xb.T @ (W[:, None] * Xb) + 1e-6 * np.eye(Xb.shape[1])
        beta_new = beta + np.linalg.solve(XWX, Xb.T @ (T_vals - p))
        if np.linalg.norm(beta_new - beta) < 1e-6:
            break
        beta = beta_new
    e_X = 1.0 / (1.0 + np.exp(-(Xb @ beta)))

    treated_idx = np.where(T_vals == 1)[0]
    control_idx = np.where(T_vals == 0)[0]
    if len(treated_idx) < 5 or len(control_idx) < 5:
        return {"error": "too few treated/control"}

    ate_terms = []
    for ti in treated_idx:
        # Find nearest-neighbor control units by propensity score
        dists = np.abs(e_X[control_idx] - e_X[ti])
        nn = np.argsort(dists)[:n_matches]
        matched_Y = Y_vals[control_idx[nn]].mean()
        ate_terms.append(Y_vals[ti] - matched_Y)

    ate_array = np.array(ate_terms)
    return {
        "ate": float(ate_array.mean()),
        "ate_se": float(ate_array.std(ddof=1) / np.sqrt(len(ate_array))),
        "n_treated": int(len(treated_idx)),
        "n_control": int(len(control_idx)),
    }


__all__ = ["DMLResult", "double_ml", "propensity_score_matching"]
