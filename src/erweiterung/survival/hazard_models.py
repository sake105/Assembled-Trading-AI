"""Survival-Analysis (Cox PH + Kaplan-Meier).

Anwendung in Trading
--------------------
- **Delisting-Hazard**: Kleine Caps, hohe Vola → höhere Delisting-Wahrscheinlichkeit.
- **Default-Probability**: Credit-Spread-basierte Survival-Curves.
- **Drawdown-Recovery-Time**: time-to-event = Recovery von Drawdown.

Modelle
-------
- **Kaplan-Meier**: nicht-parametrische Survival-Function.
- **Cox-PH**: Hazard λ(t|x) = λ₀(t) exp(β'x). Semi-parametrisch.

Implementation: lifelines-Bibliothek empfohlen.  Fallback: einfache K-M ohne deps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def kaplan_meier_estimate(durations: np.ndarray, events: np.ndarray) -> pd.DataFrame:
    """Kaplan-Meier Survival-Function-Schätzer.

    Args:
        durations: time-to-event or censoring time.
        events: 1 = event occurred, 0 = censored.

    Returns:
        DataFrame [time, n_at_risk, n_events, survival].
    """
    df = pd.DataFrame({"t": durations, "e": events})
    df = df.sort_values("t").reset_index(drop=True)

    rows = []
    n_at_risk = len(df)
    survival = 1.0
    for t in sorted(df["t"].unique()):
        sub = df[df["t"] == t]
        d = int(sub["e"].sum())
        c = int(len(sub) - d)
        if n_at_risk == 0:
            break
        survival *= 1 - d / n_at_risk
        rows.append(
            {
                "time": t,
                "n_at_risk": n_at_risk,
                "n_events": d,
                "n_censored": c,
                "survival": survival,
            }
        )
        n_at_risk -= d + c
    return pd.DataFrame(rows)


def cox_ph_partial_likelihood_grad(
    beta: np.ndarray,
    X: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Negativer log-partial-likelihood + Gradient für Cox PH (Newton-Schritt).

    Used internally by ``fit_cox_ph_simple``.
    """
    n, p = X.shape
    # sort by duration descending
    order = np.argsort(-durations)
    X_s = X[order]
    e_s = events[order]
    eta = X_s @ beta
    exp_eta = np.exp(eta)
    cum_exp = np.cumsum(exp_eta)
    cum_X = np.cumsum(X_s * exp_eta[:, None], axis=0)

    # log partial likelihood = sum_{e_i=1} [ eta_i - log(sum_{j: t_j>=t_i} exp(eta_j)) ]
    log_pl = (e_s * (eta - np.log(cum_exp))).sum()

    # Gradient
    mean_X = cum_X / cum_exp[:, None]
    grad = (e_s[:, None] * (X_s - mean_X)).sum(axis=0)
    return -float(log_pl), -grad


def fit_cox_ph_simple(
    X: pd.DataFrame,
    durations: pd.Series,
    events: pd.Series,
    n_iter: int = 50,
    lr: float = 0.05,
) -> pd.Series:
    """Vereinfachter Cox-PH-Fit via Gradient-Descent.

    Returns:
        Series of β-coefficients indexed by feature name.
    """
    feats = list(X.columns)
    Xv = X.values.astype(float)
    Xv = (Xv - Xv.mean(axis=0)) / (Xv.std(axis=0) + 1e-9)
    dur = durations.values.astype(float)
    ev = events.values.astype(int)
    beta = np.zeros(Xv.shape[1])
    for _ in range(n_iter):
        nll, grad = cox_ph_partial_likelihood_grad(beta, Xv, dur, ev)
        beta = beta - lr * grad
    return pd.Series(beta, index=feats)


def hazard_score(beta: pd.Series, x: pd.Series) -> float:
    """Compute hazard rate exp(β'x). Higher = more likely to default/delist."""
    return float(np.exp(beta.values @ x.reindex(beta.index).fillna(0).values))


__all__ = [
    "kaplan_meier_estimate",
    "fit_cox_ph_simple",
    "cox_ph_partial_likelihood_grad",
    "hazard_score",
]
