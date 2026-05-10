"""Bayesian Model Averaging (BMA) für Ensemble-Forecasting.

Theorie
-------
Gegeben K Modelle M_1, ..., M_K mit Posterior-Wahrscheinlichkeiten p(M_k | data):
- Posterior für quantity θ: p(θ | data) = Σ_k p(θ | M_k, data) × p(M_k | data)

Vorteil: berücksichtigt Modell-Unsicherheit + reduziert Overfitting.

Approximation der Modell-Posteriors via BIC:
    p(M_k | data) ∝ exp(-0.5 × BIC_k)

mit BIC_k = -2 log L_k + p_k log n.

Reference
---------
- Hoeting, J., Madigan, D., Raftery, A. & Volinsky, C. (1999). Bayesian
  Model Averaging: A Tutorial. *Statistical Science* 14.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BMAResult:
    model_weights: pd.Series  # posterior probabilities
    predictions: pd.Series  # weighted forecast per timepoint
    log_likelihoods: pd.Series
    bics: pd.Series


def bma_predict(
    model_predictions: pd.DataFrame,
    log_likelihoods: dict[str, float] | None = None,
    n_params: dict[str, int] | None = None,
    n_obs: int | None = None,
    y_true: pd.Series | None = None,
) -> BMAResult:
    """Bayesian Model Averaging über Modell-Vorhersagen.

    Args:
        model_predictions: DataFrame (T, K) — predictions je Modell.
        log_likelihoods: dict model_name -> log-likelihood. Wenn None und y_true
            given → wird empirisch (Gauss-LL) berechnet.
        n_params: dict model_name -> Parameter-Anzahl. Pflicht für BIC.
        n_obs: number of observations (default = len(model_predictions)).
        y_true: optional ground truth für empirische LL.

    Returns:
        BMAResult.
    """
    T, K = model_predictions.shape
    n_obs = n_obs or T

    # Compute log-likelihoods if needed
    ll: dict[str, float] = {}
    if log_likelihoods is None:
        if y_true is None:
            raise ValueError("either log_likelihoods or y_true required")
        for col in model_predictions.columns:
            resid = y_true.reindex(model_predictions.index) - model_predictions[col]
            sigma2 = float(np.var(resid.dropna()))
            n_valid = int(resid.notna().sum())
            if sigma2 <= 0 or n_valid == 0:
                ll[col] = -np.inf
            else:
                ll[col] = float(-0.5 * n_valid * (np.log(2 * np.pi * sigma2) + 1))
    else:
        ll = log_likelihoods

    # Compute BICs
    bics: dict[str, float] = {}
    for col in model_predictions.columns:
        p = (n_params or {}).get(col, 1)
        bics[col] = -2 * ll[col] + p * np.log(n_obs)

    # Posterior model probabilities via BIC approximation
    bic_arr = np.array([bics[c] for c in model_predictions.columns])
    bic_min = bic_arr.min()
    weights_raw = np.exp(-0.5 * (bic_arr - bic_min))
    weights = weights_raw / weights_raw.sum()
    weight_series = pd.Series(weights, index=model_predictions.columns)

    # BMA prediction
    bma_pred = (model_predictions * weights).sum(axis=1)

    return BMAResult(
        model_weights=weight_series,
        predictions=bma_pred,
        log_likelihoods=pd.Series(ll),
        bics=pd.Series(bics),
    )


__all__ = ["BMAResult", "bma_predict"]
