"""Permutation Feature Importance + SHAP-like Attribution.

Permutation Importance (Breiman 2001)
-------------------------------------
Idee: shuffle die Werte eines Features → wenn Score signifikant fällt, war
das Feature wichtig.

Vorteile vs. native model.feature_importances_:
- Model-agnostisch (funktioniert mit XGB, RF, LGBM, NN, ...)
- Berücksichtigt Modell-Interaktionen
- Out-of-sample-konsistent (test set, nicht train)

SHAP-Lite via Shapley-Sampling
------------------------------
Echte Shapley-Werte über alle Permutationen: O(2^n). Sampling-Approximation:
- Random Permutation der Features → marginal contribution.
- Mittelwert über Samples → Shapley-Value-Estimate.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def permutation_importance(
    model_predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    scoring: Callable[[np.ndarray, np.ndarray], float] | None = None,
    n_repeats: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute permutation feature importance.

    Args:
        model_predict: callable f(X) -> y_pred. Already-trained model.
        X: feature matrix (n, p).
        y: targets.
        scoring: function (y_true, y_pred) -> score. Higher = better. Default = -RMSE.
        n_repeats: shuffles per feature.
        seed: RNG.

    Returns:
        DataFrame [feature_idx, importance_mean, importance_std].
    """

    def _neg_rmse(y_true, y_pred):
        return -float(np.sqrt(((y_true - y_pred) ** 2).mean()))

    if scoring is None:
        scoring = _neg_rmse

    rng = np.random.default_rng(seed)
    base_pred = model_predict(X)
    base_score = scoring(y, base_pred)
    n, p = X.shape

    rows = []
    for j in range(p):
        diffs = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_pred = model_predict(X_perm)
            perm_score = scoring(y, perm_pred)
            diffs.append(base_score - perm_score)  # score drop = importance
        rows.append(
            {
                "feature_idx": j,
                "importance_mean": float(np.mean(diffs)),
                "importance_std": float(np.std(diffs)),
            }
        )
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def shapley_sampling_values(
    model_predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    background_X: np.ndarray | None = None,
    n_samples: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """Approximate Shapley-Values via Random-Permutation-Sampling.

    Reference: Strumbelj/Kononenko 2014.

    Args:
        model_predict: trained model.
        X: instances to explain (n_inst, p).
        background_X: baseline distribution (default = X itself).
        n_samples: number of random permutations.

    Returns:
        Array (n_inst, p) — Shapley value for each (instance, feature).
    """
    if background_X is None:
        background_X = X
    rng = np.random.default_rng(seed)
    n_inst, p = X.shape
    phi = np.zeros((n_inst, p))

    for _ in range(n_samples):
        perm = rng.permutation(p)
        # For each instance, sample a baseline from background
        bg_idx = rng.integers(0, len(background_X), n_inst)
        baseline = background_X[bg_idx]
        # Start at baseline
        running = baseline.copy()
        # Predict baseline once
        prev_pred = model_predict(running)
        for j in perm:
            # Reveal feature j (i.e. swap baseline-j with X-j)
            running[:, j] = X[:, j]
            new_pred = model_predict(running)
            phi[:, j] += new_pred - prev_pred
            prev_pred = new_pred

    return phi / n_samples


__all__ = ["permutation_importance", "shapley_sampling_values"]
