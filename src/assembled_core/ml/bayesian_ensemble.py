"""Bayesian Model Averaging (BMA) über mehrere ML-Modelle.

Problem mit Single-Best-Model: Selection ist rauschbehaftet; das "beste"
Modell in CV ist oft nicht robust.

BMA-Lösung:
- Gewichte jedes Modells nach posterior P(M|data) ∝ P(data|M) · P(M)
- Final-Prediction = weighted Average über Modelle
- Häufige Approximation: BIC-gewichtet oder log-likelihood-gewichtet

Hier implementiert:
1. Likelihood-based weights (aus Validation-Set-Fehler)
2. Softmax-Weighting über negative Log-Likelihood

Ergänzt stacking_ensemble.py:
- Stacking: diskriminativer Level-2-Lernen, kann overfitten
- BMA: generative posterior, robust bei kleinem Validation-Set

PIT-Invariante: Weights werden auf Validation-Set berechnet, nicht Test-Set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BMAResult:
    model_weights: dict[str, float]
    model_validation_scores: dict[str, float]
    """Raw Log-Likelihoods oder MSEs pro Modell."""

    fitted_models: dict[str, object] = field(default_factory=dict)
    feature_cols: list[str] = field(default_factory=list)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Gewichtete Ensemble-Prediction."""
        X_vals = X[self.feature_cols].fillna(0.0).values
        preds = np.zeros(len(X))
        for name, model in self.fitted_models.items():
            w = self.model_weights.get(name, 0.0)
            if w <= 0:
                continue
            try:
                preds += w * model.predict(X_vals)  # type: ignore[attr-defined]
            except Exception as exc:
                logger.warning("[BMA] %s predict failed: %s", name, exc)
        return pd.Series(preds, index=X.index, name="bma_prediction")

    def summary(self) -> dict:
        return {
            "n_models": len(self.fitted_models),
            "weights": {k: round(v, 4) for k, v in self.model_weights.items()},
            "val_scores": {k: round(v, 4) for k, v in self.model_validation_scores.items()},
            "n_features": len(self.feature_cols),
        }


def compute_bma_weights(
    validation_scores: dict[str, float],
    temperature: float = 1.0,
    score_type: str = "neg_log_loss",
) -> dict[str, float]:
    """Softmax-Weights über Validation-Scores.

    Args:
        validation_scores: dict{model_name: score}.
                           Je NACH score_type: "neg_log_loss" oder "neg_mse" = höher ist besser;
                           "mse" / "log_loss" = niedriger ist besser.
        temperature: Höher → flachere Weights (mehr Diversifikation), niedriger → Winner-takes-all.
        score_type: 'neg_log_loss' / 'log_loss' / 'neg_mse' / 'mse' / 'ic' / 'auc'

    Returns:
        dict{model_name: weight}, Summe = 1.0
    """
    if not validation_scores:
        return {}

    scores = np.array(list(validation_scores.values()))
    names = list(validation_scores.keys())

    # Einheitliche Konvention: höher ist besser
    if score_type in ("mse", "log_loss"):
        utilities = -scores
    else:
        utilities = scores

    # Softmax mit Temperature
    scaled = utilities / max(temperature, 1e-6)
    scaled = scaled - scaled.max()  # Stabilität
    exp_u = np.exp(scaled)
    weights = exp_u / exp_u.sum()

    return dict(zip(names, [float(w) for w in weights]))


def run_bayesian_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_factories: dict[str, callable],
    feature_cols: list[str] | None = None,
    temperature: float = 1.0,
    score_type: str = "neg_mse",
) -> BMAResult:
    """Trainiert mehrere Modelle und berechnet BMA-Weights via Validation.

    Args:
        X_train, y_train: Training-Daten
        X_val, y_val: Validation-Set für Weight-Bestimmung (muss zeitlich NACH Train liegen)
        model_factories: dict{name: callable()->model}
        feature_cols: Features (None = alle numerischen)
        temperature: Softmax-Temperature
        score_type: 'neg_mse' für Regression (höher ist besser)

    Returns:
        BMAResult mit gefitteten Modellen + Weights.
    """
    feat_cols = feature_cols or list(X_train.select_dtypes(include="number").columns)
    X_train_vals = X_train[feat_cols].fillna(0.0).values
    y_train_vals = y_train.values
    X_val_vals = X_val[feat_cols].fillna(0.0).values
    y_val_vals = y_val.values

    fitted: dict[str, object] = {}
    val_scores: dict[str, float] = {}

    for name, factory in model_factories.items():
        try:
            model = factory()
            model.fit(X_train_vals, y_train_vals)  # type: ignore[attr-defined]
            preds = model.predict(X_val_vals)  # type: ignore[attr-defined]
            mse = float(np.mean((preds - y_val_vals) ** 2))
            if score_type == "neg_mse":
                val_scores[name] = -mse
            elif score_type == "mse":
                val_scores[name] = mse
            elif score_type == "ic":
                if np.std(preds) < 1e-9:
                    val_scores[name] = 0.0
                else:
                    ic = np.corrcoef(preds, y_val_vals)[0, 1]
                    val_scores[name] = float(ic) if not np.isnan(ic) else 0.0
            else:
                val_scores[name] = -mse

            fitted[name] = model
            logger.info("[BMA] %s validation_score=%.4f", name, val_scores[name])
        except Exception as exc:
            logger.warning("[BMA] %s training failed: %s", name, exc)

    if not fitted:
        raise ValueError("Keine Modelle erfolgreich trainiert")

    weights = compute_bma_weights(val_scores, temperature=temperature, score_type=score_type)
    logger.info("[BMA] Weights: %s", {k: round(v, 4) for k, v in weights.items()})

    return BMAResult(
        model_weights=weights,
        model_validation_scores=val_scores,
        fitted_models=fitted,
        feature_cols=feat_cols,
    )


__all__ = [
    "BMAResult",
    "compute_bma_weights",
    "run_bayesian_ensemble",
]
