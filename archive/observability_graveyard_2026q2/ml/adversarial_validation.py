"""Adversarial Validation: Detektiert Distribution-Shift zwischen Train und Test.

Kernidee: Trainiere einen Binary-Classifier der Train-Samples (Label=0) von
Test-Samples (Label=1) unterscheidet. Gut trennbar → Distribution-Shift.

Anwendung:
- Vor jedem Retrain: ist die aktuelle Feature-Verteilung noch wie zur
  Trainingszeit? AUC > 0.7 → ernstes Regime-Shift-Warnsignal
- Feature-Ebene: welche Features trennen Train/Test? → Kandidaten für Drift
- Sample-Weighting: Predict-Proba auf Train kann als Stichproben-Gewicht
  genutzt werden (höheres Gewicht für Train-Samples die "testähnlich" sind)

Unterscheidet sich von qa/adversarial_testing.py, das Input-Perturbation-
Robustness testet, nicht Distribution-Shift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AdversarialResult:
    """Adversarial-Validation-Ergebnis."""

    auc: float
    """AUC des Train-vs-Test-Classifiers. 0.5 = keine Trennung, 1.0 = perfekt trennbar."""

    top_drift_features: list[tuple[str, float]] = field(default_factory=list)
    """Features mit höchster Feature-Importance im Drift-Classifier."""

    n_train: int = 0
    n_test: int = 0

    def interpret(self) -> str:
        if self.auc < 0.55:
            return "NO_SHIFT (AUC < 0.55)"
        if self.auc < 0.65:
            return "MILD_SHIFT (0.55 ≤ AUC < 0.65)"
        if self.auc < 0.80:
            return "STRONG_SHIFT (0.65 ≤ AUC < 0.80)"
        return "EXTREME_SHIFT (AUC ≥ 0.80) — Retrain dringend empfohlen"


def run_adversarial_validation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_cols: list[str] | None = None,
    classifier: object | None = None,
    top_k_features: int = 10,
) -> AdversarialResult:
    """Adversarial Validation via Binary-Classifier.

    Args:
        X_train: Training-Features
        X_test: Test-Features (aktuelle Inferenz-Daten)
        feature_cols: Features (None = alle numerischen, gemeinsam in beiden DFs)
        classifier: sklearn-kompatibler Classifier. Default: RandomForest.
        top_k_features: Anzahl Top-Drift-Features in Report.

    Returns:
        AdversarialResult
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    if feature_cols is None:
        num_train = set(X_train.select_dtypes(include="number").columns)
        num_test = set(X_test.select_dtypes(include="number").columns)
        feature_cols = sorted(num_train & num_test)

    if not feature_cols:
        raise ValueError("Keine gemeinsamen numerischen Feature-Spalten gefunden")

    X_all = pd.concat([
        X_train[feature_cols].assign(_is_test=0),
        X_test[feature_cols].assign(_is_test=1),
    ], ignore_index=True)
    y_all = X_all.pop("_is_test").values
    X_arr = X_all.fillna(0.0).values

    clf = classifier or RandomForestClassifier(
        n_estimators=100, max_depth=6, random_state=42, n_jobs=-1,
    )

    # Held-out split für AUC-Schätzung
    X_fit, X_eval, y_fit, y_eval = train_test_split(
        X_arr, y_all, test_size=0.3, random_state=42, stratify=y_all,
    )
    clf.fit(X_fit, y_fit)  # type: ignore[attr-defined]
    proba = clf.predict_proba(X_eval)[:, 1]  # type: ignore[attr-defined]
    auc = float(roc_auc_score(y_eval, proba))

    # Feature-Importance (wenn verfügbar)
    top_drift: list[tuple[str, float]] = []
    importances = getattr(clf, "feature_importances_", None)
    if importances is not None:
        paired = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
        top_drift = [(f, float(imp)) for f, imp in paired[:top_k_features]]

    result = AdversarialResult(
        auc=auc,
        top_drift_features=top_drift,
        n_train=len(X_train),
        n_test=len(X_test),
    )
    logger.info(
        "[AdvVal] AUC=%.3f (%s); top-drift: %s",
        auc,
        result.interpret(),
        [f for f, _ in top_drift[:3]],
    )
    return result


def sample_weight_from_adversarial(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_cols: list[str] | None = None,
    classifier: object | None = None,
    max_weight: float = 10.0,
) -> pd.Series:
    """Berechnet Sample-Weights für Train-Samples basierend auf Test-Ähnlichkeit.

    Likelihood-Ratio: P(test|x) / P(train|x). Samples die "testähnlich" sind
    bekommen höheres Gewicht, sodass das Training auf distribution-shifted-
    Test-Set gezielter anpasst.

    Returns:
        pd.Series mit Weights für X_train (Index passt).
    """
    from sklearn.ensemble import RandomForestClassifier

    if feature_cols is None:
        num_train = set(X_train.select_dtypes(include="number").columns)
        num_test = set(X_test.select_dtypes(include="number").columns)
        feature_cols = sorted(num_train & num_test)

    X_all = pd.concat([
        X_train[feature_cols].assign(_is_test=0),
        X_test[feature_cols].assign(_is_test=1),
    ], ignore_index=True)
    y_all = X_all.pop("_is_test").values
    X_arr = X_all.fillna(0.0).values

    clf = classifier or RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, n_jobs=-1,
    )
    clf.fit(X_arr, y_all)  # type: ignore[attr-defined]

    # Proba nur für Train-Samples
    train_proba = clf.predict_proba(  # type: ignore[attr-defined]
        X_train[feature_cols].fillna(0.0).values
    )[:, 1]

    eps = 1e-6
    ratio = train_proba / np.maximum(1.0 - train_proba, eps)
    weights = np.clip(ratio, 0.0, max_weight)
    return pd.Series(weights, index=X_train.index, name="adv_sample_weight")


__all__ = [
    "AdversarialResult",
    "run_adversarial_validation",
    "sample_weight_from_adversarial",
]
