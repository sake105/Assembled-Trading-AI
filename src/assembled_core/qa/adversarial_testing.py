"""Adversarial Robustness Testing (M25 Task 25.5).

Tests model robustness against:
1. FGSM-style input perturbations
2. Data poisoning detection
3. Input validation for anomalous features

If small feature changes cause large prediction changes → overfitting risk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PerturbationResult:
    """Result of adversarial perturbation test."""
    original_prediction: float
    perturbed_prediction: float
    prediction_change: float
    perturbation_norm: float
    sensitivity: float         # change / perturbation
    is_robust: bool


@dataclass
class AdversarialReport:
    """Full adversarial robustness report."""
    avg_sensitivity: float
    max_sensitivity: float
    pct_robust: float           # Fraction of samples that are robust
    stale_features_detected: int
    out_of_bounds_detected: int
    sudden_jumps_detected: int
    overall_robust: bool


def fgsm_perturbation(
    features: np.ndarray,
    predict_fn,
    epsilon: float = 1.0,
    n_samples: int = 100,
) -> list[PerturbationResult]:
    """FGSM-style perturbation test (Task 25.5 part 1).

    Perturbs each feature by ±epsilon standard deviations and measures
    how much the prediction changes. High sensitivity = fragile model.

    Args:
        features: (N, d) feature matrix.
        predict_fn: Callable that takes (N, d) array → (N,) predictions.
        epsilon: Perturbation size in standard deviations.
        n_samples: Number of samples to test.

    Returns:
        List of PerturbationResult per sample.
    """
    n = min(n_samples, len(features))
    indices = np.random.choice(len(features), n, replace=False)
    std = np.std(features, axis=0) + 1e-8

    results = []
    for idx in indices:
        x_orig = features[idx:idx + 1]
        pred_orig = float(predict_fn(x_orig)[0])

        # Random direction perturbation
        direction = np.random.randn(1, features.shape[1])
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        perturbation = epsilon * direction * std

        x_perturbed = x_orig + perturbation
        pred_perturbed = float(predict_fn(x_perturbed)[0])

        change = abs(pred_perturbed - pred_orig)
        pert_norm = float(np.linalg.norm(perturbation / std))
        sensitivity = change / max(pert_norm, 1e-8)

        results.append(PerturbationResult(
            original_prediction=round(pred_orig, 6),
            perturbed_prediction=round(pred_perturbed, 6),
            prediction_change=round(change, 6),
            perturbation_norm=round(pert_norm, 4),
            sensitivity=round(sensitivity, 6),
            is_robust=sensitivity < 0.1,
        ))

    return results


def detect_stale_features(
    features: pd.DataFrame,
    max_identical_days: int = 5,
) -> dict[str, list[int]]:
    """Detect stale (unchanged) features (Task 25.5 part 3c).

    Features identical for >max_identical_days consecutive days
    indicate data feed issues.

    Args:
        features: Feature DataFrame with dates as index.
        max_identical_days: Threshold for staleness.

    Returns:
        {column: [list of stale start indices]}.
    """
    stale = {}
    for col in features.columns:
        vals = features[col].values
        runs = []
        run_start = 0
        for i in range(1, len(vals)):
            if vals[i] == vals[i - 1] or (np.isnan(vals[i]) and np.isnan(vals[i - 1])):
                continue
            if i - run_start > max_identical_days:
                runs.append(run_start)
            run_start = i
        if len(vals) - run_start > max_identical_days:
            runs.append(run_start)
        if runs:
            stale[col] = runs

    return stale


def detect_out_of_bounds(
    features: pd.DataFrame,
    historical_features: pd.DataFrame,
    n_sigma: float = 5.0,
) -> dict[str, int]:
    """Detect features outside historical bounds (Task 25.5 part 3a).

    Args:
        features: Current feature values.
        historical_features: Historical feature values for bounds.
        n_sigma: Number of standard deviations for bounds.

    Returns:
        {column: count_of_oob_values}.
    """
    oob = {}
    for col in features.columns:
        if col not in historical_features.columns:
            continue
        hist = historical_features[col].dropna()
        if len(hist) < 30:
            continue
        mean = hist.mean()
        std = hist.std()
        lower = mean - n_sigma * std
        upper = mean + n_sigma * std
        current = features[col].dropna()
        n_oob = int(((current < lower) | (current > upper)).sum())
        if n_oob > 0:
            oob[col] = n_oob

    return oob


def detect_sudden_jumps(
    features: pd.DataFrame,
    max_daily_change_sigma: float = 5.0,
) -> dict[str, int]:
    """Detect sudden overnight jumps in features (Task 25.5 part 3b).

    Args:
        features: Feature DataFrame with dates as index.
        max_daily_change_sigma: Max allowed daily change in sigma.

    Returns:
        {column: count_of_jumps}.
    """
    jumps = {}
    for col in features.columns:
        vals = features[col].dropna()
        if len(vals) < 30:
            continue
        daily_change = vals.diff()
        std_change = daily_change.std()
        if std_change < 1e-10:
            continue
        n_jumps = int((daily_change.abs() > max_daily_change_sigma * std_change).sum())
        if n_jumps > 0:
            jumps[col] = n_jumps

    return jumps


def run_adversarial_audit(
    features: pd.DataFrame,
    predict_fn=None,
    historical_features: pd.DataFrame | None = None,
    epsilon: float = 1.0,
    n_perturbation_samples: int = 50,
) -> AdversarialReport:
    """Run full adversarial robustness audit.

    Args:
        features: Current feature DataFrame.
        predict_fn: Model prediction function (optional).
        historical_features: Historical features for bounds (optional).
        epsilon: Perturbation size.
        n_perturbation_samples: Samples for perturbation test.

    Returns:
        AdversarialReport.
    """
    # Perturbation test
    sensitivities = []
    pct_robust = 1.0
    if predict_fn is not None:
        results = fgsm_perturbation(
            features.values, predict_fn, epsilon, n_perturbation_samples,
        )
        sensitivities = [r.sensitivity for r in results]
        pct_robust = sum(1 for r in results if r.is_robust) / max(len(results), 1)

    # Stale features
    stale = detect_stale_features(features)
    n_stale = sum(len(v) for v in stale.values())

    # Out of bounds
    n_oob = 0
    if historical_features is not None:
        oob = detect_out_of_bounds(features, historical_features)
        n_oob = sum(oob.values())

    # Sudden jumps
    jump_dict = detect_sudden_jumps(features)
    n_jumps = sum(jump_dict.values())

    avg_sens = float(np.mean(sensitivities)) if sensitivities else 0.0
    max_sens = float(np.max(sensitivities)) if sensitivities else 0.0

    overall = pct_robust > 0.8 and n_stale == 0 and n_oob == 0

    report = AdversarialReport(
        avg_sensitivity=round(avg_sens, 6),
        max_sensitivity=round(max_sens, 6),
        pct_robust=round(pct_robust, 4),
        stale_features_detected=n_stale,
        out_of_bounds_detected=n_oob,
        sudden_jumps_detected=n_jumps,
        overall_robust=overall,
    )

    logger.info("[Adversarial] Audit: robust=%.0f%%, stale=%d, OOB=%d, jumps=%d → %s",
                pct_robust * 100, n_stale, n_oob, n_jumps,
                "PASS" if overall else "FAIL")

    return report


__all__ = [
    "PerturbationResult",
    "AdversarialReport",
    "fgsm_perturbation",
    "detect_stale_features",
    "detect_out_of_bounds",
    "detect_sudden_jumps",
    "run_adversarial_audit",
]
