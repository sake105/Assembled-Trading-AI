"""Bayesian Signal Confidence estimation (Plan 1.9).

Uses conjugate Normal-Normal Bayesian updating to produce posterior
confidence intervals for each signal score.

Prior: historical distribution Normal(mu_0, sigma_0²)
Likelihood: current cross-section of scores
Posterior: Normal(mu_post, sigma_post²)

Narrow CI → high confidence → larger position.
Wide CI → low confidence → smaller position (or skip).

Position sizing uses ``1 / confidence_width`` as scaling factor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SignalConfidence:
    """Posterior confidence for a signal score."""

    point_estimate: float
    ci_lower: float
    ci_upper: float
    confidence_width: float  # ci_upper - ci_lower
    prior_mean: float
    prior_std: float
    n_obs: int


def bayesian_update_normal(
    prior_mean: float,
    prior_var: float,
    observations: np.ndarray,
) -> tuple[float, float]:
    """Conjugate Normal-Normal Bayesian update.

    Prior: N(prior_mean, prior_var)
    Likelihood: observations ~ N(mu, sigma²) with known sigma² = sample_var

    Posterior: N(posterior_mean, posterior_var)

    Args:
        prior_mean: Prior mean for the signal.
        prior_var: Prior variance.
        observations: Current cross-sectional observations.

    Returns:
        (posterior_mean, posterior_var).
    """
    n = len(observations)
    if n == 0:
        return prior_mean, prior_var

    sample_mean = float(np.mean(observations))
    sample_var = float(np.var(observations, ddof=1)) if n > 1 else prior_var

    if sample_var < 1e-15:
        return sample_mean, prior_var / (n + 1)

    # Conjugate update
    posterior_var = 1.0 / (1.0 / prior_var + n / sample_var)
    posterior_mean = posterior_var * (prior_mean / prior_var + n * sample_mean / sample_var)

    return posterior_mean, posterior_var


def compute_signal_confidence(
    current_scores: pd.Series,
    historical_scores: pd.Series | None = None,
    *,
    ci_level: float = 0.90,
    min_history: int = 30,
) -> dict[str, SignalConfidence]:
    """Compute Bayesian confidence for each symbol's signal score.

    Args:
        current_scores: Series indexed by symbol with current signal scores.
        historical_scores: Series of historical cross-sectional signal means
            (for prior estimation).  If None, uses current cross-section.
        ci_level: Credible interval level (default 90%).
        min_history: Minimum history for prior estimation.

    Returns:
        Dict mapping symbol → SignalConfidence.
    """
    from scipy.stats import norm

    z = norm.ppf(0.5 + ci_level / 2)

    # Estimate prior from history or current cross-section
    if historical_scores is not None and len(historical_scores) >= min_history:
        prior_mean = float(historical_scores.mean())
        prior_std = float(historical_scores.std()) if len(historical_scores) > 1 else 1.0
    else:
        valid = current_scores.dropna()
        prior_mean = float(valid.mean()) if len(valid) > 0 else 0.0
        prior_std = float(valid.std()) if len(valid) > 1 else 1.0

    prior_var = max(prior_std ** 2, 1e-10)

    # Cross-sectional observations
    observations = current_scores.dropna().values

    # Hoist constant (symbol-independent) computations outside the loop
    post_mean, post_var = bayesian_update_normal(prior_mean, prior_var, observations)
    individual_var = prior_var / 2.0
    combined_precision = 1.0 / individual_var + 1.0 / post_var
    individual_post_var = 1.0 / combined_precision
    post_std = np.sqrt(individual_post_var)
    half_ci = z * post_std
    n_obs = len(observations)

    results: dict[str, SignalConfidence] = {}

    for symbol, score in current_scores.items():
        if pd.isna(score):
            continue

        individual_mean = (score / individual_var + post_mean / post_var) / combined_precision
        ci_lower = individual_mean - half_ci
        ci_upper = individual_mean + half_ci

        results[symbol] = SignalConfidence(
            point_estimate=round(individual_mean, 6),
            ci_lower=round(ci_lower, 6),
            ci_upper=round(ci_upper, 6),
            confidence_width=round(ci_upper - ci_lower, 6),
            prior_mean=round(prior_mean, 6),
            prior_std=round(prior_std, 6),
            n_obs=n_obs,
        )

    return results


def confidence_position_scaler(
    confidence: SignalConfidence,
    *,
    max_scale: float = 2.0,
    min_scale: float = 0.2,
    reference_width: float | None = None,
) -> float:
    """Convert confidence into position size scaler.

    ``scale = min(max_scale, reference_width / confidence_width)``

    Narrow CI → larger position, wide CI → smaller position.

    Args:
        confidence: SignalConfidence from Bayesian update.
        max_scale: Maximum scaling factor.
        min_scale: Minimum scaling factor.
        reference_width: Reference CI width for unit scale.
            If None, uses ``2 * prior_std``.

    Returns:
        Position size multiplier in [min_scale, max_scale].
    """
    if reference_width is None:
        reference_width = 2 * confidence.prior_std

    if confidence.confidence_width <= 0 or reference_width <= 0:
        return 1.0

    scale = reference_width / confidence.confidence_width
    return float(np.clip(scale, min_scale, max_scale))


__all__ = [
    "SignalConfidence",
    "bayesian_update_normal",
    "compute_signal_confidence",
    "confidence_position_scaler",
]
