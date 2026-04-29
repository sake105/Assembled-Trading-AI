"""Differential Privacy utilities for model training — stub / skeleton.

Tier 4 item: DP-SGD training (via Google's dp-accounting / Opacus) is a
complex integration that requires careful epsilon-delta budgeting per training
run and is not yet needed for paper-trading.

This stub exposes the intended interface and a pure-Python Gaussian mechanism
for scalar statistics (no ML training). The full Opacus integration is deferred.

Implements:
  - Gaussian mechanism for scalar/vector statistics (release mean, quantile, etc.)
  - Laplace mechanism for scalar statistics
  - Privacy budget accounting (epsilon-delta bookkeeping)
  - Stub for DP-SGD gradient clipping (Opacus when available)

References:
  - Dwork & Roth (2014) "The Algorithmic Foundations of Differential Privacy"
  - Abadi et al. (2016) "Deep Learning with Differential Privacy" (DP-SGD)
  - Mironov (2017) "Rényi Differential Privacy"
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_OPACUS_AVAILABLE = False
try:
    import opacus  # type: ignore[import]
    _OPACUS_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Privacy budget tracker
# ---------------------------------------------------------------------------

@dataclass
class PrivacyBudget:
    """Tracks cumulative epsilon-delta consumption across queries.

    Args:
        epsilon_total: Total privacy budget (e.g. 1.0 for epsilon-1 DP).
        delta: Failure probability (e.g. 1e-5).
    """
    epsilon_total: float = 1.0
    delta: float = 1e-5
    epsilon_used: float = 0.0
    query_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def epsilon_remaining(self) -> float:
        return max(0.0, self.epsilon_total - self.epsilon_used)

    @property
    def is_exhausted(self) -> bool:
        return self.epsilon_used >= self.epsilon_total

    def consume(self, epsilon: float, mechanism: str = "unknown") -> bool:
        """Record consumption of epsilon budget. Returns False if budget exceeded."""
        if self.epsilon_used + epsilon > self.epsilon_total:
            logger.warning(
                "[DP] budget exhausted: used=%.4f total=%.4f requested=%.4f",
                self.epsilon_used, self.epsilon_total, epsilon,
            )
            return False
        self.epsilon_used += epsilon
        self.query_log.append({"mechanism": mechanism, "epsilon": epsilon})
        return True


# ---------------------------------------------------------------------------
# Gaussian mechanism
# ---------------------------------------------------------------------------

def gaussian_noise_scale(
    sensitivity: float,
    epsilon: float,
    delta: float,
) -> float:
    """Compute the Gaussian noise standard deviation for (epsilon, delta)-DP.

    Using the analytic Gaussian mechanism (Balle & Wang 2018):
        sigma >= sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon

    Args:
        sensitivity: L2 global sensitivity of the query.
        epsilon: Privacy parameter.
        delta: Failure probability.

    Returns:
        Noise standard deviation sigma.
    """
    if epsilon <= 0 or delta <= 0 or delta >= 1:
        raise ValueError(f"Invalid DP parameters: epsilon={epsilon}, delta={delta}")
    sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    return sigma


def gaussian_mechanism(
    value: float | np.ndarray,
    sensitivity: float,
    epsilon: float,
    delta: float,
    rng: np.random.Generator | None = None,
) -> float | np.ndarray:
    """Add calibrated Gaussian noise for (epsilon, delta)-DP.

    Args:
        value: True statistic (scalar or array).
        sensitivity: L2 global sensitivity.
        epsilon: Privacy parameter.
        delta: Failure probability.
        rng: NumPy random generator (reproducibility).

    Returns:
        Noisy version of value with same shape.
    """
    rng = rng or np.random.default_rng()
    sigma = gaussian_noise_scale(sensitivity, epsilon, delta)
    if np.isscalar(value):
        noise = rng.normal(0, sigma)
        return float(value) + float(noise)
    arr = np.asarray(value, dtype=float)
    noise = rng.normal(0, sigma, size=arr.shape)
    return arr + noise


# ---------------------------------------------------------------------------
# Laplace mechanism
# ---------------------------------------------------------------------------

def laplace_noise_scale(sensitivity: float, epsilon: float) -> float:
    """Laplace mechanism noise scale b = sensitivity / epsilon."""
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    return sensitivity / epsilon


def laplace_mechanism(
    value: float,
    sensitivity: float,
    epsilon: float,
    rng: np.random.Generator | None = None,
) -> float:
    """Add Laplace noise for epsilon-DP (pure DP, no delta).

    Args:
        value: True scalar statistic.
        sensitivity: L1 global sensitivity.
        epsilon: Privacy parameter.
        rng: NumPy random generator.

    Returns:
        Noisy scalar.
    """
    rng = rng or np.random.default_rng()
    b = laplace_noise_scale(sensitivity, epsilon)
    noise = rng.laplace(0, b)
    return float(value) + float(noise)


# ---------------------------------------------------------------------------
# DP-mean and DP-quantile (common analytics queries)
# ---------------------------------------------------------------------------

def dp_mean(
    data: np.ndarray,
    clip_bound: float,
    epsilon: float,
    delta: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> float:
    """Differentially private mean estimator.

    Clips values to [-clip_bound, clip_bound], computes mean, adds Gaussian noise.
    Sensitivity = 2 * clip_bound / n.

    Args:
        data: 1-D array.
        clip_bound: Clipping bound (symmetric).
        epsilon: Privacy parameter.
        delta: Failure probability.
        rng: Random generator.

    Returns:
        Noisy mean estimate.
    """
    arr = np.clip(np.asarray(data, dtype=float), -clip_bound, clip_bound)
    n = len(arr)
    if n == 0:
        return 0.0
    true_mean = float(np.mean(arr))
    sensitivity = 2.0 * clip_bound / n
    return float(gaussian_mechanism(true_mean, sensitivity, epsilon, delta, rng))


def dp_count(
    n_true: int,
    epsilon: float,
    rng: np.random.Generator | None = None,
) -> int:
    """Differentially private count (Laplace mechanism, sensitivity=1)."""
    noisy = laplace_mechanism(float(n_true), sensitivity=1.0, epsilon=epsilon, rng=rng)
    return max(0, round(noisy))


# ---------------------------------------------------------------------------
# DP-SGD stub (Opacus integration)
# ---------------------------------------------------------------------------

class DPSGDTrainer:
    """Stub for Differentially Private SGD using Opacus.

    When Opacus is installed, this will wrap a PyTorch optimizer with
    gradient clipping and Gaussian noise injection.

    Args:
        max_grad_norm: Per-sample gradient clipping norm.
        noise_multiplier: Ratio of noise std to clipping norm.
        target_epsilon: Privacy budget target.
        target_delta: Failure probability.
    """

    def __init__(
        self,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.1,
        target_epsilon: float = 1.0,
        target_delta: float = 1e-5,
    ) -> None:
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta

        if not _OPACUS_AVAILABLE:
            logger.info(
                "[DPSGDTrainer] Opacus not installed — stub mode. "
                "Install with: pip install opacus"
            )

    def make_private(self, model: Any, optimizer: Any, data_loader: Any) -> tuple[Any, Any, Any]:
        """Wrap model/optimizer for DP-SGD. Raises NotImplementedError in stub mode."""
        if not _OPACUS_AVAILABLE:
            raise NotImplementedError(
                "DPSGDTrainer.make_private() requires Opacus. "
                "Install it, then call: from opacus import PrivacyEngine"
            )

        from opacus import PrivacyEngine  # type: ignore[import]
        engine = PrivacyEngine()
        model_dp, optimizer_dp, dl_dp = engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            target_epsilon=self.target_epsilon,
            target_delta=self.target_delta,
            max_grad_norm=self.max_grad_norm,
        )
        return model_dp, optimizer_dp, dl_dp

    @property
    def opacus_available(self) -> bool:
        return _OPACUS_AVAILABLE


OPACUS_AVAILABLE = _OPACUS_AVAILABLE
