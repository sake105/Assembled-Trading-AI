"""F2 — Regime posterior + EWMA smoothing (Plan v3 Part F2).

Replaces hardcoded ``{regime: weights}`` step functions with a probabilistic
posterior × base-weight mix::

    final_weights = Σ_k P(regime=k | x_t) * base_weights[k]

The posterior ``P(regime=k | x_t)`` is expected from an HMM (or any other
state-inference module). To prevent whipsaw on rapid regime flickers, the
posterior is smoothed with an EWMA whose half-life defaults to 5 days per
the plan:

    α = 1 - exp(ln(0.5) / half_life)

Design
------

* Input posteriors must sum to ~1 per timestep. Minor float drift is
  re-normalised; gross violations raise ``ValueError``.
* Base weights are specified as ``{regime: {factor: weight}}``. Any
  factor that appears in one regime dict but not another defaults to 0 in
  the others — *not* to the current regime's value. This forces callers
  to declare intent for every factor per regime.
* The smoother is stateful. Callers pass the previous smoothed posterior
  in and receive the new one out, so the smoothing is deterministic and
  easy to persist across paper-engine cycles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

DEFAULT_HALF_LIFE_DAYS = 5.0
_POSTERIOR_SUM_TOLERANCE = 1e-3


def _alpha_for_half_life(half_life_days: float) -> float:
    if half_life_days <= 0:
        return 1.0
    return 1.0 - math.exp(math.log(0.5) / float(half_life_days))


def smooth_posterior(
    new_posterior: Mapping[str, float],
    prev_smoothed: Mapping[str, float] | None,
    *,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
) -> dict[str, float]:
    """Apply an EWMA smoother to a regime posterior.

    On the first call (``prev_smoothed is None``) the new posterior is
    returned verbatim after normalisation.
    """
    new_total = sum(new_posterior.values())
    if new_total <= 0:
        raise ValueError("posterior entries sum to 0")
    new_norm = {k: float(v) / new_total for k, v in new_posterior.items()}

    if prev_smoothed is None:
        return new_norm

    alpha = _alpha_for_half_life(half_life_days)
    keys = set(new_norm) | set(prev_smoothed)
    blended = {
        k: alpha * new_norm.get(k, 0.0)
        + (1.0 - alpha) * float(prev_smoothed.get(k, 0.0))
        for k in keys
    }
    total = sum(blended.values())
    if total <= 0:
        raise ValueError("smoothed posterior sums to 0")
    return {k: v / total for k, v in blended.items()}


@dataclass(frozen=True)
class RegimeBlendResult:
    weights: dict[str, float]
    posterior_used: dict[str, float]


def blend_weights_by_regime_posterior(
    posterior: Mapping[str, float],
    base_weights_per_regime: Mapping[str, Mapping[str, float]],
) -> RegimeBlendResult:
    """Return weights = Σ_k P(k) * base_weights[k]."""
    total = sum(posterior.values())
    if total <= 0:
        raise ValueError("posterior sums to 0")
    if abs(total - 1.0) > _POSTERIOR_SUM_TOLERANCE:
        posterior = {k: float(v) / total for k, v in posterior.items()}
    else:
        posterior = {k: float(v) for k, v in posterior.items()}

    # Union of every factor mentioned in any regime.
    factors: set[str] = set()
    for m in base_weights_per_regime.values():
        factors.update(m.keys())

    blended: dict[str, float] = {f: 0.0 for f in factors}
    for regime, prob in posterior.items():
        bw = base_weights_per_regime.get(regime)
        if bw is None:
            continue
        for f in factors:
            blended[f] += prob * float(bw.get(f, 0.0))

    return RegimeBlendResult(weights=blended, posterior_used=dict(posterior))
