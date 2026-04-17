"""F1 — IC-decay-weighted factor combination (Plan v3 Part F1).

Replaces static factor weights with an IC-aware, decay-adjusted mix:

    raw_weight_i = rolling_90d_IC_i * exp(-lag_i / half_life_i)
    weight_i     = clip(raw_weight_i, 0, max_w_per_factor)
    normalized   = weight_i / sum(weights)

Inputs
------

* ``ic_snapshot``: mapping of factor → most-recent 90d rolling IC (signed).
* ``lags``: mapping of factor → days since that factor last rebalanced.
* ``half_lives``: mapping of factor → IC half-life in days (from D5 profile).
* ``max_w_per_factor``: cap per factor (default 0.25).

Design notes
------------

* Negative IC factors get a weight of **0** — this gate rejects factors
  whose predictive sign flipped rather than shorting them blindly.
* Missing half-life defaults to 30 days, matching D5 defaults, and is
  logged once per call for auditability.
* When every factor has non-positive IC, the caller-supplied
  ``fallback_weights`` are returned (equal-weight if unspecified). The
  plan explicitly rejects "silent muting" — a missing signal is always a
  visible event.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Mapping

logger = logging.getLogger(__name__)

DEFAULT_MAX_W_PER_FACTOR = 0.25
DEFAULT_HALF_LIFE_DAYS = 30.0


@dataclass(frozen=True)
class ICDecayWeightResult:
    weights: dict[str, float]
    raw_weights: dict[str, float]
    fallback_used: bool
    clipped_factors: tuple[str, ...]


def compute_ic_decay_weights(
    ic_snapshot: Mapping[str, float],
    *,
    lags: Mapping[str, float] | None = None,
    half_lives: Mapping[str, float] | None = None,
    max_w_per_factor: float = DEFAULT_MAX_W_PER_FACTOR,
    fallback_weights: Mapping[str, float] | None = None,
) -> ICDecayWeightResult:
    """Return normalised per-factor weights from IC + decay inputs.

    Args:
        ic_snapshot: ``{factor: rolling_ic}`` — a single IC observation per
            factor. Callers decide the rolling window (plan uses 90d).
        lags: ``{factor: days_since_last_rebalance}``; missing defaults to 0.
        half_lives: ``{factor: ic_half_life_days}``; missing defaults to 30d.
        max_w_per_factor: upper bound on any single factor's weight.
        fallback_weights: returned verbatim when no factor has positive IC.

    Returns:
        :class:`ICDecayWeightResult` with normalised weights, raw pre-clip
        weights, a fallback flag, and the set of clipped factors.
    """
    lags = lags or {}
    half_lives = half_lives or {}

    raw: dict[str, float] = {}
    for name, ic in ic_snapshot.items():
        try:
            ic_val = float(ic)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(ic_val) or ic_val <= 0.0:
            raw[name] = 0.0
            continue
        lag = float(lags.get(name, 0.0))
        hl = float(half_lives.get(name, DEFAULT_HALF_LIFE_DAYS))
        if hl <= 0.0:
            hl = DEFAULT_HALF_LIFE_DAYS
        decay = math.exp(-lag / hl)
        raw[name] = ic_val * decay

    clipped: list[str] = []
    clipped_vals: dict[str, float] = {}
    for name, w in raw.items():
        if w > max_w_per_factor:
            clipped.append(name)
            clipped_vals[name] = max_w_per_factor
        else:
            clipped_vals[name] = max(w, 0.0)

    total = sum(clipped_vals.values())
    if total <= 0.0:
        fallback = dict(fallback_weights or {})
        if not fallback and ic_snapshot:
            even = 1.0 / len(ic_snapshot)
            fallback = {k: even for k in ic_snapshot}
        logger.info(
            "[IC-DECAY] all IC non-positive — using fallback weights (n=%d)",
            len(fallback),
        )
        return ICDecayWeightResult(
            weights=fallback,
            raw_weights=raw,
            fallback_used=True,
            clipped_factors=tuple(clipped),
        )

    normalised = {k: v / total for k, v in clipped_vals.items()}
    return ICDecayWeightResult(
        weights=normalised,
        raw_weights=raw,
        fallback_used=False,
        clipped_factors=tuple(sorted(clipped)),
    )
