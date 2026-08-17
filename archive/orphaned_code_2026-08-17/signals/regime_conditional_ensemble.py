"""Regime-aware conditional ensemble (audit C2-055).

A small helper that combines per-model predictions into a regime-
specific weighted average. Different base models perform differently
across market regimes — momentum signals tend to dominate in trending
bull markets, mean-reversion in choppy sideways periods, and
defensive/short-bias in crises. A single global weight vector smears
these conditional edges; a regime-conditional weight matrix preserves
them.

Inputs:

* ``per_regime_weights``: dict[regime_label, np.ndarray] — each weight
  vector is length M (number of base models), sums to 1.
* ``current_regime``: which regime row to use; if absent, fall back to
  ``default_weights`` or to a uniform 1/M.

The helper is **stateless and pure**: it does NOT make a regime call
on its own. The caller (typically a pipeline step that consumes
``risk/regime_models.py``) passes the current regime label in. This
keeps the helper composable and trivially testable.

Activation criterion (in line with §4.3 of KNOWN_ISSUES): switching
from a global weight vector to a per-regime one requires evidence
that the regime-mean OOS Sharpe of each base model is materially
different across regimes — otherwise you are just over-fitting to
noise. The helper exposes :func:`regime_dispersion` so callers can
mechanically check this before turning the overlay on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ConditionalEnsembleResult:
    combined: np.ndarray
    regime_used: str
    weights_used: np.ndarray
    fell_back: bool


def conditional_ensemble(
    *,
    model_predictions: Sequence[np.ndarray],
    current_regime: str,
    per_regime_weights: Mapping[str, np.ndarray],
    default_weights: np.ndarray | None = None,
) -> ConditionalEnsembleResult:
    """Combine M base-model prediction arrays under the active regime.

    Args:
        model_predictions: length-M sequence of (T,) prediction arrays.
        current_regime: regime label to look up in ``per_regime_weights``.
        per_regime_weights: mapping ``regime_label -> length-M weight``
            array. Weights are renormalized to sum to 1 internally.
        default_weights: fallback when ``current_regime`` is absent;
            if also None, uses uniform 1/M.

    Returns:
        :class:`ConditionalEnsembleResult` with the combined prediction
        and a flag indicating whether the fallback path was taken.
    """
    M = len(model_predictions)
    if M < 1:
        raise ValueError("need at least one base model")

    fell_back = False
    if current_regime in per_regime_weights:
        w = np.asarray(per_regime_weights[current_regime], dtype=float).ravel()
        regime_used = current_regime
    elif default_weights is not None:
        w = np.asarray(default_weights, dtype=float).ravel()
        regime_used = "default"
        fell_back = True
    else:
        w = np.full(M, 1.0 / M, dtype=float)
        regime_used = "uniform"
        fell_back = True

    if w.size != M:
        raise ValueError(
            f"weights length ({w.size}) does not match number of models ({M})"
        )
    if (w < 0).any():
        raise ValueError("weights must be non-negative")
    s = w.sum()
    if s <= 0.0:
        raise ValueError("weights must have positive sum")
    w = w / s

    P = np.stack(
        [np.asarray(p, dtype=float).ravel() for p in model_predictions], axis=1
    )
    combined = P @ w

    return ConditionalEnsembleResult(
        combined=combined,
        regime_used=regime_used,
        weights_used=w,
        fell_back=fell_back,
    )


def regime_dispersion(per_regime_weights: Mapping[str, np.ndarray]) -> float:
    """Quantify how strongly the per-regime weights diverge.

    Returns the **mean pair-wise L2 distance** between regime weight
    vectors. A low value (e.g. < 0.05) means the regimes barely
    differ — using a global weight vector would lose nothing.
    A high value (e.g. > 0.30) means regime-conditioning is doing real
    work. Callers can use this as a mechanical activation gate.
    """
    labels = list(per_regime_weights.keys())
    if len(labels) < 2:
        return 0.0
    vecs = []
    for lab in labels:
        v = np.asarray(per_regime_weights[lab], dtype=float).ravel()
        s = v.sum()
        if s <= 0:
            raise ValueError(f"non-positive weight sum for regime {lab!r}")
        vecs.append(v / s)
    M = vecs[0].size
    if any(v.size != M for v in vecs):
        raise ValueError("all regime weight vectors must have the same length")
    dists: list[float] = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            dists.append(float(np.linalg.norm(vecs[i] - vecs[j])))
    return float(np.mean(dists))


__all__ = [
    "ConditionalEnsembleResult",
    "conditional_ensemble",
    "regime_dispersion",
]
