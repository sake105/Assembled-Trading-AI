"""Composite-score attribution builder.

From 38_FEATURE_ATTRIBUTION_DASHBOARD.md §2.2.

Bridges the composite_score module output (score + per-dimension dict)
into a structured CompositeAttribution that can be stored and analysed.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from assembled_core.attribution.schemas import CompositeAttribution

logger = logging.getLogger(__name__)


def build_attribution(
    ticker: str,
    composite_score: float,
    dimension_raw_scores: dict[str, float],
    dimension_weights: dict[str, float],
    regime: str,
    strategy_id: str = "default",
    model_version: str = "0.0.0",
    timestamp: datetime | None = None,
) -> CompositeAttribution:
    """Build a CompositeAttribution from composite_score output.

    Args:
        ticker: Instrument symbol.
        composite_score: Final weighted score in [-1, +1].
        dimension_raw_scores: Per-dimension scores before weighting.
        dimension_weights: Per-dimension weights (should sum ~1.0).
        regime: Market regime label ('calm', 'normal', 'elevated', 'crisis').
        strategy_id: Strategy identifier string.
        model_version: Model version tag.
        timestamp: Moment of computation; defaults to now UTC.

    Returns:
        CompositeAttribution ready for storage or inspection.
    """
    ts = timestamp or datetime.now(tz=timezone.utc)
    contributions = {
        dim: dimension_weights.get(dim, 0.0) * score
        for dim, score in dimension_raw_scores.items()
    }
    return CompositeAttribution(
        timestamp=ts,
        ticker=ticker,
        composite_score=composite_score,
        dimension_contributions=contributions,
        dimension_raw_scores=dimension_raw_scores,
        dimension_weights=dimension_weights,
        strategy_id=strategy_id,
        model_version=model_version,
        regime=regime,
    )


def attribution_to_dict(attr: CompositeAttribution) -> dict[str, Any]:
    return attr.to_dict()


def compute_composite_with_attribution(
    ticker: str,
    dimension_raw_scores: dict[str, float],
    dimension_weights: dict[str, float],
    regime: str,
    strategy_id: str = "default",
    model_version: str = "0.0.0",
    timestamp: datetime | None = None,
) -> tuple[float, CompositeAttribution]:
    """Compute weighted composite score and return attribution alongside.

    The attribution is a zero-cost by-product — computed in the same pass.

    Args:
        ticker: Instrument symbol.
        dimension_raw_scores: Per-dimension scores before weighting.
        dimension_weights: Per-dimension weights.
        regime: Market regime label.
        strategy_id: Strategy identifier string.
        model_version: Model version tag.
        timestamp: Moment of computation; defaults to now UTC.

    Returns:
        (composite_score, CompositeAttribution) tuple.
    """
    contributions = {
        dim: dimension_weights.get(dim, 0.0) * score
        for dim, score in dimension_raw_scores.items()
    }
    composite_score = float(sum(contributions.values()))

    attr = build_attribution(
        ticker=ticker,
        composite_score=composite_score,
        dimension_raw_scores=dimension_raw_scores,
        dimension_weights=dimension_weights,
        regime=regime,
        strategy_id=strategy_id,
        model_version=model_version,
        timestamp=timestamp,
    )
    return composite_score, attr


__all__ = ["build_attribution", "attribution_to_dict", "compute_composite_with_attribution"]
