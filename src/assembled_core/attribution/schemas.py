"""Attribution data models.

From 38_FEATURE_ATTRIBUTION_DASHBOARD.md §2.1.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class CompositeAttribution:
    """Per-decision attribution for a single composite score."""
    timestamp: datetime
    ticker: str
    composite_score: float
    dimension_contributions: dict[str, float]  # dim → weighted contribution
    dimension_raw_scores: dict[str, float]      # dim → raw score pre-weight
    dimension_weights: dict[str, float]         # dim → weight
    strategy_id: str
    model_version: str
    regime: str

    def top_contributors(self, n: int = 3) -> dict[str, float]:
        """Return the n dimensions with the largest absolute contribution."""
        return dict(
            sorted(
                self.dimension_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:n]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "ticker": self.ticker,
            "composite_score": self.composite_score,
            "dimension_contributions": self.dimension_contributions,
            "dimension_raw_scores": self.dimension_raw_scores,
            "dimension_weights": self.dimension_weights,
            "strategy_id": self.strategy_id,
            "model_version": self.model_version,
            "regime": self.regime,
        }


__all__ = ["CompositeAttribution"]
