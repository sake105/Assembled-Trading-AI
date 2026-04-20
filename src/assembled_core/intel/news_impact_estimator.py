"""News impact estimator — maps event classification to expected return impact.

Converts NewsClassification attributes (severity, event_type, market_direction,
time_horizon, geo_tags) into an estimated return impact in basis points (BPS).

This is a rule-based heuristic, NOT a statistical model. It provides a
calibrated starting point for portfolio-layer consumption.

Usage:
    estimator = NewsImpactEstimator()
    impact = estimator.estimate(classification, geo_tags=["RU"], source_tier="T1")
    print(f"Expected impact: {impact.bps:.0f} BPS, horizon: {impact.horizon_days} days")
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Event-type base impact table (in BPS on affected assets)
# ---------------------------------------------------------------------------

_EVENT_TYPE_BASE_BPS: dict[str, float] = {
    "war_escalation": -180.0,
    "military_strike": -120.0,
    "sanctions": -90.0,
    "energy_disruption": -75.0,
    "political_crisis": -60.0,
    "market_stress": -80.0,
    "cyber_attack": -40.0,
    "natural_disaster": -35.0,
    "trade_policy": -50.0,
    "regulatory": -30.0,
    "central_bank": -25.0,   # rate hike context
    "diplomatic": 20.0,      # peace talks / agreement
    "earnings": 0.0,         # direction-dependent
    "ma_activity": 60.0,     # M&A target premium
}

# Time horizon → scaling factor for impact
_HORIZON_SCALE: dict[str, float] = {
    "intraday": 1.0,   # full immediate impact
    "short": 0.6,      # partial realisation
    "medium": 0.35,
    "long": 0.15,
}

# Time horizon → expected days to mean-revert
_HORIZON_DAYS: dict[str, int] = {
    "intraday": 1,
    "short": 5,
    "medium": 20,
    "long": 60,
}

# Severity (0-10) → multiplier
def _severity_mult(severity: float) -> float:
    return 0.5 + 0.5 * math.sqrt(min(severity, 10.0) / 10.0)

# Source tier → confidence multiplier
_TIER_CONF: dict[str, float] = {
    "T0": 1.0, "T1": 0.90, "T2": 0.70, "T3": 0.50,
}

# Country → base geo-risk premium (BPS additional discount on bearish events)
_GEO_RISK_PREMIUM: dict[str, float] = {
    "RU": 50.0,   # Russia — high geo risk
    "IR": 40.0,   # Iran
    "KP": 35.0,   # North Korea
    "CN": 20.0,   # China — systemic risk
    "IL": 15.0,
    "UA": 30.0,
    "TR": 12.0,
    "VE": 25.0,   # Venezuela
}


@dataclass
class ImpactEstimate:
    """Estimated news impact on affected assets."""
    bps: float               # expected basis point impact (negative = bearish)
    direction: str           # "bearish" / "bullish" / "neutral"
    confidence: float        # 0-1 confidence in estimate
    horizon_days: int        # expected days until mean-reversion
    dominant_event_type: str # most impactful event type
    geo_premium_bps: float   # additional geo-risk component


class NewsImpactEstimator:
    """Maps news classification to expected asset return impact in BPS."""

    def estimate(
        self,
        classification: object,
        *,
        geo_tags: list[str] | None = None,
        source_tier: str = "T2",
    ) -> ImpactEstimate:
        """Estimate the return impact of a news classification.

        Args:
            classification: NewsClassification with event_types, severity,
                            market_direction, time_horizon, confidence fields.
            geo_tags: ISO-2 country codes for geo-risk premium.
            source_tier: Source tier string for confidence scaling.

        Returns:
            ImpactEstimate with BPS impact and metadata.
        """
        event_types: list[str] = list(getattr(classification, "event_types", []))
        severity: float = float(getattr(classification, "severity", 0.0))
        market_direction: str = getattr(classification, "market_direction", "neutral")
        time_horizon: str = getattr(classification, "time_horizon", "short")
        confidence: float = float(getattr(classification, "confidence", 0.5))
        geo_tags = geo_tags or []

        # Find dominant event type (highest absolute base BPS)
        dominant_etype = "earnings"
        best_abs = 0.0
        for et in event_types:
            abs_val = abs(_EVENT_TYPE_BASE_BPS.get(et, 0.0))
            if abs_val > best_abs:
                best_abs = abs_val
                dominant_etype = et

        base_bps = _EVENT_TYPE_BASE_BPS.get(dominant_etype, 0.0)

        # For earnings, direction determines sign
        if dominant_etype == "earnings":
            if market_direction == "bullish":
                base_bps = 80.0
            elif market_direction == "bearish":
                base_bps = -60.0
            else:
                base_bps = 0.0

        # For central_bank, direction determines sign
        if dominant_etype == "central_bank":
            if market_direction == "bearish":
                base_bps = -25.0  # rate hike
            elif market_direction == "bullish":
                base_bps = 30.0   # rate cut

        # Scale by severity, horizon, source tier, and confidence
        sev_mult = _severity_mult(severity)
        horizon_scale = _HORIZON_SCALE.get(time_horizon, 0.6)
        tier_conf = _TIER_CONF.get(source_tier, 0.7)
        final_bps = base_bps * sev_mult * horizon_scale * tier_conf * confidence

        # Geo-risk premium (additional negative BPS for bearish events in high-risk countries)
        geo_premium = 0.0
        if final_bps < 0:  # only for bearish events
            for iso2 in geo_tags:
                geo_premium += _GEO_RISK_PREMIUM.get(iso2.upper(), 0.0)
            geo_premium = min(geo_premium, 80.0)  # cap at 80 BPS
            final_bps -= geo_premium

        final_bps = round(final_bps, 1)
        geo_premium = round(geo_premium, 1)

        # Effective confidence
        eff_confidence = round(confidence * tier_conf, 4)

        return ImpactEstimate(
            bps=final_bps,
            direction=market_direction if market_direction != "mixed" else "neutral",
            confidence=eff_confidence,
            horizon_days=_HORIZON_DAYS.get(time_horizon, 5),
            dominant_event_type=dominant_etype,
            geo_premium_bps=geo_premium,
        )

    def estimate_batch(
        self,
        classifications: list,
        *,
        geo_tags_list: list[list[str]] | None = None,
        source_tier: str = "T2",
    ) -> list[ImpactEstimate]:
        """Estimate impact for a batch of classifications."""
        if geo_tags_list is None:
            geo_tags_list = [[] for _ in classifications]
        return [
            self.estimate(clf, geo_tags=geo, source_tier=source_tier)
            for clf, geo in zip(classifications, geo_tags_list)
        ]


__all__ = ["NewsImpactEstimator", "ImpactEstimate"]
