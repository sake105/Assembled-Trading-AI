"""EDCL Phase B — Trigger-Basket Activation.

Connects geo_trigger.score_event() output with news_classifier sector/country
asset maps to produce a TriggerBasket: a structured, conviction-scored bundle
of fired triggers, affected sectors, and candidate assets.

The basket is the primary EDCL input object consumed by:
  - conviction_engine.py (Phase C)
  - composite_score.compute_news_dim_with_edcl() (Phase D)
  - _tc_sizing._sp_compute_final_multiplier (Phase A wired this)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .geo_trigger import KEYWORD_RULES, score_event
from .models import NewsEvent, TriggerType
from .news_classifier import COUNTRY_TO_ASSETS, SECTOR_TO_ETFS

# ---------------------------------------------------------------------------
# TriggerType → primary sectors
# ---------------------------------------------------------------------------

_TRIGGER_SECTOR_MAP: dict[TriggerType, list[str]] = {
    TriggerType.CHOKEPOINT_STRESS: ["energy", "industrials"],
    TriggerType.WAR_ESCALATION: ["defense", "energy"],
    TriggerType.SANCTIONS_ESCALATION: ["financials", "energy"],
    TriggerType.ENERGY_SUPPLY_RISK: ["energy"],
    TriggerType.SHIPPING_DISRUPTION: ["industrials", "energy"],
    TriggerType.CYBER_ESCALATION: ["tech"],
    TriggerType.COUP_RISK: ["financials"],
    TriggerType.POLICY_SHIFT: ["industrials", "financials"],
    TriggerType.BANKING_CRISIS: ["financials"],
    TriggerType.CREDIT_DOWNGRADE: ["financials"],
    TriggerType.RATE_SURPRISE: ["financials"],
    TriggerType.PEG_STRESS: ["financials"],
    TriggerType.FISCAL_CLIFF: ["financials"],
    TriggerType.MILITARY_BUILDUP: ["defense"],
    TriggerType.NUCLEAR_THREAT: ["defense", "energy"],
    TriggerType.CAPABILITY_SHIFT: ["defense", "tech"],
    TriggerType.NEW_EXPORT_CONTROL: ["tech"],
    TriggerType.ENTITY_LISTING: ["tech", "financials"],
    TriggerType.ZERO_DAY_DISCLOSURE: ["tech"],
    TriggerType.TRADE_WAR_ESCALATION: ["industrials"],
    TriggerType.RESOURCE_NATIONALIZATION: ["materials", "energy"],
    TriggerType.SUPPLY_CHAIN_BREAK: ["tech", "industrials"],
    TriggerType.DIPLOMATIC_CRISIS: ["financials"],
    TriggerType.ALLIANCE_SHIFT: ["defense"],
    TriggerType.TERRITORIAL_ESCALATION: ["defense", "energy"],
    TriggerType.STRAIT_BLOCKADE: ["energy", "industrials"],
    # --- Previously missing entries ---
    TriggerType.HEGEMONIC_CHALLENGE: ["defense", "tech"],
    TriggerType.PROXY_WAR_EXPANSION: ["defense", "energy"],
    TriggerType.RESERVE_DRAIN: ["financials"],
    TriggerType.TECHNOLOGY_GAP_WIDENING: ["tech", "defense"],
    TriggerType.MAJOR_BREACH_DETECTED: ["cybersecurity", "technology"],
    TriggerType.STATE_ACTOR_ACTIVITY: ["defense", "geopolitics"],
    TriggerType.SEVERE_WEATHER_ALERT: ["energy", "agriculture", "insurance"],
    TriggerType.LOGISTICS_DISRUPTION: ["industrials", "energy"],
    TriggerType.CASUALTY_SPIKE: ["defense", "healthcare"],
}

# Severity thresholds for conviction aggregation
_HIGH_CONVICTION_THRESHOLD = 0.60
_MIN_SCORE_TO_INCLUDE = 0.05


# ---------------------------------------------------------------------------
# TriggerBasket dataclass
# ---------------------------------------------------------------------------


@dataclass
class TriggerBasket:
    """Structured output of the trigger-basket builder.

    Attributes:
        fired_triggers: List of (TriggerType, score) pairs for all triggers that
            fired above _MIN_SCORE_TO_INCLUDE on at least one event.
        affected_sectors: Dict of sector -> aggregate conviction-weighted score.
        affected_assets: Deduplicated list of candidate ETF/ticker symbols derived
            from sector and country mappings.
        geo_tags: Union of all ISO-2 geo_tags from the input events.
        conviction: Aggregate basket conviction in [0, 1]. Calculated as the
            weighted max across fired triggers, adjusted for trigger diversity.
        n_events: Number of input NewsEvents processed.
        n_high_conviction: Number of events with score > _HIGH_CONVICTION_THRESHOLD.
    """

    fired_triggers: list[tuple[TriggerType, float]] = field(default_factory=list)
    affected_sectors: dict[str, float] = field(default_factory=dict)
    affected_assets: list[str] = field(default_factory=list)
    geo_tags: set[str] = field(default_factory=set)
    conviction: float = 0.0
    n_events: int = 0
    n_high_conviction: int = 0

    def is_active(self, threshold: float = _HIGH_CONVICTION_THRESHOLD) -> bool:
        """Return True if basket conviction exceeds threshold."""
        return self.conviction >= threshold

    def top_trigger(self) -> TriggerType | None:
        """Return the highest-scoring TriggerType, or None if empty."""
        if not self.fired_triggers:
            return None
        return max(self.fired_triggers, key=lambda t: t[1])[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "conviction": self.conviction,
            "n_events": self.n_events,
            "n_high_conviction": self.n_high_conviction,
            "fired_triggers": [(t.name, s) for t, s in self.fired_triggers],
            "affected_sectors": self.affected_sectors,
            "affected_assets": self.affected_assets,
            "geo_tags": sorted(self.geo_tags),
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_trigger_basket(
    events: list[NewsEvent],
    keyword_rules: dict[TriggerType, list[str]] | None = None,
    min_score: float = _MIN_SCORE_TO_INCLUDE,
) -> TriggerBasket:
    """Score a list of NewsEvents and aggregate them into a TriggerBasket.

    Algorithm:
    1. For each event, call geo_trigger.score_event() → raw_score [0,1].
    2. For each TriggerType keyword group that matched, record trigger type + score.
    3. Derive affected sectors from _TRIGGER_SECTOR_MAP and geo_tags from event.
    4. Map sectors → ETFs (SECTOR_TO_ETFS) and geo_tags → country assets
       (COUNTRY_TO_ASSETS).
    5. Compute basket conviction:
       - Start with max single-event score (best individual signal).
       - Boost by sqrt(n_high_conviction / n_events) to reward corroborated events.
       - Clamp to [0.0, 1.0].

    Args:
        events: List of NewsEvent objects (title + geo_tags required for scoring).
        keyword_rules: Override KEYWORD_RULES for testing / custom regimes.
        min_score: Minimum geo_trigger score to include a trigger in the basket.

    Returns:
        TriggerBasket with all fields populated.
    """
    rules = keyword_rules if keyword_rules is not None else KEYWORD_RULES

    # Accumulate per-trigger scores across all events
    trigger_scores: dict[TriggerType, float] = {}
    all_geo_tags: set[str] = set()
    n_events = 0
    n_high = 0
    per_event_max: list[float] = []

    from .geo_trigger import _kw_in_text

    for event in events:
        n_events += 1
        raw_score = score_event(event, keyword_rules=rules)
        per_event_max.append(raw_score)
        if raw_score >= _HIGH_CONVICTION_THRESHOLD:
            n_high += 1
        all_geo_tags.update(g.upper() for g in (event.geo_tags or []))

        # Per-trigger scoring: use keyword hit-density for each trigger type,
        # independent of the holistic score_event score. This avoids penalising
        # specialised events that only activate one or two trigger types.
        text = _build_text(event)
        for ttype, keywords in rules.items():
            matched_kws = sum(1 for kw in keywords if _kw_in_text(kw, text))
            if matched_kws == 0:
                continue
            hit_density = matched_kws / max(len(keywords), 1)
            trigger_scores[ttype] = max(trigger_scores.get(ttype, 0.0), hit_density)

    # Filter triggers below min_score
    fired: list[tuple[TriggerType, float]] = [
        (t, s) for t, s in trigger_scores.items() if s >= min_score
    ]
    fired.sort(key=lambda x: x[1], reverse=True)

    # Aggregate sectors from fired triggers
    sector_scores: dict[str, float] = {}
    for ttype, score in fired:
        for sector in _TRIGGER_SECTOR_MAP.get(ttype, []):
            sector_scores[sector] = max(sector_scores.get(sector, 0.0), score)

    # Derive affected assets from sectors + geo_tags
    seen_assets: set[str] = set()
    affected_assets: list[str] = []
    for sector in sector_scores:
        for asset in SECTOR_TO_ETFS.get(sector, []):
            if asset not in seen_assets:
                affected_assets.append(asset)
                seen_assets.add(asset)
    for iso2 in all_geo_tags:
        for asset in COUNTRY_TO_ASSETS.get(iso2, []):
            if asset not in seen_assets:
                affected_assets.append(asset)
                seen_assets.add(asset)

    # Basket conviction
    conviction = 0.0
    if per_event_max:
        best = max(per_event_max)
        corroboration_boost = (n_high / n_events) ** 0.5 if n_events > 0 else 0.0
        conviction = min(1.0, best * (0.7 + 0.3 * corroboration_boost))

    return TriggerBasket(
        fired_triggers=fired,
        affected_sectors=sector_scores,
        affected_assets=affected_assets,
        geo_tags=all_geo_tags,
        conviction=conviction,
        n_events=n_events,
        n_high_conviction=n_high,
    )


def _build_text(event: NewsEvent) -> str:
    """Combine all searchable text from an event into one lowercase string."""
    parts = [event.title.lower()]
    parts.extend(k.lower() for k in (event.keywords or []))
    parts.extend(g.lower() for g in (event.geo_tags or []))
    parts.extend(e.lower() for e in (event.entities or []))
    return " ".join(parts)


def compute_basket_score(basket: TriggerBasket) -> float:
    """Return the composite score for a TriggerBasket [0, 1].

    Combines conviction with sector diversity (more sectors → higher edge potential).
    """
    if not basket.fired_triggers:
        return 0.0
    diversity_bonus = min(0.1 * len(basket.affected_sectors), 0.3)
    return min(1.0, basket.conviction + diversity_bonus)


__all__ = [
    "TriggerBasket",
    "build_trigger_basket",
    "compute_basket_score",
]
