"""Tier-weighted source voting for news direction / event-type consensus.

Given a group of NewsEvents about (presumably) the same story, returns the
weighted majority direction and event type. Source weights default to:

* T0 = 3.0  (regulators, OFAC etc.)
* T1 = 2.0  (Reuters, AP, Bloomberg)
* T2 = 1.0  (open / aggregator)
* T3 = 0.4  (scrapes / social)

Bias-adjusted: events from a source classified as "state media" or
"pro_government" via `news_classifier.get_source_bias` get an additional
configurable discount, since they tend to spin stories.

Returns a `VoteResult` with the winning label and a margin score.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from src.assembled_core.intel.news_classifier import get_source_bias, is_state_media

logger = logging.getLogger(__name__)


_TIER_WEIGHTS: dict[str, float] = {
    "T0": 3.0,
    "T1": 2.0,
    "T2": 1.0,
    "T3": 0.4,
}


@dataclass
class VoteResult:
    winner: str
    margin: float
    total_weight: float
    counts: dict[str, float] = field(default_factory=dict)


def _weight_for_event(evt, *, state_media_discount: float = 0.5) -> float:
    tier_obj = getattr(evt, "source_tier", None)
    tier_val = getattr(
        tier_obj, "value", str(tier_obj) if tier_obj is not None else "T3"
    )
    base = _TIER_WEIGHTS.get(tier_val, 0.4)
    src = (getattr(evt, "source_id", "") or "").lower().strip()
    if is_state_media(src):
        base *= state_media_discount
    else:
        bias = get_source_bias(src) or {}
        if bias.get("editorial_bias") == "pro_government":
            base *= state_media_discount
    return base


def vote_direction(events: list, *, state_media_discount: float = 0.5) -> VoteResult:
    """Return the tier-weighted majority `market_direction` across events."""
    return _vote_field(
        events, "market_direction", "neutral", state_media_discount=state_media_discount
    )


def vote_event_type(events: list, *, state_media_discount: float = 0.5) -> VoteResult:
    """Return the tier-weighted majority single event_type across events.

    `event_types` is a list per event. Each list contributes its first
    (highest-priority) entry to the vote.
    """
    counts: dict[str, float] = defaultdict(float)
    total = 0.0
    for evt in events or []:
        try:
            etypes = list(getattr(evt, "event_types", []) or [])
            if not etypes:
                continue
            label = etypes[0]
            w = _weight_for_event(evt, state_media_discount=state_media_discount)
            counts[label] += w
            total += w
        except Exception as exc:
            logger.debug("[SKIP] vote_event_type: %s", exc)
    return _build_result(counts, total)


def _vote_field(
    events: list,
    field_name: str,
    default_label: str,
    *,
    state_media_discount: float = 0.5,
) -> VoteResult:
    counts: dict[str, float] = defaultdict(float)
    total = 0.0
    for evt in events or []:
        try:
            label = getattr(evt, field_name, default_label) or default_label
            w = _weight_for_event(evt, state_media_discount=state_media_discount)
            counts[label] += w
            total += w
        except Exception as exc:
            logger.debug("[SKIP] _vote_field: %s", exc)
    return _build_result(counts, total)


def _build_result(counts: dict[str, float], total: float) -> VoteResult:
    if not counts or total <= 0:
        return VoteResult(winner="", margin=0.0, total_weight=0.0, counts={})
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    top_label, top_w = sorted_items[0]
    runner_w = sorted_items[1][1] if len(sorted_items) > 1 else 0.0
    margin = (top_w - runner_w) / total
    return VoteResult(
        winner=top_label,
        margin=round(margin, 4),
        total_weight=round(total, 4),
        counts={k: round(v, 4) for k, v in counts.items()},
    )


__all__ = ["vote_direction", "vote_event_type", "VoteResult"]
