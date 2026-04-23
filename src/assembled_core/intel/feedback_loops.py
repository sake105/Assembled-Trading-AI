"""Feedback loop tracker for geopolitical and macro cascade risk.

Tracks whether individual elements of a systemic feedback loop
have been activated across observed event histories.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FeedbackLoop:
    loop_id: str
    name: str
    chain: list[str] = field(default_factory=list)
    alert_threshold: float = 0.7


def track_loop_activation(
    loop: FeedbackLoop,
    history: list[list[str]],
) -> dict[str, Any]:
    """Count which chain elements appear in the event history.

    Args:
        loop: FeedbackLoop definition with chain elements.
        history: List of event sets (each a list of activated element IDs).

    Returns:
        dict with activated_elements, activation_score, alert.
    """
    seen: set[str] = set()
    for events in history:
        seen.update(events)

    activated = [e for e in loop.chain if e in seen]
    n_activated = len(activated)
    score = n_activated / len(loop.chain) if loop.chain else 0.0

    return {
        "activated_elements": n_activated,
        "activation_score": round(score, 6),
        "activated": activated,
        "alert": score > loop.alert_threshold,
    }
