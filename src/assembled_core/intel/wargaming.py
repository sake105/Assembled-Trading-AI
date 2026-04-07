"""Lightweight Wargaming Engine (Plan 4.9).

2-3 actors per scenario with defined preferences.
Payoff matrix → Nash Equilibrium (2x2 or 3x3 games).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WargameResult:
    """Result of a wargame analysis."""
    actor_actions: dict[str, str]  # actor → most likely action
    equilibrium_type: str  # "pure", "mixed", "dominant"
    confidence: float  # 0-1
    payoff_summary: dict[str, float]  # actor → expected payoff


def find_nash_2x2(
    payoff_a: np.ndarray,
    payoff_b: np.ndarray,
) -> WargameResult:
    """Find Nash equilibrium for a 2x2 game.

    Args:
        payoff_a: 2x2 payoff matrix for player A.
        payoff_b: 2x2 payoff matrix for player B.

    Returns:
        WargameResult with equilibrium strategy.
    """
    actions_a = ["action_0", "action_1"]
    actions_b = ["action_0", "action_1"]

    # Check for pure strategy Nash equilibria
    pure_ne = []
    for i in range(2):
        for j in range(2):
            # i is best response for A given j?
            a_br = payoff_a[i, j] >= payoff_a[1 - i, j]
            # j is best response for B given i?
            b_br = payoff_b[i, j] >= payoff_b[i, 1 - j]
            if a_br and b_br:
                pure_ne.append((i, j))

    if pure_ne:
        i, j = pure_ne[0]
        return WargameResult(
            actor_actions={"A": actions_a[i], "B": actions_b[j]},
            equilibrium_type="pure",
            confidence=0.8,
            payoff_summary={"A": float(payoff_a[i, j]), "B": float(payoff_b[i, j])},
        )

    # Mixed strategy NE
    # p = probability A plays action_0
    # q = probability B plays action_0
    denom_a = (payoff_a[0, 0] - payoff_a[1, 0]) - (payoff_a[0, 1] - payoff_a[1, 1])
    denom_b = (payoff_b[0, 0] - payoff_b[0, 1]) - (payoff_b[1, 0] - payoff_b[1, 1])

    if abs(denom_a) < 1e-10 or abs(denom_b) < 1e-10:
        return WargameResult(
            actor_actions={"A": "action_0", "B": "action_0"},
            equilibrium_type="dominant",
            confidence=0.5,
            payoff_summary={"A": float(payoff_a[0, 0]), "B": float(payoff_b[0, 0])},
        )

    q = (payoff_a[1, 1] - payoff_a[1, 0]) / denom_a  # B's mixing probability
    p = (payoff_b[1, 1] - payoff_b[0, 1]) / denom_b  # A's mixing probability

    p = np.clip(p, 0, 1)
    q = np.clip(q, 0, 1)

    best_a = "action_0" if p >= 0.5 else "action_1"
    best_b = "action_0" if q >= 0.5 else "action_1"

    exp_a = p * q * payoff_a[0, 0] + p * (1 - q) * payoff_a[0, 1] + (1 - p) * q * payoff_a[1, 0] + (1 - p) * (1 - q) * payoff_a[1, 1]
    exp_b = p * q * payoff_b[0, 0] + p * (1 - q) * payoff_b[0, 1] + (1 - p) * q * payoff_b[1, 0] + (1 - p) * (1 - q) * payoff_b[1, 1]

    return WargameResult(
        actor_actions={"A": best_a, "B": best_b},
        equilibrium_type="mixed",
        confidence=round(max(p, 1 - p) * max(q, 1 - q), 3),
        payoff_summary={"A": round(float(exp_a), 4), "B": round(float(exp_b), 4)},
    )


__all__ = ["WargameResult", "find_nash_2x2"]
