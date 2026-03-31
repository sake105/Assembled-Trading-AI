"""Crisis-Alpha simple entry signals — M5.

Entry logic for the crisis sub-portfolio.  Only runs when the crisis
state machine is ACTIVE.

Philosophy:
- Simple and transparent: rule-based momentum/breakout signals only.
- No complex ML models.
- ETF baskets only (as defined in baskets.py).
- Conservative sizing: equal-weight across basket instruments, subject to
  per-instrument and gross caps from risk_budget.py.
- No overnight positions (exit_rules.py handles the no-overnight rule).

Entry signal types (v1):
    equal_weight:  All active basket instruments get equal weight
                   (sum <= max_gross_exposure, split equally).
    geo_weighted:  Weight instruments proportionally by basket priority
                   (DEFENSIVE > INVERSE_EQUITY > VOLATILITY).

Policy config keys (crisis_alpha.entry.*):
    method:             "equal_weight" | "geo_weighted" (default: "equal_weight")
    active_baskets:     list of basket names to include (default: all)
    scale_by_geo_score: If True, scale total exposure by geo_score/activate_threshold
                        (more aggressive with higher geo score).
"""

from __future__ import annotations

import logging
from typing import Any

from src.assembled_core.events.crisis_alpha.baskets import get_baskets
from src.assembled_core.events.crisis_alpha.context import CrisisAlphaContext
from src.assembled_core.events.crisis_alpha.risk_budget import apply_risk_budget

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------


def _get(d: dict, *keys, default=None):
    node = d
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key, default)
        if node is None:
            return default
    return node


# ---------------------------------------------------------------------------
# Basket priority weights for geo_weighted method
# ---------------------------------------------------------------------------

_BASKET_PRIORITY: dict[str, float] = {
    "DEFENSIVE": 1.0,
    "INVERSE_EQUITY": 0.60,
    "VOLATILITY": 0.30,
}


# ---------------------------------------------------------------------------
# Entry signal generation
# ---------------------------------------------------------------------------


def generate_crisis_entry(
    ctx: CrisisAlphaContext,
    policy: dict | None = None,
) -> tuple[dict[str, float], list[str]]:
    """Generate target weights for crisis-alpha positions when state is ACTIVE.

    Returns:
        (target_weights, reasons) — target_weights is {symbol: weight},
        reasons is a list of audit log strings.

    The returned weights already have risk budget applied (per-instrument cap
    + gross exposure cap).  Caller must still check open positions before
    generating orders (to avoid re-entering already held positions).
    """
    policy = policy or {}
    reasons: list[str] = []

    cfg = _get(policy, "crisis_alpha", "entry", default={})
    method = _get(cfg, "method", default="equal_weight")
    active_basket_names: list[str] | None = _get(cfg, "active_baskets", default=None)
    scale_by_geo: bool = bool(_get(cfg, "scale_by_geo_score", default=False))

    baskets = get_baskets(policy)

    # Filter to active baskets
    if active_basket_names is not None:
        baskets = [b for b in baskets if b.get("basket") in active_basket_names]

    if not baskets:
        reasons.append("no active basket instruments — empty entry")
        return {}, reasons

    # Build raw weights
    raw_weights: dict[str, float] = {}

    if method == "equal_weight":
        weight_per_instrument = 1.0 / len(baskets)
        for basket in baskets:
            symbol = basket["symbol"]
            raw_weights[symbol] = weight_per_instrument
        reasons.append(
            f"equal_weight: {len(baskets)} instruments, weight={weight_per_instrument:.4f}"
        )

    elif method == "geo_weighted":
        total_priority = sum(
            _BASKET_PRIORITY.get(b.get("basket", ""), 1.0) for b in baskets
        )
        if total_priority == 0:
            total_priority = 1.0
        for basket in baskets:
            symbol = basket["symbol"]
            priority = _BASKET_PRIORITY.get(basket.get("basket", ""), 1.0)
            raw_weights[symbol] = priority / total_priority
        reasons.append(f"geo_weighted: {len(baskets)} instruments by basket priority")

    else:
        reasons.append(
            f"unknown entry method '{method}' — falling back to equal_weight"
        )
        weight_per_instrument = 1.0 / len(baskets) if baskets else 0.0
        for basket in baskets:
            raw_weights[basket["symbol"]] = weight_per_instrument

    # Optional: scale total exposure by geo_score
    if scale_by_geo:
        activate_threshold = float(
            _get(
                policy, "crisis_alpha", "hysteresis", "activate_geo_score", default=2.0
            )
        )
        scale = min(1.0, ctx.geo_score / max(activate_threshold, 0.01))
        raw_weights = {sym: w * scale for sym, w in raw_weights.items()}
        reasons.append(
            f"geo_score scale applied: {scale:.3f} (geo_score={ctx.geo_score:.2f})"
        )

    # Apply risk budget (caps + gross exposure scaling)
    final_weights, budget_reasons = apply_risk_budget(raw_weights, baskets, policy)
    reasons.extend(budget_reasons)

    logger.info(
        "[CRISIS_ENTRY] generated %d position(s): %s",
        len(final_weights),
        {s: f"{w:.4f}" for s, w in final_weights.items()},
    )

    return final_weights, reasons
