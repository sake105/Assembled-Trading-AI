"""Regime-Conditional Portfolio Templates (Plan 5.9).

Pre-defined allocation templates per regime, blended via HMM probabilities.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Template weights: asset_class → weight
REGIME_TEMPLATES: dict[str, dict[str, float]] = {
    "bull": {
        "equity_high_beta": 0.30,
        "equity_momentum": 0.25,
        "equity_quality": 0.15,
        "equity_growth": 0.15,
        "bonds_corporate": 0.10,
        "gold": 0.05,
    },
    "bear": {
        "equity_low_vol": 0.20,
        "equity_quality": 0.20,
        "bonds_treasury": 0.25,
        "gold": 0.20,
        "cash": 0.15,
    },
    "crisis": {
        "bonds_treasury": 0.40,
        "gold": 0.30,
        "cash": 0.20,
        "equity_speculative": 0.10,
    },
    "recovery": {
        "equity_value": 0.30,
        "equity_small_cap": 0.20,
        "equity_quality": 0.15,
        "equity_growth": 0.15,
        "bonds_corporate": 0.15,
        "gold": 0.05,
    },
    "sideways": {
        "equity_quality": 0.20,
        "equity_value": 0.15,
        "equity_low_vol": 0.15,
        "bonds_treasury": 0.20,
        "bonds_corporate": 0.15,
        "gold": 0.10,
        "cash": 0.05,
    },
}


def blend_regime_templates(
    regime_probabilities: dict[str, float],
    templates: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """Blend portfolio templates across regimes using HMM probabilities.

    Args:
        regime_probabilities: Regime → probability.
        templates: Override default templates.

    Returns:
        Asset class → blended target weight.
    """
    tpl = templates or REGIME_TEMPLATES
    total_p = sum(regime_probabilities.values())
    if total_p < 1e-10:
        return tpl.get("sideways", {})

    blended: dict[str, float] = {}
    for regime, prob in regime_probabilities.items():
        norm_prob = prob / total_p
        weights = tpl.get(regime, tpl.get("sideways", {}))
        for asset, w in weights.items():
            blended[asset] = blended.get(asset, 0.0) + norm_prob * w

    return {k: round(v, 4) for k, v in blended.items()}


__all__ = ["REGIME_TEMPLATES", "blend_regime_templates"]
