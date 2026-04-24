"""Barbell Strategy (Taleb) — convex portfolio in high tail-risk environments.

When tail risk is elevated:
- 80-90% allocation to safe assets (treasuries, gold, short-term bonds)
- 10-20% allocation to speculative positions (highest alpha, tightest stops)

This creates convexity: bounded downside (safe assets floor the portfolio)
with unbounded upside (speculative sleeve captures asymmetric payoffs).

Activation triggers:
- EVT VaR > 2× historical average
- HMM P(crisis) > 0.4
- VIX > 30 and rising
- Copula tail dependence spike

When NOT in barbell mode, returns None and the normal portfolio
construction logic takes over.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class BarbellConfig:
    """Configuration for barbell activation."""

    evt_var_threshold_multiplier: float = 2.0
    hmm_crisis_probability_threshold: float = 0.4
    vix_threshold: float = 30.0
    copula_tail_dep_threshold: float = 0.5
    safe_allocation_pct: float = 0.85  # 85% to safe assets
    speculative_allocation_pct: float = 0.15  # 15% to speculative
    max_speculative_positions: int = 5
    speculative_stop_loss_pct: float = 0.05  # 5% stop-loss on each speculative position


@dataclass
class BarbellAllocation:
    """Result of barbell allocation decision."""

    active: bool  # whether barbell mode is triggered
    trigger_reasons: list[str]  # what caused activation
    safe_weight: float  # total weight in safe sleeve
    speculative_weight: float  # total weight in speculative sleeve
    safe_symbols: list[str]  # symbols in safe sleeve
    speculative_symbols: list[str]  # symbols in speculative sleeve
    safe_weights: dict[str, float]  # per-symbol weights in safe sleeve
    speculative_weights: dict[str, float]  # per-symbol weights in speculative sleeve
    tail_risk_score: float  # composite trigger score


# Default safe-haven assets (ETFs)
DEFAULT_SAFE_ASSETS = [
    "TLT",  # 20+ Year Treasury Bond
    "IEF",  # 7-10 Year Treasury Bond
    "SHY",  # 1-3 Year Treasury Bond
    "GLD",  # Gold
    "IAU",  # Gold (alternative)
    "BIL",  # 1-3 Month T-Bill
]


def compute_tail_risk_score(
    *,
    evt_var_99: float = 0.0,
    evt_var_99_historical_avg: float = 0.0,
    hmm_crisis_prob: float = 0.0,
    vix_current: float = 0.0,
    vix_5d_change: float = 0.0,
    avg_copula_tail_dep: float = 0.0,
    config: BarbellConfig | None = None,
) -> tuple[float, list[str]]:
    """Compute composite tail risk score from multiple indicators.

    Returns:
        Tuple of (score [0-1], list of triggered reasons).
    """
    if config is None:
        config = BarbellConfig()

    score = 0.0
    reasons: list[str] = []

    # EVT component (30% weight)
    if evt_var_99_historical_avg > 0 and evt_var_99 > 0:
        evt_ratio = evt_var_99 / evt_var_99_historical_avg
        if evt_ratio > config.evt_var_threshold_multiplier:
            evt_score = min(1.0, (evt_ratio - 1.0) / 3.0)
            score += 0.30 * evt_score
            reasons.append(f"EVT VaR ratio {evt_ratio:.1f}x (>{config.evt_var_threshold_multiplier}x)")

    # HMM crisis probability (30% weight)
    if hmm_crisis_prob > config.hmm_crisis_probability_threshold:
        hmm_score = min(1.0, hmm_crisis_prob / 0.8)
        score += 0.30 * hmm_score
        reasons.append(f"HMM P(crisis)={hmm_crisis_prob:.2f} (>{config.hmm_crisis_probability_threshold})")

    # VIX (20% weight)
    if vix_current > config.vix_threshold and vix_5d_change > 0:
        vix_score = min(1.0, (vix_current - 20) / 40)
        score += 0.20 * vix_score
        reasons.append(f"VIX={vix_current:.1f} rising (>{config.vix_threshold})")

    # Copula tail dependence (20% weight)
    if avg_copula_tail_dep > config.copula_tail_dep_threshold:
        copula_score = min(1.0, avg_copula_tail_dep)
        score += 0.20 * copula_score
        reasons.append(f"Copula tail dep={avg_copula_tail_dep:.3f} (>{config.copula_tail_dep_threshold})")

    return round(min(1.0, score), 4), reasons


def build_barbell_allocation(
    *,
    tail_risk_score: float,
    trigger_reasons: list[str],
    alpha_scores: dict[str, float],
    available_safe_assets: list[str] | None = None,
    config: BarbellConfig | None = None,
) -> BarbellAllocation:
    """Build barbell allocation if tail risk exceeds threshold.

    Args:
        tail_risk_score: Composite tail risk score from
            ``compute_tail_risk_score``.
        trigger_reasons: Reasons for activation.
        alpha_scores: Dict mapping symbol → alpha/signal score.
            Top-scoring symbols go into the speculative sleeve.
        available_safe_assets: Safe-haven symbols available in universe.
        config: Barbell configuration.

    Returns:
        BarbellAllocation (``active=False`` if barbell not triggered).
    """
    if config is None:
        config = BarbellConfig()

    if available_safe_assets is None:
        available_safe_assets = DEFAULT_SAFE_ASSETS

    # Barbell activates when tail risk score > 0.3 (at least one trigger hit)
    if tail_risk_score < 0.3 or not trigger_reasons:
        return BarbellAllocation(
            active=False,
            trigger_reasons=[],
            safe_weight=0.0,
            speculative_weight=0.0,
            safe_symbols=[],
            speculative_symbols=[],
            safe_weights={},
            speculative_weights={},
            tail_risk_score=tail_risk_score,
        )

    # Scale allocation by severity
    # Higher tail risk → more safe, less speculative
    severity_scale = min(1.0, tail_risk_score / 0.8)
    safe_pct = config.safe_allocation_pct + (1.0 - config.safe_allocation_pct) * severity_scale * 0.5
    spec_pct = 1.0 - safe_pct

    # Safe sleeve: equal weight across available safe assets
    n_safe = max(1, len(available_safe_assets))
    safe_weights = {s: round(safe_pct / n_safe, 6) for s in available_safe_assets}

    # Speculative sleeve: top alpha scores with tight stops
    if alpha_scores:
        sorted_alpha = sorted(alpha_scores.items(), key=lambda x: x[1], reverse=True)
        n_spec = min(config.max_speculative_positions, len(sorted_alpha))
        # Only take positive alpha
        spec_symbols = [
            s for s, score in sorted_alpha[:n_spec] if score > 0
        ]
    else:
        spec_symbols = []

    if spec_symbols:
        spec_weight_each = spec_pct / len(spec_symbols)
        spec_weights = {s: round(spec_weight_each, 6) for s in spec_symbols}
    else:
        spec_weights = {}
        # Reallocate unused speculative budget to safe
        for s in safe_weights:
            safe_weights[s] = round(safe_weights[s] + spec_pct / n_safe, 6)

    logger.info(
        "[Barbell] ACTIVATED: score=%.3f, safe=%.0f%%, spec=%.0f%%, triggers=%s",
        tail_risk_score, safe_pct * 100, spec_pct * 100,
        ", ".join(trigger_reasons),
    )

    return BarbellAllocation(
        active=True,
        trigger_reasons=trigger_reasons,
        safe_weight=round(safe_pct, 4),
        speculative_weight=round(spec_pct, 4),
        safe_symbols=list(safe_weights.keys()),
        speculative_symbols=spec_symbols,
        safe_weights=safe_weights,
        speculative_weights=spec_weights,
        tail_risk_score=tail_risk_score,
    )


__all__ = [
    "BarbellAllocation",
    "BarbellConfig",
    "DEFAULT_SAFE_ASSETS",
    "build_barbell_allocation",
    "compute_tail_risk_score",
]
