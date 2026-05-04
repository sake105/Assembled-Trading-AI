"""3-layer News-TA Fusion architecture.

From 30_NEWS_TA_FUSION.md.

Layer 1: News as 9th BaseSignal — 6 sub-features → news_z_score
Layer 2: Meta-labeling gate — P(take | side, context) via news features
Layer 3: 2D Decision Matrix — TA-score × news-score → size multiplier

The three layers work in PARALLEL, not sequentially:
- Layer 1: continuous composite contribution
- Layer 2: binary filter (take/skip false positives)
- Layer 3: multiplicative conviction scaler
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layer 1: News Z-Score (6 sub-features)
# ---------------------------------------------------------------------------

_NEWS_WEIGHTS: dict[str, float] = {
    "sentiment_vw": 0.30,
    "novelty": 0.15,
    "surprise": 0.20,
    "event_volume_z": 0.10,
    "velocity": 0.15,
    "dispersion": -0.10,  # high dispersion = uncertainty penalty
}


def news_z_score(features: dict[str, float]) -> float:
    """Aggregate 6 news sub-features into a single Z-score clipped to [-3, +3].

    Args:
        features: Dict with keys matching _NEWS_WEIGHTS.  Missing keys default to 0.

    Returns:
        float in [-3.0, +3.0].
    """
    raw = sum(_NEWS_WEIGHTS[k] * features.get(k, 0.0) for k in _NEWS_WEIGHTS)
    return float(max(-3.0, min(3.0, raw)))


def news_score_normalized(features: dict[str, float]) -> float:
    """news_z_score / 3 → [-1, +1] for use in composite_score."""
    return news_z_score(features) / 3.0


# ---------------------------------------------------------------------------
# Layer 2: Meta-labeling helpers
# ---------------------------------------------------------------------------


def size_from_meta(p_meta: float, theta_meta: float = 0.55) -> float:
    """Kelly-like size from calibrated posterior probability.

    Args:
        p_meta: Predicted probability from meta-model.
        theta_meta: Minimum probability to take the trade.

    Returns:
        Position size fraction in [0, 1].
    """
    if p_meta < theta_meta:
        return 0.0
    return float(max(0.0, min(1.0, (p_meta - theta_meta) / (1.0 - theta_meta))))


def news_veto(news_z: float, primary_side: float, tau_veto: float = 1.5) -> bool:
    """Return True if news strongly contradicts the primary signal direction.

    Args:
        news_z: News Z-score in [-3, +3].
        primary_side: +1 (long) or -1 (short).
        tau_veto: Magnitude threshold for veto.

    Returns:
        True = veto (skip trade), False = proceed.
    """
    if primary_side == 0:
        return False
    return (
        float(np.sign(news_z)) != float(np.sign(primary_side))
        and abs(news_z) > tau_veto
    )


# ---------------------------------------------------------------------------
# Layer 3: 2D Decision Matrix
# ---------------------------------------------------------------------------


def bayesian_update(ta_score: float, news_z: float, kappa: float = 10.0) -> float:
    """Beta-Binomial Bayesian update of TA-prior with news evidence.

    Args:
        ta_score: TA composite score in [-1, +1].
        news_z: News Z-score in [-3, +3].
        kappa: Prior strength (effective sample size).

    Returns:
        Posterior mean in [0, 1] (>0.5 = bullish).
    """
    ta_prob = 1.0 / (1.0 + np.exp(-2.0 * ta_score))
    alpha_prior = ta_prob * kappa
    beta_prior = (1.0 - ta_prob) * kappa

    news_prob = 1.0 / (1.0 + np.exp(-news_z))
    alpha_post = alpha_prior + news_prob
    beta_post = beta_prior + (1.0 - news_prob)

    return float(alpha_post / (alpha_post + beta_post))


def agreement_multiplier(ta_score: float, news_z: float) -> float:
    """Size multiplier based on signal agreement.

    - Agreement → up to 1.5×
    - Conflict (both strong) → 0.5×
    - One-sided weak → 1.0×

    Args:
        ta_score: TA composite score in [-1, +1].
        news_z: News Z-score in [-3, +3].

    Returns:
        Multiplier in [0.5, 1.5].
    """
    sign_match = np.sign(ta_score) == np.sign(news_z)
    magnitude_avg = (abs(ta_score) + abs(news_z) / 3.0) / 2.0

    if sign_match:
        return float(min(1.5, 1.0 + 0.5 * magnitude_avg))
    elif abs(ta_score) < 0.3 or abs(news_z) < 0.3:
        return 1.0  # one-sided weak signal, neutral
    else:
        return 0.5  # strong conflict


# ---------------------------------------------------------------------------
# Unified decision function (all 3 layers)
# ---------------------------------------------------------------------------


def decide_trade(
    composite_score: float,
    news_features: dict[str, float],
    meta_probability: float,
    theta_meta: float = 0.55,
    tau_veto: float = 1.5,
    sector_sentiment: float = 0.0,
) -> dict[str, Any]:
    """Combine all 3 layers into a single trade decision.

    Args:
        composite_score: 9-dim composite in [-1, +1] (includes news as dim 9).
        news_features: Raw news sub-features for Z-score computation.
        meta_probability: P(take | context) from meta-model.
        theta_meta: Minimum meta-probability to take the trade.
        tau_veto: News veto threshold.
        sector_sentiment: Optional cross-impact graph output in [-1, +1].

    Returns:
        Dict with keys: action, size, composite_score, news_z, p_meta, multiplier, reason.
    """
    primary_side = float(np.sign(composite_score))
    nz = news_z_score(news_features)

    # Layer 2: meta gate
    if meta_probability < theta_meta:
        return {
            "action": "skip",
            "size": 0.0,
            "composite_score": composite_score,
            "news_z": nz,
            "p_meta": meta_probability,
            "multiplier": 0.0,
            "reason": "meta_below_threshold",
        }

    if news_veto(nz, primary_side, tau_veto):
        return {
            "action": "skip",
            "size": 0.0,
            "composite_score": composite_score,
            "news_z": nz,
            "p_meta": meta_probability,
            "multiplier": 0.0,
            "reason": "news_veto",
        }

    # Base size from meta (Layer 2)
    base_size = size_from_meta(meta_probability, theta_meta)

    # Layer 3: size multiplier
    multiplier = agreement_multiplier(composite_score, nz)

    final_size = base_size * multiplier

    # Optional Layer 4: cross-impact sector headwind
    if abs(sector_sentiment) > 0.7 and np.sign(sector_sentiment) != primary_side:
        final_size *= 0.5

    action = "long" if primary_side > 0 else ("short" if primary_side < 0 else "skip")

    return {
        "action": action,
        "size": float(max(0.0, min(1.0, final_size))),
        "composite_score": composite_score,
        "news_z": nz,
        "p_meta": meta_probability,
        "multiplier": multiplier,
        "reason": "ok",
    }


# ---------------------------------------------------------------------------
# Meta-features builder (Layer 2 input — spec §30 "Die 12-15 Meta-Features")
# ---------------------------------------------------------------------------

META_FEATURES = [
    "sentiment_z",
    "novelty_z",
    "surprise_z",
    "event_vol_z",
    "velocity_z",
    "dispersion_z",
    "event_earnings",
    "event_m_and_a",
    "event_mgmt",
    "event_regulatory",
    "event_analyst",
    "event_product",
    "event_legal",
    "event_macro",
    "days_since_earnings",
    "days_to_next_earnings",
    "macro_shock_flag",
    "vix_level",
    "vix_regime_ord",
    "hy_oas",
    "corroboration_count",
    "primary_strength",
    "news_vs_primary_agree",
]


def build_meta_features(
    ticker: str,
    news_features: dict,
    composite_score: float,
) -> list:
    """Assemble the 12-15 meta-feature vector for the meta-labeling model.

    Args:
        ticker: Unused directly — reserved for per-ticker lookup enrichment.
        news_features: Dict with news-derived floats (see META_FEATURES list).
        composite_score: TA composite score in [-1, +1]; used for primary_strength
                         and news_vs_primary_agree.

    Returns:
        List of floats in the META_FEATURES order. Missing keys default to 0.
    """
    nz = news_features.get("aggregate_z", 0.0)
    row = {k: float(news_features.get(k, 0.0)) for k in META_FEATURES}
    row["primary_strength"] = abs(composite_score)
    agree = 1.0 if (composite_score * nz > 0) else (0.0 if nz == 0 else -1.0)
    row["news_vs_primary_agree"] = agree
    return [row[k] for k in META_FEATURES]


__all__ = [
    "news_z_score",
    "news_score_normalized",
    "size_from_meta",
    "news_veto",
    "bayesian_update",
    "agreement_multiplier",
    "decide_trade",
    "META_FEATURES",
    "build_meta_features",
]
