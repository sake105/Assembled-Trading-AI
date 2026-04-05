"""Central bank policy divergence modeling.

Tracks major central bank policy stances and models the market impact
of policy divergence, synchronized tightening, and liquidity shocks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Central Bank Profile
# ---------------------------------------------------------------------------


@dataclass
class CentralBankProfile:
    """Profile for a major central bank."""

    cb_id: str
    name: str
    currency: str
    current_rate: float       # Policy rate (%)
    terminal_rate_estimate: float  # Market-implied terminal rate
    inflation_target: float   # Official inflation target (%)
    current_inflation: float  # Current CPI (%)
    qe_status: str            # "active", "tapering", "ended", "qt"  (QT = quantitative tightening)
    bias: str                 # "hawkish", "dovish", "neutral"
    fx_intervention_active: bool = False
    forward_guidance: str = ""


# ---------------------------------------------------------------------------
# Central Bank Database
# ---------------------------------------------------------------------------

CENTRAL_BANK_PROFILES: dict[str, CentralBankProfile] = {
    "FED": CentralBankProfile(
        cb_id="FED",
        name="Federal Reserve",
        currency="USD",
        current_rate=5.25,
        terminal_rate_estimate=4.50,   # Market pricing cut path
        inflation_target=2.0,
        current_inflation=3.3,
        qe_status="qt",                # Active QT ($95B/month runoff)
        bias="hawkish",
        fx_intervention_active=False,
        forward_guidance="Higher for longer, data dependent",
    ),
    "ECB": CentralBankProfile(
        cb_id="ECB",
        name="European Central Bank",
        currency="EUR",
        current_rate=4.50,
        terminal_rate_estimate=3.50,
        inflation_target=2.0,
        current_inflation=3.3,
        qe_status="qt",
        bias="neutral",               # Less hawkish than FED
        fx_intervention_active=False,
        forward_guidance="Data dependent, watching services inflation",
    ),
    "BOJ": CentralBankProfile(
        cb_id="BOJ",
        name="Bank of Japan",
        currency="JPY",
        current_rate=0.10,            # Just exited ZIRP in 2024
        terminal_rate_estimate=0.75,
        inflation_target=2.0,
        current_inflation=2.9,
        qe_status="tapering",         # Slowly reducing JGB purchases
        bias="dovish",                # Still most dovish G7
        fx_intervention_active=True,   # Yen weakness interventions
        forward_guidance="Gradual normalization, watching wages",
    ),
    "PBOC": CentralBankProfile(
        cb_id="PBOC",
        name="People's Bank of China",
        currency="CNY",
        current_rate=3.45,            # LPR 1-year
        terminal_rate_estimate=3.00,
        inflation_target=3.0,
        current_inflation=1.5,
        qe_status="active",           # Easing to support economy
        bias="dovish",                # Property sector crisis response
        fx_intervention_active=True,
        forward_guidance="Supportive monetary policy, property sector support",
    ),
    "BOE": CentralBankProfile(
        cb_id="BOE",
        name="Bank of England",
        currency="GBP",
        current_rate=5.25,
        terminal_rate_estimate=4.25,
        inflation_target=2.0,
        current_inflation=4.0,
        qe_status="qt",
        bias="hawkish",               # Sticky services inflation
        fx_intervention_active=False,
        forward_guidance="Inflation still above target, restrictive for longer",
    ),
    "RBI": CentralBankProfile(
        cb_id="RBI",
        name="Reserve Bank of India",
        currency="INR",
        current_rate=6.50,
        terminal_rate_estimate=6.00,
        inflation_target=4.0,
        current_inflation=4.5,
        qe_status="ended",
        bias="neutral",
        fx_intervention_active=True,  # Manages INR stability
        forward_guidance="Inflation within tolerance, growth focus",
    ),
}

# Policy divergence impact on asset classes
# (hawkish_cb, dovish_cb): {asset: direction +/-}
_DIVERGENCE_ASSET_IMPACT: dict[tuple[str, str], dict[str, float]] = {
    ("FED", "BOJ"): {
        "USD_JPY": +0.8,       # USD strengthens vs JPY
        "US_EQUITIES": +0.2,   # Mild positive (USD strength, carry)
        "JP_EQUITIES": +0.3,   # Export benefit for Japan
        "GOLD": -0.2,          # USD strength weighs on gold
        "EM_EQUITIES": -0.4,   # USD strength = EM outflows
        "EM_BONDS": -0.5,
    },
    ("FED", "PBOC"): {
        "USD_CNY": +0.6,
        "CHINA_EQUITIES": -0.3,
        "US_EQUITIES": +0.1,
        "COMMODITIES": -0.2,   # CNY weakness = lower commodity demand
        "EM_EQUITIES": -0.3,
    },
    ("ECB", "FED"): {
        "EUR_USD": -0.5,       # EUR weakens vs USD
        "EU_EQUITIES": +0.2,   # Export competitive advantage
        "ENERGY": +0.1,        # EUR weakness = higher oil import cost
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_policy_divergence_matrix() -> dict[tuple[str, str], float]:
    """Compute pairwise policy divergence scores for all central bank pairs.

    Divergence = |rate_a - rate_b| / max_rate, adjusted by bias direction.
    Returns {(cb_a, cb_b): divergence_score 0-1}.
    """
    cbs = list(CENTRAL_BANK_PROFILES.keys())
    matrix: dict[tuple[str, str], float] = {}
    max_rate = max(p.current_rate for p in CENTRAL_BANK_PROFILES.values())

    for i, cb_a in enumerate(cbs):
        for cb_b in cbs[i + 1:]:
            pa = CENTRAL_BANK_PROFILES[cb_a]
            pb = CENTRAL_BANK_PROFILES[cb_b]
            rate_diff = abs(pa.current_rate - pb.current_rate)
            base_divergence = rate_diff / max(max_rate, 1.0)

            # Bias divergence amplification
            bias_factor = 1.0
            if (pa.bias == "hawkish" and pb.bias == "dovish") or \
               (pa.bias == "dovish" and pb.bias == "hawkish"):
                bias_factor = 1.5
            elif pa.bias == pb.bias:
                bias_factor = 0.7

            matrix[(cb_a, cb_b)] = min(base_divergence * bias_factor, 1.0)

    return matrix


def estimate_capital_flow_impact(
    cb_hawkish: str,
    cb_dovish: str,
) -> dict[str, float]:
    """Estimate capital flow implications from a central bank divergence pair.

    Higher-rate CB attracts capital → currency strengthens, assets reprice.
    """
    pair = (cb_hawkish, cb_dovish)
    known = _DIVERGENCE_ASSET_IMPACT.get(pair, {})

    if known:
        return known

    # Generic divergence logic
    ha = CENTRAL_BANK_PROFILES.get(cb_hawkish)
    da = CENTRAL_BANK_PROFILES.get(cb_dovish)
    if ha is None or da is None:
        return {}

    rate_gap = ha.current_rate - da.current_rate
    strength = min(abs(rate_gap) / 5.0, 1.0)

    return {
        f"{ha.currency}_{da.currency}": strength,  # Hawkish currency strengthens
        f"{da.currency}_equities": strength * 0.3,  # Dovish country exports benefit
        "em_outflows": strength * 0.4 if ha.cb_id == "FED" else 0.0,
        "gold": -strength * 0.2,  # USD strength weighs on gold if FED is hawkish
    }


def detect_synchronized_tightening() -> bool:
    """Return True if multiple major CBs are simultaneously tightening."""
    tightening_count = sum(
        1 for p in CENTRAL_BANK_PROFILES.values()
        if p.bias == "hawkish" or p.qe_status == "qt"
    )
    return tightening_count >= 3


def compute_liquidity_shock_risk() -> float:
    """Compute global liquidity shock risk based on aggregate CB stance.

    Returns 0.0 (easy/accommodative) to 1.0 (highly restrictive/shock risk).
    """
    scores = []
    for profile in CENTRAL_BANK_PROFILES.values():
        # Rate vs neutral estimate (~2% for developed markets)
        rate_restrictiveness = max(0, (profile.current_rate - 2.0) / 6.0)
        qt_score = 0.3 if profile.qe_status == "qt" else 0.0
        hawkish_score = 0.2 if profile.bias == "hawkish" else 0.0
        scores.append(min(rate_restrictiveness + qt_score + hawkish_score, 1.0))

    return sum(scores) / len(scores) if scores else 0.0


def get_policy_stance(cb_name: str) -> str:
    """Return current policy stance for a central bank."""
    profile = CENTRAL_BANK_PROFILES.get(cb_name.upper())
    if profile is None:
        return "unknown"
    return profile.bias


def get_most_divergent_pair() -> tuple[str, str, float]:
    """Return the central bank pair with the highest current divergence."""
    matrix = compute_policy_divergence_matrix()
    if not matrix:
        return ("", "", 0.0)
    pair, score = max(matrix.items(), key=lambda x: x[1])
    return (pair[0], pair[1], score)
