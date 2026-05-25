"""Asset router: maps news event topic_id → directional trade signals.

Each routing rule specifies:
    topic_id:           matches trigger_scoring.TOPIC_RULES topic_id
    trigger_type:       category for grouping / logging
    long_etfs:          ETFs to go LONG (1x, always allowed)
    long_etfs_2x:       2x leveraged long ETFs (requires leverage_etfs_allowed=True)
    inverse_etfs:       1x inverse ETFs used as hedges (direction="long", always allowed)
    inverse_etfs_2x:    2x leveraged inverse ETFs (requires leverage_etfs_allowed=True)
    size_multiplier:    relative sizing vs base_weight (1.0 = normal, 2.0 = double)
    min_severity:       minimum trigger severity to activate (1=watch, 2=elevated, 3=critical)
    hold_days:          expected holding period in trading days
    rationale:          human-readable explanation

Design principles:
- ETF-only in v1 (no single-name equity exposure)
- 2x ETFs allowed when policy.news_alpha.leverage_etfs_allowed = true
- Standard ETFs always available as fallback
- Reverse-alpha protection: if event narrative reverses, exit_rules handles it
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Routing table
# Each entry is a dict matching fields of NewsAlphaRoute (see models.py)
# ---------------------------------------------------------------------------

ROUTING_TABLE: list[dict[str, Any]] = [
    # ------------------------------------------------------------------
    # OIL / ENERGY SUPPLY SHOCK
    # Triggers: Hormuz blockade, OPEC cut, pipeline attack, Red Sea
    # "Short" side uses inverse ETFs (direction="long") — no naked shorts.
    # ------------------------------------------------------------------
    {
        "topic_id": "shipping_disruption",
        "trigger_type": "supply_chain",
        "long_etfs": ["XLE", "XOM"],  # standard
        "long_etfs_2x": ["UCO"],  # 2x crude ETF — only if leverage_etfs_allowed
        "inverse_etfs": [],  # no liquid inverse for airlines in v1
        "size_multiplier": 1.5,
        "min_severity": 2,
        "hold_days": 5,
        "rationale": "Supply chain / shipping disruption raises energy costs → long oil sector",
    },
    {
        "topic_id": "energy_crisis",
        "trigger_type": "commodity",
        "long_etfs": ["XLE", "XOM"],
        "long_etfs_2x": ["UCO"],
        "inverse_etfs": [],
        "size_multiplier": 1.5,
        "min_severity": 2,
        "hold_days": 5,
        "rationale": "Energy price spike → long energy sector",
    },
    # ------------------------------------------------------------------
    # GEOPOLITICAL CONFLICT / WAR
    # Triggers: invasion, missile strike, NATO activation
    # ------------------------------------------------------------------
    {
        "topic_id": "geopolitical_conflict",
        "trigger_type": "geo_risk",
        "long_etfs": ["LMT", "NOC", "RTX", "GLD"],  # defense + gold
        "long_etfs_2x": [],
        "inverse_etfs": [],  # international short via inverse in v2
        "size_multiplier": 1.2,
        "min_severity": 2,
        "hold_days": 7,
        "rationale": "Armed conflict → long defense + gold",
    },
    {
        "topic_id": "taiwan_strait",
        "trigger_type": "geo_risk",
        "long_etfs": ["LMT", "NOC", "RTX", "GLD"],
        "long_etfs_2x": [],
        "inverse_etfs": [],  # FXI inverse (YANG) deferred to v2
        "size_multiplier": 2.0,
        "min_severity": 2,
        "hold_days": 10,
        "rationale": "Taiwan scenario: highest severity geo-event; long US defense + gold",
    },
    {
        "topic_id": "nuclear_risk",
        "trigger_type": "geo_risk",
        "long_etfs": ["GLD", "SHY"],  # flight to safety only
        "long_etfs_2x": [],
        "inverse_etfs": ["SH"],  # SH = ProShares Short S&P500 (long inverse)
        "size_multiplier": 2.0,
        "min_severity": 3,  # only at critical severity
        "hold_days": 3,
        "rationale": "Nuclear risk: extreme flight to safety, small size (tail hedge only)",
    },
    # ------------------------------------------------------------------
    # CENTRAL BANK SURPRISE
    # Rate hike: TBT (inverse long-bond) + XLF, NOT naked short TLT
    # Rate cut: long TLT + QQQ
    # ------------------------------------------------------------------
    {
        "topic_id": "central_bank_hike",  # dynamically split from central_bank
        "trigger_type": "macro",
        "long_etfs": ["XLF", "SHY"],
        "long_etfs_2x": [],
        "inverse_etfs": [],
        "inverse_etfs_2x": [
            "TBT"
        ],  # TBT = ProShares UltraShort 20+ Yr (2x leveraged inverse) — requires leverage_etfs_allowed
        "size_multiplier": 1.0,
        "min_severity": 2,
        "hold_days": 3,
        "rationale": "Surprise rate hike → short duration (TBT inverse), long financials",
    },
    {
        "topic_id": "central_bank_cut",
        "trigger_type": "macro",
        "long_etfs": ["TLT", "QQQ"],
        "long_etfs_2x": [],
        "inverse_etfs": [],
        "size_multiplier": 1.0,
        "min_severity": 2,
        "hold_days": 3,
        "rationale": "Surprise rate cut → long duration + growth",
    },
    # ------------------------------------------------------------------
    # MARKET CRASH / PANIC
    # SH = ProShares Short S&P500 (1x inverse, direction="long")
    # ------------------------------------------------------------------
    {
        "topic_id": "market_crash",
        "trigger_type": "market_stress",
        "long_etfs": ["GLD", "TLT", "SHY"],
        "long_etfs_2x": ["UVXY"],  # vol spike — only if leverage allowed
        "inverse_etfs": ["SH"],  # SH = short S&P500 proxy
        "size_multiplier": 1.5,
        "min_severity": 2,
        "hold_days": 5,
        "rationale": "Market crash: flight to safety + equity hedge via SH",
    },
]

# Build lookup dict by topic_id for O(1) access
_ROUTE_BY_TOPIC: dict[str, dict[str, Any]] = {r["topic_id"]: r for r in ROUTING_TABLE}


def get_route(topic_id: str | None) -> dict[str, Any] | None:
    """Return routing rule for a topic_id, or None if not mapped."""
    return _ROUTE_BY_TOPIC.get(topic_id)  # type: ignore[arg-type]


def get_all_routes() -> list[dict[str, Any]]:
    return list(ROUTING_TABLE)


def split_central_bank_topic(trigger: dict[str, Any]) -> str | None:
    """Refine 'central_bank' topic into hike/cut based on text evidence.

    Returns None when direction cannot be determined — caller should skip the trade.
    """
    text = (trigger.get("topic", "") + " " + trigger.get("source", "")).lower()
    if any(kw in text for kw in ["hike", "increase", "raise", "hawkish", "tighten"]):
        return "central_bank_hike"
    if any(kw in text for kw in ["cut", "lower", "reduce", "dovish", "ease"]):
        return "central_bank_cut"
    # Also check trigger metadata if present
    details = trigger.get("details", "")
    if details:
        details_lower = str(details).lower()
        if any(kw in details_lower for kw in ["hike", "hawkish"]):
            return "central_bank_hike"
        if any(kw in details_lower for kw in ["cut", "dovish"]):
            return "central_bank_cut"
    return None  # ambiguous — no trade rather than wrong-direction trade
