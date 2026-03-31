"""Crisis-Alpha ETF basket definitions — M5.

Crisis alpha trades high-liquidity ETFs only.  No single-name equities.
No leverage products in v1.

Basket philosophy:
- DEFENSIVE: bonds, gold, cash-proxies — go long when crisis escalates
- VOLATILITY: VIX-related ETFs — go long on vol spike (higher risk, smaller size)
- INVERSE_EQUITY: broad equity shorts — hedge equity exposure during crisis

Each basket entry contains:
    symbol:      ETF ticker
    basket:      basket name
    direction:   "long" or "short" (from the crisis alpha perspective)
    max_weight:  Maximum portfolio weight for this instrument (as fraction of sub-portfolio)
    liquidity:   "high" / "medium" (informational, for future filtering)
    description: Brief description for audit logs

Policy overrides can replace or extend these defaults via configs/crisis_alpha/crisis_alpha.yaml.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Default basket definitions (can be overridden by config)
# ---------------------------------------------------------------------------

DEFAULT_BASKETS: list[dict[str, Any]] = [
    # --- DEFENSIVE basket ---
    {
        "symbol": "GLD",
        "basket": "DEFENSIVE",
        "direction": "long",
        "max_weight": 0.20,
        "liquidity": "high",
        "description": "SPDR Gold Shares — safe-haven long",
    },
    {
        "symbol": "TLT",
        "basket": "DEFENSIVE",
        "direction": "long",
        "max_weight": 0.20,
        "liquidity": "high",
        "description": "iShares 20+ Year Treasury Bond — long-duration safe-haven",
    },
    {
        "symbol": "SHY",
        "basket": "DEFENSIVE",
        "direction": "long",
        "max_weight": 0.15,
        "liquidity": "high",
        "description": "iShares 1-3 Year Treasury Bond — near-cash defensive",
    },
    # --- INVERSE_EQUITY basket ---
    {
        "symbol": "SH",
        "basket": "INVERSE_EQUITY",
        "direction": "long",
        "max_weight": 0.10,
        "liquidity": "high",
        "description": "ProShares Short S&P500 — 1x inverse equity hedge",
    },
    # --- VOLATILITY basket (smaller sizing — higher risk) ---
    {
        "symbol": "VIXY",
        "basket": "VOLATILITY",
        "direction": "long",
        "max_weight": 0.05,
        "liquidity": "medium",
        "description": "ProShares VIX Short-Term Futures ETF — vol long (small size only)",
    },
]


def get_baskets(policy: dict | None = None) -> list[dict[str, Any]]:
    """Return the active basket definitions, optionally overridden by policy.

    If policy contains ``crisis_alpha.baskets``, that list replaces the defaults.
    If policy contains ``crisis_alpha.basket_overrides``, entries are merged by symbol
    (symbol match → update fields).

    Args:
        policy: Policy dict (from crisis_alpha.yaml). None → use defaults.

    Returns:
        List of basket definition dicts.
    """
    if policy is None:
        return list(DEFAULT_BASKETS)

    cfg = policy.get("crisis_alpha", {})

    # Full override: if "baskets" key present, use it instead of defaults
    if "baskets" in cfg:
        return list(cfg["baskets"])

    # Partial override: merge by symbol
    baskets = [dict(b) for b in DEFAULT_BASKETS]
    overrides: list[dict] = cfg.get("basket_overrides", [])
    override_map = {o["symbol"]: o for o in overrides if "symbol" in o}
    for basket in baskets:
        sym = basket.get("symbol")
        if sym in override_map:
            basket.update(override_map[sym])

    return baskets


def get_basket_symbols(policy: dict | None = None) -> list[str]:
    """Return just the list of symbols in the active baskets."""
    return [b["symbol"] for b in get_baskets(policy)]


def get_basket_by_name(name: str, policy: dict | None = None) -> list[dict[str, Any]]:
    """Return all basket entries for a given basket name (e.g. 'DEFENSIVE')."""
    return [b for b in get_baskets(policy) if b.get("basket") == name]
