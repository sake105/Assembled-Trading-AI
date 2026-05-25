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

Regime-aware basket selection (v2):
    detect_rate_regime() uses PIT-safe TLT/SHY relative performance to distinguish
    rate-hike crises (2022-style) from liquidity/geopolitical crises (2008/2020-style).
    In a rate-hike regime, bonds are replaced with energy (XLE) and utilities (XLU)
    which historically outperform in inflationary/supply-shock environments.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default basket definitions (liquidity/geopolitical crisis — 2008/2020 style)
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

# ---------------------------------------------------------------------------
# Rate-hike / inflationary-crisis basket (2022-style)
# Used when TLT significantly underperforms SHY — long bonds are toxic.
# XLE: energy benefits from supply shocks; XLU: utilities are defensive +
# semi-inflation-protected; GLD: partial geopolitical hedge even in rate-hike env;
# SHY: near-cash preserves capital when duration risk is high.
# ---------------------------------------------------------------------------

RATE_HIKE_BASKETS: list[dict[str, Any]] = [
    {
        "symbol": "XLE",
        "basket": "DEFENSIVE",
        "direction": "long",
        "max_weight": 0.30,
        "liquidity": "high",
        "description": "Energy Select Sector SPDR — supply-shock / inflation hedge",
    },
    {
        "symbol": "XLU",
        "basket": "DEFENSIVE",
        "direction": "long",
        "max_weight": 0.25,
        "liquidity": "high",
        "description": "Utilities Select Sector SPDR — defensive / inflation-resilient",
    },
    {
        "symbol": "GLD",
        "basket": "DEFENSIVE",
        "direction": "long",
        "max_weight": 0.25,
        "liquidity": "high",
        "description": "SPDR Gold Shares — partial geopolitical hedge",
    },
    {
        "symbol": "SHY",
        "basket": "DEFENSIVE",
        "direction": "long",
        "max_weight": 0.20,
        "liquidity": "high",
        "description": "iShares 1-3 Year Treasury Bond — near-cash / low duration",
    },
]

# Minimum number of price rows needed to compute the regime indicator
_MIN_REGIME_ROWS = 5

# Default TLT-minus-SHY 90-day return gap threshold for declaring rate_hike regime.
# Callers should pass their own threshold rather than relying on this constant.
_DEFAULT_RATE_HIKE_THRESHOLD = -0.05


def detect_rate_regime(
    prices_pit: "pd.DataFrame | None",
    as_of: Any,
    lookback_days: int = 90,
    threshold: float = _DEFAULT_RATE_HIKE_THRESHOLD,
) -> str:
    """Classify the current interest-rate regime using PIT-safe price data.

    Returns ``"rate_hike"`` when TLT has underperformed SHY by more than
    ``threshold`` over the last ``lookback_days`` days, signalling that
    long-duration bonds are under pressure and the default flight-to-quality
    basket would be harmful.  Returns ``"neutral"`` otherwise (including when
    data for TLT or SHY is unavailable).

    Args:
        prices_pit: Price DataFrame already filtered to ``<= as_of`` (caller's
            responsibility).  Must contain columns ``timestamp``, ``symbol``,
            ``close``.
        as_of: Reference timestamp (used only for the log message).
        lookback_days: Rolling window for relative-performance check.
        threshold: TLT-minus-SHY gap below which regime is ``"rate_hike"``.
            Default -0.05 (-5pp).  Pass explicitly to avoid module-global state.

    Returns:
        ``"rate_hike"`` or ``"neutral"``.
    """
    try:
        import pandas as pd

        if prices_pit is None or (hasattr(prices_pit, "empty") and prices_pit.empty):
            return "neutral"

        cutoff = pd.to_datetime(as_of, utc=True) - pd.Timedelta(days=lookback_days)
        ts_col = pd.to_datetime(prices_pit["timestamp"], utc=True)

        def _ret(sym: str) -> float | None:
            mask = (prices_pit["symbol"] == sym) & (ts_col >= cutoff)
            sub = prices_pit[mask].sort_values("timestamp")
            if len(sub) < _MIN_REGIME_ROWS:
                return None
            start = float(sub["close"].iloc[0])
            end = float(sub["close"].iloc[-1])
            if start == 0:
                return None
            return end / start - 1.0

        tlt_ret = _ret("TLT")
        shy_ret = _ret("SHY")

        if tlt_ret is None or shy_ret is None:
            logger.debug(
                "detect_rate_regime: insufficient data for TLT/SHY at %s — defaulting neutral",
                as_of,
            )
            return "neutral"

        gap = tlt_ret - shy_ret
        regime = "rate_hike" if gap < threshold else "neutral"
        logger.debug(
            "detect_rate_regime: TLT=%+.2f%% SHY=%+.2f%% gap=%+.2f%% threshold=%+.2f%% => %s",
            tlt_ret * 100,
            shy_ret * 100,
            gap * 100,
            threshold * 100,
            regime,
        )
        return regime

    except Exception as exc:  # noqa: BLE001
        logger.warning("detect_rate_regime failed (%s) — defaulting neutral", exc)
        return "neutral"


def _get_regime_cfg(policy: dict | None) -> dict:
    """Extract basket_regime_detection config from either flat or nested policy dict.

    Supports both:
    - ``policy["crisis_alpha"]["basket_regime_detection"]`` (scoped / test format)
    - ``policy["intel"]["crisis_alpha"]["basket_regime_detection"]`` (full production format)
    """
    p = policy or {}
    # Scoped format: {"crisis_alpha": {...}}
    ca = p.get("crisis_alpha")
    if ca is None:
        # Full format: {"intel": {"crisis_alpha": {...}}}
        ca = p.get("intel", {}).get("crisis_alpha")
    return (ca or {}).get("basket_regime_detection", {})


def get_regime_aware_baskets(
    prices_pit: "pd.DataFrame | None",
    as_of: Any,
    policy: dict | None = None,
    lookback_days: int = 90,
) -> list[dict[str, Any]]:
    """Return the appropriate crisis basket for the current rate regime.

    If ``crisis_alpha.basket_regime_detection.enabled`` is False (or absent),
    falls back to the standard ``get_baskets(policy)`` path — no behaviour
    change for callers that have not opted in.

    Args:
        prices_pit: PIT-filtered price DataFrame (``<= as_of``).
        as_of: Reference timestamp.
        policy: Policy dict (same as ``get_baskets``). Supports both scoped
            ``{"crisis_alpha": {...}}`` and full ``{"intel": {"crisis_alpha": {...}}}``
            formats.
        lookback_days: Passed to ``detect_rate_regime``.

    Returns:
        List of basket definition dicts.
    """
    cfg = _get_regime_cfg(policy)
    if not cfg.get("enabled", False):
        return get_baskets(policy)

    threshold = float(cfg.get("rate_hike_threshold", _DEFAULT_RATE_HIKE_THRESHOLD))
    lb = int(cfg.get("lookback_days", lookback_days))

    regime = detect_rate_regime(
        prices_pit, as_of, lookback_days=lb, threshold=threshold
    )

    if regime == "rate_hike":
        logger.info(
            "get_regime_aware_baskets: rate_hike regime detected at %s — "
            "switching to RATE_HIKE_BASKETS (XLE/XLU/GLD/SHY)",
            as_of,
        )
        return list(RATE_HIKE_BASKETS)

    return get_baskets(policy)


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
