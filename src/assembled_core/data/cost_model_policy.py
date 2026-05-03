"""Policy-driven cost model: estimate rebalancing cost for a weight change.

Complements the full TCA module (risk/transaction_costs.py) with a simpler,
policy-configurable wrapper suitable for use in the trading cycle and backtests.

The model estimates one-way cost in basis points:
    cost_bps = commission_bps + half_spread_bps + slippage_bps

Total rebalancing cost as a fraction of portfolio value:
    cost_fraction = turnover * cost_bps / 10000

where turnover = sum of absolute weight changes across all symbols.

M7-T03: policy-driven cost model wrapper.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core cost estimation
# ---------------------------------------------------------------------------


def estimate_rebalance_cost_fraction(
    old_weights: dict[str, float],
    new_weights: dict[str, float],
    policy: dict[str, Any] | None = None,
    commission_bps: float = 0.5,
    half_spread_bps: float = 2.5,
    slippage_bps: float = 3.0,
) -> float:
    """Estimate total rebalancing cost as a fraction of portfolio value.

    Cost model:
        one_way_cost_bps = commission_bps + half_spread_bps + slippage_bps
        turnover = sum |new_weight_i - old_weight_i|
        cost_fraction = turnover * one_way_cost_bps / 10000

    Args:
        old_weights: symbol -> weight before rebalancing.
        new_weights: symbol -> weight after rebalancing.
        policy: Optional policy dict. Reads from ``cost_model`` section:
            - commission_bps (float, default 0.5)
            - half_spread_bps (float, default 2.5)
            - slippage_bps (float, default 3.0)
            - enabled (bool, default True)
        commission_bps: Fallback commission if not in policy.
        half_spread_bps: Fallback half-spread if not in policy.
        slippage_bps: Fallback slippage if not in policy.

    Returns:
        Estimated total cost as a fraction of portfolio value (e.g. 0.0012 = 12 bps).
        Returns 0.0 if cost model is disabled in policy or both weight dicts are empty.
    """
    cm = (policy or {}).get("cost_model") or {}
    if not cm.get("enabled", True):
        return 0.0

    c_bps = float(cm.get("commission_bps", commission_bps) or commission_bps)
    s_bps = float(cm.get("half_spread_bps", half_spread_bps) or half_spread_bps)
    sl_bps = float(cm.get("slippage_bps", slippage_bps) or slippage_bps)

    one_way_cost_bps = c_bps + s_bps + sl_bps

    # Collect all symbols
    symbols = set(old_weights or {}) | set(new_weights or {})
    if not symbols:
        return 0.0

    turnover = sum(
        abs((new_weights or {}).get(s, 0.0) - (old_weights or {}).get(s, 0.0))
        for s in symbols
    )

    return turnover * one_way_cost_bps / 10_000.0


def compute_cost_drag_per_period(
    turnover_series: "list[float]",
    policy: dict[str, Any] | None = None,
    commission_bps: float = 0.5,
    half_spread_bps: float = 2.5,
    slippage_bps: float = 3.0,
) -> list[float]:
    """Compute per-period cost drag from a turnover series.

    Args:
        turnover_series: List of per-period turnover values (0–2 range typical).
        policy: Optional policy dict with ``cost_model`` section.
        commission_bps: Fallback commission.
        half_spread_bps: Fallback half-spread.
        slippage_bps: Fallback slippage.

    Returns:
        List of cost fractions matching the length of *turnover_series*.
    """
    cm = (policy or {}).get("cost_model") or {}
    if not cm.get("enabled", True):
        return [0.0] * len(turnover_series)

    c_bps = float(cm.get("commission_bps", commission_bps) or commission_bps)
    s_bps = float(cm.get("half_spread_bps", half_spread_bps) or half_spread_bps)
    sl_bps = float(cm.get("slippage_bps", slippage_bps) or slippage_bps)
    one_way = (c_bps + s_bps + sl_bps) / 10_000.0

    return [t * one_way for t in turnover_series]


def get_effective_cost_params(
    policy: dict[str, Any] | None = None,
    commission_bps: float = 0.5,
    half_spread_bps: float = 2.5,
    slippage_bps: float = 3.0,
) -> dict[str, float]:
    """Return the effective cost parameters after applying policy overrides.

    Args:
        policy: Policy dict with optional ``cost_model`` section.
        commission_bps: Default commission.
        half_spread_bps: Default half-spread.
        slippage_bps: Default slippage.

    Returns:
        Dict with keys: ``commission_bps``, ``half_spread_bps``, ``slippage_bps``,
        ``one_way_cost_bps``, ``enabled``.
    """
    cm = (policy or {}).get("cost_model") or {}
    c = float(cm.get("commission_bps", commission_bps) or commission_bps)
    s = float(cm.get("half_spread_bps", half_spread_bps) or half_spread_bps)
    sl = float(cm.get("slippage_bps", slippage_bps) or slippage_bps)
    enabled = bool(cm.get("enabled", True))
    return {
        "commission_bps": c,
        "half_spread_bps": s,
        "slippage_bps": sl,
        "one_way_cost_bps": c + s + sl,
        "enabled": enabled,
    }


# ---------------------------------------------------------------------------
# Per-symbol adaptive cost model (V2)
# ---------------------------------------------------------------------------

_DEFAULT_TIERS = {
    "mega_cap":  {"adv_min_usd": 100_000_000, "commission_bps": 0.2, "half_spread_bps": 1.0, "slippage_bps": 1.5},
    "large_cap": {"adv_min_usd": 20_000_000,  "commission_bps": 0.5, "half_spread_bps": 2.0, "slippage_bps": 2.5},
    "mid_cap":   {"adv_min_usd": 5_000_000,   "commission_bps": 0.8, "half_spread_bps": 3.5, "slippage_bps": 5.0},
    "small_cap": {"adv_min_usd": 1_000_000,   "commission_bps": 1.0, "half_spread_bps": 5.0, "slippage_bps": 8.0},
    "micro_cap": {"adv_min_usd": 0,           "commission_bps": 1.5, "half_spread_bps": 8.0, "slippage_bps": 12.0},
}


def load_cost_tiers(yaml_path: str | Path | None = None) -> dict[str, dict]:
    """Load cost tiers from YAML or return built-in defaults.

    Args:
        yaml_path: Path to cost_tiers.yaml. If None, uses built-in defaults.

    Returns:
        Dict mapping tier_name -> {adv_min_usd, commission_bps, half_spread_bps, slippage_bps}.
    """
    if yaml_path is not None:
        yaml_path = Path(yaml_path)
        if yaml_path.exists():
            try:
                import yaml
                with open(yaml_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                return cfg.get("tiers", _DEFAULT_TIERS)
            except Exception as exc:
                logger.warning("[CostModelPolicy] failed to load tiers from %s: %s", yaml_path, exc)
    return _DEFAULT_TIERS


def compute_adv_usd(
    prices: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """Compute per-symbol Average Daily Volume in USD.

    Args:
        prices: DataFrame with columns: symbol, close, volume.
        window: Rolling lookback window.

    Returns:
        Series indexed by symbol with mean ADV in USD over the last *window* bars.
    """
    if prices.empty or "volume" not in prices.columns:
        return pd.Series(dtype=float)

    df = prices.copy()
    close_col = "close" if "close" in df.columns else "Close"
    vol_col = "volume" if "volume" in df.columns else "Volume"

    if close_col not in df.columns or vol_col not in df.columns:
        return pd.Series(dtype=float)

    df["_dollar_vol"] = df[close_col].abs() * df[vol_col].abs()

    # Per-symbol mean of the last *window* bars (tail+mean avoids per-group apply overhead)
    sort_cols = ["symbol", "timestamp"] if "timestamp" in df.columns else ["symbol"]
    result = (
        df.sort_values(sort_cols)
        .groupby("symbol")
        .tail(window)
        .groupby("symbol")["_dollar_vol"]
        .mean()
    )
    return result


def classify_symbol_tier(
    adv_usd: float,
    tiers: dict[str, dict] | None = None,
) -> str:
    """Classify a single symbol into a cost tier based on ADV.

    Args:
        adv_usd: Average Daily Volume in USD.
        tiers: Tier definitions (from load_cost_tiers).

    Returns:
        Tier name string.
    """
    tiers = tiers or _DEFAULT_TIERS
    # Sort tiers by adv_min_usd descending so we match highest tier first
    sorted_tiers = sorted(tiers.items(), key=lambda x: x[1].get("adv_min_usd", 0), reverse=True)
    for name, t in sorted_tiers:
        if adv_usd >= t.get("adv_min_usd", 0):
            return name
    return "mid_cap"


def get_per_symbol_costs(
    prices: pd.DataFrame,
    tiers: dict[str, dict] | None = None,
    adv_window: int = 20,
) -> pd.DataFrame:
    """Compute per-symbol cost parameters based on ADV tier classification.

    Args:
        prices: OHLCV DataFrame with symbol, close, volume columns.
        tiers: Cost tier definitions. Uses defaults if None.
        adv_window: Rolling window for ADV calculation.

    Returns:
        DataFrame with columns: symbol, tier, commission_bps, half_spread_bps,
        slippage_bps, one_way_cost_bps, adv_usd.
    """
    tiers = tiers or _DEFAULT_TIERS
    adv_series = compute_adv_usd(prices, window=adv_window)

    rows = []
    for sym, adv in adv_series.items():
        tier_name = classify_symbol_tier(float(adv), tiers)
        tier = tiers[tier_name]
        c = tier["commission_bps"]
        s = tier["half_spread_bps"]
        sl = tier["slippage_bps"]
        rows.append({
            "symbol": sym,
            "tier": tier_name,
            "commission_bps": c,
            "half_spread_bps": s,
            "slippage_bps": sl,
            "one_way_cost_bps": c + s + sl,
            "adv_usd": float(adv),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["symbol", "tier", "commission_bps", "half_spread_bps",
                 "slippage_bps", "one_way_cost_bps", "adv_usd"]
    )


def estimate_rebalance_cost_per_symbol(
    old_weights: dict[str, float],
    new_weights: dict[str, float],
    symbol_costs: pd.DataFrame | None = None,
    fallback_cost_bps: float = 6.0,
) -> float:
    """Estimate rebalancing cost using per-symbol cost tiers.

    Args:
        old_weights: symbol -> old weight.
        new_weights: symbol -> new weight.
        symbol_costs: DataFrame from get_per_symbol_costs (needs symbol, one_way_cost_bps).
        fallback_cost_bps: Fallback one-way cost if symbol not in symbol_costs.

    Returns:
        Estimated total cost as a fraction of portfolio value.
    """
    symbols = set(old_weights or {}) | set(new_weights or {})
    if not symbols:
        return 0.0

    cost_map: dict[str, float] = {}
    if symbol_costs is not None and not symbol_costs.empty:
        cost_map = dict(zip(symbol_costs["symbol"], symbol_costs["one_way_cost_bps"]))

    total_cost = 0.0
    for s in symbols:
        delta = abs((new_weights or {}).get(s, 0.0) - (old_weights or {}).get(s, 0.0))
        cost_bps = cost_map.get(s, fallback_cost_bps)
        total_cost += delta * cost_bps / 10_000.0

    return total_cost


__all__ = [
    "estimate_rebalance_cost_fraction",
    "compute_cost_drag_per_period",
    "get_effective_cost_params",
    "load_cost_tiers",
    "compute_adv_usd",
    "classify_symbol_tier",
    "get_per_symbol_costs",
    "estimate_rebalance_cost_per_symbol",
]
