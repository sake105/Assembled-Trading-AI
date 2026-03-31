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

from typing import Any


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


__all__ = [
    "estimate_rebalance_cost_fraction",
    "compute_cost_drag_per_period",
    "get_effective_cost_params",
]
