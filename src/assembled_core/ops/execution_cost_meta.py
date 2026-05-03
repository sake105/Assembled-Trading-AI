"""Part B deeper wiring: pre-trade execution cost estimates.

Bridges the M20 Almgren-Chriss + Smart-Order-Router modules into the live
paper cycle so every order carries:

- ``impact_cost_bps`` — Almgren-Chriss permanent + temporary impact estimate
- ``venue_allocation`` — SOR-chosen venue(s) and expected routing cost

Opt-in via policy:

```yaml
execution:
  cost_meta:
    enabled: true       # annotate result.meta.execution_cost with per-order estimates
    impact_limit_bps: 50  # orders beyond this get tagged high_impact=true (shadow)
    enforce: false        # when true, high-impact orders are dropped
  smart_order_router:
    enabled: true
    urgency: 0.5
    allow_dark_pools: true
    max_venues: 3
```

All paths are defensive no-ops on missing deps/bad data; never block the cycle.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


def _safe_price_map(prices: pd.DataFrame) -> dict[str, float]:
    if prices is None or prices.empty or "symbol" not in prices.columns:
        return {}
    price_col = "close" if "close" in prices.columns else "price"
    if price_col not in prices.columns:
        return {}
    try:
        return (
            prices.groupby("symbol")[price_col].last().astype(float).to_dict()
        )
    except Exception:
        return {}


def _safe_adv_map(prices: pd.DataFrame, default_adv: float = 1_000_000.0) -> dict[str, float]:
    """Estimate ADV from trailing N days of volume per symbol if available."""
    if prices is None or prices.empty or "symbol" not in prices.columns:
        return {}
    if "volume" not in prices.columns:
        return {}
    try:
        return (
            prices.groupby("symbol")["volume"]
            .apply(lambda s: float(s.tail(20).mean()) if len(s) > 0 else default_adv)
            .to_dict()
        )
    except Exception:
        return {}


def annotate_execution_cost(
    orders: pd.DataFrame,
    prices: pd.DataFrame,
    policy: dict[str, Any],
    *,
    regime: str = "bull",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach impact + routing cost estimates to each order.

    Returns ``(orders_out, meta_summary)``. ``orders_out`` may be a filtered
    copy when ``enforce=true`` removes high-impact orders; otherwise it is
    the input unchanged.
    """
    exec_cfg = (policy.get("execution") or {})
    cm_cfg = exec_cfg.get("cost_meta") or {}
    sor_cfg = exec_cfg.get("smart_order_router") or {}

    cm_enabled = bool(cm_cfg.get("enabled", False))
    sor_enabled = bool(sor_cfg.get("enabled", False))

    meta: dict[str, Any] = {
        "enabled": cm_enabled,
        "sor_enabled": sor_enabled,
        "n_orders_in": int(len(orders)) if orders is not None else 0,
        "per_order": [],
        "total_est_cost_bps": 0.0,
        "high_impact_count": 0,
        "dropped_high_impact": 0,
    }

    if not cm_enabled and not sor_enabled:
        return orders, meta

    if orders is None or orders.empty:
        return orders, meta

    impact_limit_bps = float(cm_cfg.get("impact_limit_bps", 50.0))
    enforce = bool(cm_cfg.get("enforce", False))

    try:
        from src.assembled_core.execution.almgren_chriss import estimate_impact_cost
    except Exception as exc:
        log.debug("[EXEC-COST] almgren_chriss import failed: %s", exc)
        estimate_impact_cost = None  # type: ignore

    try:
        from src.assembled_core.execution.smart_order_router import route_order
    except Exception as exc:
        log.debug("[EXEC-COST] smart_order_router import failed: %s", exc)
        route_order = None  # type: ignore

    price_map = _safe_price_map(prices)
    adv_map = _safe_adv_map(prices)
    default_adv = float(cm_cfg.get("default_adv", 1_000_000.0))
    default_sigma = float(cm_cfg.get("default_sigma", 0.02))
    sor_urgency = float(sor_cfg.get("urgency", 0.5))
    sor_allow_dark = bool(sor_cfg.get("allow_dark_pools", True))
    sor_max_venues = int(sor_cfg.get("max_venues", 3))

    drop_mask: list[bool] = []
    total_cost_bps = 0.0
    high_impact_count = 0

    for row in orders.itertuples(index=False):
        sym = str(getattr(row, "symbol", "")).strip()
        try:
            qty = abs(float(getattr(row, "qty", 0.0)))
        except (TypeError, ValueError):
            drop_mask.append(False)
            continue

        if not sym or qty <= 0:
            drop_mask.append(False)
            continue

        price = float(price_map.get(sym, 0.0))
        if price <= 0:
            drop_mask.append(False)
            continue

        adv = float(adv_map.get(sym, default_adv))
        if adv <= 0:
            adv = default_adv

        per = {"symbol": sym, "qty": qty}

        if cm_enabled and estimate_impact_cost is not None:
            try:
                cost = estimate_impact_cost(
                    total_shares=qty,
                    price=price,
                    adv=adv,
                    sigma=default_sigma,
                )
                per["impact_bps"] = float(cost.get("total_bps", 0.0))
                per["impact_usd"] = float(cost.get("total_cost_usd", 0.0))
                total_cost_bps += per["impact_bps"]
                if per["impact_bps"] > impact_limit_bps:
                    per["high_impact"] = True
                    high_impact_count += 1
            except Exception as exc:
                log.debug("[EXEC-COST] impact estimate failed for %s: %s", sym, exc)

        if sor_enabled and route_order is not None:
            try:
                routing = route_order(
                    order_size=qty,
                    signal_urgency=sor_urgency,
                    adv=adv,
                    regime=regime,
                    price=price,
                    allow_dark_pools=sor_allow_dark,
                    max_venues=sor_max_venues,
                )
                per["venues"] = "|".join(a.venue for a in routing.allocations)
                per["sor_cost_bps"] = float(routing.total_expected_cost_bps)
            except Exception as exc:
                log.debug("[EXEC-COST] SOR routing failed for %s: %s", sym, exc)

        meta["per_order"].append(per)

        if enforce and per.get("high_impact"):
            drop_mask.append(True)
        else:
            drop_mask.append(False)

    meta["total_est_cost_bps"] = round(total_cost_bps, 2)
    meta["high_impact_count"] = int(high_impact_count)

    orders_out = orders
    if enforce and any(drop_mask):
        keep = [not d for d in drop_mask]
        orders_out = orders.iloc[keep].reset_index(drop=True)
        meta["dropped_high_impact"] = int(sum(drop_mask))
        log.warning(
            "[EXEC-COST] dropped %d high-impact orders (>%.0f bps)",
            meta["dropped_high_impact"],
            impact_limit_bps,
        )

    if cm_enabled or sor_enabled:
        log.info(
            "[EXEC-COST] annotated %d orders | total_est=%.1f bps | high_impact=%d | dropped=%d",
            len(meta["per_order"]),
            meta["total_est_cost_bps"],
            meta["high_impact_count"],
            meta["dropped_high_impact"],
        )

    return orders_out, meta


__all__ = ["annotate_execution_cost"]
