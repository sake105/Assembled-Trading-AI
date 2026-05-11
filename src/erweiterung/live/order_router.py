"""Order-Router — konvertiert Allocator-Weights zu konkreten Trade-Orders.

Pipeline
--------
1. ``decide_next()`` liefert Weights pro Symbol (z. B. {"NVDA": 0.143, "GOOGL": 0.143, ...})
2. ``compute_target_notionals(weights, equity, exposure_cap)`` → $-Beträge
3. ``compute_orders(targets, current_positions, prices, lot_size)`` → Order-List
4. ``apply_pre_trade_checks(orders, policy)`` → filtered Orders
5. ``orders_to_dataframe(orders)`` → DataFrame für Logging/Execution

Diese API ist Mainline-Style: kompatibel mit ``trading_cycle_v2`` Order-Flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Order:
    """Eine konkrete Trade-Order."""

    symbol: str
    side: str  # "BUY" or "SELL"
    qty: float  # absolute units (positive)
    target_notional: float  # signed $-Betrag
    current_position: float
    target_position: float
    price: float
    reason: str = ""
    pre_trade_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "qty": self.qty,
            "target_notional": self.target_notional,
            "current_position": self.current_position,
            "target_position": self.target_position,
            "price": self.price,
            "reason": self.reason,
            "pre_trade_flags": ",".join(self.pre_trade_flags),
        }


@dataclass
class OrderRouterConfig:
    equity: float = 100_000.0  # Portfolio-Equity in USD
    exposure_cap: float = 1.0  # max gross exposure (1.0 = 100% long-only)
    lot_size: float = 1.0  # share-rounding (1.0 = whole shares)
    min_order_notional: float = 100.0  # Skip Orders < $100 (Cost-effizient)
    min_order_qty: float = 1.0  # Skip < 1 share
    rebalance_threshold: float = 0.005  # 0.5% min-change vs current weight
    """Wenn |target - current| weight-Diff < threshold → kein Trade (Anti-Churn)."""


def compute_target_notionals(
    weights: pd.Series,
    equity: float,
    exposure_cap: float = 1.0,
) -> pd.Series:
    """Konvertiere Weights → Ziel-Notional pro Symbol.

    Args:
        weights: Series mit Float-Weights (long-only: alle >= 0).
        equity: Portfolio-Equity-Wert in USD.
        exposure_cap: max gross exposure (Sum of |weights| × cap).

    Returns:
        Series mit Notional in USD pro Symbol.
    """
    if weights.empty:
        return pd.Series(dtype=float)
    # Renormalize wenn Sum überschreitet Cap
    gross = weights.abs().sum()
    if gross > exposure_cap and gross > 0:
        weights = weights * (exposure_cap / gross)
    return weights * equity


def compute_orders(
    target_notionals: pd.Series,
    current_positions: pd.Series,
    prices: pd.Series,
    config: OrderRouterConfig | None = None,
) -> list[Order]:
    """Erzeuge konkrete Buy/Sell-Orders aus Targets vs Current.

    Args:
        target_notionals: Series Symbol → Ziel-Notional in USD.
        current_positions: Series Symbol → aktuelle Position in Shares.
        prices: Series Symbol → Last-Price in USD.
        config: OrderRouterConfig.

    Returns:
        Liste von Order-Objekten (nur actionable Trades).
    """
    cfg = config or OrderRouterConfig()
    orders: list[Order] = []

    # Union of symbols (target + current)
    all_syms = sorted(set(target_notionals.index) | set(current_positions.index))

    for sym in all_syms:
        target_notional = float(target_notionals.get(sym, 0.0))
        current_pos = float(current_positions.get(sym, 0.0))
        price = float(prices.get(sym, np.nan))
        if not np.isfinite(price) or price <= 0:
            continue

        # Target position in shares
        target_pos_raw = target_notional / price
        # Round to lot-size
        target_pos = round(target_pos_raw / cfg.lot_size) * cfg.lot_size

        # Position diff
        qty_diff = target_pos - current_pos
        notional_diff = qty_diff * price

        # Anti-Churn: Skip wenn Notional-Diff zu klein
        if abs(notional_diff) < cfg.min_order_notional:
            continue
        if abs(qty_diff) < cfg.min_order_qty:
            continue

        # Rebalance-Threshold check (weight-basiert)
        current_weight = current_pos * price / cfg.equity if cfg.equity > 0 else 0
        target_weight = target_notional / cfg.equity if cfg.equity > 0 else 0
        if abs(target_weight - current_weight) < cfg.rebalance_threshold:
            continue

        side = "BUY" if qty_diff > 0 else "SELL"
        reason_parts = []
        if abs(current_pos) < 1e-6:
            reason_parts.append("NEW_POSITION")
        if target_pos == 0 and current_pos > 0:
            reason_parts.append("EXIT_POSITION")
        if not reason_parts:
            reason_parts.append("REBALANCE")
        reason = "|".join(reason_parts)

        orders.append(
            Order(
                symbol=sym,
                side=side,
                qty=abs(qty_diff),
                target_notional=target_notional,
                current_position=current_pos,
                target_position=target_pos,
                price=price,
                reason=reason,
            )
        )

    return orders


def apply_pre_trade_checks(
    orders: list[Order],
    policy: dict[str, Any] | None = None,
) -> list[Order]:
    """Pre-Trade-Checks: max-position-size, blacklist, total-exposure.

    Setzt ``pre_trade_flags`` auf jeder Order. Orders mit kritischen Flags
    werden NICHT entfernt, sondern markiert — entscheidet Caller.

    Args:
        orders: Liste Order-Objekten.
        policy: dict mit Optional-Keys:
            - max_position_pct (z.B. 0.10 = max 10% pro Symbol)
            - blacklist (List[str])
            - total_equity_for_checks (für %-Berechnung; default sum of all notionals)
    """
    pol = policy or {}
    max_pos_pct = float(pol.get("max_position_pct", 0.20))
    blacklist = set(pol.get("blacklist", []) or [])
    total_eq = float(pol.get("total_equity_for_checks") or 0.0)
    if total_eq <= 0:
        total_eq = sum(abs(o.target_notional) for o in orders)
        if total_eq == 0:
            total_eq = 1.0

    for o in orders:
        flags = []
        if o.symbol in blacklist:
            flags.append("BLACKLISTED")
        target_pct = abs(o.target_notional) / total_eq if total_eq > 0 else 0
        if target_pct > max_pos_pct:
            flags.append(f"OVER_MAX_POS_{target_pct:.1%}")
        if o.price <= 0:
            flags.append("INVALID_PRICE")
        o.pre_trade_flags = flags
    return orders


def orders_to_dataframe(orders: list[Order]) -> pd.DataFrame:
    """Konvertiere Order-Liste → DataFrame für Logging/Execution-Pipeline."""
    if not orders:
        return pd.DataFrame()
    return pd.DataFrame([o.to_dict() for o in orders])


def decision_to_orders(
    decision: dict,
    current_positions: pd.Series,
    prices: pd.Series,
    config: OrderRouterConfig | None = None,
    pre_trade_policy: dict[str, Any] | None = None,
) -> list[Order]:
    """End-to-End: LiveDecisionEngine.decide_next() Output → Order-List.

    Erwartet das ``decision``-Dict mit ``eq_top_weights`` und ``xa_hybrid_weights``
    (Output von LiveDecisionEngine.decide_next()).

    Args:
        decision: Output von ``LiveDecisionEngine.decide_next()``.
        current_positions: Series Symbol → aktuelle Shares.
        prices: Series Symbol → Last-Prices.
        config: OrderRouterConfig.
        pre_trade_policy: Pre-Trade-Check-Policy.

    Returns:
        Liste von Order-Objekten mit Pre-Trade-Flags.
    """
    cfg = config or OrderRouterConfig()
    sa_w = float(decision.get("sa_weight", 0.70))
    eq_top = decision.get("eq_top_weights", pd.Series(dtype=float))
    xa_hyb = decision.get("xa_hybrid_weights", pd.Series(dtype=float))

    # Combine (sa_weight × eq + (1-sa_weight) × xa)
    sa_lev = float(decision.get("sa_leverage", 1.0))
    combined_eq = sa_w * sa_lev * eq_top
    combined_xa = (1 - sa_w) * xa_hyb
    combined = combined_eq.add(combined_xa, fill_value=0)
    combined = combined[combined != 0]

    target_notionals = compute_target_notionals(combined, cfg.equity, cfg.exposure_cap)
    orders = compute_orders(target_notionals, current_positions, prices, cfg)
    orders = apply_pre_trade_checks(orders, pre_trade_policy)
    return orders


__all__ = [
    "Order",
    "OrderRouterConfig",
    "compute_target_notionals",
    "compute_orders",
    "apply_pre_trade_checks",
    "orders_to_dataframe",
    "decision_to_orders",
]
