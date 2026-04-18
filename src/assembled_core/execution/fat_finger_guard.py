"""Fat-finger guard (Sprint 4 / Plan C29).

Rejects individual orders whose notional or quantity is abnormally large
compared to recent history. Designed as an additive pre-trade filter that
does not touch the existing ``pre_trade_checks`` chain — callers opt in
explicitly.

Two orthogonal checks:
  1. ``max_notional_usd``   — hard cap on absolute order notional
  2. ``max_qty_multiple``   — dynamic cap relative to the per-symbol recent
                              maximum quantity (``history_qty_by_symbol``)

Both checks can be disabled individually. The guard never mutates the
input DataFrame; it returns a new DataFrame plus a list of human-readable
reasons describing each rejection.

Typical use:

    from src.assembled_core.execution.fat_finger_guard import apply_fat_finger_guard

    filtered, reasons = apply_fat_finger_guard(
        orders,
        max_notional_usd=250_000.0,
        max_qty_multiple=3.0,
        history_qty_by_symbol={"AAPL": 100, "MSFT": 80},
    )
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def apply_fat_finger_guard(
    orders: pd.DataFrame,
    *,
    max_notional_usd: float | None = None,
    max_qty_multiple: float | None = None,
    history_qty_by_symbol: dict[str, float] | None = None,
    qty_col: str = "qty",
    price_col: str = "price",
    symbol_col: str = "symbol",
) -> tuple[pd.DataFrame, list[str]]:
    """Filter orders that look like fat-finger errors.

    Args:
        orders: DataFrame with at least ``symbol``, ``qty``, ``price`` columns.
        max_notional_usd: Hard cap on ``|qty * price|``. ``None`` disables.
        max_qty_multiple: Multiplier vs ``history_qty_by_symbol``. An order is
            rejected when ``|qty| > max_qty_multiple * history_qty[symbol]``.
            Requires ``history_qty_by_symbol`` to be provided.
        history_qty_by_symbol: Per-symbol recent max quantity (for example the
            rolling 20-session max order qty). Missing symbols are skipped
            from the multiple check.
        qty_col / price_col / symbol_col: Column names (overridable for tests).

    Returns:
        ``(filtered_orders, reasons)`` — ``filtered_orders`` is a new
        DataFrame with rejected rows removed. ``reasons`` lists a human-
        readable message per rejection. When nothing is rejected, the
        frame is an unchanged *copy*.
    """
    if orders is None or orders.empty:
        return orders.copy() if orders is not None else pd.DataFrame(), []

    # Missing required columns used to silently return the orders unchanged
    # with an empty reasons list — indistinguishable from "passed all checks".
    # A refactor renaming ``price`` → ``limit_price`` (or an upstream stage
    # dropping the column) would then disable the only notional fat-finger cap
    # without any trace. Raise so the caller — which always wraps this call in
    # a try/except at the paper-engine layer — logs a loud ERROR and the
    # operator sees a schema drift instead of invisibly unbounded notional.
    missing = [c for c in (qty_col, price_col) if c not in orders.columns]
    if missing:
        raise ValueError(
            f"apply_fat_finger_guard: required column(s) {missing} missing "
            f"from orders; available columns: {sorted(orders.columns)}"
        )

    reasons: list[str] = []
    keep_mask = pd.Series(True, index=orders.index)
    history = history_qty_by_symbol or {}

    for idx, row in orders.iterrows():
        qty = float(row.get(qty_col, 0.0) or 0.0)
        price = float(row.get(price_col, 0.0) or 0.0)
        notional = abs(qty * price)
        symbol = str(row.get(symbol_col, ""))

        if max_notional_usd is not None and notional > float(max_notional_usd):
            keep_mask.at[idx] = False
            reasons.append(
                f"fat_finger: {symbol} rejected — notional={notional:.2f} "
                f"> max_notional_usd={float(max_notional_usd):.2f}"
            )
            continue

        if max_qty_multiple is not None and symbol in history:
            hist_qty = float(history[symbol] or 0.0)
            if hist_qty > 0 and abs(qty) > float(max_qty_multiple) * hist_qty:
                keep_mask.at[idx] = False
                reasons.append(
                    f"fat_finger: {symbol} rejected — qty={qty:.2f} > "
                    f"{float(max_qty_multiple):.2f} * history_max={hist_qty:.2f}"
                )

    filtered = orders.loc[keep_mask].copy()
    return filtered, reasons


def apply_fat_finger_guard_from_policy(
    orders: pd.DataFrame,
    policy: dict[str, Any],
    *,
    history_qty_by_symbol: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Read fat-finger settings from ``policy['fat_finger_guard']`` and apply.

    Config shape::

        policy:
          fat_finger_guard:
            enabled: true
            max_notional_usd: 250000
            max_qty_multiple: 3.0

    When disabled or missing, returns ``(orders.copy(), [])``.
    """
    cfg = (policy or {}).get("fat_finger_guard") or {}
    if not cfg.get("enabled", False):
        return (orders.copy() if orders is not None else pd.DataFrame()), []

    return apply_fat_finger_guard(
        orders,
        max_notional_usd=cfg.get("max_notional_usd"),
        max_qty_multiple=cfg.get("max_qty_multiple"),
        history_qty_by_symbol=history_qty_by_symbol,
    )


__all__ = ["apply_fat_finger_guard", "apply_fat_finger_guard_from_policy"]
