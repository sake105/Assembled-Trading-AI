# src/assembled_core/execution/fill_model_pipeline.py
"""Fill model pipeline: central function for applying all fill model components."""

from __future__ import annotations

import pandas as pd

from src.assembled_core.execution.fill_model import (
    PartialFillModel,
    _ensure_reject_reason_filled,
    apply_cash_gate,
    apply_limit_order_fills,
    apply_partial_fills,
    apply_session_gate,
    ensure_fill_schema,
)


def apply_fill_model_pipeline(
    orders: pd.DataFrame,
    *,
    prices: pd.DataFrame,
    freq: str,
    partial_fill_model: PartialFillModel | None = None,
    strict_session_gate: bool = True,
    available_cash: float | None = None,
) -> pd.DataFrame:
    """Apply complete fill model pipeline: cash gate -> session gate -> limit -> partial.

    Order of steps:
    1. Cash gate (optional): reject BUY orders with notional > available_cash -> INSUFFICIENT_CASH
    2. Session gate: reject orders outside trading sessions
    3. Limit order eligibility: reject unreachable limits
    4. Partial fill model: apply ADV-based partial fills
    """
    if orders.empty:
        return orders

    # Step 0: Cash gate (when available_cash provided)
    if available_cash is not None and available_cash > 0:
        fills = apply_cash_gate(orders, available_cash)
    else:
        fills = orders.copy()

    # Step 1: Apply session gate (if exchange_calendars available)
    try:
        fills = apply_session_gate(fills, freq=freq, strict=strict_session_gate)
    except (ImportError, RuntimeError):
        # exchange_calendars not available or raised RuntimeError: skip session gate if strict=False
        if strict_session_gate:
            raise  # Re-raise if strict=True
        # Permissive fallback: allow all orders (keep fill schema if coming from cash gate)
        fills = ensure_fill_schema(fills, default_full_fill=True)

    # Step 2: Apply limit order fills (if limit orders present)
    # This will check limit eligibility and apply partial fills if provided
    if "order_type" in fills.columns and (fills["order_type"] == "limit").any():
        fills = apply_limit_order_fills(
            fills,
            prices=prices,
            partial_fill_model=partial_fill_model,
        )
    else:
        # No limit orders: apply partial fills directly
        fills = apply_partial_fills(
            fills,
            prices=prices,
            partial_fill_model=partial_fill_model,
        )

    # Final guard: no rejected row leaves the pipeline with empty reject_reason
    _ensure_reject_reason_filled(fills)

    return fills
