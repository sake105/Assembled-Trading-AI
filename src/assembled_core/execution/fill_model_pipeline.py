# src/assembled_core/execution/fill_model_pipeline.py
"""Fill model pipeline: central function for applying all fill model components."""

from __future__ import annotations

import pandas as pd

from src.assembled_core.execution.fill_model import (
    PartialFillModel,
    apply_limit_order_fills,
    apply_partial_fills,
    apply_session_gate,
)


def apply_fill_model_pipeline(
    orders: pd.DataFrame,
    *,
    prices: pd.DataFrame,
    freq: str,
    partial_fill_model: PartialFillModel | None = None,
    strict_session_gate: bool = True,
) -> pd.DataFrame:
    """Apply complete fill model pipeline: session gate -> limit -> partial.
    
    This is the central function for applying all fill model components in the correct order:
    1. Session gate (reject orders outside trading sessions)
    2. Limit order eligibility (reject limit orders not reachable)
    3. Partial fill model (apply ADV-based partial fills)
    
    Args:
        orders: DataFrame with columns: timestamp, symbol, side, qty, price
            Optional columns: order_type, limit_price
        prices: Prices DataFrame with columns: timestamp, symbol, close
            Optional columns: open, high, low, volume
        freq: Trading frequency ("1d" or "5min") for session gate
        partial_fill_model: Optional PartialFillModel instance.
            If None, assumes full fills (no partial fill constraints)
        strict_session_gate: If True, reject orders outside sessions (default: True)
            If False and exchange_calendars missing, allow all orders (permissive fallback)
    
    Returns:
        DataFrame with fill model applied:
        - fill_qty: Filled quantity (0 for rejected, <= qty for partial/full)
        - fill_price: Fill price (limit_price for limit orders, else order price)
        - status: "filled", "partial", or "rejected"
        - remaining_qty: qty - fill_qty
    
    Note:
        This function applies the fill model pipeline in the correct order:
        1. Session gate first (reject weekends/holidays)
        2. Limit eligibility second (reject unreachable limits)
        3. Partial fills third (apply ADV cap)
    """
    if orders.empty:
        return orders
    
    # Step 1: Apply session gate (if exchange_calendars available)
    try:
        fills = apply_session_gate(orders, freq=freq, strict=strict_session_gate)
    except ImportError:
        # exchange_calendars not available: skip session gate if strict=False
        if strict_session_gate:
            raise  # Re-raise if strict=True
        # Permissive fallback: allow all orders
        fills = orders.copy()
    
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
    
    return fills
