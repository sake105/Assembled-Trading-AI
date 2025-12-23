# src/assembled_core/api/routers/orders.py
"""Orders endpoints."""

from __future__ import annotations


from fastapi import APIRouter, HTTPException

from src.assembled_core.api.models import Frequency, OrderPreview, OrdersResponse
from src.assembled_core.config import OUTPUT_DIR
from src.assembled_core.config.constants import MAX_ORDERS_PER_RESPONSE
from src.assembled_core.pipeline.io import load_orders

router = APIRouter()


@router.get("/orders/{freq}", response_model=OrdersResponse)
def get_orders(freq: Frequency) -> OrdersResponse:
    """Get orders for a given frequency.

    Args:
        freq: Trading frequency ("1d" or "5min")

    Returns:
        OrdersResponse with list of orders

    Raises:
        HTTPException: 400 if frequency is invalid, 404 if orders file not found,
                       413 if too many orders (DoS protection), 500 if data is malformed

    """
    # Validate frequency
    if freq.value not in ["1d", "5min"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid frequency: {freq.value}. Must be '1d' or '5min'",
        )

    try:
        df = load_orders(freq.value, output_dir=OUTPUT_DIR, strict=True)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Malformed orders data: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading orders: {e}")

    # Limit response size (DoS protection)
    if len(df) > MAX_ORDERS_PER_RESPONSE:
        raise HTTPException(
            status_code=413,
            detail=f"Too many orders ({len(df)}). Maximum: {MAX_ORDERS_PER_RESPONSE}. Consider pagination or filtering.",
        )

    # Convert DataFrame rows to OrderPreview models - vectorized
    # Compute notionals vectorized
    notionals = df["qty"] * df["price"]
    total_notional = float(notionals.sum())

    # Build orders list using vectorized operations
    orders_list = [
        OrderPreview(
            timestamp=ts,
            symbol=str(sym),
            side=side,  # Already validated as "BUY" or "SELL"
            qty=float(qty),
            price=float(px),
            notional=float(notional),
        )
        for ts, sym, side, qty, px, notional in zip(
            df["timestamp"],
            df["symbol"],
            df["side"],
            df["qty"],
            df["price"],
            notionals,
        )
    ]

    # Get first and last timestamps
    first_ts = df["timestamp"].min() if not df.empty else None
    last_ts = df["timestamp"].max() if not df.empty else None

    return OrdersResponse(
        frequency=freq,
        orders=orders_list,
        count=len(orders_list),
        total_notional=total_notional,
        first_timestamp=first_ts,
        last_timestamp=last_ts,
    )
