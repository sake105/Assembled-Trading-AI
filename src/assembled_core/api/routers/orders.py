# src/assembled_core/api/routers/orders.py
"""Orders endpoints."""
from __future__ import annotations


from fastapi import APIRouter, HTTPException

from src.assembled_core.api.models import Frequency, OrderPreview, OrdersResponse
from src.assembled_core.config import OUTPUT_DIR
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
        HTTPException: 404 if orders file not found, 500 if data is malformed
    """
    try:
        df = load_orders(freq.value, output_dir=OUTPUT_DIR, strict=True)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Malformed orders data: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading orders: {e}")
    
    # Convert DataFrame rows to OrderPreview models
    orders_list = []
    total_notional = 0.0
    
    for _, row in df.iterrows():
        qty = float(row["qty"])
        price = float(row["price"])
        notional = qty * price
        total_notional += notional
        
        orders_list.append(
            OrderPreview(
                timestamp=row["timestamp"],
                symbol=str(row["symbol"]),
                side=row["side"],  # Already validated as "BUY" or "SELL"
                qty=qty,
                price=price,
                notional=notional
            )
        )
    
    # Get first and last timestamps
    first_ts = df["timestamp"].min() if not df.empty else None
    last_ts = df["timestamp"].max() if not df.empty else None
    
    return OrdersResponse(
        frequency=freq,
        orders=orders_list,
        count=len(orders_list),
        total_notional=total_notional,
        first_timestamp=first_ts,
        last_timestamp=last_ts
    )

