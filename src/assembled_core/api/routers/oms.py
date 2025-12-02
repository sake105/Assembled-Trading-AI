"""OMS (Order Management System) endpoints for blotter and execution views.

This module provides OMS-Light functionality over the paper trading engine,
including order blotter, execution views, and route management.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.assembled_core.api.models import OmsExecution, OmsOrderView, OmsRoute, OrderSide
from src.assembled_core.api.routers.paper_trading import _engine

router = APIRouter()


def _convert_paper_order_to_oms_view(order) -> OmsOrderView:
    """Convert PaperOrder to OmsOrderView."""
    from src.assembled_core.api.models import OrderSide
    
    return OmsOrderView(
        order_id=order.order_id,
        symbol=order.symbol,
        side=OrderSide(order.side),
        quantity=order.quantity,
        price=order.price,
        status=order.status,
        route=order.route,
        source=order.source,
        client_order_id=order.client_order_id,
        created_at=order.created_at,
    )


@router.get("/blotter", response_model=list[OmsOrderView])
def get_oms_blotter(
    symbol: str | None = None,
    status: str | None = None,
    route: str | None = None,
    limit: int = 100,
) -> list[OmsOrderView]:
    """Get OMS blotter view of all orders.
    
    The blotter shows all orders with filtering and sorting capabilities.
    
    Args:
        symbol: Filter by symbol (exact match)
        status: Filter by status (e.g., "FILLED", "REJECTED")
        route: Filter by route (e.g., "PAPER")
        limit: Maximum number of orders to return (default: 100)
    
    Returns:
        List of OmsOrderView objects (newest first, after filtering)
    """
    if limit < 0:
        raise HTTPException(status_code=400, detail="limit must be >= 0")
    
    try:
        # Get all orders from paper trading engine
        all_orders = _engine.list_orders(limit=None)  # Get all orders
        
        # Convert to OMS view
        blotter_orders = [_convert_paper_order_to_oms_view(order) for order in all_orders]
        
        # Apply filters
        if symbol is not None:
            symbol_upper = symbol.strip().upper()
            blotter_orders = [o for o in blotter_orders if o.symbol.upper() == symbol_upper]
        
        if status is not None:
            status_upper = status.strip().upper()
            blotter_orders = [o for o in blotter_orders if o.status.upper() == status_upper]
        
        if route is not None:
            route_upper = route.strip().upper()
            blotter_orders = [o for o in blotter_orders if o.route and o.route.upper() == route_upper]
        
        # Sort by created_at descending (newest first)
        blotter_orders.sort(key=lambda o: o.created_at, reverse=True)
        
        # Apply limit
        if limit > 0:
            blotter_orders = blotter_orders[:limit]
        
        return blotter_orders
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving blotter: {e}")


@router.get("/executions", response_model=list[OmsExecution])
def get_oms_executions(
    symbol: str | None = None,
    route: str | None = None,
    limit: int = 100,
) -> list[OmsExecution]:
    """Get OMS execution view (fills).
    
    For OMS-Light, each FILLED order is treated as a single execution.
    
    Args:
        symbol: Filter by symbol (exact match)
        route: Filter by route (e.g., "PAPER")
        limit: Maximum number of executions to return (default: 100)
    
    Returns:
        List of OmsExecution objects (newest first, after filtering)
    """
    if limit < 0:
        raise HTTPException(status_code=400, detail="limit must be >= 0")
    
    try:
        # Get all orders from paper trading engine
        all_orders = _engine.list_orders(limit=None)  # Get all orders
        
        # Filter to only FILLED orders
        filled_orders = [order for order in all_orders if order.status == "FILLED"]
        
        # Convert to executions
        executions = []
        for order in filled_orders:
            exec_id = f"EXEC-{order.order_id}"
            execution = OmsExecution(
                exec_id=exec_id,
                order_id=order.order_id,
                symbol=order.symbol,
                side=OrderSide(order.side),
                quantity=order.quantity,
                price=order.price,
                timestamp=order.created_at,  # Use created_at as execution timestamp
                route=order.route,
            )
            executions.append(execution)
        
        # Apply filters
        if symbol is not None:
            symbol_upper = symbol.strip().upper()
            executions = [e for e in executions if e.symbol.upper() == symbol_upper]
        
        if route is not None:
            route_upper = route.strip().upper()
            executions = [e for e in executions if e.route and e.route.upper() == route_upper]
        
        # Sort by timestamp descending (newest first)
        executions.sort(key=lambda e: e.timestamp, reverse=True)
        
        # Apply limit
        if limit > 0:
            executions = executions[:limit]
        
        return executions
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving executions: {e}")


@router.get("/routes", response_model=list[OmsRoute])
def get_oms_routes() -> list[OmsRoute]:
    """Get available OMS routes.
    
    Returns a list of configured routes, including the default paper trading route
    and placeholders for future broker routes.
    
    Returns:
        List of OmsRoute objects
    """
    routes = [
        OmsRoute(
            route_id="PAPER",
            description="Internal paper trading route",
            is_default=True
        ),
        # Placeholder for future routes
        # OmsRoute(
        #     route_id="IBKR",
        #     description="Interactive Brokers API route",
        #     is_default=False
        # ),
    ]
    
    return routes

