"""Paper trading endpoints.

This module provides endpoints for paper trading with integrated risk controls.
By default, all orders are validated through pre-trade checks and kill switch
before execution. This ensures safe paper trading simulation.

Risk Controls:
- Pre-trade checks: Position size limits, gross exposure limits
- Kill switch: Emergency block via ASSEMBLED_KILL_SWITCH environment variable
- Default: Both checks are enabled (safe by default)
"""
from __future__ import annotations

import uuid
from typing import Any, Union

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.assembled_core.api.models import (
    PaperOrderRequest,
    PaperOrderResponse,
    PaperPosition,
    PaperResetResponse,
)
from src.assembled_core.execution.kill_switch import is_kill_switch_engaged
from src.assembled_core.execution.paper_trading_engine import (
    PaperOrder,
    PaperTradingEngine,
)
from src.assembled_core.execution.pre_trade_checks import PreTradeConfig
from src.assembled_core.execution.risk_controls import filter_orders_with_risk_controls
from src.assembled_core.logging_utils import setup_logging

logger = setup_logging(level="INFO")

router = APIRouter()

# Global engine instance (singleton for this process)
_engine = PaperTradingEngine()

# Paper trading configuration
# Default: Risk controls enabled (safe by default)
# Can be disabled for testing/development
_ENABLE_RISK_CONTROLS = True
_DEFAULT_PRE_TRADE_CONFIG: PreTradeConfig | None = None  # None = no limits by default


def _convert_order_to_response(order: PaperOrder) -> PaperOrderResponse:
    """Convert PaperOrder to PaperOrderResponse."""
    from src.assembled_core.api.models import OrderSide
    
    return PaperOrderResponse(
        order_id=order.order_id,
        symbol=order.symbol,
        side=OrderSide(order.side),
        quantity=order.quantity,
        price=order.price,
        status=order.status,
        reason=order.reason,
        client_order_id=order.client_order_id,
    )


def _build_portfolio_snapshot_from_engine() -> pd.DataFrame | None:
    """Build portfolio snapshot from current paper trading positions.
    
    Returns:
        DataFrame with columns: symbol, qty (or None if no positions)
    """
    positions = _engine.get_positions()
    if not positions:
        return None
    
    return pd.DataFrame({
        "symbol": [pos.symbol for pos in positions],
        "qty": [pos.quantity for pos in positions]
    })


def _apply_risk_controls_to_paper_orders(
    paper_orders: list[PaperOrder],
    enable_pre_trade_checks: bool = True,
    enable_kill_switch: bool = True,
    pre_trade_config: PreTradeConfig | None = None
) -> tuple[list[PaperOrder], list[PaperOrder]]:
    """Apply risk controls to paper orders.
    
    This function converts paper orders to DataFrame format, applies risk controls,
    and returns filtered orders (passed) and rejected orders (blocked).
    
    Args:
        paper_orders: List of PaperOrder objects to validate
        enable_pre_trade_checks: Enable pre-trade checks (default: True)
        enable_kill_switch: Enable kill switch (default: True)
        pre_trade_config: Optional PreTradeConfig (default: None = no limits)
    
    Returns:
        Tuple of (passed_orders, rejected_orders):
        - passed_orders: Orders that passed risk controls (will be FILLED)
        - rejected_orders: Orders that were blocked (status=REJECTED, reason set)
    """
    if not paper_orders:
        return [], []
    
    # Convert paper orders to DataFrame
    orders_df = pd.DataFrame({
        "symbol": [order.symbol for order in paper_orders],
        "side": [order.side for order in paper_orders],
        "qty": [order.quantity for order in paper_orders],
        "price": [order.price if order.price is not None else 0.0 for order in paper_orders]
    })
    
    # Build portfolio snapshot from current positions
    portfolio = _build_portfolio_snapshot_from_engine()
    
    # Apply risk controls
    try:
        filtered_df, risk_result = filter_orders_with_risk_controls(
            orders_df,
            portfolio=portfolio,
            qa_status=None,  # QA status not available in paper trading
            risk_summary=None,
            pre_trade_config=pre_trade_config or _DEFAULT_PRE_TRADE_CONFIG,
            enable_pre_trade_checks=enable_pre_trade_checks,
            enable_kill_switch=enable_kill_switch
        )
    except Exception as e:
        logger.error(f"Error applying risk controls: {e}", exc_info=True)
        # On error, reject all orders
        rejected = paper_orders.copy()
        for order in rejected:
            order.status = "REJECTED"
            order.reason = f"Risk control error: {str(e)}"
        return [], rejected
    
    # Determine which orders passed and which were rejected
    passed_orders = []
    rejected_orders = []
    
    # Check kill switch first (before matching)
    if enable_kill_switch and is_kill_switch_engaged():
        # All orders rejected due to kill switch
        rejected = paper_orders.copy()
        for order in rejected:
            order.status = "REJECTED"
            order.reason = "KILL_SWITCH: Kill switch is engaged"
        return [], rejected
    
    # Create a mapping from (symbol, side, qty, price) to order index in filtered_df
    # Use a more robust matching approach: match by symbol, side, qty, and approximate price
    filtered_set = set()
    for _, row in filtered_df.iterrows():
        # Normalize for matching
        symbol = str(row["symbol"]).strip().upper()
        side = str(row["side"]).strip().upper()
        qty = float(row["qty"])
        price = float(row["price"])
        filtered_set.add((symbol, side, qty, price))
    
    # Categorize orders
    for order in paper_orders:
        # Normalize for matching
        symbol = order.symbol.strip().upper()
        side = order.side.upper()
        qty = order.quantity
        price = order.price if order.price is not None else 0.0
        
        key = (symbol, side, qty, price)
        
        if key in filtered_set:
            # Order passed risk controls
            passed_orders.append(order)
        else:
            # Order was blocked
            order.status = "REJECTED"
            
            # Determine reason from risk result
            if risk_result.kill_switch_engaged:
                order.reason = "KILL_SWITCH: Kill switch is engaged"
            elif risk_result.pre_trade_result and not risk_result.pre_trade_result.is_ok:
                # Use first blocked reason (or generic message)
                if risk_result.pre_trade_result.blocked_reasons:
                    reason_text = risk_result.pre_trade_result.blocked_reasons[0]
                    # Truncate if too long
                    if len(reason_text) > 200:
                        reason_text = reason_text[:197] + "..."
                    order.reason = f"PRE_TRADE_CHECK_FAILED: {reason_text}"
                else:
                    order.reason = "PRE_TRADE_CHECK_FAILED: Order blocked by pre-trade checks"
            else:
                order.reason = "PRE_TRADE_CHECK_FAILED: Order blocked by risk controls"
            
            rejected_orders.append(order)
    
    return passed_orders, rejected_orders


@router.post("/orders", response_model=list[PaperOrderResponse])
def submit_paper_orders(
    orders: Union[PaperOrderRequest, list[PaperOrderRequest]]
) -> list[PaperOrderResponse]:
    """Submit orders to paper trading engine.
    
    Accepts either a single order or a list of orders. Orders are validated through
    risk controls (pre-trade checks and kill switch) before execution. Orders that
    pass risk controls are immediately filled and positions are updated.
    
    **Risk Controls (enabled by default):**
    - Pre-trade checks: Position size limits, gross exposure limits
    - Kill switch: Emergency block via ASSEMBLED_KILL_SWITCH environment variable
    
    **Order Status:**
    - FILLED: Order passed risk controls and was executed
    - REJECTED: Order was blocked by risk controls (reason provided)
    
    Args:
        orders: Single PaperOrderRequest or list of PaperOrderRequest objects
    
    Returns:
        List of PaperOrderResponse objects with status updated (FILLED or REJECTED)
    """
    # Normalize to list
    if isinstance(orders, PaperOrderRequest):
        order_list = [orders]
    else:
        order_list = orders
    
    # Convert requests to PaperOrder objects
    paper_orders = []
    for req in order_list:
        paper_order = PaperOrder(
            order_id=str(uuid.uuid4()),
            symbol=req.symbol,
            side=req.side.value,
            quantity=req.quantity,
            price=req.price,
            status="NEW",
            client_order_id=req.client_order_id,
            route=req.route if req.route is not None else "PAPER",  # Use provided route or default to "PAPER"
            source=req.source,  # Pass through source from request
        )
        paper_orders.append(paper_order)
    
    # Apply risk controls (if enabled)
    if _ENABLE_RISK_CONTROLS:
        passed_orders, rejected_orders = _apply_risk_controls_to_paper_orders(
            paper_orders,
            enable_pre_trade_checks=True,
            enable_kill_switch=True,
            pre_trade_config=_DEFAULT_PRE_TRADE_CONFIG
        )
        
        # Only submit orders that passed risk controls
        if passed_orders:
            try:
                filled_orders = _engine.submit_orders(passed_orders)
            except ValueError as e:
                # If engine rejects, mark as rejected
                for order in passed_orders:
                    order.status = "REJECTED"
                    order.reason = f"Engine validation failed: {str(e)}"
                filled_orders = []
                rejected_orders.extend(passed_orders)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error submitting orders: {e}")
        else:
            filled_orders = []
        
        # Combine filled and rejected orders
        all_orders = filled_orders + rejected_orders
    else:
        # Risk controls disabled - submit all orders directly
        try:
            all_orders = _engine.submit_orders(paper_orders)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error submitting orders: {e}")
    
    # Convert to responses
    responses = [_convert_order_to_response(order) for order in all_orders]
    return responses


@router.get("/orders", response_model=list[PaperOrderResponse])
def list_paper_orders(limit: int | None = 50) -> list[PaperOrderResponse]:
    """List recent paper trading orders.
    
    Args:
        limit: Maximum number of orders to return (default: 50)
    
    Returns:
        List of PaperOrderResponse objects (newest first)
    """
    if limit is not None and limit < 0:
        raise HTTPException(status_code=400, detail="limit must be >= 0")
    
    try:
        orders = _engine.list_orders(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing orders: {e}")
    
    responses = [_convert_order_to_response(order) for order in orders]
    return responses


@router.get("/positions", response_model=list[PaperPosition])
def get_paper_positions() -> list[PaperPosition]:
    """Get current paper trading positions.
    
    Returns:
        List of PaperPosition objects (only non-zero positions)
    """
    try:
        positions = _engine.get_positions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting positions: {e}")
    
    return [
        PaperPosition(symbol=pos.symbol, quantity=pos.quantity)
        for pos in positions
    ]


@router.post("/reset", response_model=PaperResetResponse)
def reset_paper_trading() -> PaperResetResponse:
    """Reset paper trading engine (clear all orders and positions).
    
    This endpoint is primarily for testing and development purposes.
    
    Returns:
        PaperResetResponse with status
    """
    try:
        _engine.reset()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting engine: {e}")
    
    return PaperResetResponse(status="ok")

