# src/assembled_core/execution/fill_model.py
"""Fill model: Schema and contract for trades/fills with partial fill support.

This module provides functions to convert orders to fills and ensure fill schema compliance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PartialFillModel:
    """Partial fill model based on liquidity (ADV proxy).

    This model determines the maximum fill quantity based on Average Daily Volume (ADV)
    and a participation rate cap. Orders exceeding the cap are partially filled.

    Attributes:
        adv_window: Rolling window size for ADV calculation (default: 20)
        participation_cap: Maximum participation rate as fraction of ADV (default: 0.05 = 5%)
            Example: 0.05 means max 5% of ADV can be filled per bar
        min_fill_qty: Minimum fill quantity (default: 0.0, no minimum)
            If computed fill_qty < min_fill_qty, fill is rejected (fill_qty=0)
        fallback_fill_ratio: Fallback fill ratio when ADV is missing (default: 1.0 = full fill)
            If ADV is missing/NaN, use this ratio of order qty as fill_qty
    """

    adv_window: int = 20
    participation_cap: float = 0.05
    min_fill_qty: float = 0.0
    fallback_fill_ratio: float = 1.0

    def __post_init__(self) -> None:
        """Validate partial fill model parameters."""
        if self.adv_window < 1:
            raise ValueError(f"adv_window must be >= 1, got {self.adv_window}")
        if self.participation_cap <= 0.0 or self.participation_cap > 1.0:
            raise ValueError(
                f"participation_cap must be in (0, 1], got {self.participation_cap}"
            )
        if self.min_fill_qty < 0.0:
            raise ValueError(f"min_fill_qty must be >= 0.0, got {self.min_fill_qty}")
        if self.fallback_fill_ratio < 0.0 or self.fallback_fill_ratio > 1.0:
            raise ValueError(
                f"fallback_fill_ratio must be in [0, 1], got {self.fallback_fill_ratio}"
            )


def ensure_fill_schema(
    trades: pd.DataFrame,
    *,
    default_full_fill: bool = True,
) -> pd.DataFrame:
    """Ensure trades DataFrame conforms to fill schema contract.

    Adds missing fill columns (fill_qty, fill_price, status, remaining_qty) if not present.
    For backward compatibility, assumes full fills by default.

    Args:
        trades: DataFrame with columns: timestamp, symbol, side, qty, price
            (may have additional columns like commission_cash, etc.)
        default_full_fill: If True, assume full fills for missing fill_qty/fill_price/status/remaining_qty.
            If False, raise ValueError if fill columns are missing.

    Returns:
        DataFrame with all required fill columns:
        - timestamp, symbol, side, qty, price (required)
        - fill_qty, fill_price, status, remaining_qty (added if missing)
        - commission_cash, spread_cash, slippage_cash, total_cost_cash (if present, preserved)

    Raises:
        ValueError: If required columns (timestamp, symbol, side, qty, price) are missing
        ValueError: If default_full_fill=False and fill columns are missing
    """
    # Validate required columns
    required_cols = ["timestamp", "symbol", "side", "qty", "price"]
    missing_cols = [col for col in required_cols if col not in trades.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in trades DataFrame: {missing_cols}")

    # Make a copy to avoid modifying original
    fills = trades.copy()

    # Add fill columns if missing (backward compatibility: assume full fills)
    if "fill_qty" not in fills.columns:
        if not default_full_fill:
            raise ValueError("fill_qty column missing and default_full_fill=False")
        fills["fill_qty"] = fills["qty"].copy()
    
    if "fill_price" not in fills.columns:
        if not default_full_fill:
            raise ValueError("fill_price column missing and default_full_fill=False")
        fills["fill_price"] = fills["price"].copy()
    
    if "status" not in fills.columns:
        if not default_full_fill:
            raise ValueError("status column missing and default_full_fill=False")
        # Determine status based on fill_qty vs qty
        fills["status"] = fills.apply(
            lambda row: (
                "filled" if row["fill_qty"] == row["qty"]
                else "partial" if row["fill_qty"] > 0
                else "rejected"
            ),
            axis=1,
        )
    
    if "remaining_qty" not in fills.columns:
        if not default_full_fill:
            raise ValueError("remaining_qty column missing and default_full_fill=False")
        fills["remaining_qty"] = fills["qty"] - fills["fill_qty"]

    # Validate fill constraints
    _validate_fill_constraints(fills)

    # Ensure deterministic sorting (by timestamp, symbol)
    if not fills.empty:
        fills = fills.sort_values(["timestamp", "symbol"], ignore_index=True)

    return fills


def _validate_fill_constraints(fills: pd.DataFrame) -> None:
    """Validate fill constraints (internal helper).

    Args:
        fills: DataFrame with fill columns

    Raises:
        ValueError: If constraints are violated
    """
    # Check fill_qty <= qty
    if not (fills["fill_qty"] <= fills["qty"]).all():
        raise ValueError("fill_qty must be <= qty for all rows")
    
    # Check fill_qty >= 0
    if not (fills["fill_qty"] >= 0).all():
        raise ValueError("fill_qty must be >= 0 for all rows")
    
    # Check remaining_qty = qty - fill_qty
    expected_remaining = fills["qty"] - fills["fill_qty"]
    if not np.allclose(fills["remaining_qty"], expected_remaining, rtol=1e-9, atol=1e-9):
        raise ValueError("remaining_qty must equal qty - fill_qty")
    
    # Check status consistency
    status_filled = fills["status"] == "filled"
    status_partial = fills["status"] == "partial"
    status_rejected = fills["status"] == "rejected"
    
    # filled: fill_qty == qty
    if not (fills.loc[status_filled, "fill_qty"] == fills.loc[status_filled, "qty"]).all():
        raise ValueError("status='filled' requires fill_qty == qty")
    
    # partial: 0 < fill_qty < qty
    if status_partial.any():
        partial_fills = fills.loc[status_partial]
        if not ((partial_fills["fill_qty"] > 0) & (partial_fills["fill_qty"] < partial_fills["qty"])).all():
            raise ValueError("status='partial' requires 0 < fill_qty < qty")
    
    # rejected: fill_qty == 0
    if not (fills.loc[status_rejected, "fill_qty"] == 0).all():
        raise ValueError("status='rejected' requires fill_qty == 0")
    
    # Check fill_price > 0 if fill_qty > 0
    if not (fills.loc[fills["fill_qty"] > 0, "fill_price"] > 0).all():
        raise ValueError("fill_price must be > 0 if fill_qty > 0")


def create_full_fill_from_order(order: dict | pd.Series) -> dict:
    """Create a full fill from an order (helper for backward compatibility).

    Args:
        order: Order dict or Series with columns: timestamp, symbol, side, qty, price

    Returns:
        Dict with fill columns: timestamp, symbol, side, qty, price, fill_qty, fill_price, status, remaining_qty
    """
    if isinstance(order, pd.Series):
        order = order.to_dict()
    
    fill = {
        "timestamp": order["timestamp"],
        "symbol": order["symbol"],
        "side": order["side"],
        "qty": order["qty"],
        "price": order["price"],
        "fill_qty": order["qty"],  # Full fill
        "fill_price": order["price"],  # No slippage (can be adjusted later)
        "status": "filled",
        "remaining_qty": 0.0,
    }
    return fill


def create_partial_fill_from_order(
    order: dict | pd.Series,
    fill_qty: float,
    fill_price: float | None = None,
) -> dict:
    """Create a partial fill from an order.

    Args:
        order: Order dict or Series with columns: timestamp, symbol, side, qty, price
        fill_qty: Filled quantity (must be 0 < fill_qty < qty)
        fill_price: Fill price (default: order price)

    Returns:
        Dict with fill columns: timestamp, symbol, side, qty, price, fill_qty, fill_price, status, remaining_qty

    Raises:
        ValueError: If fill_qty is invalid (<= 0 or >= qty)
    """
    if isinstance(order, pd.Series):
        order = order.to_dict()
    
    qty = order["qty"]
    if fill_qty <= 0 or fill_qty >= qty:
        raise ValueError(f"fill_qty must be 0 < fill_qty < qty (got fill_qty={fill_qty}, qty={qty})")
    
    if fill_price is None:
        fill_price = order["price"]
    
    fill = {
        "timestamp": order["timestamp"],
        "symbol": order["symbol"],
        "side": order["side"],
        "qty": qty,
        "price": order["price"],
        "fill_qty": fill_qty,
        "fill_price": fill_price,
        "status": "partial",
        "remaining_qty": qty - fill_qty,
    }
    return fill


def create_rejected_fill_from_order(order: dict | pd.Series) -> dict:
    """Create a rejected fill from an order.

    Args:
        order: Order dict or Series with columns: timestamp, symbol, side, qty, price

    Returns:
        Dict with fill columns: timestamp, symbol, side, qty, price, fill_qty, fill_price, status, remaining_qty
    """
    if isinstance(order, pd.Series):
        order = order.to_dict()
    
    fill = {
        "timestamp": order["timestamp"],
        "symbol": order["symbol"],
        "side": order["side"],
        "qty": order["qty"],
        "price": order["price"],
        "fill_qty": 0.0,  # Rejected
        "fill_price": order["price"],  # Use order price for consistency
        "status": "rejected",
        "remaining_qty": order["qty"],  # All remaining
    }
    return fill


def apply_session_gate(
    trades: pd.DataFrame,
    *,
    freq: str,
    strict: bool = True,
) -> pd.DataFrame:
    """Apply session gate to trades: reject orders outside trading sessions.

    For freq="1d": only accept fills at session close.
    For freq="5min": only accept fills within trading session (9:30 ET - 16:00 ET).

    Args:
        trades: DataFrame with columns: timestamp, symbol, side, qty, price
            (may have fill_qty, fill_price, status, remaining_qty if already processed)
        freq: Trading frequency ("1d" or "5min")
        strict: If True, reject orders outside sessions. If False, warn but allow (for fallback).

    Returns:
        DataFrame with session gate applied:
        - Orders outside sessions: status="rejected", fill_qty=0, remaining_qty=qty, costs=0
        - Orders within sessions: unchanged (or set to full fill if not already processed)

    Note:
        If exchange_calendars is not available and strict=False, warns and allows all orders
        (permissive fallback). If strict=True, raises ImportError.
    """
    try:
        from src.assembled_core.data.calendar import (
            get_nyse_calendar,
            is_trading_day,
            session_close_utc,
        )
    except ImportError as e:
        if strict:
            raise ImportError(
                "exchange_calendars is required for session gate. "
                "Install with: pip install exchange-calendars"
            ) from e
        logger.warning(
            "exchange_calendars not available, session gate disabled (permissive fallback)"
        )
        # Permissive fallback: allow all orders
        return trades.copy()

    # Ensure fill schema (add fill columns if missing)
    fills = ensure_fill_schema(trades, default_full_fill=True)
    
    # Make a copy to avoid modifying original
    fills = fills.copy()
    
    # Get calendar for intraday checks
    cal = get_nyse_calendar()
    
    # Apply session gate per row
    for idx, row in fills.iterrows():
        timestamp = row["timestamp"]
        
        # Check if trading day
        if not is_trading_day(timestamp):
            # Not a trading day (weekend/holiday): reject
            fills.loc[idx, "status"] = "rejected"
            fills.loc[idx, "fill_qty"] = 0.0
            fills.loc[idx, "remaining_qty"] = row["qty"]
            fills.loc[idx, "fill_price"] = row["price"]  # Use order price for consistency
            # Costs should be 0 for rejected (will be set later if not already)
            if "commission_cash" in fills.columns:
                fills.loc[idx, "commission_cash"] = 0.0
            if "spread_cash" in fills.columns:
                fills.loc[idx, "spread_cash"] = 0.0
            if "slippage_cash" in fills.columns:
                fills.loc[idx, "slippage_cash"] = 0.0
            if "total_cost_cash" in fills.columns:
                fills.loc[idx, "total_cost_cash"] = 0.0
            continue
        
        # For freq="1d": only accept at session close
        if freq == "1d":
            try:
                session_close = session_close_utc(timestamp.date())
                # Check if timestamp is at session close (within 1 minute tolerance)
                time_diff = abs((timestamp - session_close).total_seconds())
                if time_diff > 60:  # More than 1 minute away from session close
                    # Not at session close: reject
                    fills.loc[idx, "status"] = "rejected"
                    fills.loc[idx, "fill_qty"] = 0.0
                    fills.loc[idx, "remaining_qty"] = row["qty"]
                    fills.loc[idx, "fill_price"] = row["price"]
                    if "commission_cash" in fills.columns:
                        fills.loc[idx, "commission_cash"] = 0.0
                    if "spread_cash" in fills.columns:
                        fills.loc[idx, "spread_cash"] = 0.0
                    if "slippage_cash" in fills.columns:
                        fills.loc[idx, "slippage_cash"] = 0.0
                    if "total_cost_cash" in fills.columns:
                        fills.loc[idx, "total_cost_cash"] = 0.0
            except ValueError:
                # Not a trading day (already handled above, but catch for safety)
                fills.loc[idx, "status"] = "rejected"
                fills.loc[idx, "fill_qty"] = 0.0
                fills.loc[idx, "remaining_qty"] = row["qty"]
                fills.loc[idx, "fill_price"] = row["price"]
                if "commission_cash" in fills.columns:
                    fills.loc[idx, "commission_cash"] = 0.0
                if "spread_cash" in fills.columns:
                    fills.loc[idx, "spread_cash"] = 0.0
                if "slippage_cash" in fills.columns:
                    fills.loc[idx, "slippage_cash"] = 0.0
                if "total_cost_cash" in fills.columns:
                    fills.loc[idx, "total_cost_cash"] = 0.0
        
        # For freq="5min": only accept within trading session
        elif freq == "5min":
            try:
                # Get session open and close for the date
                session_date = timestamp.date()
                session_ts = pd.Timestamp(session_date)
                
                # Get session open and close times (in exchange timezone)
                session_open_local = cal.session_open(session_ts)
                session_close_local = cal.session_close(session_ts)
                
                # Convert to UTC
                if session_open_local.tz is None:
                    session_open_local = session_open_local.tz_localize("America/New_York")
                if session_close_local.tz is None:
                    session_close_local = session_close_local.tz_localize("America/New_York")
                
                session_open_utc = session_open_local.tz_convert("UTC")
                session_close_utc = session_close_local.tz_convert("UTC")
                
                # Check if timestamp is within session (inclusive)
                if timestamp < session_open_utc or timestamp > session_close_utc:
                    # Outside session: reject
                    fills.loc[idx, "status"] = "rejected"
                    fills.loc[idx, "fill_qty"] = 0.0
                    fills.loc[idx, "remaining_qty"] = row["qty"]
                    fills.loc[idx, "fill_price"] = row["price"]
                    if "commission_cash" in fills.columns:
                        fills.loc[idx, "commission_cash"] = 0.0
                    if "spread_cash" in fills.columns:
                        fills.loc[idx, "spread_cash"] = 0.0
                    if "slippage_cash" in fills.columns:
                        fills.loc[idx, "slippage_cash"] = 0.0
                    if "total_cost_cash" in fills.columns:
                        fills.loc[idx, "total_cost_cash"] = 0.0
            except Exception as e:
                # Fallback: if calendar check fails, reject (deterministic)
                logger.warning(f"Session check failed for {timestamp}: {e}, rejecting order")
                fills.loc[idx, "status"] = "rejected"
                fills.loc[idx, "fill_qty"] = 0.0
                fills.loc[idx, "remaining_qty"] = row["qty"]
                fills.loc[idx, "fill_price"] = row["price"]
                if "commission_cash" in fills.columns:
                    fills.loc[idx, "commission_cash"] = 0.0
                if "spread_cash" in fills.columns:
                    fills.loc[idx, "spread_cash"] = 0.0
                if "slippage_cash" in fills.columns:
                    fills.loc[idx, "slippage_cash"] = 0.0
                if "total_cost_cash" in fills.columns:
                    fills.loc[idx, "total_cost_cash"] = 0.0
    
    # Ensure deterministic sorting
    if not fills.empty:
        fills = fills.sort_values(["timestamp", "symbol"], ignore_index=True)
    
    return fills


def compute_max_fill_qty(
    adv_usd: float,
    price: float,
    model: PartialFillModel,
) -> float:
    """Compute maximum fill quantity based on ADV and participation cap.

    Args:
        adv_usd: Average Daily Volume in USD (from compute_adv_proxy)
        price: Order price
        model: PartialFillModel instance

    Returns:
        Maximum fill quantity (always >= 0)
        If adv_usd is NaN or <= 0, returns NaN (caller should use fallback)

    Formula:
        max_notional = adv_usd * participation_cap
        max_qty = max_notional / price
    """
    if np.isnan(adv_usd) or adv_usd <= 0.0:
        return np.nan

    if price <= 0.0:
        return 0.0

    max_notional = adv_usd * model.participation_cap
    max_qty = max_notional / price

    # Apply minimum fill quantity constraint
    if max_qty < model.min_fill_qty:
        return 0.0  # Reject if below minimum

    return max_qty


def apply_partial_fills(
    trades: pd.DataFrame,
    *,
    prices: pd.DataFrame,
    partial_fill_model: PartialFillModel | None = None,
) -> pd.DataFrame:
    """Apply partial fill model to trades based on liquidity (ADV proxy).

    Computes fill_qty based on ADV and participation cap. Orders exceeding the cap
    are partially filled.

    Args:
        trades: DataFrame with columns: timestamp, symbol, side, qty, price
            (may have fill_qty, fill_price, status, remaining_qty if already processed)
        prices: Prices DataFrame with columns: timestamp, symbol, close, volume (optional)
            Required for ADV calculation
        partial_fill_model: Optional PartialFillModel instance.
            If None, assumes full fills (no partial fill constraints)

    Returns:
        DataFrame with partial fills applied:
        - fill_qty: Computed fill quantity (min(qty, max_fill_qty))
        - fill_price: Order price (no price impact here)
        - status: "filled" if fill_qty == qty, else "partial"
        - remaining_qty: qty - fill_qty

    Edge Cases:
        - Missing volume/ADV: Uses fallback_fill_ratio (default: 1.0 = full fill)
        - qty == 0: Row is dropped (no trade)
        - fill_qty < min_fill_qty: Rejected (fill_qty=0, status="rejected")
    """
    if partial_fill_model is None:
        # No partial fill model: assume full fills
        return ensure_fill_schema(trades, default_full_fill=True)

    # Ensure fill schema (add fill columns if missing)
    fills = ensure_fill_schema(trades, default_full_fill=True)
    
    # Drop rows with qty == 0 (no trade)
    fills = fills[fills["qty"] != 0.0].copy()
    
    if fills.empty:
        return fills

    # Compute ADV proxy
    try:
        from src.assembled_core.execution.transaction_costs import compute_adv_proxy

        adv_df = compute_adv_proxy(prices, adv_window=partial_fill_model.adv_window)
    except Exception as e:
        logger.warning(f"Failed to compute ADV proxy: {e}, using fallback")
        adv_df = pd.DataFrame(columns=["timestamp", "symbol", "adv_usd"])

    # Merge ADV with fills
    fills_with_adv = fills.merge(
        adv_df,
        on=["timestamp", "symbol"],
        how="left",
        suffixes=("", "_adv"),
    )

    # Compute max fill qty for each row
    max_fill_qtys = []
    for idx, row in fills_with_adv.iterrows():
        adv_usd = row.get("adv_usd", np.nan)
        price = row["price"]
        qty = row["qty"]

        if np.isnan(adv_usd) or adv_usd <= 0.0:
            # Missing ADV: use fallback
            max_fill_qty = qty * partial_fill_model.fallback_fill_ratio
        else:
            # Compute max fill qty from ADV
            max_fill_qty = compute_max_fill_qty(adv_usd, price, partial_fill_model)
            if np.isnan(max_fill_qty):
                # Fallback if computation failed
                max_fill_qty = qty * partial_fill_model.fallback_fill_ratio

        max_fill_qtys.append(max_fill_qty)

    fills_with_adv["max_fill_qty"] = max_fill_qtys

    # Compute actual fill_qty (min of qty and max_fill_qty)
    fills_with_adv["fill_qty"] = np.minimum(
        fills_with_adv["qty"].abs(), fills_with_adv["max_fill_qty"]
    )

    # Apply minimum fill quantity constraint
    min_fill_mask = fills_with_adv["fill_qty"] < partial_fill_model.min_fill_qty
    fills_with_adv.loc[min_fill_mask, "fill_qty"] = 0.0

    # Update status
    fills_with_adv["status"] = fills_with_adv.apply(
        lambda row: (
            "rejected" if row["fill_qty"] == 0.0
            else "filled" if row["fill_qty"] == abs(row["qty"])
            else "partial"
        ),
        axis=1,
    )

    # Update remaining_qty
    fills_with_adv["remaining_qty"] = fills_with_adv["qty"].abs() - fills_with_adv["fill_qty"]

    # Keep fill_price = price (no price impact here)
    fills_with_adv["fill_price"] = fills_with_adv["price"]

    # Drop helper columns
    result = fills_with_adv.drop(columns=["adv_usd", "max_fill_qty"], errors="ignore")

    # Ensure deterministic sorting
    if not result.empty:
        result = result.sort_values(["timestamp", "symbol"], ignore_index=True)

    return result


def apply_limit_order_fills(
    trades: pd.DataFrame,
    *,
    prices: pd.DataFrame,
    partial_fill_model: PartialFillModel | None = None,
) -> pd.DataFrame:
    """Apply limit order fill logic to trades.

    For limit orders, checks if limit price is reachable based on OHLC data.
    If limit not reachable, order is rejected. If reachable, applies partial fill model.

    Args:
        trades: DataFrame with columns: timestamp, symbol, side, qty, price
            Optional columns: order_type, limit_price
            (may have fill_qty, fill_price, status, remaining_qty if already processed)
        prices: Prices DataFrame with columns: timestamp, symbol, close
            Optional columns: open, high, low (for limit order checks)
        partial_fill_model: Optional PartialFillModel instance.
            If None, assumes full fills (no partial fill constraints)

    Returns:
        DataFrame with limit order fills applied:
        - fill_qty: Computed fill quantity (0 if limit not reached, else min(qty, max_fill_qty))
        - fill_price: limit_price if limit order filled, else order price
        - status: "filled", "partial", or "rejected"
        - remaining_qty: qty - fill_qty

    Edge Cases:
        - Missing OHLC: Uses close as both high and low (deterministic fallback)
        - Missing limit_price: Treated as market order (full fill if eligible)
        - Limit not reached: Rejected (fill_qty=0, status="rejected")
    """
    # Ensure fill schema (add fill columns if missing)
    fills = ensure_fill_schema(trades, default_full_fill=True)
    
    # Drop rows with qty == 0 (no trade)
    fills = fills[fills["qty"] != 0.0].copy()
    
    if fills.empty:
        return fills

    # Determine order types (default to "market" if missing)
    if "order_type" not in fills.columns:
        fills["order_type"] = "market"
    fills["order_type"] = fills["order_type"].fillna("market").astype(str).str.lower()

    # Get limit prices (if present)
    has_limit_price = "limit_price" in fills.columns
    if not has_limit_price:
        fills["limit_price"] = np.nan

    # Merge with prices to get OHLC data
    fills_with_prices = fills.merge(
        prices[["timestamp", "symbol", "close"] + (["high", "low"] if "high" in prices.columns and "low" in prices.columns else [])],
        on=["timestamp", "symbol"],
        how="left",
        suffixes=("", "_price"),
    )

    # Fallback: if high/low missing, use close as both
    if "high" not in fills_with_prices.columns:
        fills_with_prices["high"] = fills_with_prices["close"]
    if "low" not in fills_with_prices.columns:
        fills_with_prices["low"] = fills_with_prices["close"]

    # Check limit eligibility
    limit_eligible = pd.Series(True, index=fills_with_prices.index)
    
    # For limit orders, check if limit is reachable
    limit_mask = fills_with_prices["order_type"] == "limit"
    if limit_mask.any():
        # Check if limit_price is present and valid
        limit_price_valid = ~fills_with_prices["limit_price"].isna() & (fills_with_prices["limit_price"] > 0)
        limit_mask = limit_mask & limit_price_valid
        
        if limit_mask.any():
            buy_limit_mask = limit_mask & (fills_with_prices["side"] == "BUY")
            sell_limit_mask = limit_mask & (fills_with_prices["side"] == "SELL")
            
            # BUY limit: fill only if bar_low <= limit_price
            if buy_limit_mask.any():
                buy_limit_eligible = (
                    fills_with_prices.loc[buy_limit_mask, "low"] <= 
                    fills_with_prices.loc[buy_limit_mask, "limit_price"]
                )
                limit_eligible.loc[buy_limit_mask] = buy_limit_eligible.values
            
            # SELL limit: fill only if bar_high >= limit_price
            if sell_limit_mask.any():
                sell_limit_eligible = (
                    fills_with_prices.loc[sell_limit_mask, "high"] >= 
                    fills_with_prices.loc[sell_limit_mask, "limit_price"]
                )
                limit_eligible.loc[sell_limit_mask] = sell_limit_eligible.values
        
        # Reject limit orders with missing/invalid limit_price
        limit_invalid_mask = (fills_with_prices["order_type"] == "limit") & ~limit_price_valid
        if limit_invalid_mask.any():
            limit_eligible.loc[limit_invalid_mask] = False

    # Reject orders where limit not reached
    fills_with_prices.loc[~limit_eligible, "fill_qty"] = 0.0
    fills_with_prices.loc[~limit_eligible, "status"] = "rejected"
    fills_with_prices.loc[~limit_eligible, "remaining_qty"] = fills_with_prices.loc[~limit_eligible, "qty"]
    fills_with_prices.loc[~limit_eligible, "fill_price"] = fills_with_prices.loc[~limit_eligible, "price"]

    # For eligible orders, apply partial fill model if provided
    eligible_mask = limit_eligible
    if eligible_mask.any() and partial_fill_model is not None:
        # Apply partial fills to eligible orders
        eligible_fills = fills_with_prices[eligible_mask].copy()
        eligible_fills = apply_partial_fills(
            eligible_fills,
            prices=prices,
            partial_fill_model=partial_fill_model,
        )
        
        # Update fills_with_prices with partial fill results
        fills_with_prices.loc[eligible_mask, "fill_qty"] = eligible_fills["fill_qty"].values
        fills_with_prices.loc[eligible_mask, "status"] = eligible_fills["status"].values
        fills_with_prices.loc[eligible_mask, "remaining_qty"] = eligible_fills["remaining_qty"].values
    elif eligible_mask.any():
        # No partial fill model: full fills for eligible orders
        fills_with_prices.loc[eligible_mask, "fill_qty"] = fills_with_prices.loc[eligible_mask, "qty"].abs()
        fills_with_prices.loc[eligible_mask, "status"] = "filled"
        fills_with_prices.loc[eligible_mask, "remaining_qty"] = 0.0

    # Set fill_price for limit orders (conservative: use limit_price)
    limit_filled_mask = limit_mask & (fills_with_prices["fill_qty"] > 0)
    if limit_filled_mask.any():
        fills_with_prices.loc[limit_filled_mask, "fill_price"] = fills_with_prices.loc[limit_filled_mask, "limit_price"]

    # Drop helper columns
    result = fills_with_prices.drop(
        columns=["high", "low", "order_type", "limit_price"],
        errors="ignore"
    )

    # Ensure deterministic sorting
    if not result.empty:
        result = result.sort_values(["timestamp", "symbol"], ignore_index=True)

    return result
