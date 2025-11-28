"""SAFE-Bridge order file generation.

This module creates SAFE-compatible order files for human-in-the-loop review.
SAFE-Bridge ensures that no orders are executed automatically - all orders
must be reviewed and approved by a human before execution.

Zukünftige Integration:
- Integrates with execution.order_generation for order creation
- Provides standardized SAFE-Bridge CSV format
- Supports different order types (MARKET, LIMIT) and comments
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.assembled_core.config import OUTPUT_DIR


def write_safe_orders_csv(
    orders: pd.DataFrame,
    output_path: Path | str | None = None,
    date: datetime | None = None,
    price_type: str = "MARKET",
    comment: str = "EOD Strategy"
) -> Path:
    """Write orders to SAFE-Bridge compatible CSV file.
    
    SAFE-Bridge format:
    - File name: orders_YYYYMMDD.csv (e.g., orders_20251128.csv)
    - Columns: Ticker, Side, Quantity, PriceType, Comment
    
    Args:
        orders: DataFrame with columns: timestamp, symbol, side, qty, price
        output_path: Optional explicit output path. If None, uses output/orders_YYYYMMDD.csv
        date: Date for filename (default: current date or latest order timestamp)
        price_type: Order price type (default: "MARKET", can be "LIMIT")
        comment: Comment to add to each order (default: "EOD Strategy")
    
    Returns:
        Path to written CSV file
    
    Raises:
        ValueError: If required columns are missing
    
    Side effects:
        Creates output directory if it doesn't exist
        Writes CSV file with SAFE-Bridge format
    """
    if orders.empty:
        # Create empty file with correct schema
        safe_df = pd.DataFrame(columns=["Ticker", "Side", "Quantity", "PriceType", "Comment"])
    else:
        # Ensure required columns
        required = ["symbol", "side", "qty"]
        missing = [c for c in required if c not in orders.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Available: {list(orders.columns)}")
        
        # Map to SAFE-Bridge format
        safe_df = pd.DataFrame({
            "Ticker": orders["symbol"].astype(str).str.upper(),
            "Side": orders["side"].astype(str).str.upper(),
            "Quantity": orders["qty"].astype(float),
            "PriceType": price_type,
            "Comment": comment
        })
        
        # Validate sides
        valid_sides = {"BUY", "SELL"}
        invalid = safe_df[~safe_df["Side"].isin(valid_sides)]
        if not invalid.empty:
            raise ValueError(f"Invalid side values: {invalid['Side'].unique()}. Must be BUY or SELL")
    
    # Determine output path
    if output_path is None:
        # Use date for filename
        if date is None:
            if not orders.empty and "timestamp" in orders.columns:
                date = pd.to_datetime(orders["timestamp"]).max().to_pydatetime()
            else:
                date = datetime.utcnow()
        
        date_str = date.strftime("%Y%m%d")
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"orders_{date_str}.csv"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write CSV
    safe_df.to_csv(output_path, index=False)
    
    return output_path


def write_safe_orders_csv_from_targets(
    target_positions: pd.DataFrame,
    current_positions: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    output_path: Path | str | None = None,
    date: datetime | None = None,
    price_type: str = "MARKET",
    comment: str = "EOD Strategy"
) -> Path:
    """Write SAFE-Bridge orders from target positions (convenience function).
    
    This function combines order generation and SAFE-Bridge file writing.
    
    Args:
        target_positions: DataFrame with columns: symbol, target_weight, target_qty
        current_positions: Optional current positions DataFrame
        prices: Optional prices DataFrame for price lookup
        output_path: Optional explicit output path
        date: Date for filename (default: current date)
        price_type: Order price type (default: "MARKET")
        comment: Comment to add to each order (default: "EOD Strategy")
    
    Returns:
        Path to written CSV file
    """
    from src.assembled_core.execution.order_generation import generate_orders_from_targets
    
    # Generate orders
    orders = generate_orders_from_targets(
        target_positions,
        current_positions=current_positions,
        timestamp=date,
        prices=prices
    )
    
    # Write SAFE-Bridge CSV
    return write_safe_orders_csv(
        orders,
        output_path=output_path,
        date=date,
        price_type=price_type,
        comment=comment
    )

