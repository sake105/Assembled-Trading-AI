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


def validate_safe_orders_df(df: pd.DataFrame) -> dict[str, bool | list[str]]:
    """Validate SAFE orders DataFrame before writing.

    This function performs sanity checks on orders to ensure they are valid
    before writing to CSV. Invalid orders are rejected to prevent writing
    obviously broken data.

    Validation checks:
    - No rows with Quantity <= 0
    - Side must be in {BUY, SELL}
    - No duplicate rows (same Ticker, Side, Comment) - these are aggregated

    Args:
        df: DataFrame with SAFE-Bridge format columns: Ticker, Side, Quantity, PriceType, Comment

    Returns:
        Dictionary with:
        - valid: bool - True if all checks pass
        - issues: list[str] - List of descriptive issue messages (empty if valid=True)
        - df_cleaned: pd.DataFrame - Cleaned DataFrame (duplicates aggregated, invalid rows removed)

    Note:
        If duplicates are found, they are aggregated (quantities summed).
        If invalid rows are found (Quantity <= 0, invalid Side), they are removed.
        The function returns the cleaned DataFrame, but if all rows are invalid,
        valid=False and df_cleaned will be empty.
    """
    issues = []
    df_cleaned = df.copy()

    if df_cleaned.empty:
        return {"valid": True, "issues": [], "df_cleaned": df_cleaned}

    # Check 1: Quantity <= 0
    invalid_qty = df_cleaned[df_cleaned["Quantity"] <= 0]
    if not invalid_qty.empty:
        count = len(invalid_qty)
        symbols = invalid_qty["Ticker"].unique().tolist()
        issues.append(
            f"{count} order(s) with Quantity <= 0: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}"
        )
        df_cleaned = df_cleaned[df_cleaned["Quantity"] > 0].copy()

    # Check 2: Side must be BUY or SELL
    valid_sides = {"BUY", "SELL"}
    invalid_sides = df_cleaned[~df_cleaned["Side"].isin(valid_sides)]
    if not invalid_sides.empty:
        count = len(invalid_sides)
        invalid_values = invalid_sides["Side"].unique().tolist()
        issues.append(
            f"{count} order(s) with invalid Side: {', '.join(invalid_values)}"
        )
        df_cleaned = df_cleaned[df_cleaned["Side"].isin(valid_sides)].copy()

    # Check 3: Duplicate rows (same Ticker, Side, Comment) - aggregate
    if not df_cleaned.empty:
        duplicates = df_cleaned.duplicated(
            subset=["Ticker", "Side", "Comment"], keep=False
        )
        if duplicates.any():
            dup_count = duplicates.sum()
            # Aggregate duplicates: sum quantities
            df_cleaned = df_cleaned.groupby(
                ["Ticker", "Side", "Comment", "PriceType"], as_index=False
            ).agg({"Quantity": "sum"})
            issues.append(
                f"{dup_count} duplicate order(s) aggregated (quantities summed)"
            )

    # Final check: If all rows were invalid, mark as invalid
    if df_cleaned.empty and not df.empty:
        issues.append("All orders were invalid and removed - no valid orders remain")
        return {"valid": False, "issues": issues, "df_cleaned": df_cleaned}

    # If we have issues but some valid rows remain, it's a warning but still valid
    # (we'll log warnings but allow the file to be written)
    valid = len(issues) == 0 or not df_cleaned.empty

    return {"valid": valid, "issues": issues, "df_cleaned": df_cleaned}


def write_safe_orders_csv(
    orders: pd.DataFrame,
    output_path: Path | str | None = None,
    date: datetime | None = None,
    price_type: str = "MARKET",
    comment: str = "EOD Strategy",
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
        safe_df = pd.DataFrame(
            columns=["Ticker", "Side", "Quantity", "PriceType", "Comment"]
        )
    else:
        # Ensure required columns
        required = ["symbol", "side", "qty"]
        missing = [c for c in required if c not in orders.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. Available: {list(orders.columns)}"
            )

        # Map to SAFE-Bridge format
        safe_df = pd.DataFrame(
            {
                "Ticker": orders["symbol"].astype(str).str.upper(),
                "Side": orders["side"].astype(str).str.upper(),
                "Quantity": orders["qty"].astype(float),
                "PriceType": price_type,
                "Comment": comment,
            }
        )

        # Validate orders
        validation_result = validate_safe_orders_df(safe_df)

        if not validation_result["valid"]:
            issues_str = "; ".join(validation_result["issues"])
            raise ValueError(
                f"SAFE orders validation failed - no file will be written. Issues: {issues_str}"
            )

        # Use cleaned DataFrame (duplicates aggregated, invalid rows removed)
        safe_df = validation_result["df_cleaned"]

        # Log warnings if there were issues (but file is still written)
        if validation_result["issues"]:
            from src.assembled_core.logging_utils import get_logger

            logger = get_logger()
            for issue in validation_result["issues"]:
                logger.warning(f"SAFE orders validation warning: {issue}")

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
    comment: str = "EOD Strategy",
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
    from src.assembled_core.execution.order_generation import (
        generate_orders_from_targets,
    )

    # Generate orders
    orders = generate_orders_from_targets(
        target_positions,
        current_positions=current_positions,
        timestamp=date,
        prices=prices,
    )

    # Write SAFE-Bridge CSV
    return write_safe_orders_csv(
        orders,
        output_path=output_path,
        date=date,
        price_type=price_type,
        comment=comment,
    )
