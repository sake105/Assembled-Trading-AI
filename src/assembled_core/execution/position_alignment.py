"""Position alignment utilities for order generation.

This module provides utilities to align current and target position DataFrames
to ensure deterministic symbol ordering and handle missing symbols consistently.
This enables the fast-path order generation to trigger more often.
"""

from __future__ import annotations

import pandas as pd


def align_current_and_target(
    current_positions: pd.DataFrame,
    target_positions: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    qty_col: str = "qty",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align current and target positions to same symbol set and order.
    
    This function ensures that both DataFrames have:
    - The same set of symbols (union of both)
    - The same symbol order (sorted alphabetically)
    - Missing symbols filled with 0 quantity
    
    This alignment enables the fast-path order generation to trigger
    (which requires exact symbol match and order).
    
    Args:
        current_positions: DataFrame with columns: symbol_col, qty_col (and optionally others)
            Current portfolio positions (may be empty)
        target_positions: DataFrame with columns: symbol_col, qty_col (and optionally others)
            Target positions (may be empty)
        symbol_col: Name of the symbol column (default: "symbol")
        qty_col: Name of the quantity column (default: "qty")
    
    Returns:
        Tuple of (current_aligned, target_aligned):
        - current_aligned: Current positions with aligned symbols (sorted, missing = 0)
        - target_aligned: Target positions with aligned symbols (sorted, missing = 0)
        
        Both DataFrames have:
        - Same symbol set (union of symbols from both inputs)
        - Same symbol order (sorted alphabetically)
        - Missing symbols have qty = 0
        - Same columns as input (preserved from first non-empty DataFrame, or minimal if both empty)
    
    Examples:
        >>> current = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "qty": [10, 20]})
        >>> target = pd.DataFrame({"symbol": ["MSFT", "GOOGL"], "qty": [25, 5]})
        >>> curr_aligned, tgt_aligned = align_current_and_target(current, target)
        >>> # Both have symbols: ["AAPL", "GOOGL", "MSFT"] (sorted)
        >>> # curr_aligned: AAPL=10, GOOGL=0, MSFT=20
        >>> # tgt_aligned: AAPL=0, GOOGL=5, MSFT=25
    """
    # Handle empty inputs
    if current_positions.empty and target_positions.empty:
        # Return empty DataFrames with correct schema
        return (
            pd.DataFrame(columns=[symbol_col, qty_col]),
            pd.DataFrame(columns=[symbol_col, qty_col]),
        )
    
    # Get symbol sets from both DataFrames
    current_symbols = set(current_positions[symbol_col].unique()) if not current_positions.empty else set()
    target_symbols = set(target_positions[symbol_col].unique()) if not target_positions.empty else set()
    
    # Union of all symbols (sorted for deterministic order)
    all_symbols = sorted(current_symbols | target_symbols)
    
    # Determine output columns (use minimal set: symbol and qty)
    # Additional columns are not preserved for alignment (focus on qty alignment)
    output_columns = [symbol_col, qty_col]
    
    # Create aligned DataFrames
    current_aligned = _align_positions_to_symbols(
        positions=current_positions,
        symbols=all_symbols,
        symbol_col=symbol_col,
        qty_col=qty_col,
        output_columns=output_columns,
    )
    
    target_aligned = _align_positions_to_symbols(
        positions=target_positions,
        symbols=all_symbols,
        symbol_col=symbol_col,
        qty_col=qty_col,
        output_columns=output_columns,
    )
    
    return current_aligned, target_aligned


def _align_positions_to_symbols(
    positions: pd.DataFrame,
    symbols: list[str],
    *,
    symbol_col: str,
    qty_col: str,
    output_columns: list[str],
) -> pd.DataFrame:
    """Align a positions DataFrame to a specific symbol set.
    
    Args:
        positions: Input positions DataFrame (may be empty)
        symbols: List of symbols to include (sorted)
        symbol_col: Name of symbol column
        qty_col: Name of quantity column
        output_columns: List of column names for output DataFrame
    
    Returns:
        DataFrame with all symbols from `symbols` list, sorted, with missing symbols = 0
    """
    if positions.empty:
        # Create DataFrame with all symbols and qty=0
        aligned = pd.DataFrame({symbol_col: symbols, qty_col: 0.0})
        
        # Add other columns with default values (NaN or 0, depending on type)
        for col in output_columns:
            if col not in aligned.columns:
                aligned[col] = 0.0 if col == qty_col else pd.NA
        
        # Ensure correct column order
        aligned = aligned[output_columns]
        return aligned
    
    # Create a DataFrame with all symbols
    aligned = pd.DataFrame({symbol_col: symbols})
    
    # Merge with existing positions (left join, so all symbols are preserved)
    aligned = aligned.merge(
        positions,
        on=symbol_col,
        how="left",
        suffixes=("", "_drop"),
    )
    
    # Drop duplicate columns from merge (keep original column names)
    cols_to_drop = [c for c in aligned.columns if c.endswith("_drop")]
    aligned = aligned.drop(columns=cols_to_drop)
    
    # Fill missing quantities with 0
    if qty_col in aligned.columns:
        aligned[qty_col] = aligned[qty_col].fillna(0.0).astype(float)
    else:
        # If qty_col doesn't exist after merge, create it with 0
        aligned[qty_col] = 0.0
    
    # Ensure deterministic sorting by symbol
    aligned = aligned.sort_values(symbol_col).reset_index(drop=True)
    
    # Ensure correct column order
    aligned = aligned[output_columns]
    
    return aligned

