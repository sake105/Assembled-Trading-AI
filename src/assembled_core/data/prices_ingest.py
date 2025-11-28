"""Price data ingestion module.

This module provides functions to load EOD (End-of-Day) price data with full OHLCV information.
It extends the basic I/O functionality from pipeline.io to support complete price bars.

Zukünftige Integration:
- Nutzt intern pipeline.io.load_prices für Basis-I/O
- Erweitert um Multi-Source-Support (Yahoo, Alpha Vantage, lokale Dateien)
- Validiert Datenqualität (Gaps, Outliers, Schema)
- Normalisiert auf Standardformat: timestamp (UTC), symbol, open, high, low, close, volume
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from src.assembled_core.config import OUTPUT_DIR, get_base_dir
from src.assembled_core.pipeline.io import coerce_price_types, ensure_cols, load_prices


def load_eod_prices(
    symbols: list[str] | None = None,
    price_file: Path | str | None = None,
    data_dir: Path | str | None = None,
    freq: Literal["1d", "5min"] = "1d"
) -> pd.DataFrame:
    """Load EOD price data with full OHLCV information.
    
    This function loads price data with complete bar information (open, high, low, close, volume).
    It can load from aggregated files (output/aggregates/) or raw data files (data/raw/).
    
    Args:
        symbols: Optional list of symbols to filter. If None, loads all symbols.
        price_file: Optional explicit path to price file. If None, uses default path for freq.
        data_dir: Optional base data directory. If None, uses config.OUTPUT_DIR for aggregates.
        freq: Frequency string ("1d" or "5min"), default "1d"
    
    Returns:
        DataFrame with columns: timestamp (UTC), symbol, open, high, low, close, volume
        Sorted by symbol, then timestamp
    
    Raises:
        FileNotFoundError: If price file does not exist
        ValueError: If required columns are missing or data is invalid
    
    Examples:
        >>> # Load from default aggregated file
        >>> df = load_eod_prices(freq="1d")
        >>> # Load specific symbols
        >>> df = load_eod_prices(symbols=["AAPL", "MSFT"], freq="1d")
        >>> # Load from explicit file
        >>> df = load_eod_prices(price_file=Path("data/sample/eod_sample.parquet"))
    """
    # Determine source file
    if price_file:
        source_path = Path(price_file)
    else:
        # Use default path from pipeline.io
        from src.assembled_core.pipeline.io import get_default_price_path
        base = Path(data_dir) if data_dir else OUTPUT_DIR
        source_path = get_default_price_path(freq, output_dir=base)
    
    if not source_path.exists():
        raise FileNotFoundError(
            f"Price file not found: {source_path}. "
            f"Run data ingestion or resampling first."
        )
    
    # Load base data (timestamp, symbol, close)
    # This uses the existing pipeline.io logic
    df = pd.read_parquet(source_path)
    
    # Ensure minimum required columns
    df = ensure_cols(df, ["timestamp", "symbol", "close"])
    df = coerce_price_types(df)
    
    # Check if we have full OHLCV data
    has_ohlcv = all(col in df.columns for col in ["open", "high", "low", "volume"])
    
    if not has_ohlcv:
        # If only 'close' is available, create synthetic OHLCV from close
        # This is a fallback for compatibility with existing aggregated files
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]
        df["volume"] = 0.0  # Default volume if not available
    
    # Ensure OHLCV columns have correct types
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).astype("float64")
    
    # Filter by symbols if provided
    if symbols:
        symbols_upper = [s.upper().strip() for s in symbols]
        df = df[df["symbol"].str.upper().isin(symbols_upper)].copy()
        if df.empty:
            raise ValueError(f"No data found for symbols: {symbols}")
    
    # Select and order columns
    required_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    df = df[required_cols].copy()
    
    # Validate OHLC relationships
    invalid = (
        (df["high"] < df["low"]) |
        (df["high"] < df["open"]) |
        (df["high"] < df["close"]) |
        (df["low"] > df["open"]) |
        (df["low"] > df["close"])
    )
    if invalid.any():
        invalid_count = invalid.sum()
        print(f"[PRICES] WARNING: {invalid_count} rows with invalid OHLC relationships (high < low, etc.)")
        # Don't fail, but log warning
    
    # Sort and return
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    
    return df


def load_eod_prices_for_universe(
    universe_file: Path | str | None = None,
    price_file: Path | str | None = None,
    freq: Literal["1d", "5min"] = "1d"
) -> pd.DataFrame:
    """Load EOD prices for symbols from a universe file (e.g., watchlist.txt).
    
    Args:
        universe_file: Path to file with symbols (one per line). If None, uses watchlist.txt.
        price_file: Optional explicit path to price file. If None, uses default path for freq.
        freq: Frequency string ("1d" or "5min"), default "1d"
    
    Returns:
        DataFrame with columns: timestamp (UTC), symbol, open, high, low, close, volume
        Sorted by symbol, then timestamp
    
    Raises:
        FileNotFoundError: If universe file or price file not found
        ValueError: If no symbols found in universe file
    """
    # Determine universe file
    if universe_file:
        universe_path = Path(universe_file)
    else:
        # Default to watchlist.txt in repo root
        base = get_base_dir()
        universe_path = base / "watchlist.txt"
    
    if not universe_path.exists():
        raise FileNotFoundError(f"Universe file not found: {universe_path}")
    
    # Read symbols from file
    symbols = []
    with open(universe_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Handle symbols with suffixes (e.g., "SRT3.DE" -> "SRT3.DE")
                symbols.append(line.upper())
    
    if not symbols:
        raise ValueError(f"No symbols found in universe file: {universe_path}")
    
    # Load prices for these symbols
    return load_eod_prices(symbols=symbols, price_file=price_file, freq=freq)


def validate_price_data(df: pd.DataFrame) -> dict[str, bool | int | str]:
    """Validate price data quality.
    
    Args:
        df: DataFrame with price data (must have timestamp, symbol, open, high, low, close, volume)
    
    Returns:
        Dictionary with validation results:
        - valid: bool - Overall validity
        - row_count: int - Number of rows
        - symbol_count: int - Number of unique symbols
        - date_range: str - Date range (ISO format)
        - issues: list[str] - List of validation issues
    """
    issues = []
    
    # Check required columns
    required = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
        return {
            "valid": False,
            "row_count": 0,
            "symbol_count": 0,
            "date_range": "N/A",
            "issues": issues
        }
    
    # Check for empty DataFrame
    if df.empty:
        issues.append("DataFrame is empty")
        return {
            "valid": False,
            "row_count": 0,
            "symbol_count": 0,
            "date_range": "N/A",
            "issues": issues
        }
    
    # Check OHLC relationships
    invalid_ohlc = (
        (df["high"] < df["low"]) |
        (df["high"] < df["open"]) |
        (df["high"] < df["close"]) |
        (df["low"] > df["open"]) |
        (df["low"] > df["close"])
    )
    if invalid_ohlc.any():
        issues.append(f"Invalid OHLC relationships in {invalid_ohlc.sum()} rows")
    
    # Check for NaNs in critical columns
    for col in ["timestamp", "symbol", "close"]:
        if df[col].isna().any():
            issues.append(f"NaNs found in column: {col}")
    
    # Check for negative prices
    if (df[["open", "high", "low", "close"]] < 0).any().any():
        issues.append("Negative prices found")
    
    # Check for zero volume (might be valid, but log as info)
    zero_volume = (df["volume"] == 0).sum()
    if zero_volume > len(df) * 0.5:  # More than 50% zero volume
        issues.append(f"High percentage of zero volume: {zero_volume}/{len(df)} rows")
    
    # Get date range
    date_range = f"{df['timestamp'].min().isoformat()} to {df['timestamp'].max().isoformat()}"
    
    return {
        "valid": len(issues) == 0,
        "row_count": len(df),
        "symbol_count": df["symbol"].nunique(),
        "date_range": date_range,
        "issues": issues
    }
