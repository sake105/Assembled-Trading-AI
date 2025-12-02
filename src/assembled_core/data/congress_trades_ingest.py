"""Congress trading data ingestion module (Phase 6 Skeleton).

This module provides functions to load congressional trading data.
Currently provides skeleton implementations with dummy data.

Zukünftige Integration:
- Echte Datenquellen: House Stock Watcher, Senate financial disclosures, etc.
- Normalisierung auf Standardformat: timestamp, symbol, politician, party, amount
- Validierung und Datenqualitätsprüfungen
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd


def load_congress_sample(path: Path | str | None = None) -> pd.DataFrame:
    """Load congressional trading sample data.
    
    If path is provided, loads from CSV/Parquet file.
    If path is None, generates a small dummy DataFrame with sample congressional trades.
    
    Args:
        path: Optional path to congress data file (CSV or Parquet). If None, generates dummy data.
    
    Returns:
        DataFrame with columns:
        - timestamp: UTC timestamp of the trade
        - symbol: Stock symbol
        - politician: Name of the politician
        - party: Political party (e.g., "D", "R")
        - amount: Trade amount in USD
    
    Examples:
        >>> # Generate dummy data
        >>> df = load_congress_sample()
        >>> # Load from file
        >>> df = load_congress_sample(path="data/congress/congress_trades.parquet")
    """
    if path is not None:
        path = Path(path)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Use .parquet or .csv")
    
    # Generate dummy data
    now = datetime.now(timezone.utc)
    symbols = ["AAPL", "MSFT", "TSLA"]
    politicians = ["John Doe", "Jane Smith", "Bob Johnson"]
    parties = ["D", "R", "D"]
    
    data = []
    for i, symbol in enumerate(symbols):
        for j in range(2):
            data.append({
                "timestamp": now - timedelta(days=i*3 + j*2),
                "symbol": symbol,
                "politician": politicians[i],
                "party": parties[i],
                "amount": (50000 + i*10000) * (1.0 + j*0.5)  # Varying amounts
            })
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def normalize_congress(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize congressional trading DataFrame to standard format.
    
    Ensures columns: timestamp (UTC), symbol, politician, party, amount.
    
    Args:
        df: Raw congressional trading DataFrame
    
    Returns:
        Normalized DataFrame with standard columns
    
    Raises:
        KeyError: If required columns are missing
    """
    df = df.copy()
    
    # Ensure timestamp is UTC-aware
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    else:
        raise KeyError("timestamp column is required")
    
    # Ensure symbol exists
    if "symbol" not in df.columns:
        raise KeyError("symbol column is required")
    
    # Ensure politician and party
    if "politician" not in df.columns:
        df["politician"] = "Unknown"
    if "party" not in df.columns:
        df["party"] = "Unknown"
    
    # Ensure amount is numeric
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0).astype("float64")
    else:
        df["amount"] = 0.0
    
    # Sort by symbol, then timestamp
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    
    return df

