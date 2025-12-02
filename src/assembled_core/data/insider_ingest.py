"""Insider trading data ingestion module (Phase 6 Skeleton).

This module provides functions to load insider trading data.
Currently provides skeleton implementations with dummy data.

Zukünftige Integration:
- Echte Datenquellen: SEC Form 4 filings, InsiderMonkey API, etc.
- Normalisierung auf Standardformat: timestamp, symbol, trades_count, net_shares, role
- Validierung und Datenqualitätsprüfungen
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd


def load_insider_sample(path: Path | str | None = None) -> pd.DataFrame:
    """Load insider trading sample data.
    
    If path is provided, loads from CSV/Parquet file.
    If path is None, generates a small dummy DataFrame with sample insider trading events.
    
    Args:
        path: Optional path to insider data file (CSV or Parquet). If None, generates dummy data.
    
    Returns:
        DataFrame with columns:
        - timestamp: UTC timestamp of the trade
        - symbol: Stock symbol
        - trades_count: Number of trades in this event
        - net_shares: Net shares bought/sold (positive = buy, negative = sell)
        - role: Insider role (e.g., "CEO", "CFO", "Director")
    
    Examples:
        >>> # Generate dummy data
        >>> df = load_insider_sample()
        >>> # Load from file
        >>> df = load_insider_sample(path="data/insider/insider_trades.parquet")
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
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    data = []
    for i, symbol in enumerate(symbols):
        for j in range(2):
            data.append({
                "timestamp": now - timedelta(days=i*2 + j),
                "symbol": symbol,
                "trades_count": 1 + j,
                "net_shares": (1000 + i*500) * (1 if j == 0 else -1),  # Buy then sell
                "role": ["CEO", "CFO", "Director"][i % 3]
            })
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def normalize_insider(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize insider trading DataFrame to standard format.
    
    Ensures columns: timestamp (UTC), symbol, trades_count, net_shares, role.
    
    Args:
        df: Raw insider trading DataFrame
    
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
    
    # Ensure numeric columns
    if "trades_count" in df.columns:
        df["trades_count"] = pd.to_numeric(df["trades_count"], errors="coerce").fillna(0).astype("int64")
    else:
        df["trades_count"] = 1
    
    if "net_shares" in df.columns:
        df["net_shares"] = pd.to_numeric(df["net_shares"], errors="coerce").fillna(0.0).astype("float64")
    else:
        df["net_shares"] = 0.0
    
    if "role" not in df.columns:
        df["role"] = "Unknown"
    
    # Sort by symbol, then timestamp
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    
    return df

