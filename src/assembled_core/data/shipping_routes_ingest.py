"""Shipping routes data ingestion module (Phase 6 Skeleton).

This module provides functions to load shipping route and congestion data.
Currently provides skeleton implementations with dummy data.

Zukünftige Integration:
- Echte Datenquellen: MarineTraffic API, Port Authority data, etc.
- Normalisierung auf Standardformat: timestamp, route_id, port_from, port_to, ships, congestion_score
- Validierung und Datenqualitätsprüfungen
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd


def load_shipping_sample(path: Path | str | None = None) -> pd.DataFrame:
    """Load shipping routes sample data.
    
    If path is provided, loads from CSV/Parquet file.
    If path is None, generates a small dummy DataFrame with sample shipping route events.
    
    Args:
        path: Optional path to shipping data file (CSV or Parquet). If None, generates dummy data.
    
    Returns:
        DataFrame with columns:
        - timestamp: UTC timestamp of the event
        - route_id: Unique route identifier
        - port_from: Origin port code
        - port_to: Destination port code
        - ships: Number of ships on this route
        - congestion_score: Congestion score (0-100, higher = more congested)
    
    Examples:
        >>> # Generate dummy data
        >>> df = load_shipping_sample()
        >>> # Load from file
        >>> df = load_shipping_sample(path="data/shipping/shipping_routes.parquet")
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
    routes = [
        {"route_id": "US-CN-001", "port_from": "LAX", "port_to": "SHG", "symbol": "AAPL"},
        {"route_id": "US-EU-002", "port_from": "NYC", "port_to": "HAM", "symbol": "MSFT"},
        {"route_id": "US-AS-003", "port_from": "SEA", "port_to": "SIN", "symbol": "GOOGL"},
    ]
    
    data = []
    for i, route in enumerate(routes):
        for j in range(2):
            data.append({
                "timestamp": now - timedelta(days=i*2 + j),
                "route_id": route["route_id"],
                "port_from": route["port_from"],
                "port_to": route["port_to"],
                "symbol": route["symbol"],  # Link to stock symbol for feature engineering
                "ships": 10 + i*5 + j*2,
                "congestion_score": 30 + i*10 + j*5  # 0-100 scale
            })
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def normalize_shipping(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize shipping routes DataFrame to standard format.
    
    Ensures columns: timestamp (UTC), route_id, port_from, port_to, ships, congestion_score.
    
    Args:
        df: Raw shipping routes DataFrame
    
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
    
    # Ensure route_id exists
    if "route_id" not in df.columns:
        raise KeyError("route_id column is required")
    
    # Ensure port columns
    if "port_from" not in df.columns:
        df["port_from"] = "Unknown"
    if "port_to" not in df.columns:
        df["port_to"] = "Unknown"
    
    # Ensure numeric columns
    if "ships" in df.columns:
        df["ships"] = pd.to_numeric(df["ships"], errors="coerce").fillna(0).astype("int64")
    else:
        df["ships"] = 0
    
    if "congestion_score" in df.columns:
        df["congestion_score"] = pd.to_numeric(df["congestion_score"], errors="coerce").fillna(0.0).astype("float64")
        # Clamp to 0-100
        df["congestion_score"] = df["congestion_score"].clip(0, 100)
    else:
        df["congestion_score"] = 0.0
    
    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    return df

