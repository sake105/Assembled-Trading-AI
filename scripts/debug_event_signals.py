# scripts/debug_event_signals.py
"""Debug-Skript für Event-Signal-Generierung.

Dieses Skript testet die generate_event_signals Funktion mit minimalen Dummy-Daten.
Nützlich für schnelle Sanity-Checks ohne komplexe Test-Setups.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.signals.rules_event_insider_shipping import generate_event_signals


def main() -> None:
    """Test generate_event_signals mit minimalen Dummy-Daten."""
    # Minimaler Dummy-Input für die Event-Strategie
    df = pd.DataFrame(
        {
            "timestamp": [datetime.now(timezone.utc)],
            "symbol": ["AAPL"],
            "close": [100.0],
            "insider_net_buy_20d": [2000.0],
            "shipping_congestion_score_7d": [20.0],
        }
    )
    
    # Safety: sicherstellen, dass Timestamp wirklich UTC-aware ist
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    result = generate_event_signals(df)
    
    # Je nach Implementierung heißt die Spalte bei dir vermutlich "direction" und nicht mehr "signal"
    cols = [c for c in result.columns if c.lower() in ("direction", "signal")]
    col = cols[0] if cols else "direction"
    
    row = result.iloc[0]
    print("=== Event-Signal-Debug ===")
    print(f"symbol    : {row['symbol']}")
    print(f"timestamp : {row['timestamp']}")
    print(f"{col:9}: {row[col]}")
    if "score" in result.columns:
        print(f"score     : {row['score']:.4f}")
    print("===========================")


if __name__ == "__main__":
    main()

