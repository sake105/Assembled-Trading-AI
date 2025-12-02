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


def test_scenario(name: str, insider_net_buy: float, shipping_congestion: float) -> None:
    """Test ein einzelnes Szenario."""
    df = pd.DataFrame(
        {
            "timestamp": [datetime.now(timezone.utc)],
            "symbol": ["AAPL"],
            "close": [100.0],
            "insider_net_buy_20d": [insider_net_buy],
            "shipping_congestion_score_7d": [shipping_congestion],
        }
    )
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    result = generate_event_signals(df)
    
    cols = [c for c in result.columns if c.lower() in ("direction", "signal")]
    col = cols[0] if cols else "direction"
    
    row = result.iloc[0]
    print(f"\n=== {name} ===")
    print(f"Input: insider_net_buy_20d={insider_net_buy:.1f}, shipping_congestion={shipping_congestion:.1f}")
    print(f"symbol    : {row['symbol']}")
    print(f"timestamp : {row['timestamp']}")
    print(f"{col:9}: {row[col]}")
    if "score" in result.columns:
        print(f"score     : {row['score']:.4f}")
    print("=" * 30)


def main() -> None:
    """Test generate_event_signals mit verschiedenen Szenarien."""
    print("Event-Signal-Debug: Teste LONG, SHORT und FLAT Szenarien\n")
    
    # LONG: Starker Insider-Kauf + niedrige Congestion
    test_scenario("LONG Szenario", insider_net_buy=2000.0, shipping_congestion=20.0)
    
    # SHORT: Starker Insider-Verkauf + hohe Congestion
    test_scenario("SHORT Szenario", insider_net_buy=-3000.0, shipping_congestion=80.0)
    
    # FLAT: Neutrale Werte
    test_scenario("FLAT Szenario", insider_net_buy=0.0, shipping_congestion=50.0)
    
    print("\n✅ Alle drei Szenarien getestet!")


if __name__ == "__main__":
    main()

