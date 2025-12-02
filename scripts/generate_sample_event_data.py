"""Generate sample event data that matches existing price sample data.

This script reads the existing eod_sample.parquet and generates matching
insider and shipping events for the same symbols and time periods.
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import timedelta

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.data.insider_ingest import normalize_insider
from src.assembled_core.data.shipping_routes_ingest import normalize_shipping

SAMPLE_PRICE_FILE = ROOT / "data" / "sample" / "eod_sample.parquet"
OUTPUT_DIR = ROOT / "data" / "sample" / "events"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Generate sample event data matching price sample."""
    if not SAMPLE_PRICE_FILE.exists():
        print(f"Error: Price sample file not found: {SAMPLE_PRICE_FILE}")
        print("Please ensure data/sample/eod_sample.parquet exists.")
        sys.exit(1)
    
    print(f"Reading price data from {SAMPLE_PRICE_FILE}...")
    prices = pd.read_parquet(SAMPLE_PRICE_FILE)
    
    # Ensure timestamp is UTC-aware
    if prices["timestamp"].dtype != "datetime64[ns, UTC]":
        prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    
    # Get unique symbols and dates
    symbols = prices["symbol"].unique()
    dates = prices["timestamp"].sort_values().unique()
    
    print(f"Found {len(symbols)} symbols: {symbols}")
    print(f"Date range: {dates.min()} to {dates.max()} ({len(dates)} days)")
    
    # Use first symbol for events (or all symbols if you want)
    first_symbol = symbols[0]
    
    # Take dates from middle of range (5-10 days)
    mid = len(dates) // 2
    event_dates = dates[mid:mid + 7]  # 7 days of events
    
    print(f"\nGenerating events for symbol {first_symbol} on {len(event_dates)} dates...")
    
    # ---------- Insider-Events ----------
    insider_raw = pd.DataFrame({
        "timestamp": event_dates,
        "symbol": [first_symbol] * len(event_dates),
        # Alternating net buy/sell to generate both LONG and SHORT signals
        "trades_count": [5, 3, 4, 6, 2, 5, 3],
        "net_shares": [2000, -3000, 1500, -2500, 3000, -2000, 2500],  # Mix of buy/sell
        "role": ["CEO", "CFO", "Director", "CEO", "Director", "CFO", "CEO"],
    })
    
    insider_events = normalize_insider(insider_raw)
    insider_path = OUTPUT_DIR / "insider_sample.parquet"
    insider_events.to_parquet(insider_path, index=False)
    
    print(f"  ✓ Insider events: {len(insider_events)} rows")
    print(f"    Net shares range: {insider_events['net_shares'].min():.0f} to {insider_events['net_shares'].max():.0f}")
    
    # ---------- Shipping-Events ----------
    shipping_raw = pd.DataFrame({
        "timestamp": event_dates,
        "route_id": [f"R{i:03d}" for i in range(len(event_dates))],
        "port_from": ["LAX"] * len(event_dates),
        "port_to": ["SHG"] * len(event_dates),
        "symbol": [first_symbol] * len(event_dates),
        "ships": [10, 20, 5, 30, 15, 25, 12],
        "congestion_score": [20, 80, 40, 90, 50, 75, 30],  # Low = bullish, High = bearish
    })
    
    shipping_events = normalize_shipping(shipping_raw)
    shipping_path = OUTPUT_DIR / "shipping_sample.parquet"
    shipping_events.to_parquet(shipping_path, index=False)
    
    print(f"  ✓ Shipping events: {len(shipping_events)} rows")
    print(f"    Congestion range: {shipping_events['congestion_score'].min():.0f} to {shipping_events['congestion_score'].max():.0f}")
    
    print(f"\n✅ Sample events written to:")
    print(f"  {insider_path}")
    print(f"  {shipping_path}")
    print(f"\nYou can now run backtests with --strategy event_insider_shipping")


if __name__ == "__main__":
    main()

