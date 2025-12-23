# scripts/dev/create_sample_eod_data.py
"""Create sample EOD price data for testing."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.assembled_core.config import get_base_dir


def create_sample_eod_data(output_path: Path | None = None) -> Path:
    """Create sample EOD price data.

    Args:
        output_path: Path to output file. If None, uses data/sample/eod_sample.parquet

    Returns:
        Path to created file
    """
    if output_path is None:
        base = get_base_dir()
        output_path = base / "data" / "sample" / "eod_sample.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create sample data: 3 symbols, 30 trading days each
    base_date = datetime(2025, 1, 1, 0, 0, 0)
    symbols = ["AAPL", "MSFT", "GOOGL"]
    data = []

    for sym in symbols:
        # Different base prices per symbol
        price_base = 100.0 if sym == "AAPL" else (200.0 if sym == "MSFT" else 150.0)

        for i in range(30):
            ts = base_date + timedelta(days=i)
            open_p = price_base + i * 0.1
            high_p = open_p + 0.5
            low_p = open_p - 0.3
            close_p = open_p + 0.2
            volume = 1000000.0 + i * 10000.0

            data.append(
                {
                    "timestamp": ts,
                    "symbol": sym,
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "close": close_p,
                    "volume": volume,
                }
            )

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Save to parquet
    df.to_parquet(output_path, index=False)

    print(f"[SAMPLE] Created sample EOD data: {output_path}")
    print(
        f"[SAMPLE] Rows: {len(df)}, Symbols: {df['symbol'].nunique()}, Date range: {df['timestamp'].min()} to {df['timestamp'].max()}"
    )

    return output_path


if __name__ == "__main__":
    create_sample_eod_data()
