"""Price data source abstraction.

Provides get_price_data_source() which returns a callable that loads
price data for the pipeline (local parquet by default).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def get_price_data_source(
    freq: str = "1d",
    output_dir: str | Path = "output",
) -> pd.DataFrame:
    """Load price data from local parquet/CSV files.

    Returns a DataFrame with columns: timestamp, symbol, close (+ optional OHLCV).
    """
    output_dir = Path(output_dir)

    candidates = [
        output_dir / "aggregates" / f"{freq}.parquet",
        output_dir / "aggregates" / f"assembled_intraday_{freq}.parquet",
        Path("data") / "raw" / "1min" / "demo_1min.csv",
    ]

    for path in candidates:
        if path.exists():
            if path.suffix == ".csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_parquet(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            logger.info("[data_source] Loaded %d rows from %s", len(df), path)
            return df

    raise FileNotFoundError(
        f"No price data found for freq={freq} in {output_dir}"
    )
