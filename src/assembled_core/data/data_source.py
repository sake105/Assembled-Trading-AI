"""Price data source abstraction.

Provides get_price_data_source() which returns a data source object with
a get_history() method for loading price data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class LocalParquetPriceDataSource:
    """Load price data from a local Parquet (or CSV) file."""

    def __init__(self, settings, price_file: str | Path | None = None) -> None:
        self._settings = settings
        self.price_file = str(price_file) if price_file is not None else None

    def get_history(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        freq: str = "1d",
    ) -> pd.DataFrame:
        """Load and filter price data from local file."""
        path = self._resolve_path(freq)

        if path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_parquet(path)

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        if symbols:
            df = df[df["symbol"].isin(symbols)]

        # Parse date bounds
        today = pd.Timestamp.now(tz="UTC").normalize()
        start_ts = pd.Timestamp(start_date, tz="UTC") if start_date else None
        if end_date == "today":
            end_ts = today
        else:
            end_ts = pd.Timestamp(end_date, tz="UTC") if end_date else None

        if start_ts is not None:
            df = df[df["timestamp"] >= start_ts]
        if end_ts is not None:
            df = df[df["timestamp"] <= end_ts]

        return df.reset_index(drop=True)

    def _resolve_path(self, freq: str) -> Path:
        if self.price_file is not None:
            return Path(self.price_file)

        output_dir = Path(getattr(self._settings, "output_dir", "output"))
        candidates = [
            output_dir / "aggregates" / f"{freq}.parquet",
            output_dir / "aggregates" / f"assembled_intraday_{freq}.parquet",
            Path("data") / "raw" / "1min" / "demo_1min.csv",
        ]
        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(f"No price data found for freq={freq} in {output_dir}")


class YahooPriceDataSource:
    """Load price data from Yahoo Finance via yfinance."""

    def __init__(self, settings) -> None:
        try:
            import yfinance  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "yfinance is required for YahooPriceDataSource. "
                "Install it with: pip install yfinance"
            ) from exc
        self._settings = settings

    def get_history(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        freq: str = "1d",
    ) -> pd.DataFrame:
        """Download price data from Yahoo Finance."""
        import yfinance as yf

        if not symbols:
            raise ValueError("Symbols list cannot be empty")

        today_str = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
        end = today_str if end_date == "today" else end_date

        tickers = yf.download(
            tickers=symbols,
            start=start_date,
            end=end,
            interval=freq,
            auto_adjust=True,
            progress=False,
        )

        if tickers.empty:
            return pd.DataFrame(columns=["timestamp", "symbol", "close"])

        # Flatten multi-level columns if multiple symbols
        if isinstance(tickers.columns, pd.MultiIndex):
            rows = []
            for sym in symbols:
                if sym in tickers["Close"].columns:
                    sub = tickers["Close"][[sym]].copy()
                    sub.columns = ["close"]
                    sub["symbol"] = sym
                    sub.index.name = "timestamp"
                    rows.append(sub.reset_index())
            if not rows:
                return pd.DataFrame(columns=["timestamp", "symbol", "close"])
            df = pd.concat(rows, ignore_index=True)
        else:
            df = tickers[["Close"]].copy()
            df.columns = ["close"]
            df["symbol"] = symbols[0]
            df.index.name = "timestamp"
            df = df.reset_index()

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df[["timestamp", "symbol", "close"]].reset_index(drop=True)


def get_price_data_source(
    settings,
    data_source: str | None = None,
    price_file: str | Path | None = None,
    allow_external_fetch: bool = True,
):
    """Factory: return the appropriate PriceDataSource based on settings.

    Args:
        settings: Settings object with .data_source and .output_dir.
        data_source: Override the settings.data_source value.
        price_file: For local source, explicit file path.

    Returns:
        A data source instance with a get_history() method.

    Raises:
        ValueError: If data_source is unknown.
        ImportError: If required optional package is not installed.
    """
    resolved = (
        data_source
        if data_source is not None
        else getattr(settings, "data_source", "local")
    )

    if resolved == "local":
        return LocalParquetPriceDataSource(settings, price_file=price_file)
    elif resolved == "yahoo":
        return YahooPriceDataSource(settings)
    else:
        raise ValueError(f"Unknown data_source: {resolved!r}")
