# src/assembled_core/ports/prices_repository.py
"""PricesRepository port — domain-facing read of historical / current prices.

The actual data source (yfinance, Polygon, Alpaca, FactorStore cache)
is the adapter's problem. The domain just asks for prices keyed by
(symbol, date range, frequency) and gets a pandas DataFrame back.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import pandas as pd


@runtime_checkable
class PricesRepository(Protocol):
    """PIT-safe price reader."""

    def get_panel(
        self,
        symbols: Sequence[str],
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
        *,
        freq: str = "1d",
        as_of: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Return a DataFrame indexed by (timestamp, symbol) with at
        least an ``adj_close`` column. ``as_of`` filters strictly to
        rows whose ``timestamp <= as_of``.
        """
        ...
