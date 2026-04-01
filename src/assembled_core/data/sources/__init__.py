"""Free API data sources for Assembled-Trading-AI.

Available fetch functions
-------------------------
- :func:`fetch_prices_yfinance`  — daily OHLCV via yfinance (no key required)
- :func:`fetch_prices_polygon`   — daily OHLCV via Polygon.io (POLYGON_API_KEY)
- :func:`fetch_fred_series`      — macro time-series via FRED (FRED_API_KEY)

All functions return an empty :class:`pandas.DataFrame` on failure rather than
raising, to allow graceful degradation in pipeline contexts.
"""

from assembled_core.data.sources.fred_source import fetch_fred_series
from assembled_core.data.sources.polygon_source import fetch_prices_polygon
from assembled_core.data.sources.yfinance_source import fetch_prices_yfinance

__all__ = [
    "fetch_prices_yfinance",
    "fetch_prices_polygon",
    "fetch_fred_series",
]
