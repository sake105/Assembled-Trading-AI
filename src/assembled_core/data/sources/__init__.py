"""Free API data sources for Assembled-Trading-AI.

Available fetch functions
-------------------------
- :func:`fetch_prices_yfinance`      — daily OHLCV via yfinance (no key required)
- :func:`fetch_prices_polygon`       — daily OHLCV via Polygon.io (POLYGON_API_KEY)
- :func:`fetch_prices_alphavantage`  — daily OHLCV via Alpha Vantage (ALPHAVANTAGE_KEY)
- :func:`fetch_fred_series`          — macro time-series via FRED (FRED_API_KEY)
- :func:`fetch_news_headlines`       — news articles via NewsAPI (NEWSAPI_KEY)
- :func:`fetch_worldbank_indicator`  — country macro indicators via World Bank (no key)
- :func:`fetch_bls_series`           — US labor/price statistics via BLS (no key)
- :func:`fetch_insider_trades`       — SEC EDGAR Form 4 insider trades (no key)

All functions return an empty :class:`pandas.DataFrame` on failure rather than
raising, to allow graceful degradation in pipeline contexts.
"""

from src.assembled_core.data.sources.alphavantage_source import (
    fetch_prices_alphavantage,
)
from src.assembled_core.data.sources.bls_source import fetch_bls_series
from src.assembled_core.data.sources.edgar_source import fetch_insider_trades
from src.assembled_core.data.sources.fred_source import fetch_fred_series
from src.assembled_core.data.sources.newsapi_source import fetch_news_headlines
from src.assembled_core.data.sources.polygon_source import fetch_prices_polygon
from src.assembled_core.data.sources.worldbank_source import fetch_worldbank_indicator
from src.assembled_core.data.sources.yfinance_source import fetch_prices_yfinance

__all__ = [
    "fetch_prices_yfinance",
    "fetch_prices_polygon",
    "fetch_prices_alphavantage",
    "fetch_fred_series",
    "fetch_news_headlines",
    "fetch_worldbank_indicator",
    "fetch_bls_series",
    "fetch_insider_trades",
]
