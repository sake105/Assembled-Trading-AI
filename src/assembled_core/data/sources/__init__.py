"""Free API data sources for Assembled-Trading-AI.

Available fetch functions
-------------------------
- :func:`fetch_prices_yfinance`      — daily OHLCV via yfinance (no key required)
- :func:`fetch_prices_polygon`       — daily OHLCV via Polygon.io (POLYGON_API_KEY)
- :func:`fetch_fred_series`          — macro time-series via FRED (FRED_API_KEY)
- :func:`fetch_bls_series`           — US labor/price statistics via BLS (no key)

All functions return an empty :class:`pandas.DataFrame` on failure rather than
raising, to allow graceful degradation in pipeline contexts.

Reexporte fetch_prices_alphavantage + fetch_insider_trades ENTFERNT 2026-08-17
(Audit-Plan 6.4 Tranche 3): alphavantage_source + edgar_source nach
archive/orphaned_code_2026-08-17/sources/ archiviert — beide lebten nur als
__init__-Reexport ohne Konsument. Der LEBENDE Form-4-Ingest ist
data/edgar_form4_ingest.py (anderes Modul, unberuehrt).
"""

from src.assembled_core.data.sources.bls_source import fetch_bls_series
from src.assembled_core.data.sources.fred_source import fetch_fred_series
from src.assembled_core.data.sources.polygon_source import fetch_prices_polygon
from src.assembled_core.data.sources.yfinance_source import fetch_prices_yfinance

__all__ = [
    "fetch_prices_yfinance",
    "fetch_prices_polygon",
    "fetch_fred_series",
    "fetch_bls_series",
]
