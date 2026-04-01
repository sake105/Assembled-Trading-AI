"""World Bank macro indicator source.

Fetches country-level macro indicators from the World Bank Open Data API.
No API key required.

API docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/898581

Commonly used indicators
-------------------------
- ``NY.GDP.MKTP.KD.ZG`` — GDP growth rate (annual %)
- ``FP.CPI.TOTL.ZG``    — Inflation, consumer prices (annual %)
- ``SL.UEM.TOTL.ZS``    — Unemployment rate (% of total labor force)
- ``GC.DOD.TOTL.GD.ZS`` — Central government debt (% of GDP)
- ``BX.KLT.DINV.WD.GD.ZS`` — Foreign direct investment, net inflows (% of GDP)
- ``NY.GDP.PCAP.KD``    — GDP per capita (constant 2015 USD)

Usage::

    from assembled_core.data.sources.worldbank_source import fetch_worldbank_indicator

    df = fetch_worldbank_indicator(
        countries=["US", "DE", "CN"],
        indicator="NY.GDP.MKTP.KD.ZG",
        start_year=2015,
        end_year=2023,
    )
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_EMPTY = pd.DataFrame(columns=["year", "country_code", "country_name", "indicator", "value"])
_BASE_URL = "https://api.worldbank.org/v2"


def fetch_worldbank_indicator(
    countries: list[str],
    indicator: str,
    start_year: int,
    end_year: int,
    *,
    per_page: int = 500,
) -> pd.DataFrame:
    """Fetch a World Bank indicator for one or more countries.

    Args:
        countries: ISO 3166-1 alpha-2 country codes, e.g. ["US", "DE", "CN"].
                   Use "all" to fetch all available countries.
        indicator:  World Bank indicator code, e.g. "NY.GDP.MKTP.KD.ZG".
        start_year: First year of data (inclusive).
        end_year:   Last year of data (inclusive).
        per_page:   Max rows per API page (default: 500).

    Returns:
        DataFrame with columns: year (int), country_code, country_name, indicator, value.
        Empty DataFrame on error or if no data available.
    """
    try:
        import requests  # noqa: PLC0415
    except ImportError:
        logger.error("[ERROR] requests not installed.")
        return _EMPTY.copy()

    if not countries:
        return _EMPTY.copy()

    country_str = ";".join(c.upper() for c in countries)
    url = (
        f"{_BASE_URL}/country/{country_str}/indicator/{indicator}"
        f"?format=json&date={start_year}:{end_year}&per_page={per_page}"
    )

    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.error("[ERROR] worldbank: request failed for indicator %s — %s", indicator, exc)
        return _EMPTY.copy()

    if not isinstance(data, list) or len(data) < 2:
        logger.warning("[WARN] worldbank: unexpected response format for indicator %s", indicator)
        return _EMPTY.copy()

    records = data[1]
    if not records:
        logger.warning("[WARN] worldbank: no data for indicator %s (%d–%d)", indicator, start_year, end_year)
        return _EMPTY.copy()

    rows = []
    for rec in records:
        value = rec.get("value")
        if value is None:
            continue
        try:
            year = int(rec.get("date", 0))
        except (ValueError, TypeError):
            continue
        rows.append({
            "year": year,
            "country_code": (rec.get("countryiso3code") or rec.get("country", {}).get("id") or "").upper(),
            "country_name": (rec.get("country") or {}).get("value") or "",
            "indicator": indicator,
            "value": float(value),
        })

    if not rows:
        return _EMPTY.copy()

    result = pd.DataFrame(rows)
    result = result.sort_values(["country_code", "year"]).reset_index(drop=True)
    logger.info(
        "[OK] worldbank: %d rows for indicator %s (%d countries).",
        len(result), indicator, result["country_code"].nunique(),
    )
    return result
