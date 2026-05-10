"""World Bank Open Data — Cross-Country Macro (frei, kein Key).

Quelle
------
World Bank API: https://api.worldbank.org/v2/country/{code}/indicator/{ind}?format=json

Anwendung
---------
Country-level Macro-Indikatoren für GeoRisk- und Country-Beta-Signale:
- NY.GDP.MKTP.CD: GDP (current US$)
- NY.GDP.MKTP.KD.ZG: GDP growth (annual %)
- FP.CPI.TOTL.ZG: Inflation, consumer prices
- FR.INR.RINR: Real interest rate
- BX.KLT.DINV.WD.GD.ZS: FDI net inflows (% of GDP)
- DT.DOD.DECT.CD: External debt stocks

Frequenz: i. d. R. jährlich; einige quarterly.
"""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

from erweiterung._base import (
    FetchResult,
    get_cache_dir,
    rate_limited,
    retry_with_backoff,
    stable_hash,
)

logger = logging.getLogger(__name__)


@rate_limited(min_interval_s=0.5)
@retry_with_backoff(max_attempts=3, base_delay=2.0)
def _wb_indicator(country: str, indicator: str, years: tuple[int, int]) -> list[dict]:
    import requests

    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
    params = {"format": "json", "date": f"{years[0]}:{years[1]}", "per_page": 200}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, list) or len(payload) < 2:
        return []
    return payload[1] or []


def fetch_wb_indicators(
    countries: Sequence[str] = ("US", "DE", "JP", "CN", "GB"),
    indicators: Sequence[str] = (
        "NY.GDP.MKTP.KD.ZG",
        "FP.CPI.TOTL.ZG",
        "FR.INR.RINR",
    ),
    year_range: tuple[int, int] = (2000, 2026),
    use_cache: bool = True,
) -> FetchResult:
    """Hole Cross-Country-Macroindikatoren."""
    cache_key = stable_hash(
        "wb",
        tuple(sorted(countries)),
        tuple(sorted(indicators)),
        year_range[0],
        year_range[1],
    )
    cache_path = get_cache_dir("worldbank") / f"{cache_key}.parquet"
    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        return FetchResult(df, "worldbank", pd.Timestamp.utcnow(), len(df), "cache")

    rows = []
    for c in countries:
        for ind in indicators:
            try:
                data = _wb_indicator(c, ind, year_range)
            except Exception as e:  # noqa: BLE001
                logger.warning("[wb] %s/%s skip: %s", c, ind, e)
                continue
            for row in data:
                rows.append(
                    {
                        "country": c,
                        "indicator": ind,
                        "year": int(row["date"]) if row.get("date") else None,
                        "value": row.get("value"),
                    }
                )
    df = pd.DataFrame(rows).dropna(subset=["year", "value"])
    if use_cache:
        df.to_parquet(cache_path, index=False)
    return FetchResult(df, "worldbank", pd.Timestamp.utcnow(), len(df), "")


__all__ = ["fetch_wb_indicators"]
