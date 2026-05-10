"""US Bureau of Labor Statistics — frei via API v2.

API: https://api.bls.gov/publicAPI/v2/timeseries/data/

Frei mit registriertem Key (höhere Limits) oder ohne Key (10 series, 25 calls/day).

Standard-Series (Series-ID-Beispiele)
--------------------------------------
- LNS14000000 : Unemployment Rate (16+)
- CES0000000001 : Total Nonfarm Payrolls
- CUUR0000SA0 : Consumer Price Index (CPI-U)
- WPSFD4 : PPI Final Demand
- LES1252881600Q : Labor Force Participation
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from erweiterung._base import (
    FetchResult,
    get_cache_dir,
    rate_limited,
    retry_with_backoff,
    stable_hash,
)

logger = logging.getLogger(__name__)


@rate_limited(min_interval_s=2.0)
@retry_with_backoff(max_attempts=3, base_delay=2.0)
def _bls_post(
    series_ids: list[str], start_year: int, end_year: int, api_key: Optional[str]
) -> dict:
    import requests

    payload = {
        "seriesid": series_ids,
        "startyear": str(start_year),
        "endyear": str(end_year),
    }
    if api_key:
        payload["registrationkey"] = api_key
    r = requests.post(
        "https://api.bls.gov/publicAPI/v2/timeseries/data/",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def fetch_bls_series(
    series_ids: list[str],
    start_year: int = 2000,
    end_year: int = 2026,
    api_key: Optional[str] = None,
    use_cache: bool = True,
) -> FetchResult:
    """Hole BLS-Series.

    Returns:
        FetchResult mit DataFrame [date, series_id, value, period].
    """
    cache_key = stable_hash("bls", tuple(sorted(series_ids)), start_year, end_year)
    cache_path = get_cache_dir("bls") / f"{cache_key}.parquet"
    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        return FetchResult(df, "bls", pd.Timestamp.utcnow(), len(df), "cache")

    data = _bls_post(series_ids, start_year, end_year, api_key)
    if data.get("status") != "REQUEST_SUCCEEDED":
        return FetchResult(pd.DataFrame(), "bls", pd.Timestamp.utcnow(), 0, "error")
    rows = []
    for series in data.get("Results", {}).get("series", []):
        sid = series.get("seriesID")
        for entry in series.get("data", []):
            year = int(entry["year"])
            period = entry["period"]  # M01 .. M12, Q01..Q04, A01
            if period.startswith("M"):
                month = int(period[1:])
                d = pd.Timestamp(year, month, 1, tz="UTC")
            elif period.startswith("Q"):
                qtr = int(period[1:])
                d = pd.Timestamp(year, (qtr - 1) * 3 + 1, 1, tz="UTC")
            else:
                d = pd.Timestamp(year, 1, 1, tz="UTC")
            rows.append(
                {
                    "date": d,
                    "series_id": sid,
                    "value": (
                        float(entry["value"])
                        if entry.get("value") not in (None, "-")
                        else None
                    ),
                    "period": period,
                }
            )
    df = pd.DataFrame(rows).sort_values(["series_id", "date"])
    if use_cache and not df.empty:
        df.to_parquet(cache_path, index=False)
    return FetchResult(df, "bls", pd.Timestamp.utcnow(), len(df), "")


__all__ = ["fetch_bls_series"]
