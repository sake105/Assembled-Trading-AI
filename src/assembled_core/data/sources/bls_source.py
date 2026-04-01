"""Bureau of Labor Statistics (BLS) data source.

Fetches US economic time-series from the BLS Public Data API.
No API key required (25 queries/day without registration).

API docs: https://www.bls.gov/developers/api_signature_v2.htm

Commonly used series IDs
-------------------------
- ``LNS14000000``  — US Unemployment Rate (seasonally adjusted, monthly)
- ``CUUR0000SA0``  — CPI-U All Items (not seasonally adjusted, monthly)
- ``CUSR0000SA0``  — CPI-U All Items (seasonally adjusted, monthly)
- ``PRS85006092``  — Nonfarm Business Sector: Real Output Per Hour (quarterly)
- ``CEU0000000001`` — Total Nonfarm Payroll Employment (monthly)
- ``LNS12000000``  — Civilian Employment Level (monthly)

Usage::

    from assembled_core.data.sources.bls_source import fetch_bls_series

    df = fetch_bls_series(
        series_ids=["LNS14000000", "CUUR0000SA0"],
        start_year=2020,
        end_year=2024,
    )
"""

from __future__ import annotations

import json
import logging

import pandas as pd

logger = logging.getLogger(__name__)

_EMPTY = pd.DataFrame(columns=["timestamp", "series_id", "value", "period", "year"])
_BLS_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

_MONTH_MAP = {
    "M01": 1, "M02": 2, "M03": 3, "M04": 4, "M05": 5, "M06": 6,
    "M07": 7, "M08": 8, "M09": 9, "M10": 10, "M11": 11, "M12": 12,
    "Q01": 1, "Q02": 4, "Q03": 7, "Q04": 10,  # quarterly -> first month
    "A01": 1,  # annual -> January
}


def fetch_bls_series(
    series_ids: list[str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Fetch BLS time-series data.

    Args:
        series_ids: List of BLS series IDs, e.g. ["LNS14000000", "CUUR0000SA0"].
                    Max 25 series per request (BLS API limit without registration key).
        start_year: First year of data (inclusive).
        end_year:   Last year of data (inclusive).

    Returns:
        DataFrame with columns: timestamp (UTC, first day of period), series_id,
        value (float), period (raw BLS period code), year (int).
        Empty DataFrame on error.
    """
    try:
        import requests  # noqa: PLC0415
    except ImportError:
        logger.error("[ERROR] requests not installed.")
        return _EMPTY.copy()

    if not series_ids:
        return _EMPTY.copy()

    # BLS v2 API allows up to 50 series per request with key, 25 without
    chunks = [series_ids[i:i + 25] for i in range(0, len(series_ids), 25)]
    frames: list[pd.DataFrame] = []

    for chunk in chunks:
        payload = json.dumps({
            "seriesid": chunk,
            "startyear": str(start_year),
            "endyear": str(end_year),
        })
        try:
            resp = requests.post(
                _BLS_URL,
                data=payload,
                headers={"Content-type": "application/json"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("[ERROR] bls: request failed — %s", exc)
            continue

        if data.get("status") != "REQUEST_SUCCEEDED":
            logger.warning("[WARN] bls: API returned status '%s': %s", data.get("status"), data.get("message"))
            continue

        for series in data.get("Results", {}).get("series", []):
            sid = series.get("seriesID", "")
            rows = []
            for obs in series.get("data", []):
                try:
                    year = int(obs["year"])
                    period = obs.get("period", "M01")
                    month = _MONTH_MAP.get(period, 1)
                    ts = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
                    value = float(obs["value"])
                    rows.append({"timestamp": ts, "series_id": sid, "value": value,
                                 "period": period, "year": year})
                except (KeyError, ValueError):
                    continue
            if rows:
                frames.append(pd.DataFrame(rows))
                logger.debug("[OK] bls: %d observations for series %s", len(rows), sid)

    if not frames:
        logger.warning("[WARN] bls: no data returned for any of %d requested series.", len(series_ids))
        return _EMPTY.copy()

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    logger.info("[OK] bls: %d rows for %d series.", len(result), result["series_id"].nunique())
    return result
