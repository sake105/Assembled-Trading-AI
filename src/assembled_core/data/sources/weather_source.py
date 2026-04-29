"""Weather data for commodity/energy signals via Open-Meteo and NOAA.

From 10_FREE_DATEN.md §10.13.
Use cases:
- HDD (Heating Degree Days) for natural gas demand forecasting
- Temperature anomalies for agricultural commodity signals
- Drought index proxy for Corn/Wheat exposure

Sources:
- Open-Meteo (non-commercial, no key, ECMWF/GFS data)
- NOAA NWS API (public domain, no key, User-Agent required)

Install: pip install openmeteo-requests requests-cache retry-requests
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# NOAA NWS User-Agent (required for public API)
_NOAA_USER_AGENT = "AssembledTradingAI research@example.com"

# Key US energy demand cities and their coordinates
US_ENERGY_CITIES = {
    "chicago": {"lat": 41.85, "lon": -87.65},
    "new_york": {"lat": 40.71, "lon": -74.01},
    "houston": {"lat": 29.76, "lon": -95.37},
    "boston": {"lat": 42.36, "lon": -71.06},
    "minneapolis": {"lat": 44.98, "lon": -93.27},
    "atlanta": {"lat": 33.75, "lon": -84.39},
    "los_angeles": {"lat": 34.05, "lon": -118.24},
    "denver": {"lat": 39.74, "lon": -104.98},
}


def _try_openmeteo():
    try:
        import openmeteo_requests
        return openmeteo_requests
    except ImportError:
        logger.warning(
            "openmeteo-requests not installed — pip install openmeteo-requests"
        )
        return None


def fetch_temperature_openmeteo(
    latitude: float,
    longitude: float,
    start_date: date | None = None,
    end_date: date | None = None,
    lookback_days: int = 30,
) -> pd.Series:
    """Fetch daily mean temperature from Open-Meteo API (no key required).

    Args:
        latitude: Location latitude.
        longitude: Location longitude.
        start_date: Start date (defaults to lookback_days ago).
        end_date: End date (defaults to today).
        lookback_days: Lookback window if start_date not specified.

    Returns:
        Daily temperature Series (°C). Empty if unavailable.
    """
    if start_date is None:
        start_date = date.today() - timedelta(days=lookback_days)
    if end_date is None:
        end_date = date.today()

    try:
        import requests
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": "temperature_2m_mean",
            "timezone": "auto",
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        dates = data["daily"]["time"]
        temps = data["daily"]["temperature_2m_mean"]
        idx = pd.to_datetime(dates)
        return pd.Series(temps, index=idx, name="temp_c").dropna()

    except Exception as exc:
        logger.debug("Open-Meteo fetch failed (%.2f, %.2f): %s", latitude, longitude, exc)
        return pd.Series(dtype=float)


def compute_hdd(
    temperatures: pd.Series,
    base_temp_c: float = 18.0,
) -> pd.Series:
    """Compute Heating Degree Days (HDD).

    HDD = max(0, base_temp - actual_temp)
    High HDD → high natural gas demand.

    Args:
        temperatures: Daily mean temperature in °C.
        base_temp_c: Base temperature (default 18°C = 65°F).

    Returns:
        Daily HDD series.
    """
    hdd = (base_temp_c - temperatures).clip(lower=0)
    return hdd.rename("hdd")


def compute_cdd(
    temperatures: pd.Series,
    base_temp_c: float = 18.0,
) -> pd.Series:
    """Compute Cooling Degree Days (CDD).

    CDD = max(0, actual_temp - base_temp)
    High CDD → high electricity demand (A/C).
    """
    cdd = (temperatures - base_temp_c).clip(lower=0)
    return cdd.rename("cdd")


def us_energy_demand_signal(
    lookback_days: int = 30,
    base_temp_c: float = 18.0,
) -> dict[str, float]:
    """Compute HDD/CDD signals across major US energy demand cities.

    Returns:
        Dict with keys: avg_hdd, avg_cdd, hdd_z, cdd_z (vs 30d average).
        All 0.0 if Open-Meteo unavailable.
    """
    all_hdd: list[float] = []
    all_cdd: list[float] = []
    all_baseline_hdd: list[float] = []
    all_baseline_cdd: list[float] = []

    for city, coords in US_ENERGY_CITIES.items():
        temps = fetch_temperature_openmeteo(
            coords["lat"], coords["lon"],
            lookback_days=lookback_days + 5,
        )
        if temps.empty:
            continue

        recent = temps.tail(5)
        baseline = temps.iloc[:-5] if len(temps) > 10 else temps

        hdd = compute_hdd(recent, base_temp_c)
        cdd = compute_cdd(recent, base_temp_c)
        baseline_hdd = compute_hdd(baseline, base_temp_c)
        baseline_cdd = compute_cdd(baseline, base_temp_c)

        all_hdd.append(float(hdd.mean()))
        all_cdd.append(float(cdd.mean()))
        all_baseline_hdd.append(float(baseline_hdd.mean()))
        all_baseline_cdd.append(float(baseline_cdd.mean()))

    if not all_hdd:
        return {"avg_hdd": 0.0, "avg_cdd": 0.0, "hdd_anomaly": 0.0, "cdd_anomaly": 0.0}

    avg_hdd = float(np.mean(all_hdd))
    avg_cdd = float(np.mean(all_cdd))
    baseline_avg_hdd = float(np.mean(all_baseline_hdd))
    baseline_avg_cdd = float(np.mean(all_baseline_cdd))

    return {
        "avg_hdd": avg_hdd,
        "avg_cdd": avg_cdd,
        "hdd_anomaly": avg_hdd - baseline_avg_hdd,
        "cdd_anomaly": avg_cdd - baseline_avg_cdd,
    }


def fetch_temperature_noaa(
    station_id: str,
    lookback_days: int = 30,
) -> pd.Series:
    """Fetch temperature from NOAA Climate Data Online (CDO) API.

    Note: Requires free API token from ncdc.noaa.gov/cdo-web/token.
    Falls back to empty Series if token not available.

    Args:
        station_id: NOAA station ID (e.g. 'GHCND:USW00094728' = NYC Central Park).
        lookback_days: Days of history to fetch.

    Returns:
        Daily mean temperature Series (°C). Empty on failure.
    """
    import os
    token = os.environ.get("NOAA_CDO_TOKEN")
    if not token:
        logger.debug("NOAA_CDO_TOKEN not set — using Open-Meteo fallback")
        return pd.Series(dtype=float)

    try:
        import requests
        start = (date.today() - timedelta(days=lookback_days)).isoformat()
        end = date.today().isoformat()

        resp = requests.get(
            "https://www.ncei.noaa.gov/cdo-web/api/v2/data",
            headers={"token": token},
            params={
                "datasetid": "GHCND",
                "stationid": station_id,
                "datatypeid": "TAVG",
                "startdate": start,
                "enddate": end,
                "limit": 1000,
                "units": "metric",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if "results" not in data:
            return pd.Series(dtype=float)

        rows = [(r["date"], r["value"] / 10.0) for r in data["results"]]
        idx = pd.to_datetime([r[0] for r in rows])
        vals = [r[1] for r in rows]
        return pd.Series(vals, index=idx, name="temp_c").dropna()

    except Exception as exc:
        logger.debug("NOAA CDO fetch failed for %s: %s", station_id, exc)
        return pd.Series(dtype=float)


__all__ = [
    "US_ENERGY_CITIES",
    "fetch_temperature_openmeteo",
    "compute_hdd",
    "compute_cdd",
    "us_energy_demand_signal",
    "fetch_temperature_noaa",
]
