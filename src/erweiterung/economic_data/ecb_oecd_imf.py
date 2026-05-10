"""ECB SDW + OECD-Stats + IMF-DataMapper — alle frei.

ECB Statistical Data Warehouse
------------------------------
https://sdw-wsrest.ecb.europa.eu/service/data/<flowRef>/<key>?format=jsondata
Beispiel: ICP/M.U2.N.000000.4.ANR (HICP)

OECD Stats
----------
https://stats.oecd.org/SDMX-JSON/data/<dataset>/<filter>/all
Beispiel: KEI/PRINTO01.USA.GP.M (industrial production)

IMF DataMapper API
------------------
https://www.imf.org/external/datamapper/api/v1/<indicator>/<country>
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


@rate_limited(min_interval_s=1.5)
@retry_with_backoff(max_attempts=3, base_delay=2.0)
def fetch_ecb_series(
    flow_ref: str,
    key: str,
    start: str = "2000-01-01",
    end: Optional[str] = None,
    use_cache: bool = True,
) -> FetchResult:
    """Hole ECB-SDW-Series."""
    import requests

    cache_key = stable_hash("ecb", flow_ref, key, start, end)
    cache_path = get_cache_dir("ecb") / f"{cache_key}.parquet"
    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        return FetchResult(df, "ecb", pd.Timestamp.utcnow(), len(df), "cache")

    url = f"https://sdw-wsrest.ecb.europa.eu/service/data/{flow_ref}/{key}"
    params = {"format": "jsondata", "startPeriod": start}
    if end:
        params["endPeriod"] = end
    r = requests.get(
        url, params=params, headers={"Accept": "application/json"}, timeout=30
    )
    if r.status_code == 404:
        return FetchResult(pd.DataFrame(), "ecb", pd.Timestamp.utcnow(), 0, "not_found")
    r.raise_for_status()
    payload = r.json()

    obs = payload.get("dataSets", [{}])[0].get("series", {})
    if not obs:
        return FetchResult(pd.DataFrame(), "ecb", pd.Timestamp.utcnow(), 0, "empty")
    structure = payload.get("structure", {})
    time_dim = (
        structure.get("dimensions", {}).get("observation", [{}])[0].get("values", [])
    )
    rows = []
    for series_key, series_data in obs.items():
        for t_idx, val in series_data.get("observations", {}).items():
            t = int(t_idx)
            if t < len(time_dim):
                period = time_dim[t]["id"]
                d = pd.to_datetime(period, errors="coerce", utc=True)
                rows.append(
                    {
                        "date": d,
                        "series_key": series_key,
                        "period": period,
                        "value": float(val[0]) if val else None,
                    }
                )
    df = pd.DataFrame(rows).dropna(subset=["date"]).sort_values("date")
    if use_cache and not df.empty:
        df.to_parquet(cache_path, index=False)
    return FetchResult(df, "ecb", pd.Timestamp.utcnow(), len(df), "")


@rate_limited(min_interval_s=1.5)
@retry_with_backoff(max_attempts=3, base_delay=2.0)
def fetch_imf_indicator(
    indicator: str,
    country: str = "USA",
    use_cache: bool = True,
) -> FetchResult:
    """IMF DataMapper API."""
    import requests

    cache_key = stable_hash("imf", indicator, country)
    cache_path = get_cache_dir("imf") / f"{cache_key}.parquet"
    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        return FetchResult(df, "imf", pd.Timestamp.utcnow(), len(df), "cache")

    url = f"https://www.imf.org/external/datamapper/api/v1/{indicator}/{country}"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return FetchResult(
            pd.DataFrame(), "imf", pd.Timestamp.utcnow(), 0, f"status={r.status_code}"
        )
    payload = r.json()
    values = payload.get("values", {}).get(indicator, {}).get(country, {})
    rows = [
        {"year": int(y), "value": float(v) if v is not None else None}
        for y, v in values.items()
    ]
    df = pd.DataFrame(rows).dropna(subset=["value"]).sort_values("year")
    if use_cache and not df.empty:
        df.to_parquet(cache_path, index=False)
    return FetchResult(
        df, "imf", pd.Timestamp.utcnow(), len(df), f"indicator={indicator}"
    )


__all__ = ["fetch_ecb_series", "fetch_imf_indicator"]
