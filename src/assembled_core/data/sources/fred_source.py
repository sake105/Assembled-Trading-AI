"""FRED macro data source (Federal Reserve Economic Data).

Fetches time-series data from the St. Louis Fed FRED API via the `fredapi` package.

Requires environment variable::

    FRED_API_KEY=<your key>

Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html

Commonly used series IDs
------------------------
- ``DGS10``        — 10-Year Treasury Constant Maturity Rate (daily)
- ``T10Y2Y``       — 10-Year minus 2-Year Treasury spread / yield curve (daily)
- ``VIXCLS``       — CBOE Volatility Index (VIX, daily)
- ``UNRATE``       — Civilian Unemployment Rate (monthly)
- ``CPIAUCSL``     — CPI, All Urban Consumers, All Items (monthly)
- ``BAMLH0A0HYM2`` — ICE BofA US High Yield Index OAS / spread (daily)

Usage::

    from assembled_core.data.sources.fred_source import fetch_fred_series

    df = fetch_fred_series(["DGS10", "VIXCLS"], "2024-01-01", "2024-12-31")
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level TTL cache for FRED fetches (6-hour TTL)
# Key: (series_id, start_date, end_date)  →  (fetch_timestamp, DataFrame)
# ---------------------------------------------------------------------------
_FRED_CACHE: dict[str, tuple[float, Any]] = {}
_FRED_CACHE_TTL: float = 21600.0  # 6 hours in seconds


def _get_api_key() -> str | None:
    """Return FRED_API_KEY from environment, or None if not set."""
    key = os.environ.get("FRED_API_KEY", "").strip()
    return key if key else None


def _fetch_single_series(
    series_id: str,
    start_date: str,
    end_date: str,
    fred_client: object,
) -> pd.DataFrame | None:
    """Fetch one FRED series.  Returns None on failure."""
    try:
        raw = fred_client.get_series(  # type: ignore[attr-defined]
            series_id,
            observation_start=start_date,
            observation_end=end_date,
        )
        if raw is None or raw.empty:
            logger.warning(
                "[WARN] fred: no data for series %s (%s – %s)",
                series_id,
                start_date,
                end_date,
            )
            return None

        df = raw.reset_index()
        df.columns = ["timestamp", "value"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.normalize()
        df["series_id"] = series_id
        df = df[["timestamp", "series_id", "value"]].copy()
        # Drop NaN observations (FRED sometimes returns them for missing release days)
        df = df.dropna(subset=["value"])
        return df

    except Exception as exc:  # noqa: BLE001
        logger.error("[ERROR] fred: failed to fetch series %s — %s", series_id, exc)
        return None


def fetch_fred_series(
    series_ids: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch FRED macro series. Returns empty DataFrame if key missing or on error.

    Args:
        series_ids: List of FRED series IDs, e.g. ["DGS10", "T10Y2Y", "VIXCLS"].
        start_date: Inclusive start date, "YYYY-MM-DD".
        end_date:   Inclusive end date, "YYYY-MM-DD".

    Returns:
        DataFrame with columns: timestamp (UTC date), series_id, value.
        Empty DataFrame if FRED_API_KEY is not set or all fetches fail.
    """
    _empty = pd.DataFrame(columns=["timestamp", "series_id", "value"])

    if not series_ids:
        return _empty

    api_key = _get_api_key()
    if api_key is None:
        logger.warning("[WARN] fred: FRED_API_KEY not set — returning empty DataFrame.")
        return _empty

    try:
        from fredapi import Fred  # noqa: PLC0415
    except ImportError:
        logger.error("[ERROR] fredapi not installed. Run: pip install fredapi>=0.5.0")
        return _empty

    try:
        fred = Fred(api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        logger.error("[ERROR] fred: failed to initialize FRED client — %s", exc)
        return _empty

    frames: list[pd.DataFrame] = []
    now = time.time()
    for sid in series_ids:
        cache_key = f"{sid}|{start_date}|{end_date}"
        cached = _FRED_CACHE.get(cache_key)
        if cached is not None and (now - cached[0]) < _FRED_CACHE_TTL:
            logger.debug("[OK] fred: cache hit for %s", sid)
            df = cached[1]
        else:
            df = _fetch_single_series(sid, start_date, end_date, fred)
            _FRED_CACHE[cache_key] = (now, df)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        logger.warning(
            "[WARN] fred: no data returned for any of %d requested series.",
            len(series_ids),
        )
        return _empty

    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["series_id", "timestamp"]).reset_index(drop=True)
    logger.info(
        "[OK] fred: fetched %d rows for %d series.",
        len(result),
        result["series_id"].nunique(),
    )
    return result
