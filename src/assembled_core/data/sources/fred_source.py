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
    """Return next available FRED key from the rotator pool, or None.

    Multi-key rotation (2026-05-22): the rotator pools FRED_API_KEY,
    FRED_API_KEY_2/3/..., and FRED_API_KEYS (comma-separated). Backward
    compat: when only FRED_API_KEY is set, the pool has 1 key and the
    rotator behaves identically to the prior single-key path. If the
    rotator import fails (defensive against test-only circular imports),
    falls back to direct env read.
    """
    try:
        from src.assembled_core.utils.api_key_rotator import get_rotator

        rotated = get_rotator().get_key("fred")
        if rotated:
            return rotated
    except Exception:  # noqa: BLE001
        pass
    key = os.environ.get("FRED_API_KEY", "").strip()
    return key if key else None


def _mark_429_if_applicable(key: str | None, exc_or_response: object) -> None:
    """If exc_or_response looks like a rate-limit signal, cool down `key`.

    Best-effort. Silently no-op if rotator import fails, key is None, or
    the signal is not a rate-limit pattern.
    """
    if not key:
        return
    try:
        from src.assembled_core.utils.api_key_rotator import (
            get_rotator,
            is_rate_limit_signal,
        )

        if is_rate_limit_signal(exc_or_response):
            # FRED limit is 120 req/min — short cooldown is enough to
            # rotate through the pool while a key recovers.
            get_rotator().mark_rate_limited("fred", key, cooldown_seconds=60.0)
    except Exception:  # noqa: BLE001
        pass


def _fetch_single_series(
    series_id: str,
    start_date: str,
    end_date: str,
    fred_client: object,
    api_key: str | None = None,
) -> tuple[pd.DataFrame | None, bool]:
    """Fetch one FRED series. Returns (df_or_None, was_rate_limited).

    On rate-limit signals, marks `api_key` cooled-down so subsequent
    fetches in the same run rotate to a different key from the pool.
    The bool tells the caller whether retry-with-rotated-key is justified
    (avoids wasting a second call on a missing-series or invalid-key
    failure that won't be fixed by switching keys).
    """
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
            return None, False

        df = raw.reset_index()
        df.columns = ["timestamp", "value"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.normalize()
        df["series_id"] = series_id
        df = df[["timestamp", "series_id", "value"]].copy()
        # Drop NaN observations (FRED sometimes returns them for missing release days)
        df = df.dropna(subset=["value"])
        return df, False

    except Exception as exc:  # noqa: BLE001
        from src.assembled_core.utils.api_key_rotator import is_rate_limit_signal

        rate_limited = is_rate_limit_signal(exc)
        if rate_limited:
            _mark_429_if_applicable(api_key, exc)
        logger.error("[ERROR] fred: failed to fetch series %s — %s", series_id, exc)
        return None, rate_limited


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
            df, rate_limited = _fetch_single_series(
                sid, start_date, end_date, fred, api_key=api_key
            )
            # Retry with a rotated key ONLY for rate-limit failures.
            # Missing-series / invalid-key / network errors won't benefit
            # from another key — retrying would waste a quota slot.
            if df is None and rate_limited:
                next_key = _get_api_key()
                if next_key and next_key != api_key:
                    api_key = next_key
                    try:
                        from fredapi import Fred  # noqa: PLC0415

                        fred = Fred(api_key=api_key)
                        df, _ = _fetch_single_series(
                            sid, start_date, end_date, fred, api_key=api_key
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "[WARN] fred: rotation retry failed for %s — %s", sid, exc
                        )
            # F-AKR2-10: do NOT cache None — let the next call retry.
            # Otherwise a transient failure blocks the series for 6h.
            if df is not None and not df.empty:
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
