"""ECB EUR exchange rate client.

Fetches daily EUR FX rates from the European Central Bank's public SDMX-JSON
API.  No API key required.

ECB API reference: https://data-api.ecb.europa.eu/help/

Supported series pattern: ``EXR.D.<CURRENCY>.EUR.SP00.A``

Usage::

    from assembled_core.data.fx import fetch_ecb_fx_rates

    df = fetch_ecb_fx_rates(
        currencies=["USD", "GBP", "JPY", "CHF"],
        start_date="2020-01-01",
        end_date="2024-12-31",
    )
    # Returns DataFrame with columns: [date, currency, rate]
    # rate = units of <currency> per 1 EUR
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from types import ModuleType

import pandas as pd

logger = logging.getLogger(__name__)

_BASE_URL = "https://data-api.ecb.europa.eu/service/data"
_DATASET = "EXR"

_EMPTY = pd.DataFrame(columns=["date", "currency", "rate"])

# Commonly available ECB FX currencies (EUR base)
AVAILABLE_CURRENCIES = [
    "USD",
    "GBP",
    "JPY",
    "CHF",
    "AUD",
    "CAD",
    "CNY",
    "SEK",
    "NOK",
    "DKK",
    "HKD",
    "SGD",
    "KRW",
    "MXN",
    "BRL",
    "INR",
    "ZAR",
    "TRY",
    "RUB",
    "PLN",
    "CZK",
    "HUF",
]


def fetch_ecb_fx_rates(
    currencies: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch daily EUR FX rates from the ECB public API.

    Args:
        currencies: Currency codes to fetch (e.g. ["USD", "GBP"]).
                    Defaults to a broad set of major currencies.
        start_date: Start date as "YYYY-MM-DD". Defaults to 1 year ago.
        end_date:   End date as "YYYY-MM-DD". Defaults to today.
        timeout:    HTTP timeout in seconds.

    Returns:
        DataFrame with columns:
            date (datetime64[ns]), currency (str), rate (float).
        rate = units of <currency> per 1 EUR.
        Empty DataFrame on error or if no data available.
    """
    try:
        import requests  # noqa: PLC0415
    except ImportError:
        logger.error("[ERROR] requests not installed.")
        return _EMPTY.copy()

    if currencies is None:
        currencies = ["USD", "GBP", "JPY", "CHF", "AUD", "CAD", "CNY", "SEK"]

    currencies = [c.upper() for c in currencies]

    if start_date is None:
        start_date = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    all_frames: list[pd.DataFrame] = []
    for currency in currencies:
        frame = _fetch_single_currency(
            currency, start_date, end_date, requests, timeout
        )
        if frame is not None and not frame.empty:
            all_frames.append(frame)

    if not all_frames:
        logger.warning("[WARN] ecb_fx: no data returned for any currency.")
        return _EMPTY.copy()

    result = pd.concat(all_frames, ignore_index=True)
    result = result.sort_values(["date", "currency"]).reset_index(drop=True)
    logger.info(
        "[OK] ecb_fx: %d rows, %d currencies, %s → %s.",
        len(result),
        result["currency"].nunique(),
        result["date"].min().date() if not result.empty else "n/a",
        result["date"].max().date() if not result.empty else "n/a",
    )
    return result


def _fetch_single_currency(
    currency: str,
    start_date: str,
    end_date: str,
    requests: ModuleType,
    timeout: int,
) -> pd.DataFrame | None:
    """Fetch ECB SDMX-JSON data for a single currency pair."""
    # Series key: EXR.D.<CURRENCY>.EUR.SP00.A
    series_key = f"D.{currency}.EUR.SP00.A"
    url = f"{_BASE_URL}/{_DATASET}/{series_key}"
    params = {
        "format": "csvdata",
        "startPeriod": start_date,
        "endPeriod": end_date,
        "detail": "dataonly",
    }

    try:
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 404:
            logger.debug("[SKIP] ecb_fx: currency %s not found (404).", currency)
            return None
        resp.raise_for_status()
        text = resp.text
    except Exception as exc:
        logger.warning("[WARN] ecb_fx: request failed for %s — %s.", currency, exc)
        return None

    if not text or not text.strip():
        logger.debug("[SKIP] ecb_fx: empty response for %s.", currency)
        return None

    try:
        from io import StringIO  # noqa: PLC0415

        df = pd.read_csv(StringIO(text))
    except Exception as exc:
        logger.warning("[WARN] ecb_fx: CSV parse failed for %s — %s.", currency, exc)
        return None

    # ECB CSV has columns: KEY, FREQ, CURRENCY, CURRENCY_DENOM, EXR_TYPE, EXR_SUFFIX,
    # TIME_PERIOD, OBS_VALUE, ...
    date_col = next(
        (c for c in df.columns if "TIME" in c.upper() or "PERIOD" in c.upper()), None
    )
    value_col = next(
        (c for c in df.columns if "OBS_VALUE" in c.upper() or "VALUE" in c.upper()),
        None,
    )
    if date_col is None or value_col is None:
        logger.warning(
            "[WARN] ecb_fx: unexpected CSV columns for %s: %s.",
            currency,
            df.columns.tolist(),
        )
        return None

    df = df[[date_col, value_col]].copy()
    df.columns = ["date", "rate"]
    df = df.dropna(subset=["rate"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df = df.dropna(subset=["rate"])
    df["currency"] = currency

    return df[["date", "currency", "rate"]].reset_index(drop=True)


def fetch_ecb_fx_wide(
    currencies: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch ECB FX rates in wide format (date × currency columns).

    Returns:
        DataFrame indexed by date with one column per currency code.
        Values are units of that currency per 1 EUR.
    """
    long = fetch_ecb_fx_rates(
        currencies=currencies, start_date=start_date, end_date=end_date
    )
    if long.empty:
        return pd.DataFrame()

    wide = long.pivot(index="date", columns="currency", values="rate")
    wide.index.name = "date"
    wide.columns.name = None
    return wide


__all__ = [
    "AVAILABLE_CURRENCIES",
    "fetch_ecb_fx_rates",
    "fetch_ecb_fx_wide",
]
