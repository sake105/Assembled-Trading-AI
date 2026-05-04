"""CoinGecko OHLC data source.

Fetches OHLCV-style candlestick data from the CoinGecko public API.
No API key required for the free tier (rate limited ~30 req/min).

API reference: https://www.coingecko.com/api/documentation

Usage::

    from assembled_core.data.crypto import fetch_coingecko_ohlc

    df = fetch_coingecko_ohlc(
        coin_ids=["bitcoin", "ethereum", "solana"],
        days=90,
        vs_currency="usd",
    )
    # Returns long-format DataFrame with columns:
    # [timestamp, symbol, open, high, low, close, volume]
"""

from __future__ import annotations

import logging
import time

import pandas as pd

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.coingecko.com/api/v3"

_EMPTY = pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close"])

# Mapping from CoinGecko coin_id to ticker symbol for convenience
COIN_SYMBOLS: dict[str, str] = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
    "binancecoin": "BNB",
    "ripple": "XRP",
    "cardano": "ADA",
    "avalanche-2": "AVAX",
    "polkadot": "DOT",
    "chainlink": "LINK",
    "litecoin": "LTC",
}

# CoinGecko free tier: max days per granularity bucket
# < 2 days → hourly, 2–90 days → 4-hourly, > 90 days → daily
_MAX_FREE_DAYS = 365


def fetch_coingecko_ohlc(
    coin_ids: list[str],
    days: int = 90,
    vs_currency: str = "usd",
    *,
    request_delay: float = 2.0,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch OHLC candlestick data from CoinGecko for one or more coins.

    Args:
        coin_ids:      CoinGecko coin IDs (e.g. ["bitcoin", "ethereum"]).
                       Use ``list_coins()`` to discover valid IDs.
        days:          Number of days of history to fetch (max 365 free tier).
        vs_currency:   Quote currency (default: "usd").
        request_delay: Seconds to wait between per-coin requests to respect
                       free-tier rate limit (~30 req/min).
        timeout:       HTTP timeout in seconds.

    Returns:
        Long-format DataFrame with columns:
            timestamp (datetime64[ns]), symbol (str), open, high, low, close (float).
        Volume column is absent — CoinGecko OHLC endpoint does not return volume;
        use ``fetch_coingecko_market_chart()`` for volume data.
        Empty DataFrame on error or if no data available.
    """
    try:
        import requests  # noqa: PLC0415
    except ImportError:
        logger.error("[ERROR] requests not installed.")
        return _EMPTY.copy()

    if not coin_ids:
        return _EMPTY.copy()

    days = min(max(1, days), _MAX_FREE_DAYS)
    all_frames: list[pd.DataFrame] = []

    for i, coin_id in enumerate(coin_ids):
        if i > 0:
            time.sleep(request_delay)
        frame = _fetch_ohlc_single(coin_id, days, vs_currency, requests, timeout)
        if frame is not None and not frame.empty:
            all_frames.append(frame)

    if not all_frames:
        logger.warning("[WARN] coingecko: no OHLC data returned for any coin.")
        return _EMPTY.copy()

    result = pd.concat(all_frames, ignore_index=True)
    result = result.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    logger.info(
        "[OK] coingecko: %d rows, %d coins, %s → %s.",
        len(result),
        result["symbol"].nunique(),
        result["timestamp"].min().date() if not result.empty else "n/a",
        result["timestamp"].max().date() if not result.empty else "n/a",
    )
    return result


def _fetch_ohlc_single(
    coin_id: str,
    days: int,
    vs_currency: str,
    requests: object,
    timeout: int,
) -> pd.DataFrame | None:
    """Fetch OHLC for a single coin from CoinGecko."""
    url = f"{_BASE_URL}/coins/{coin_id}/ohlc"
    params = {"vs_currency": vs_currency, "days": days}

    try:
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 404:
            logger.debug("[SKIP] coingecko: coin_id=%s not found (404).", coin_id)
            return None
        if resp.status_code == 429:
            logger.warning("[WARN] coingecko: rate limit hit for %s.", coin_id)
            return None
        resp.raise_for_status()
        raw = resp.json()
    except Exception as exc:
        logger.warning("[WARN] coingecko: request failed for %s — %s.", coin_id, exc)
        return None

    if not isinstance(raw, list) or not raw:
        logger.debug("[SKIP] coingecko: empty or invalid response for %s.", coin_id)
        return None

    # CoinGecko OHLC: [[timestamp_ms, open, high, low, close], ...]
    try:
        df = pd.DataFrame(raw, columns=["timestamp_ms", "open", "high", "low", "close"])
    except Exception as exc:
        logger.warning("[WARN] coingecko: parse failed for %s — %s.", coin_id, exc)
        return None

    df["timestamp"] = pd.to_datetime(
        df["timestamp_ms"], unit="ms", utc=True
    ).dt.tz_localize(None)
    symbol = COIN_SYMBOLS.get(coin_id, coin_id.upper()[:6])
    df["symbol"] = symbol

    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    return df[["timestamp", "symbol", "open", "high", "low", "close"]].reset_index(
        drop=True
    )


def fetch_coingecko_market_chart(
    coin_id: str,
    days: int = 90,
    vs_currency: str = "usd",
    *,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch price + volume time series for a single coin (daily granularity).

    This endpoint returns daily closes and volume — use when volume data is needed.

    Args:
        coin_id:    CoinGecko coin ID.
        days:       Days of history (max 365 free tier).
        vs_currency: Quote currency.
        timeout:    HTTP timeout in seconds.

    Returns:
        DataFrame with columns: [timestamp, close, volume].
    """
    try:
        import requests  # noqa: PLC0415
    except ImportError:
        logger.error("[ERROR] requests not installed.")
        return pd.DataFrame(columns=["timestamp", "close", "volume"])

    url = f"{_BASE_URL}/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days, "interval": "daily"}

    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning(
            "[WARN] coingecko market_chart: request failed for %s — %s.", coin_id, exc
        )
        return pd.DataFrame(columns=["timestamp", "close", "volume"])

    prices = data.get("prices", [])
    volumes = data.get("total_volumes", [])

    if not prices:
        return pd.DataFrame(columns=["timestamp", "close", "volume"])

    price_df = pd.DataFrame(prices, columns=["timestamp_ms", "close"])
    price_df["timestamp"] = pd.to_datetime(
        price_df["timestamp_ms"], unit="ms", utc=True
    ).dt.tz_localize(None)

    if volumes:
        vol_df = pd.DataFrame(volumes, columns=["timestamp_ms", "volume"])
        vol_df["timestamp"] = pd.to_datetime(
            vol_df["timestamp_ms"], unit="ms", utc=True
        ).dt.tz_localize(None)
        result = pd.merge(
            price_df[["timestamp", "close"]],
            vol_df[["timestamp", "volume"]],
            on="timestamp",
            how="left",
        )
    else:
        result = price_df[["timestamp", "close"]].copy()
        result["volume"] = float("nan")

    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result["symbol"] = COIN_SYMBOLS.get(coin_id, coin_id.upper()[:6])
    result = result.sort_values("timestamp").reset_index(drop=True)
    logger.info("[OK] coingecko market_chart: %d rows for %s.", len(result), coin_id)
    return result


def list_coins(*, timeout: int = 15) -> pd.DataFrame:
    """Fetch the full CoinGecko coin list (id, symbol, name).

    Returns:
        DataFrame with columns: [id, symbol, name].
        Useful for discovering valid coin_ids.
    """
    try:
        import requests  # noqa: PLC0415
    except ImportError:
        logger.error("[ERROR] requests not installed.")
        return pd.DataFrame(columns=["id", "symbol", "name"])

    url = f"{_BASE_URL}/coins/list"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("[WARN] coingecko list_coins: request failed — %s.", exc)
        return pd.DataFrame(columns=["id", "symbol", "name"])

    df = pd.DataFrame(data, columns=["id", "symbol", "name"])
    logger.info("[OK] coingecko list_coins: %d coins.", len(df))
    return df


__all__ = [
    "COIN_SYMBOLS",
    "fetch_coingecko_ohlc",
    "fetch_coingecko_market_chart",
    "list_coins",
]
