"""CoinGecko free tier — Crypto-Korrelations- und Risk-On/Off-Signale.

Quelle
------
https://api.coingecko.com/api/v3 (free tier, ~10-50 Calls/Min, kein Key).

Anwendung
---------
1. **Risk-On/Off-Indikator**: BTC- und ETH-Renditen korrelieren in Stress-Phasen
   stark mit Tech-Aktien (Nasdaq). Crypto-Drawdowns sind oft Frühindikator.
2. **Cross-Asset-Momentum**: Crypto vs. SPY Spread-Momentum.
3. **Stablecoin-Marketcap-Trend**: USDT/USDC-MCap-Wachstum als Liquiditätsproxy.

Frei verfügbar via CoinGecko: keine Key-Pflicht für ``/coins/{id}/market_chart``.
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


@rate_limited(min_interval_s=1.5)
@retry_with_backoff(max_attempts=3, base_delay=3.0)
def _market_chart(coin_id: str, days: int = 365, vs: str = "usd") -> dict:
    import requests

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    r = requests.get(
        url,
        params={"vs_currency": vs, "days": days, "interval": "daily"},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()


def fetch_crypto_history(
    coin_ids: Sequence[str] = ("bitcoin", "ethereum", "tether", "usd-coin"),
    days: int = 365,
    use_cache: bool = True,
) -> FetchResult:
    """Hole tägliche Preis-, Marketcap-, Volume-Series.

    Returns:
        FetchResult mit DataFrame [date, coin, price, market_cap, total_volume].
    """
    cache_key = stable_hash("coingecko", tuple(sorted(coin_ids)), days)
    cache_path = get_cache_dir("coingecko") / f"{cache_key}.parquet"
    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        return FetchResult(df, "coingecko", pd.Timestamp.utcnow(), len(df), "cache")

    frames = []
    for cid in coin_ids:
        try:
            payload = _market_chart(cid, days=days)
        except Exception as e:  # noqa: BLE001
            logger.warning("[coingecko] %s skip: %s", cid, e)
            continue
        prices = payload.get("prices", [])
        mcaps = payload.get("market_caps", [])
        vols = payload.get("total_volumes", [])
        if not prices:
            continue
        df_c = pd.DataFrame(prices, columns=["ts_ms", "price"])
        df_c["market_cap"] = [m[1] for m in mcaps]
        df_c["total_volume"] = [v[1] for v in vols]
        df_c["date"] = pd.to_datetime(df_c["ts_ms"], unit="ms", utc=True).dt.normalize()
        df_c["coin"] = cid
        frames.append(df_c.drop(columns=["ts_ms"]))
    if not frames:
        return FetchResult(
            pd.DataFrame(), "coingecko", pd.Timestamp.utcnow(), 0, "empty"
        )
    df = pd.concat(frames, ignore_index=True)
    df = df[["date", "coin", "price", "market_cap", "total_volume"]].sort_values(
        ["coin", "date"]
    )
    if use_cache:
        df.to_parquet(cache_path, index=False)
    return FetchResult(df, "coingecko", pd.Timestamp.utcnow(), len(df), "")


def crypto_risk_on_off_signal(df: pd.DataFrame, lookback: int = 14) -> pd.DataFrame:
    """Risk-On/Off-Signal aus Crypto-Marktdaten.

    Logik
    -----
    - BTC- und ETH-Returns je Tag.
    - Stablecoin-MCap-Wachstum (USDT+USDC) -> Liquiditätsproxy.
    - z-Score über lookback-Tage je Reihe.
    - Composite: ``-z(crypto_return) + z(stablecoin_growth)`` -> hoch = Risk-Off.
    """
    if df.empty:
        return pd.DataFrame()
    out = df.pivot_table(index="date", columns="coin", values="price")
    mc = df.pivot_table(index="date", columns="coin", values="market_cap")

    # Crypto returns
    btc_ret = out.get("bitcoin", pd.Series(dtype=float)).pct_change()
    eth_ret = out.get("ethereum", pd.Series(dtype=float)).pct_change()
    crypto_ret = pd.concat([btc_ret, eth_ret], axis=1).mean(axis=1)

    # Stablecoin MCap growth
    stable_mc = pd.DataFrame(
        {
            "tether": mc.get("tether", pd.Series(dtype=float)),
            "usd-coin": mc.get("usd-coin", pd.Series(dtype=float)),
        }
    ).sum(axis=1, skipna=True)
    stable_growth = stable_mc.pct_change()

    def _zr(s: pd.Series) -> pd.Series:
        return (s - s.rolling(lookback, min_periods=lookback // 2).mean()) / s.rolling(
            lookback, min_periods=lookback // 2
        ).std()

    z_ret = _zr(crypto_ret)
    z_stable = _zr(stable_growth)
    composite = (-z_ret).fillna(0) + z_stable.fillna(0)

    return pd.DataFrame(
        {
            "crypto_return": crypto_ret,
            "stable_growth": stable_growth,
            "z_crypto_return": z_ret,
            "z_stable_growth": z_stable,
            "risk_off_score": composite,
        }
    ).reset_index()


__all__ = ["fetch_crypto_history", "crypto_risk_on_off_signal"]
