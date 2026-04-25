"""CoinMetrics Community API — free crypto-macro signals.

From 10_FREE_DATEN.md §10.11.
Rate limit: 10 req/6s sliding, 1000 req/10min.

Key signals:
  - Stablecoin supply (USDT+USDC) — liquidity proxy
  - Exchange net flows — risk appetite
  - Active addresses — adoption momentum

Install: pip install coinmetrics-api-client
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

_STABLECOIN_ASSETS = ["usdt", "usdc"]
_COMMUNITY_METRICS = [
    "SplyAct180d",  # Active supply 180d (activity proxy)
    "NTVAdj",       # Adjusted transfer value (network activity)
    "FlowInExUSD",  # Exchange inflow USD
    "FlowOutExUSD", # Exchange outflow USD
    "AdrActCnt",    # Active address count
]


def _try_client():
    try:
        from coinmetrics.api_client import CoinMetricsClient
        return CoinMetricsClient()
    except ImportError:
        logger.warning("coinmetrics-api-client not installed — pip install coinmetrics-api-client")
        return None


def fetch_stablecoin_supply(
    start: str | date | None = None,
    end: str | date | None = None,
) -> pd.DataFrame:
    """Fetch combined USDT+USDC supply as liquidity proxy.

    Returns DataFrame with columns: date, usdt_supply, usdc_supply, total_supply.
    """
    client = _try_client()
    if client is None:
        return pd.DataFrame()

    if start is None:
        start = str(date.today() - timedelta(days=365))
    if end is None:
        end = str(date.today())

    dfs = []
    for asset in _STABLECOIN_ASSETS:
        try:
            df = client.get_asset_metrics(
                assets=asset,
                metrics=["SplyCur"],
                start_time=str(start),
                end_time=str(end),
                frequency="1d",
            ).to_dataframe()
            df = df[["time", "SplyCur"]].rename(columns={"time": "date", "SplyCur": f"{asset}_supply"})
            df["date"] = pd.to_datetime(df["date"]).dt.date
            dfs.append(df.set_index("date"))
        except Exception as exc:
            logger.debug("CoinMetrics %s supply failed: %s", asset, exc)

    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, axis=1)
    result["total_stablecoin_supply"] = result.sum(axis=1)
    return result.reset_index()


def fetch_exchange_net_flows(
    asset: str = "btc",
    start: str | date | None = None,
    end: str | date | None = None,
) -> pd.DataFrame:
    """Fetch exchange net flows (inflow - outflow) as risk-appetite proxy.

    Positive = crypto flowing INTO exchanges (selling pressure).
    Negative = crypto flowing OUT of exchanges (accumulation/risk-on).
    """
    client = _try_client()
    if client is None:
        return pd.DataFrame()

    if start is None:
        start = str(date.today() - timedelta(days=90))
    if end is None:
        end = str(date.today())

    try:
        df = client.get_asset_metrics(
            assets=asset,
            metrics=["FlowInExNtv", "FlowOutExNtv"],
            start_time=str(start),
            end_time=str(end),
            frequency="1d",
        ).to_dataframe()
        df["date"] = pd.to_datetime(df["time"]).dt.date
        df["net_flow"] = df["FlowInExNtv"] - df["FlowOutExNtv"]
        return df[["date", "FlowInExNtv", "FlowOutExNtv", "net_flow"]].set_index("date")
    except Exception as exc:
        logger.debug("CoinMetrics exchange flows failed: %s", exc)
        return pd.DataFrame()


def fetch_active_addresses(
    asset: str = "btc",
    start: str | date | None = None,
    end: str | date | None = None,
) -> pd.DataFrame:
    """Fetch active address count as adoption momentum signal."""
    client = _try_client()
    if client is None:
        return pd.DataFrame()

    if start is None:
        start = str(date.today() - timedelta(days=180))
    if end is None:
        end = str(date.today())

    try:
        df = client.get_asset_metrics(
            assets=asset,
            metrics=["AdrActCnt"],
            start_time=str(start),
            end_time=str(end),
            frequency="1d",
        ).to_dataframe()
        df["date"] = pd.to_datetime(df["time"]).dt.date
        return df[["date", "AdrActCnt"]].rename(columns={"AdrActCnt": "active_addresses"}).set_index("date")
    except Exception as exc:
        logger.debug("CoinMetrics active addresses failed: %s", exc)
        return pd.DataFrame()


def crypto_macro_features(lookback_days: int = 90) -> dict[str, float]:
    """Compute composite crypto-macro features for Breadth/Intermarket dimension.

    Returns dict with:
      stablecoin_supply_30d_change_pct: supply growth (positive = liquidity entering)
      exchange_net_flow_7d_avg: average net flow (negative = accumulation/risk-on)
      active_addr_30d_change_pct: address growth momentum
    """
    result: dict[str, float] = {}

    # Stablecoin supply growth
    df_supply = fetch_stablecoin_supply()
    if not df_supply.empty and "total_stablecoin_supply" in df_supply.columns:
        supply = df_supply["total_stablecoin_supply"].dropna()
        if len(supply) >= 30:
            change = (supply.iloc[-1] - supply.iloc[-30]) / (supply.iloc[-30] + 1e-9)
            result["stablecoin_supply_30d_change_pct"] = float(change)

    # Exchange net flows
    df_flows = fetch_exchange_net_flows()
    if not df_flows.empty and "net_flow" in df_flows.columns:
        flows = df_flows["net_flow"].dropna()
        if len(flows) >= 7:
            result["exchange_net_flow_7d_avg"] = float(flows.iloc[-7:].mean())

    # Active addresses growth
    df_addr = fetch_active_addresses()
    if not df_addr.empty and "active_addresses" in df_addr.columns:
        addrs = df_addr["active_addresses"].dropna()
        if len(addrs) >= 30:
            change = (addrs.iloc[-1] - addrs.iloc[-30]) / (addrs.iloc[-30] + 1e-9)
            result["active_addr_30d_change_pct"] = float(change)

    return result


__all__ = [
    "fetch_stablecoin_supply",
    "fetch_exchange_net_flows",
    "fetch_active_addresses",
    "crypto_macro_features",
]
