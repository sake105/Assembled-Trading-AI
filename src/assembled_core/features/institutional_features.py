"""Institutional Holdings Features (M26 Task 26.3).

Extracts alpha signals from SEC 13F filings:
1. Institutional ownership level and changes
2. Smart money concentration (top holders)
3. Herding detection (many institutions moving same direction)
4. New position initiation / liquidation signals

Reference:
    Yan & Zhang (2009), institutional investors and equity returns
    Sias (2004), institutional herding
    Gompers & Metrick (2001), institutional investors and equity prices
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class InstitutionalSignal:
    """Institutional holdings signal for a single stock."""
    institutional_ownership_pct: float   # Total institutional ownership
    ownership_change: float              # Quarter-over-quarter change
    n_holders: int                       # Number of institutional holders
    holder_change: int                   # Change in number of holders
    concentration_hhi: float             # Herfindahl index of top holders
    smart_money_flow: float              # Net buying by top-performing funds
    herding_measure: float               # Fraction of institutions moving same direction
    new_positions: int                   # Number of new position initiations
    liquidations: int                    # Number of full liquidations


def compute_institutional_ownership(
    holdings_data: pd.DataFrame,
    market_cap: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute institutional ownership metrics from 13F data.

    Args:
        holdings_data: DataFrame with columns [date, ticker, holder_id, shares, value].
        market_cap: Market cap per ticker for ownership percentage.

    Returns:
        DataFrame with institutional features per (date, ticker).
    """
    if holdings_data.empty:
        return pd.DataFrame()

    results = []
    for (date, ticker), group in holdings_data.groupby(["date", "ticker"]):
        total_value = group["value"].sum()
        n_holders = len(group)

        # HHI concentration
        if total_value > 0:
            shares_pct = group["value"] / total_value
            hhi = float((shares_pct ** 2).sum())
        else:
            hhi = 0.0

        # Ownership percentage
        own_pct = 0.0
        if market_cap is not None and ticker in market_cap.index:
            mcap = market_cap.get(ticker, 0)
            if mcap > 0:
                own_pct = total_value / mcap

        results.append({
            "date": date,
            "ticker": ticker,
            "inst_ownership_pct": round(own_pct, 4),
            "inst_total_value": total_value,
            "inst_n_holders": n_holders,
            "inst_concentration_hhi": round(hhi, 6),
        })

    return pd.DataFrame(results)


def compute_ownership_changes(
    current_holdings: pd.DataFrame,
    previous_holdings: pd.DataFrame,
) -> pd.DataFrame:
    """Compute quarter-over-quarter ownership changes.

    Args:
        current_holdings: Current quarter holdings [ticker, holder_id, shares, value].
        previous_holdings: Previous quarter holdings.

    Returns:
        DataFrame with change metrics per ticker.
    """
    if current_holdings.empty or previous_holdings.empty:
        return pd.DataFrame()

    results = []
    tickers = set(current_holdings["ticker"].unique()) | set(previous_holdings["ticker"].unique())

    for ticker in tickers:
        curr = current_holdings[current_holdings["ticker"] == ticker]
        prev = previous_holdings[previous_holdings["ticker"] == ticker]

        curr_holders = set(curr["holder_id"].unique())
        prev_holders = set(prev["holder_id"].unique())

        new_positions = len(curr_holders - prev_holders)
        liquidations = len(prev_holders - curr_holders)
        continuing = curr_holders & prev_holders

        # Net buying/selling among continuing holders
        n_buyers = 0
        n_sellers = 0
        for holder in continuing:
            c_shares = curr[curr["holder_id"] == holder]["shares"].sum()
            p_shares = prev[prev["holder_id"] == holder]["shares"].sum()
            if c_shares > p_shares * 1.01:
                n_buyers += 1
            elif c_shares < p_shares * 0.99:
                n_sellers += 1

        # Herding: fraction moving in same direction
        total_active = n_buyers + n_sellers
        herding = abs(n_buyers - n_sellers) / max(total_active, 1)

        # Ownership change
        curr_total = curr["value"].sum()
        prev_total = prev["value"].sum()
        ownership_change = (curr_total - prev_total) / max(prev_total, 1)

        results.append({
            "ticker": ticker,
            "inst_ownership_change": round(ownership_change, 4),
            "inst_holder_change": len(curr_holders) - len(prev_holders),
            "inst_new_positions": new_positions,
            "inst_liquidations": liquidations,
            "inst_herding": round(herding, 4),
            "inst_n_buyers": n_buyers,
            "inst_n_sellers": n_sellers,
            "inst_buy_sell_ratio": round(n_buyers / max(n_sellers, 1), 4),
        })

    return pd.DataFrame(results)


def compute_smart_money_flow(
    holdings_data: pd.DataFrame,
    fund_performance: pd.DataFrame | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """Compute smart money flow from top-performing fund managers.

    Args:
        holdings_data: Current holdings [date, ticker, holder_id, shares, value].
        fund_performance: Fund returns DataFrame [holder_id, return] (optional).
        top_n: Number of top funds to consider as "smart money".

    Returns:
        DataFrame with smart money signal per ticker.
    """
    if holdings_data.empty:
        return pd.DataFrame()

    if fund_performance is not None and not fund_performance.empty:
        # Select top-performing funds
        top_funds = set(
            fund_performance.nlargest(top_n, "return")["holder_id"].values
        )
    else:
        # Fallback: use largest holders as proxy for smart money
        holder_aum = holdings_data.groupby("holder_id")["value"].sum()
        top_funds = set(holder_aum.nlargest(top_n).index)

    # Smart money holdings
    smart = holdings_data[holdings_data["holder_id"].isin(top_funds)]
    all_holdings = holdings_data.groupby("ticker")["value"].sum()
    smart_holdings = smart.groupby("ticker")["value"].sum()

    # Smart money concentration
    smart_pct = smart_holdings / (all_holdings + 1e-8)
    smart_pct = smart_pct.fillna(0)

    result = pd.DataFrame({
        "ticker": smart_pct.index,
        "smart_money_pct": smart_pct.round(4).values,
    })

    return result


def build_institutional_features(
    holdings_history: dict[str, pd.DataFrame],
    market_cap: pd.Series | None = None,
    price_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Build complete institutional features from holdings history.

    Args:
        holdings_history: {quarter_date: holdings_df} sorted chronologically.
        market_cap: Current market cap per ticker.
        price_dates: Trading dates for output alignment.

    Returns:
        DataFrame with all institutional features.
    """
    quarters = sorted(holdings_history.keys())
    if not quarters:
        return pd.DataFrame()

    all_features = []
    for i, quarter in enumerate(quarters):
        data = holdings_history[quarter]
        if data.empty:
            continue

        # Ownership metrics
        ownership = compute_institutional_ownership(data, market_cap)

        # Changes vs previous quarter
        if i > 0:
            prev_data = holdings_history[quarters[i - 1]]
            changes = compute_ownership_changes(data, prev_data)
            if not changes.empty and not ownership.empty:
                ownership = ownership.merge(changes, on="ticker", how="left")

        # Smart money
        smart = compute_smart_money_flow(data)
        if not smart.empty and not ownership.empty:
            ownership = ownership.merge(smart, on="ticker", how="left")

        all_features.append(ownership)

    if not all_features:
        return pd.DataFrame()

    result = pd.concat(all_features, ignore_index=True)
    result = result.fillna(0.0)

    logger.info("[Institutional] Built features for %d ticker-quarter pairs", len(result))
    return result


__all__ = [
    "InstitutionalSignal",
    "compute_institutional_ownership",
    "compute_ownership_changes",
    "compute_smart_money_flow",
    "build_institutional_features",
]
