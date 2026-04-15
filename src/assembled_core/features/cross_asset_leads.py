"""Cross-Asset Lead-Lag Signals (M26 Task 26.1).

Exploits documented lead-lag relationships between asset classes:
1. Bond -> Equity: Credit spreads & yield curve predict equity returns
2. Commodity -> Sector: Oil/copper/gold lead related equity sectors
3. FX -> ADR: Currency moves predict ADR-local price adjustment

Reference:
    Fama & French (1989), bond-stock predictability
    Driesprong et al. (2008), oil price changes and stock returns
    Gagnon & Karolyi (2010), ADR pricing dynamics
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CrossAssetSignal:
    """Cross-asset lead-lag signal output."""
    bond_equity_signal: float      # Credit/yield → equity direction
    commodity_sector_signal: float  # Commodity → sector rotation
    fx_adr_signal: float           # FX → ADR mispricing
    composite_signal: float        # Weighted combination
    confidence: float              # Signal confidence (0-1)


# Default sector-commodity mappings
SECTOR_COMMODITY_MAP = {
    "Energy": ["CL=F", "NG=F"],        # Crude oil, natural gas
    "Materials": ["GC=F", "HG=F"],      # Gold, copper
    "Industrials": ["HG=F"],            # Copper (economic bellwether)
    "Technology": ["HG=F", "SI=F"],     # Copper, silver (semiconductors)
    "Utilities": ["NG=F"],              # Natural gas
    "Consumer Staples": ["ZC=F", "ZS=F"],  # Corn, soybeans
    "Financials": ["GC=F"],             # Gold (risk barometer)
    "Health Care": [],
    "Communication Services": [],
    "Real Estate": [],
    "Consumer Discretionary": ["CL=F"],  # Oil affects consumer spending
}


def compute_bond_equity_signal(
    credit_spread: pd.Series,
    term_spread: pd.Series,
    equity_returns: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """Bond market leads equity market signal.

    Rising credit spreads (widening) → bearish for equities.
    Steepening yield curve → bullish for equities.

    Args:
        credit_spread: Investment-grade credit spread (e.g., BAA-AAA or HY OAS).
        term_spread: 10Y-2Y Treasury spread.
        equity_returns: Equity index returns for calibration.
        lookback: Rolling window for z-score.

    Returns:
        Signal series (-1 to +1).
    """
    # Z-score of credit spread change (inverted: widening = bearish)
    cs_change = credit_spread.diff(5)
    cs_z = -(cs_change - cs_change.rolling(lookback).mean()) / (cs_change.rolling(lookback).std() + 1e-8)

    # Z-score of term spread level (steepening = bullish)
    ts_z = (term_spread - term_spread.rolling(lookback).mean()) / (term_spread.rolling(lookback).std() + 1e-8)

    # Combined: 60% credit, 40% term structure
    signal = 0.6 * cs_z + 0.4 * ts_z
    signal = signal.clip(-2, 2) / 2  # Normalize to [-1, 1]

    return signal.fillna(0.0)


def compute_commodity_sector_signal(
    commodity_returns: pd.DataFrame,
    sector_returns: pd.DataFrame,
    sector_commodity_map: dict[str, list[str]] | None = None,
    lag_days: int = 5,
    lookback: int = 60,
) -> pd.DataFrame:
    """Commodity returns lead related sector returns.

    Args:
        commodity_returns: Daily returns for commodity futures.
        sector_returns: Daily returns for equity sectors.
        sector_commodity_map: Mapping of sector → relevant commodities.
        lag_days: Lead-lag offset in days.
        lookback: Rolling window for correlation.

    Returns:
        DataFrame of sector signals.
    """
    if sector_commodity_map is None:
        sector_commodity_map = SECTOR_COMMODITY_MAP

    signals = pd.DataFrame(index=sector_returns.index, columns=sector_returns.columns, dtype=float)
    signals[:] = 0.0

    for sector, commodities in sector_commodity_map.items():
        if sector not in sector_returns.columns:
            continue
        if not commodities:
            continue

        available = [c for c in commodities if c in commodity_returns.columns]
        if not available:
            continue

        # Average lagged commodity return
        lagged_ret = commodity_returns[available].shift(lag_days).mean(axis=1)

        # Rolling z-score of lagged commodity return
        z = (lagged_ret - lagged_ret.rolling(lookback).mean()) / (lagged_ret.rolling(lookback).std() + 1e-8)
        signals[sector] = z.clip(-2, 2) / 2

    return signals.fillna(0.0)


def compute_fx_adr_signal(
    fx_returns: pd.Series,
    adr_returns: pd.Series,
    local_returns: pd.Series | None = None,
    lookback: int = 20,
) -> pd.Series:
    """FX movement predicts ADR-local spread adjustment.

    When local currency weakens, ADRs should decline relative to local shares.
    Slow adjustment creates a tradeable signal.

    Args:
        fx_returns: Currency returns (positive = USD strengthening).
        adr_returns: ADR returns in USD.
        local_returns: Local market returns (optional, for spread).
        lookback: Rolling window.

    Returns:
        Signal series.
    """
    # FX momentum predicts ADR returns
    fx_momentum = fx_returns.rolling(5).sum()
    fx_z = (fx_momentum - fx_momentum.rolling(lookback).mean()) / (fx_momentum.rolling(lookback).std() + 1e-8)

    if local_returns is not None:
        # ADR premium/discount relative to local
        spread = adr_returns - local_returns - fx_returns
        spread_z = (spread.rolling(5).sum()) / (spread.rolling(lookback).std() + 1e-8)
        signal = -0.5 * fx_z + 0.5 * spread_z  # Mean-revert the spread
    else:
        signal = -fx_z  # Pure FX momentum → inverse ADR signal

    return signal.clip(-1, 1).fillna(0.0)


def build_cross_asset_signals(
    equity_returns: pd.DataFrame,
    bond_data: dict[str, pd.Series] | None = None,
    commodity_returns: pd.DataFrame | None = None,
    fx_returns: pd.DataFrame | None = None,
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build composite cross-asset signals for all stocks.

    Args:
        equity_returns: (T, N) stock returns.
        bond_data: {"credit_spread": ..., "term_spread": ...} (optional).
        commodity_returns: Commodity futures returns (optional).
        fx_returns: Currency returns (optional).
        sector_map: {ticker: sector} mapping (optional).

    Returns:
        DataFrame with cross_asset_signal per stock.
    """
    signals = pd.DataFrame(index=equity_returns.index, columns=equity_returns.columns, dtype=float)
    signals[:] = 0.0

    n_components = 0

    # Bond → Equity (market-wide signal)
    if bond_data and "credit_spread" in bond_data and "term_spread" in bond_data:
        market_ret = equity_returns.mean(axis=1)
        be_signal = compute_bond_equity_signal(
            bond_data["credit_spread"], bond_data["term_spread"], market_ret,
        )
        for col in signals.columns:
            signals[col] += be_signal
        n_components += 1
        logger.info("[CrossAsset] Bond-equity signal computed")

    # Commodity → Sector
    if commodity_returns is not None and sector_map:
        # Build sector returns from stock returns
        sector_groups = {}
        for ticker, sector in sector_map.items():
            if ticker in equity_returns.columns:
                sector_groups.setdefault(sector, []).append(ticker)
        sector_ret = pd.DataFrame({
            s: equity_returns[tickers].mean(axis=1)
            for s, tickers in sector_groups.items()
        })

        cs_signals = compute_commodity_sector_signal(commodity_returns, sector_ret)
        # Map back to individual stocks
        for ticker, sector in sector_map.items():
            if ticker in signals.columns and sector in cs_signals.columns:
                signals[ticker] += cs_signals[sector]
        n_components += 1
        logger.info("[CrossAsset] Commodity-sector signal computed")

    # FX → ADR (requires FX data and knowledge of which stocks are ADRs)
    if fx_returns is not None and not fx_returns.empty:
        # For simplicity: use broad FX index as market-level signal
        fx_avg = fx_returns.mean(axis=1) if isinstance(fx_returns, pd.DataFrame) else fx_returns
        market_ret = equity_returns.mean(axis=1)
        fx_signal = compute_fx_adr_signal(fx_avg, market_ret)
        for col in signals.columns:
            signals[col] += 0.3 * fx_signal  # Lower weight for broad FX
        n_components += 1
        logger.info("[CrossAsset] FX-equity signal computed")

    # Normalize
    if n_components > 0:
        signals = signals / n_components

    return signals


__all__ = [
    "CrossAssetSignal",
    "SECTOR_COMMODITY_MAP",
    "compute_bond_equity_signal",
    "compute_commodity_sector_signal",
    "compute_fx_adr_signal",
    "build_cross_asset_signals",
]
