"""Fundamental Factor Features (M17).

Computes Barra-style fundamental factors from market data and financial
statements. These factors feed into the factor risk model and can be used
as alpha signals in multi-factor strategies.

Factors:
    1. carry_dividend_yield   — trailing 12-month dividend yield
    2. value_book_to_market   — book value / market cap (value factor)
    3. quality_gross_profit    — gross profit / total assets (profitability)
    4. quality_roe             — return on equity
    5. size_log_market_cap     — log10(market capitalization)
    6. value_earnings_yield    — trailing P/E inverted (E/P)

Data sources:
    - yfinance .info API for most fundamental data
    - Cached to avoid repeated API calls within the same session

Usage:
    from src.assembled_core.features.fundamental_factors import (
        build_fundamental_factors,
    )

    factors_df = build_fundamental_factors(["AAPL", "MSFT", "GOOG"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Module-level cache to avoid redundant yfinance calls within a session
_info_cache: dict[str, dict[str, Any]] = {}


@dataclass
class FundamentalFactorResult:
    """Result of fundamental factor computation.

    Attributes:
        factors: DataFrame with columns [symbol, carry_dividend_yield,
            value_book_to_market, quality_gross_profit, quality_roe,
            size_log_market_cap, value_earnings_yield].
        coverage: Fraction of symbols with valid data.
        errors: Symbols that failed to fetch.
    """

    factors: pd.DataFrame
    coverage: float
    errors: list[str]


FUNDAMENTAL_COLUMNS = [
    "carry_dividend_yield",
    "value_book_to_market",
    "quality_gross_profit",
    "quality_roe",
    "size_log_market_cap",
    "value_earnings_yield",
    "net_stock_issuance",
    "asset_growth",
]


def _fetch_info(symbol: str) -> dict[str, Any]:
    """Fetch yfinance info for a symbol with caching.

    Returns empty dict on failure.
    """
    if symbol in _info_cache:
        return _info_cache[symbol]

    try:
        import yfinance as yf  # type: ignore[import]

        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        _info_cache[symbol] = info
        return info
    except Exception as exc:
        logger.debug("[Fundamentals] Failed to fetch info for %s: %s", symbol, exc)
        _info_cache[symbol] = {}
        return {}


def _safe_get(info: dict, key: str, default: float = np.nan) -> float:
    """Safely extract a numeric value from yfinance info dict."""
    val = info.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def compute_single_symbol_factors(
    symbol: str,
    info: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute fundamental factors for a single symbol.

    Args:
        symbol: Ticker symbol.
        info: Pre-fetched yfinance info dict. If None, fetches from API.

    Returns:
        Dict with factor names -> values.
    """
    if info is None:
        info = _fetch_info(symbol)

    # Carry: dividend yield
    div_yield = _safe_get(info, "dividendYield", 0.0)
    # Some stocks report as percentage, normalize
    if div_yield > 1.0:
        div_yield = div_yield / 100.0

    # Value: book-to-market
    book_value = _safe_get(info, "bookValue")
    price = _safe_get(info, "currentPrice") or _safe_get(info, "previousClose")
    book_to_market = book_value / price if price and price > 0 and not np.isnan(book_value) else np.nan

    # Quality: gross profitability (gross profit / total assets)
    gross_profit = _safe_get(info, "grossProfits")
    total_assets = _safe_get(info, "totalAssets")
    gross_profitability = (
        gross_profit / total_assets
        if total_assets and total_assets > 0 and not np.isnan(gross_profit)
        else np.nan
    )

    # Quality: ROE (return on equity)
    roe = _safe_get(info, "returnOnEquity")

    # Size: log market cap
    market_cap = _safe_get(info, "marketCap")
    log_market_cap = np.log10(market_cap) if market_cap and market_cap > 0 else np.nan

    # Value: earnings yield (1/PE = E/P)
    pe_ratio = _safe_get(info, "trailingPE")
    earnings_yield = 1.0 / pe_ratio if pe_ratio and pe_ratio > 0 else np.nan

    # Net Stock Issuance (Loughran & Ritter 1995, Task 17.4)
    # share_change = shares_t / shares_t-4 - 1
    # Short diluters (capital raises), Long buyback firms
    shares_outstanding = _safe_get(info, "sharesOutstanding")
    # yfinance doesn't provide historical shares directly; use floatShares as proxy
    float_shares = _safe_get(info, "floatShares")
    # Approximate net issuance from implied buyback yield (negative = net buybacks)
    # If both available, compute ratio; otherwise NaN
    if (not np.isnan(shares_outstanding) and shares_outstanding > 0
            and not np.isnan(float_shares) and float_shares > 0):
        # Use buyback yield if available, else approximate from shares vs float
        buyback_yield = _safe_get(info, "buybackYield", np.nan)
        if np.isnan(buyback_yield):
            # Proxy: if float < shares, some are restricted/treasury → slight issuance signal
            net_stock_issuance = (shares_outstanding - float_shares) / shares_outstanding
        else:
            net_stock_issuance = -buyback_yield  # positive = dilution
    else:
        net_stock_issuance = np.nan

    # Asset Growth Anomaly (Cooper, Gulen & Schill 2008, Task 17.5)
    # asset_growth = total_assets_t / total_assets_t-4 - 1
    # Short fast-growers (empire builders), Long slow-growers
    # yfinance provides current totalAssets; for growth we use revenue growth as proxy
    revenue_growth = _safe_get(info, "revenueGrowth")
    earnings_growth = _safe_get(info, "earningsGrowth")
    # Approximate asset growth from revenue growth (strong positive correlation)
    if not np.isnan(revenue_growth):
        asset_growth = revenue_growth  # proxy: revenue growth ≈ asset growth direction
    elif not np.isnan(earnings_growth):
        asset_growth = earnings_growth * 0.7  # weaker proxy
    else:
        asset_growth = np.nan

    return {
        "carry_dividend_yield": div_yield,
        "value_book_to_market": book_to_market,
        "quality_gross_profit": gross_profitability,
        "quality_roe": roe,
        "size_log_market_cap": log_market_cap,
        "value_earnings_yield": earnings_yield,
        "net_stock_issuance": net_stock_issuance,
        "asset_growth": asset_growth,
    }


def build_fundamental_factors(
    symbols: list[str],
    info_dict: dict[str, dict[str, Any]] | None = None,
) -> FundamentalFactorResult:
    """Build fundamental factor panel for a list of symbols.

    Args:
        symbols: List of ticker symbols.
        info_dict: Optional pre-fetched {symbol: info} dict. If None,
            fetches from yfinance API.

    Returns:
        FundamentalFactorResult with factors DataFrame and diagnostics.
    """
    rows = []
    errors = []

    for sym in symbols:
        try:
            info = info_dict.get(sym, {}) if info_dict else None
            factors = compute_single_symbol_factors(sym, info)
            factors["symbol"] = sym
            rows.append(factors)
        except Exception as exc:
            logger.warning("[Fundamentals] Error computing factors for %s: %s", sym, exc)
            errors.append(sym)

    if not rows:
        return FundamentalFactorResult(
            factors=pd.DataFrame(columns=["symbol"] + FUNDAMENTAL_COLUMNS),
            coverage=0.0,
            errors=errors,
        )

    df = pd.DataFrame(rows)
    # Reorder columns
    cols = ["symbol"] + [c for c in FUNDAMENTAL_COLUMNS if c in df.columns]
    df = df[cols]

    # Compute coverage: fraction of symbols with at least 3 non-NaN factors
    valid_per_row = df[FUNDAMENTAL_COLUMNS].notna().sum(axis=1)
    coverage = float((valid_per_row >= 3).mean())

    logger.info(
        "[Fundamentals] Computed factors for %d/%d symbols (coverage=%.0f%%, errors=%d)",
        len(rows), len(symbols), coverage * 100, len(errors),
    )

    return FundamentalFactorResult(
        factors=df,
        coverage=coverage,
        errors=errors,
    )


def cross_sectional_zscore(
    factors_df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Z-score normalize fundamental factors cross-sectionally.

    Winsorizes at 3 sigma before z-scoring to limit outlier influence.

    Args:
        factors_df: DataFrame with symbol column and factor columns.
        columns: Columns to normalize. Defaults to FUNDAMENTAL_COLUMNS.

    Returns:
        DataFrame with z-scored factor values.
    """
    cols = columns or [c for c in FUNDAMENTAL_COLUMNS if c in factors_df.columns]
    df = factors_df.copy()

    for col in cols:
        vals = df[col].astype(float)
        if vals.notna().sum() < 3:
            continue
        mean = vals.mean()
        std = vals.std()
        if std < 1e-10:
            df[col] = 0.0
            continue
        # Winsorize at 3 sigma
        lower = mean - 3 * std
        upper = mean + 3 * std
        vals = vals.clip(lower, upper)
        df[col] = (vals - vals.mean()) / vals.std()

    return df


def clear_cache() -> None:
    """Clear the module-level info cache."""
    _info_cache.clear()


__all__ = [
    "FundamentalFactorResult",
    "FUNDAMENTAL_COLUMNS",
    "compute_single_symbol_factors",
    "build_fundamental_factors",
    "cross_sectional_zscore",
    "clear_cache",
]
