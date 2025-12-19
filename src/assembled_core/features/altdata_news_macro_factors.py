"""Alt-Data Factors: News Sentiment and Macro Regime Indicators.

This module implements Phase B2 factors that transform news sentiment and macro-economic
data into time-series factors for factor analysis.

**Important:** 
- Price data comes from LocalParquetPriceDataSource (local Parquet files), NOT from Finnhub.
- News/Macro data comes from Finnhub API (via downloaded Parquet files in output/altdata/).
- Finnhub is used only for news, sentiment, and macro data, NOT for price/candle data.

**Factor Columns Generated:**

**News Sentiment Factors:**
- `news_sentiment_mean_{lookback_days}d`: Rolling mean of daily sentiment scores
- `news_sentiment_trend_{lookback_days}d`: Trend in sentiment (change over lookback window)
- `news_sentiment_shock_flag`: Binary flag (1 if sentiment change exceeds threshold)
- `news_sentiment_volume_{lookback_days}d`: Rolling mean of news volume

**Macro Regime Factors:**
- `macro_growth_regime`: Growth regime indicator (+1 = expansion, -1 = recession, 0 = neutral)
- `macro_inflation_regime`: Inflation regime indicator (+1 = high inflation, -1 = low/deflation, 0 = neutral)
- `macro_risk_aversion_proxy`: Risk-on/risk-off indicator based on macro conditions

All factors are computed per symbol and aligned with the price DataFrame timestamps.
Missing values (NaN) occur when no news/macro data is available for a given symbol/date.

Integration:
- Compatible with build_core_ta_factors() and other Phase A factors
- Can be merged with price DataFrame using timestamp & symbol
- Designed for use in Phase C1/C2 factor analysis workflows
- Macro factors are the same for all symbols on a given date (market-wide)
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_news_sentiment_factors(
    news_sentiment_daily: pd.DataFrame,
    prices: pd.DataFrame,
    lookback_days: int = 20,
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
    price_col: str = "close",
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build news sentiment factors from daily sentiment data and price data.
    
    Transforms daily news sentiment into time-series factors that can be used in factor analysis.
    Price data must come from LocalParquetPriceDataSource (local Parquet files), not from Finnhub.
    News sentiment data comes from Finnhub API (via downloaded Parquet files).
    
    Args:
        news_sentiment_daily: DataFrame with daily sentiment data (news_sentiment_daily_df data contract)
            Required columns: timestamp, sentiment_score, sentiment_volume
            Optional columns: symbol (if None, treated as market-wide sentiment)
        prices: DataFrame with price data (Panel format)
            Required columns: timestamp, symbol, close
            Optional columns: open, high, low, volume
        lookback_days: Rolling window for sentiment aggregation (default: 20)
        group_col: Column name for grouping (default: "symbol")
        timestamp_col: Column name for timestamp (default: "timestamp")
        price_col: Column name for price (default: "close")
    
    Returns:
        DataFrame with columns:
        - timestamp, symbol (from prices)
        - news_sentiment_mean_{lookback_days}d: Rolling mean of sentiment scores
        - news_sentiment_trend_{lookback_days}d: Trend in sentiment (slope over lookback window)
        - news_sentiment_shock_flag: Binary flag (1 if absolute sentiment change > threshold)
        - news_sentiment_volume_{lookback_days}d: Rolling mean of news volume
        
        Sorted by symbol, then timestamp.
        Factors are NaN where no sentiment data is available.
    
    Raises:
        KeyError: If required columns are missing
        ValueError: If DataFrames are empty or invalid
    """
    # Validate inputs
    required_price_cols = [timestamp_col, group_col, price_col]
    missing_price_cols = [col for col in required_price_cols if col not in prices.columns]
    if missing_price_cols:
        raise KeyError(
            f"Missing required columns in prices: {', '.join(missing_price_cols)}. "
            f"Available: {list(prices.columns)}"
        )
    
    required_sentiment_cols = [timestamp_col, "sentiment_score", "sentiment_volume"]
    missing_sentiment_cols = [col for col in required_sentiment_cols if col not in news_sentiment_daily.columns]
    if missing_sentiment_cols:
        raise KeyError(
            f"Missing required columns in news_sentiment_daily: {', '.join(missing_sentiment_cols)}. "
            f"Available: {list(news_sentiment_daily.columns)}"
        )
    
    if prices.empty:
        raise ValueError("prices DataFrame is empty")
    
    # Prepare prices DataFrame
    result = prices.copy()
    
    # Ensure timestamps are UTC-aware datetime
    if not pd.api.types.is_datetime64_any_dtype(result[timestamp_col]):
        result[timestamp_col] = pd.to_datetime(result[timestamp_col], utc=True)
    
    if not news_sentiment_daily.empty:
        if not pd.api.types.is_datetime64_any_dtype(news_sentiment_daily[timestamp_col]):
            news_sentiment_daily = news_sentiment_daily.copy()
            news_sentiment_daily[timestamp_col] = pd.to_datetime(news_sentiment_daily[timestamp_col], utc=True)
        
        # Point-in-time handling: ensure event_date / disclosure_date exist.
        # For daily sentiment panels we treat the daily timestamp as both.
        if "event_date" not in news_sentiment_daily.columns:
            news_sentiment_daily["event_date"] = news_sentiment_daily[timestamp_col].dt.normalize()
        if "disclosure_date" not in news_sentiment_daily.columns:
            news_sentiment_daily["disclosure_date"] = news_sentiment_daily["event_date"]
        
        # If as_of is provided, restrict to sentiment that was disclosed by as_of.
        if as_of is not None:
            as_of_ts = pd.to_datetime(as_of, utc=True)
            before_filter = len(news_sentiment_daily)
            news_sentiment_daily = news_sentiment_daily[
                news_sentiment_daily["disclosure_date"] <= as_of_ts.normalize()
            ].copy()
            if len(news_sentiment_daily) < before_filter:
                logger.debug(
                    "Filtered news_sentiment_daily by as_of=%s: %d -> %d",
                    as_of_ts,
                    before_filter,
                    len(news_sentiment_daily),
                )
    
    # Sort by symbol and timestamp
    result = result.sort_values([group_col, timestamp_col]).reset_index(drop=True)
    
    if news_sentiment_daily.empty:
        logger.warning("news_sentiment_daily is empty. Returning prices with NaN factors.")
        # Add empty factor columns
        result[f"news_sentiment_mean_{lookback_days}d"] = np.nan
        result[f"news_sentiment_trend_{lookback_days}d"] = np.nan
        result["news_sentiment_shock_flag"] = 0.0
        result[f"news_sentiment_volume_{lookback_days}d"] = np.nan
        return result
    
    # Helper function to compute trend (slope over rolling window)
    def compute_trend(series: pd.Series, window: int) -> pd.Series:
        """Compute rolling trend (slope) over window."""
        trends = []
        for i in range(len(series)):
            if i < window - 1:
                trends.append(np.nan)
            else:
                y = series.iloc[i - window + 1 : i + 1].values
                x = np.arange(len(y))
                if len(y) > 1 and not np.isnan(y).all():
                    slope = np.polyfit(x, y, 1)[0]
                    trends.append(slope)
                else:
                    trends.append(np.nan)
        return pd.Series(trends, index=series.index)
    
    # Handle market-wide sentiment (symbol=None or "__MARKET__")
    # If sentiment has symbol column, join per symbol; otherwise join market-wide to all symbols
    sentiment_has_symbol = "symbol" in news_sentiment_daily.columns
    
    # Process sentiment per symbol and market-wide separately
    sentiment_per_symbol = pd.DataFrame()
    sentiment_market = pd.DataFrame()
    
    if sentiment_has_symbol:
        # Filter per-symbol sentiment
        sentiment_per_symbol = news_sentiment_daily[
            news_sentiment_daily["symbol"].notna() &
            (news_sentiment_daily["symbol"] != "__MARKET__")
        ].copy()
        
        # Filter market-wide sentiment
        sentiment_market = news_sentiment_daily[
            (news_sentiment_daily["symbol"].isna()) |
            (news_sentiment_daily["symbol"] == "__MARKET__")
        ].copy()
    else:
        # No symbol column - treat as market-wide sentiment
        sentiment_market = news_sentiment_daily.copy()
    
    # Process per-symbol sentiment
    sentiment_factors_list = []
    if not sentiment_per_symbol.empty:
        sentiment_per_symbol = sentiment_per_symbol.sort_values([group_col, timestamp_col]).reset_index(drop=True)
        
        for symbol in sentiment_per_symbol[group_col].unique():
            symbol_sentiment = sentiment_per_symbol[
                sentiment_per_symbol[group_col] == symbol
            ].copy()
            symbol_sentiment = symbol_sentiment.sort_values(timestamp_col).reset_index(drop=True)
            
            # Rolling mean of sentiment
            symbol_sentiment[f"sentiment_mean_{lookback_days}d"] = (
                symbol_sentiment["sentiment_score"]
                .rolling(window=lookback_days, min_periods=1)
                .mean()
            )
            
            # Rolling trend
            symbol_sentiment[f"sentiment_trend_{lookback_days}d"] = compute_trend(
                symbol_sentiment["sentiment_score"], lookback_days
            )
            
            # Sentiment shock flag (absolute change > 1.5 std)
            sentiment_std = symbol_sentiment["sentiment_score"].rolling(
                window=lookback_days * 2, min_periods=lookback_days
            ).std()
            sentiment_change = symbol_sentiment["sentiment_score"].diff().abs()
            symbol_sentiment["sentiment_shock_flag"] = (
                (sentiment_change > sentiment_std * 1.5).astype(float)
            )
            
            # Rolling mean of volume
            symbol_sentiment[f"sentiment_volume_{lookback_days}d"] = (
                symbol_sentiment["sentiment_volume"]
                .rolling(window=lookback_days, min_periods=1)
                .mean()
            )
            
            sentiment_factors_list.append(symbol_sentiment)
    
    # Process market-wide sentiment
    if not sentiment_market.empty:
        sentiment_market = sentiment_market.sort_values(timestamp_col).reset_index(drop=True)
        
        # Compute rolling factors for market-wide sentiment
        sentiment_market[f"sentiment_mean_{lookback_days}d"] = (
            sentiment_market["sentiment_score"]
            .rolling(window=lookback_days, min_periods=1)
            .mean()
        )
        
        sentiment_market[f"sentiment_trend_{lookback_days}d"] = compute_trend(
            sentiment_market["sentiment_score"], lookback_days
        )
        
        sentiment_std = sentiment_market["sentiment_score"].rolling(
            window=lookback_days * 2, min_periods=lookback_days
        ).std()
        sentiment_change = sentiment_market["sentiment_score"].diff().abs()
        sentiment_market["sentiment_shock_flag"] = (
            (sentiment_change > sentiment_std * 1.5).astype(float)
        )
        
        sentiment_market[f"sentiment_volume_{lookback_days}d"] = (
            sentiment_market["sentiment_volume"]
            .rolling(window=lookback_days, min_periods=1)
            .mean()
        )
    
    # Merge sentiment factors to prices (per symbol first, then market-wide)
    result_list = []
    for symbol in result[group_col].unique():
        symbol_result = result[result[group_col] == symbol].copy()
        symbol_result = symbol_result.sort_values(timestamp_col).reset_index(drop=True)
        
        # First, try per-symbol sentiment
        symbol_sentiment = None
        if sentiment_factors_list:
            for sf in sentiment_factors_list:
                if not sf.empty and sf[group_col].iloc[0] == symbol:
                    symbol_sentiment = sf.copy()
                    break
        
        if symbol_sentiment is not None and not symbol_sentiment.empty:
            symbol_sentiment = symbol_sentiment.sort_values(timestamp_col).reset_index(drop=True)
            
            # Merge per-symbol sentiment
            symbol_result = pd.merge_asof(
                symbol_result,
                symbol_sentiment[[
                    timestamp_col,
                    f"sentiment_mean_{lookback_days}d",
                    f"sentiment_trend_{lookback_days}d",
                    "sentiment_shock_flag",
                    f"sentiment_volume_{lookback_days}d",
                ]],
                on=timestamp_col,
                direction="backward",
                allow_exact_matches=True,
            )
        
        # Then, merge market-wide sentiment (fills gaps or provides default)
        if not sentiment_market.empty:
            sentiment_market_sorted = sentiment_market.sort_values(timestamp_col).reset_index(drop=True)
            
            # Merge market-wide sentiment (only fill NaN values from per-symbol)
            for col in [f"sentiment_mean_{lookback_days}d", f"sentiment_trend_{lookback_days}d", 
                       "sentiment_shock_flag", f"sentiment_volume_{lookback_days}d"]:
                if col not in symbol_result.columns:
                    symbol_result[col] = np.nan
            
            # Fill NaN values with market-wide sentiment
            market_merged = pd.merge_asof(
                symbol_result[[timestamp_col]],
                sentiment_market_sorted[[
                    timestamp_col,
                    f"sentiment_mean_{lookback_days}d",
                    f"sentiment_trend_{lookback_days}d",
                    "sentiment_shock_flag",
                    f"sentiment_volume_{lookback_days}d",
                ]],
                on=timestamp_col,
                direction="backward",
                allow_exact_matches=True,
            )
            
            # Fill NaN values in symbol_result with market values
            for col in [f"sentiment_mean_{lookback_days}d", f"sentiment_trend_{lookback_days}d", 
                       "sentiment_shock_flag", f"sentiment_volume_{lookback_days}d"]:
                mask = symbol_result[col].isna()
                if mask.any():
                    symbol_result.loc[mask, col] = market_merged.loc[mask, col].values
        
        result_list.append(symbol_result)
    
    if result_list:
        result = pd.concat(result_list, ignore_index=True)
    
    # Rename columns to match expected output format
    result = result.rename(columns={
        f"sentiment_mean_{lookback_days}d": f"news_sentiment_mean_{lookback_days}d",
        f"sentiment_trend_{lookback_days}d": f"news_sentiment_trend_{lookback_days}d",
        "sentiment_shock_flag": "news_sentiment_shock_flag",
        f"sentiment_volume_{lookback_days}d": f"news_sentiment_volume_{lookback_days}d",
    })
    
    # Ensure all factor columns exist (fill with NaN if missing)
    factor_cols = [
        f"news_sentiment_mean_{lookback_days}d",
        f"news_sentiment_trend_{lookback_days}d",
        "news_sentiment_shock_flag",
        f"news_sentiment_volume_{lookback_days}d",
    ]
    for col in factor_cols:
        if col not in result.columns:
            result[col] = np.nan
    
    # Sort by symbol, then timestamp
    result = result.sort_values([group_col, timestamp_col]).reset_index(drop=True)
    
    logger.info(
        f"Built news sentiment factors for {len(result[group_col].unique())} symbols, "
        f"{len(result)} rows. Lookback window: {lookback_days} days."
    )
    
    # Optional PIT safety check (only in strict QA mode)
    import os
    if os.getenv("ASSEMBLED_STRICT_PIT_CHECKS", "false").lower() == "true":
        from src.assembled_core.qa.point_in_time_checks import validate_feature_builder_pit_safe
        validate_feature_builder_pit_safe(
            features_df=result,
            as_of=as_of,
            builder_name="build_news_sentiment_factors",
            strict=True,
        )
    
    return result


def build_macro_regime_factors(
    macro_series: pd.DataFrame,
    prices: pd.DataFrame,
    country_filter: str | None = None,
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
    price_col: str = "close",
) -> pd.DataFrame:
    """Build macro regime factors from macro-economic indicators and price data.
    
    Transforms macro-economic indicators into regime indicators that can be used in factor analysis
    and risk modeling (Phase D). Price data must come from LocalParquetPriceDataSource (local Parquet files),
    not from Finnhub. Macro data comes from Finnhub API (via downloaded Parquet files).
    
    Args:
        macro_series: DataFrame with macro indicators (macro_series_df data contract)
            Required columns: timestamp, macro_code, value, country
            Optional columns: indicator_name, unit, previous_value, forecast_value
        prices: DataFrame with price data (Panel format)
            Required columns: timestamp, symbol, close
            Optional columns: open, high, low, volume
        country_filter: Filter macro indicators by country code (e.g., "US", "EU", "CN")
            If None, uses all countries (default: None)
        group_col: Column name for grouping (default: "symbol")
        timestamp_col: Column name for timestamp (default: "timestamp")
        price_col: Column name for price (default: "close")
    
    Returns:
        DataFrame with columns:
        - timestamp, symbol (from prices)
        - macro_growth_regime: Growth regime (+1 = expansion, -1 = recession, 0 = neutral)
        - macro_inflation_regime: Inflation regime (+1 = high inflation, -1 = low/deflation, 0 = neutral)
        - macro_risk_aversion_proxy: Risk-on/risk-off indicator
        
        Sorted by symbol, then timestamp.
        All symbols on the same date get the same macro regime values (market-wide factors).
        Factors are NaN where no macro data is available.
    
    Raises:
        KeyError: If required columns are missing
        ValueError: If DataFrames are empty or invalid
    """
    # Validate inputs
    required_price_cols = [timestamp_col, group_col, price_col]
    missing_price_cols = [col for col in required_price_cols if col not in prices.columns]
    if missing_price_cols:
        raise KeyError(
            f"Missing required columns in prices: {', '.join(missing_price_cols)}. "
            f"Available: {list(prices.columns)}"
        )
    
    required_macro_cols = [timestamp_col, "macro_code", "value", "country"]
    missing_macro_cols = [col for col in required_macro_cols if col not in macro_series.columns]
    if missing_macro_cols:
        raise KeyError(
            f"Missing required columns in macro_series: {', '.join(missing_macro_cols)}. "
            f"Available: {list(macro_series.columns)}"
        )
    
    if prices.empty:
        raise ValueError("prices DataFrame is empty")
    
    # Prepare prices DataFrame
    result = prices.copy()
    
    # Ensure timestamps are UTC-aware datetime
    if not pd.api.types.is_datetime64_any_dtype(result[timestamp_col]):
        result[timestamp_col] = pd.to_datetime(result[timestamp_col], utc=True)
    
    if not macro_series.empty:
        if not pd.api.types.is_datetime64_any_dtype(macro_series[timestamp_col]):
            macro_series = macro_series.copy()
            macro_series[timestamp_col] = pd.to_datetime(macro_series[timestamp_col], utc=True)
    
    # Sort by symbol and timestamp
    result = result.sort_values([group_col, timestamp_col]).reset_index(drop=True)
    
    if macro_series.empty:
        logger.warning("macro_series is empty. Returning prices with NaN factors.")
        # Add empty factor columns
        result["macro_growth_regime"] = np.nan
        result["macro_inflation_regime"] = np.nan
        result["macro_risk_aversion_proxy"] = np.nan
        return result
    
    # Filter by country if specified
    if country_filter:
        macro_series = macro_series[macro_series["country"] == country_filter.upper()].copy()
        if macro_series.empty:
            logger.warning(f"No macro data found for country {country_filter}")
            result["macro_growth_regime"] = np.nan
            result["macro_inflation_regime"] = np.nan
            result["macro_risk_aversion_proxy"] = np.nan
            return result
    
    # Sort macro series by timestamp
    macro_series = macro_series.sort_values(timestamp_col).reset_index(drop=True)
    
    # Compute growth regime from GDP, unemployment, PMI, etc.
    # Growth regime: +1 = expansion, -1 = recession, 0 = neutral
    growth_indicators = ["GDP", "UNEMPLOYMENT", "PMI", "INDUSTRIAL_PRODUCTION"]
    growth_data = macro_series[
        macro_series["macro_code"].isin([c.upper() for c in growth_indicators])
    ].copy()
    
    # Compute inflation regime from CPI, PPI, etc.
    # Inflation regime: +1 = high inflation, -1 = low/deflation, 0 = neutral
    inflation_indicators = ["CPI", "PPI", "INFLATION", "CORE_CPI"]
    inflation_data = macro_series[
        macro_series["macro_code"].isin([c.upper() for c in inflation_indicators])
    ].copy()
    
    # Compute risk aversion proxy from VIX, Fed rate, etc.
    # Risk aversion: +1 = risk-off, -1 = risk-on, 0 = neutral
    risk_indicators = ["FED_RATE", "VIX", "TREASURY_10Y", "CREDIT_SPREAD"]
    risk_data = macro_series[
        macro_series["macro_code"].isin([c.upper() for c in risk_indicators])
    ].copy()
    
    # Aggregate macro indicators by date
    # For each date, compute regime indicators
    all_dates = sorted(macro_series[timestamp_col].dt.date.unique())
    
    regime_factors = []
    
    for date_val in all_dates:
        date_ts = pd.Timestamp(date_val, tz="UTC")
        
        # Growth regime
        growth_values = growth_data[
            growth_data[timestamp_col].dt.date == date_val
        ]["value"].dropna()
        
        growth_regime = 0.0
        if not growth_values.empty:
            # Simple heuristic: positive GDP growth = expansion, negative = recession
            # Low unemployment = expansion, high = recession
            gdp_values = growth_data[
                (growth_data[timestamp_col].dt.date == date_val) &
                (growth_data["macro_code"] == "GDP")
            ]["value"].dropna()
            
            unemployment_values = growth_data[
                (growth_data[timestamp_col].dt.date == date_val) &
                (growth_data["macro_code"] == "UNEMPLOYMENT")
            ]["value"].dropna()
            
            if not gdp_values.empty:
                # GDP growth > 0 = expansion, < 0 = recession
                avg_gdp = gdp_values.mean()
                if avg_gdp > 2.0:  # Threshold: 2% GDP growth
                    growth_regime = 1.0
                elif avg_gdp < 0:
                    growth_regime = -1.0
            
            if not unemployment_values.empty and growth_regime == 0.0:
                # Low unemployment = expansion, high = recession
                avg_unemployment = unemployment_values.mean()
                if avg_unemployment < 4.0:  # Threshold: 4% unemployment
                    growth_regime = 1.0
                elif avg_unemployment > 7.0:  # Threshold: 7% unemployment
                    growth_regime = -1.0
        
        # Inflation regime
        inflation_values = inflation_data[
            inflation_data[timestamp_col].dt.date == date_val
        ]["value"].dropna()
        
        inflation_regime = 0.0
        if not inflation_values.empty:
            # High inflation = +1, low/deflation = -1
            avg_inflation = inflation_values.mean()
            if avg_inflation > 3.0:  # Threshold: 3% inflation
                inflation_regime = 1.0
            elif avg_inflation < 1.0:  # Threshold: 1% inflation (low/deflation)
                inflation_regime = -1.0
        
        # Risk aversion proxy
        risk_values = risk_data[
            risk_data[timestamp_col].dt.date == date_val
        ]["value"].dropna()
        
        risk_aversion = 0.0
        if not risk_values.empty:
            # High Fed rate = risk-off, low = risk-on
            # High VIX = risk-off, low = risk-on
            fed_rate_values = risk_data[
                (risk_data[timestamp_col].dt.date == date_val) &
                (risk_data["macro_code"] == "FED_RATE")
            ]["value"].dropna()
            
            vix_values = risk_data[
                (risk_data[timestamp_col].dt.date == date_val) &
                (risk_data["macro_code"] == "VIX")
            ]["value"].dropna()
            
            if not fed_rate_values.empty:
                avg_fed_rate = fed_rate_values.mean()
                if avg_fed_rate > 5.0:  # Threshold: 5% Fed rate
                    risk_aversion = 1.0
                elif avg_fed_rate < 2.0:  # Threshold: 2% Fed rate
                    risk_aversion = -1.0
            
            if not vix_values.empty and risk_aversion == 0.0:
                avg_vix = vix_values.mean()
                if avg_vix > 20.0:  # Threshold: VIX > 20
                    risk_aversion = 1.0
                elif avg_vix < 15.0:  # Threshold: VIX < 15
                    risk_aversion = -1.0
        
        regime_factors.append({
            timestamp_col: date_ts,
            "macro_growth_regime": growth_regime,
            "macro_inflation_regime": inflation_regime,
            "macro_risk_aversion_proxy": risk_aversion,
        })
    
    if not regime_factors:
        logger.warning("No regime factors computed. Returning prices with NaN factors.")
        result["macro_growth_regime"] = np.nan
        result["macro_inflation_regime"] = np.nan
        result["macro_risk_aversion_proxy"] = np.nan
        return result
    
    regime_df = pd.DataFrame(regime_factors)
    regime_df = regime_df.sort_values(timestamp_col).reset_index(drop=True)
    
    # Join regime factors to all symbols (market-wide factors)
    # All symbols on the same date get the same macro regime values
    result_list = []
    for symbol in result[group_col].unique():
        symbol_result = result[result[group_col] == symbol].copy()
        symbol_result = symbol_result.sort_values(timestamp_col).reset_index(drop=True)
        regime_df_sorted = regime_df.sort_values(timestamp_col).reset_index(drop=True)
        
        # Use merge_asof to forward-fill regime factors
        symbol_result = pd.merge_asof(
            symbol_result,
            regime_df_sorted[[
                timestamp_col,
                "macro_growth_regime",
                "macro_inflation_regime",
                "macro_risk_aversion_proxy",
            ]],
            on=timestamp_col,
            direction="backward",
            allow_exact_matches=True,
        )
        result_list.append(symbol_result)
    
    if result_list:
        result = pd.concat(result_list, ignore_index=True)
    
    # Ensure all factor columns exist
    factor_cols = [
        "macro_growth_regime",
        "macro_inflation_regime",
        "macro_risk_aversion_proxy",
    ]
    for col in factor_cols:
        if col not in result.columns:
            result[col] = np.nan
    
    # Sort by symbol, then timestamp
    result = result.sort_values([group_col, timestamp_col]).reset_index(drop=True)
    
    logger.info(
        f"Built macro regime factors for {len(result[group_col].unique())} symbols, "
        f"{len(result)} rows. Country filter: {country_filter or 'all'}."
    )
    
    return result

