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
    missing_price_cols = [
        col for col in required_price_cols if col not in prices.columns
    ]
    if missing_price_cols:
        raise KeyError(
            f"Missing required columns in prices: {', '.join(missing_price_cols)}. "
            f"Available: {list(prices.columns)}"
        )

    required_sentiment_cols = [timestamp_col, "sentiment_score", "sentiment_volume"]
    missing_sentiment_cols = [
        col
        for col in required_sentiment_cols
        if col not in news_sentiment_daily.columns
    ]
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
        if not pd.api.types.is_datetime64_any_dtype(
            news_sentiment_daily[timestamp_col]
        ):
            news_sentiment_daily = news_sentiment_daily.copy()
            news_sentiment_daily[timestamp_col] = pd.to_datetime(
                news_sentiment_daily[timestamp_col], utc=True
            )

        # Point-in-time handling: ensure event_date / disclosure_date exist.
        if "event_date" not in news_sentiment_daily.columns:
            news_sentiment_daily["event_date"] = news_sentiment_daily[
                timestamp_col
            ].dt.normalize()
        if "disclosure_date" not in news_sentiment_daily.columns:
            # Batch-12 PIT fix (Diagnostik §features MAJOR-latent): daily news
            # sentiment for day T is not fully observable until end-of-day T;
            # treating disclosure_date == event_date (T+0) leaks same-day news
            # into bar T. Derive disclosure_date = event_date + conservative
            # latency via the shared PIT helper. No "news" key exists in
            # source_latencies; GDELT/ACLED news-style feeds use 1 day, so a
            # local conservative default of 1 calendar day is used. Caller-
            # supplied disclosure_date columns are preserved (no override).
            from src.assembled_core.data.latency import apply_source_latency

            _NEWS_DISCLOSURE_LAG_DAYS = 1
            news_sentiment_daily = apply_source_latency(
                news_sentiment_daily,
                days=_NEWS_DISCLOSURE_LAG_DAYS,
                event_date_col="event_date",
                mode="derive",
            )

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
        logger.warning(
            "news_sentiment_daily is empty. Returning prices with NaN factors."
        )
        # Add empty factor columns
        result[f"news_sentiment_mean_{lookback_days}d"] = np.nan
        result[f"news_sentiment_trend_{lookback_days}d"] = np.nan
        result["news_sentiment_shock_flag"] = 0.0
        result[f"news_sentiment_volume_{lookback_days}d"] = np.nan
        return result

    # Helper function to compute trend (slope over rolling window)
    def compute_trend(series: pd.Series, window: int) -> pd.Series:
        """Compute rolling trend (slope) over window."""
        x = np.arange(window, dtype=float)
        sum_x = x.sum()
        sum_x2 = (x**2).sum()
        denom = window * sum_x2 - sum_x**2
        if denom == 0:
            return pd.Series(np.nan, index=series.index)
        rolling_xy = series.rolling(window, min_periods=window).apply(
            lambda y: np.dot(x, y), raw=True
        )
        rolling_y = series.rolling(window, min_periods=window).sum()
        return (window * rolling_xy - sum_x * rolling_y) / denom

    # Handle market-wide sentiment (symbol=None or "__MARKET__")
    # If sentiment has symbol column, join per symbol; otherwise join market-wide to all symbols
    sentiment_has_symbol = "symbol" in news_sentiment_daily.columns

    # Process sentiment per symbol and market-wide separately
    sentiment_per_symbol = pd.DataFrame()
    sentiment_market = pd.DataFrame()

    if sentiment_has_symbol:
        # Filter per-symbol sentiment
        sentiment_per_symbol = news_sentiment_daily[
            news_sentiment_daily["symbol"].notna()
            & (news_sentiment_daily["symbol"] != "__MARKET__")
        ].copy()

        # Filter market-wide sentiment
        sentiment_market = news_sentiment_daily[
            (news_sentiment_daily["symbol"].isna())
            | (news_sentiment_daily["symbol"] == "__MARKET__")
        ].copy()
    else:
        # No symbol column - treat as market-wide sentiment
        sentiment_market = news_sentiment_daily.copy()

    # Process per-symbol sentiment
    sentiment_factors_list = []
    if not sentiment_per_symbol.empty:
        sentiment_per_symbol = sentiment_per_symbol.sort_values(
            [group_col, timestamp_col]
        ).reset_index(drop=True)

        for symbol, symbol_sentiment in sentiment_per_symbol.groupby(
            group_col, sort=False
        ):
            symbol_sentiment = symbol_sentiment.reset_index(drop=True)

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
            sentiment_std = (
                symbol_sentiment["sentiment_score"]
                .rolling(window=lookback_days * 2, min_periods=lookback_days)
                .std()
            )
            sentiment_change = symbol_sentiment["sentiment_score"].diff().abs()
            symbol_sentiment["sentiment_shock_flag"] = (
                sentiment_change > sentiment_std * 1.5
            ).astype(float)

            # Rolling mean of volume
            symbol_sentiment[f"sentiment_volume_{lookback_days}d"] = (
                symbol_sentiment["sentiment_volume"]
                .rolling(window=lookback_days, min_periods=1)
                .mean()
            )

            sentiment_factors_list.append(symbol_sentiment)

    # Process market-wide sentiment
    if not sentiment_market.empty:
        sentiment_market = sentiment_market.sort_values(timestamp_col).reset_index(
            drop=True
        )

        # Compute rolling factors for market-wide sentiment
        sentiment_market[f"sentiment_mean_{lookback_days}d"] = (
            sentiment_market["sentiment_score"]
            .rolling(window=lookback_days, min_periods=1)
            .mean()
        )

        sentiment_market[f"sentiment_trend_{lookback_days}d"] = compute_trend(
            sentiment_market["sentiment_score"], lookback_days
        )

        sentiment_std = (
            sentiment_market["sentiment_score"]
            .rolling(window=lookback_days * 2, min_periods=lookback_days)
            .std()
        )
        sentiment_change = sentiment_market["sentiment_score"].diff().abs()
        sentiment_market["sentiment_shock_flag"] = (
            sentiment_change > sentiment_std * 1.5
        ).astype(float)

        sentiment_market[f"sentiment_volume_{lookback_days}d"] = (
            sentiment_market["sentiment_volume"]
            .rolling(window=lookback_days, min_periods=1)
            .mean()
        )

    # Merge sentiment factors to prices (per symbol first, then market-wide)
    # Build O(1) lookup dict from sentiment_factors_list (was O(N²) linear scan).
    _sentiment_by_symbol: dict = {}
    if sentiment_factors_list:
        for sf in sentiment_factors_list:
            if not sf.empty:
                _sentiment_by_symbol[sf[group_col].iloc[0]] = sf

    result_list = []
    for symbol, symbol_result in result.groupby(group_col, sort=False):
        symbol_result = symbol_result.sort_values(timestamp_col).reset_index(drop=True)

        # First, try per-symbol sentiment
        symbol_sentiment = None
        if _sentiment_by_symbol:
            _sf = _sentiment_by_symbol.get(symbol)
            if _sf is not None:
                symbol_sentiment = _sf.copy()

        if symbol_sentiment is not None and not symbol_sentiment.empty:
            # Batch-12 PIT fix: align sentiment to bars by *disclosure_date*
            # (PIT availability), not the raw daily timestamp.
            if "disclosure_date" not in symbol_sentiment.columns:
                symbol_sentiment = symbol_sentiment.copy()
                symbol_sentiment["disclosure_date"] = pd.to_datetime(
                    symbol_sentiment[timestamp_col], utc=True
                ).dt.normalize()
            symbol_sentiment = symbol_sentiment.sort_values(
                "disclosure_date"
            ).reset_index(drop=True)

            # Merge per-symbol sentiment on the PIT key with exact-match disabled
            # so same-disclosure-day sentiment cannot leak into that bar.
            symbol_result = pd.merge_asof(
                symbol_result,
                symbol_sentiment[
                    [
                        "disclosure_date",
                        f"sentiment_mean_{lookback_days}d",
                        f"sentiment_trend_{lookback_days}d",
                        "sentiment_shock_flag",
                        f"sentiment_volume_{lookback_days}d",
                    ]
                ],
                left_on=timestamp_col,
                right_on="disclosure_date",
                direction="backward",
                allow_exact_matches=False,
            )
            if "disclosure_date" in symbol_result.columns:
                symbol_result = symbol_result.drop(columns=["disclosure_date"])

        # Then, merge market-wide sentiment (fills gaps or provides default)
        if not sentiment_market.empty:
            # Batch-12 PIT fix: market-wide sentiment is also aligned by
            # *disclosure_date* with exact-match disabled.
            if "disclosure_date" not in sentiment_market.columns:
                sentiment_market = sentiment_market.copy()
                sentiment_market["disclosure_date"] = pd.to_datetime(
                    sentiment_market[timestamp_col], utc=True
                ).dt.normalize()
            sentiment_market_sorted = sentiment_market.sort_values(
                "disclosure_date"
            ).reset_index(drop=True)

            # Merge market-wide sentiment (only fill NaN values from per-symbol)
            for col in [
                f"sentiment_mean_{lookback_days}d",
                f"sentiment_trend_{lookback_days}d",
                "sentiment_shock_flag",
                f"sentiment_volume_{lookback_days}d",
            ]:
                if col not in symbol_result.columns:
                    symbol_result[col] = np.nan

            # Fill NaN values with market-wide sentiment (PIT key = disclosure_date)
            market_merged = pd.merge_asof(
                symbol_result[[timestamp_col]],
                sentiment_market_sorted[
                    [
                        "disclosure_date",
                        f"sentiment_mean_{lookback_days}d",
                        f"sentiment_trend_{lookback_days}d",
                        "sentiment_shock_flag",
                        f"sentiment_volume_{lookback_days}d",
                    ]
                ],
                left_on=timestamp_col,
                right_on="disclosure_date",
                direction="backward",
                allow_exact_matches=False,
            )

            # Fill NaN values in symbol_result with market values
            for col in [
                f"sentiment_mean_{lookback_days}d",
                f"sentiment_trend_{lookback_days}d",
                "sentiment_shock_flag",
                f"sentiment_volume_{lookback_days}d",
            ]:
                mask = symbol_result[col].isna()
                if mask.any():
                    symbol_result.loc[mask, col] = market_merged.loc[mask, col].values

        result_list.append(symbol_result)

    if result_list:
        result = pd.concat(result_list, ignore_index=True)

    # Rename columns to match expected output format
    result = result.rename(
        columns={
            f"sentiment_mean_{lookback_days}d": f"news_sentiment_mean_{lookback_days}d",
            f"sentiment_trend_{lookback_days}d": f"news_sentiment_trend_{lookback_days}d",
            "sentiment_shock_flag": "news_sentiment_shock_flag",
            f"sentiment_volume_{lookback_days}d": f"news_sentiment_volume_{lookback_days}d",
        }
    )

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
        from src.assembled_core.qa.point_in_time_checks import (
            validate_feature_builder_pit_safe,
        )

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
    release_lag_days: int = 32,
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
    missing_price_cols = [
        col for col in required_price_cols if col not in prices.columns
    ]
    if missing_price_cols:
        raise KeyError(
            f"Missing required columns in prices: {', '.join(missing_price_cols)}. "
            f"Available: {list(prices.columns)}"
        )

    required_macro_cols = [timestamp_col, "macro_code", "value", "country"]
    missing_macro_cols = [
        col for col in required_macro_cols if col not in macro_series.columns
    ]
    if missing_macro_cols:
        raise KeyError(
            f"Missing required columns in macro_series: {', '.join(missing_macro_cols)}. "
            f"Available: {list(macro_series.columns)}"
        )

    if prices.empty:
        raise ValueError("prices DataFrame is empty")

    # Prepare prices DataFrame
    result = prices.copy()

    # Normalize macro timestamps to match result's timezone (preserve prices tz)
    def _match_tz(series, ref: pd.Series):
        s = pd.to_datetime(series, errors="coerce")
        ref_s = pd.to_datetime(ref, errors="coerce")
        ref_tz = ref_s.dt.tz
        s_tz = s.tz if isinstance(s, pd.DatetimeIndex) else s.dt.tz
        if ref_tz is None and s_tz is not None:
            s = (
                s.tz_convert(None)
                if isinstance(s, pd.DatetimeIndex)
                else s.dt.tz_convert(None)
            )
        elif ref_tz is not None and s_tz is None:
            s = (
                s.tz_localize(str(ref_tz))
                if isinstance(s, pd.DatetimeIndex)
                else s.dt.tz_localize(str(ref_tz))
            )
        elif ref_tz is not None and s_tz is not None:
            s = (
                s.tz_convert(str(ref_tz))
                if isinstance(s, pd.DatetimeIndex)
                else s.dt.tz_convert(str(ref_tz))
            )
        return s

    if not macro_series.empty:
        macro_series = macro_series.copy()
        macro_series[timestamp_col] = _match_tz(
            macro_series[timestamp_col], result[timestamp_col]
        )

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
        macro_series = macro_series[
            macro_series["country"] == country_filter.upper()
        ].copy()
        if macro_series.empty:
            logger.warning(f"No macro data found for country {country_filter}")
            result["macro_growth_regime"] = np.nan
            result["macro_inflation_regime"] = np.nan
            result["macro_risk_aversion_proxy"] = np.nan
            return result

    # Sort macro series by timestamp
    macro_series = macro_series.sort_values(timestamp_col).reset_index(drop=True)

    # Normalize macro_code to uppercase for substring matching
    macro_series = macro_series.copy()
    _codes_upper = macro_series["macro_code"].str.upper()

    def _matches_any(keywords: list[str]) -> "pd.Series":
        pattern = "|".join(keywords)
        return _codes_upper.str.contains(pattern, regex=True, na=False)

    # Compute growth regime — GDP, unemployment, PMI, industrial production
    growth_data = macro_series[
        _matches_any(["GDP", "UNEMPLOY", "PMI", "INDUSTRIAL"])
    ].copy()

    # Compute inflation regime — CPI, PPI, inflation indicators
    inflation_data = macro_series[_matches_any(["CPI", "PPI", "INFLATION"])].copy()

    # Compute risk aversion proxy — VIX, treasury yields, credit spreads
    risk_data = macro_series[
        _matches_any(["VIX", "TNX", "TYX", "TREASURY", "YIELD", "CREDIT", "FED"])
    ].copy()

    # Aggregate macro indicators by date — vectorized via pivot_table
    all_dates = sorted(macro_series[timestamp_col].dt.date.unique())
    date_index = pd.Index(all_dates)

    for _ds in (growth_data, inflation_data, risk_data):
        _ds["_date"] = _ds[timestamp_col].dt.date

    # Growth: GDP > 2 → expansion, GDP < 0 → recession; UNEMPLOYMENT fallback
    gpivot = (
        growth_data.pivot_table(
            index="_date", columns="macro_code", values="value", aggfunc="mean"
        ).reindex(date_index)
        if not growth_data.empty
        else pd.DataFrame(index=date_index)
    )
    _gdp_cols = [c for c in gpivot.columns if "GDP" in c.upper()]
    if _gdp_cols:
        _gdp_series = gpivot[_gdp_cols].mean(axis=1)
        gdp_regime = np.where(
            _gdp_series > 2.0, 1.0, np.where(_gdp_series < 0.0, -1.0, 0.0)
        )
    else:
        gdp_regime = np.zeros(len(date_index))
    _unemp_cols = [c for c in gpivot.columns if "UNEMPLOY" in c.upper()]
    if _unemp_cols:
        _unemp_series = gpivot[_unemp_cols].mean(axis=1)
        unemp_regime = np.where(
            _unemp_series < 4.0, 1.0, np.where(_unemp_series > 7.0, -1.0, 0.0)
        )
    else:
        unemp_regime = np.zeros(len(date_index))
    growth_regime_arr = np.where(gdp_regime != 0.0, gdp_regime, unemp_regime)

    # Inflation: avg across codes > 3 → high, < 1 → low
    ipivot = (
        inflation_data.pivot_table(
            index="_date", columns="macro_code", values="value", aggfunc="mean"
        ).reindex(date_index)
        if not inflation_data.empty
        else pd.DataFrame(index=date_index)
    )
    avg_infl = (
        ipivot.mean(axis=1) if not ipivot.empty else pd.Series(np.nan, index=date_index)
    )
    inflation_regime_arr = np.where(
        avg_infl > 3.0, 1.0, np.where(avg_infl < 1.0, -1.0, 0.0)
    )

    # Risk aversion: FED_RATE > 5 → risk-off, < 2 → risk-on; VIX fallback
    rpivot = (
        risk_data.pivot_table(
            index="_date", columns="macro_code", values="value", aggfunc="mean"
        ).reindex(date_index)
        if not risk_data.empty
        else pd.DataFrame(index=date_index)
    )
    fed_regime = (
        np.where(
            rpivot["FED_RATE"] > 5.0, 1.0, np.where(rpivot["FED_RATE"] < 2.0, -1.0, 0.0)
        )
        if "FED_RATE" in rpivot.columns
        else np.zeros(len(date_index))
    )
    vix_regime = (
        np.where(rpivot["VIX"] > 20.0, 1.0, np.where(rpivot["VIX"] < 15.0, -1.0, 0.0))
        if "VIX" in rpivot.columns
        else np.zeros(len(date_index))
    )
    risk_aversion_arr = np.where(fed_regime != 0.0, fed_regime, vix_regime)

    regime_df = (
        pd.DataFrame(
            {
                timestamp_col: _match_tz(
                    pd.to_datetime(date_index), result[timestamp_col]
                ),
                "macro_growth_regime": growth_regime_arr,
                "macro_inflation_regime": inflation_regime_arr,
                "macro_risk_aversion_proxy": risk_aversion_arr,
            }
        )
        .sort_values(timestamp_col)
        .reset_index(drop=True)
    )

    if regime_df.empty:
        logger.warning("No regime factors computed. Returning prices with NaN factors.")
        result["macro_growth_regime"] = np.nan
        result["macro_inflation_regime"] = np.nan
        result["macro_risk_aversion_proxy"] = np.nan
        return result

    # Join regime factors to all symbols (market-wide factors — single merge_asof,
    # no per-symbol loop).
    # Batch-12 PIT fix (Diagnostik §features/§data, GPR reference): macro
    # indicators for observation date T are published with a release delay
    # (month-T values land during month T+1). Align the regime to bars by a
    # *release-lagged* availability date, mirroring merge_gpr_index_into_panel's
    # release_lag_days=32. allow_exact_matches=False additionally prevents a
    # regime whose availability lands exactly on a bar from feeding that bar.
    regime_df_sorted = regime_df.sort_values(timestamp_col).reset_index(drop=True)
    regime_df_sorted = regime_df_sorted.copy()
    regime_df_sorted["macro_available_date"] = regime_df_sorted[
        timestamp_col
    ] + pd.Timedelta(days=release_lag_days)
    regime_df_sorted = regime_df_sorted.sort_values("macro_available_date").reset_index(
        drop=True
    )
    result = pd.merge_asof(
        result.sort_values(timestamp_col),
        regime_df_sorted[
            [
                "macro_available_date",
                "macro_growth_regime",
                "macro_inflation_regime",
                "macro_risk_aversion_proxy",
            ]
        ],
        left_on=timestamp_col,
        right_on="macro_available_date",
        direction="backward",
        allow_exact_matches=False,
    )
    if "macro_available_date" in result.columns:
        result = result.drop(columns=["macro_available_date"])

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
