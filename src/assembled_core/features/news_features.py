"""News-derived features for the Factor Store.

Computes numeric features from classified news events that can be
used alongside technical features in signal generation.

Features per (timestamp, symbol):
  news_sentiment:    EWM-smoothed net sentiment score [-1, +1].
                     Positive = net bullish news, negative = bearish.
  news_event_count:  Number of actionable news events in the lookback window.
  news_velocity:     Rate of change in event count (today vs N-day average).
  news_confidence:   Average confidence of news events in the window.

These features are PIT-safe: only news with event_date <= as_of is used.

Usage:
    from src.assembled_core.features.news_features import compute_news_features

    events_df = pd.DataFrame({
        "event_date": [...],
        "symbol": [...],
        "direction": [...],   # "bullish" / "bearish" / "neutral"
        "confidence": [...],  # 0.0 - 1.0
    })
    features = compute_news_features(events_df, prices_dates=prices["timestamp"].unique())
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_LOOKBACK_DAYS = 5
_DEFAULT_VELOCITY_WINDOW = 10
_EWM_HALFLIFE_DAYS = 3


def compute_news_features(
    events: pd.DataFrame,
    prices_dates: "pd.DatetimeLike | None" = None,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
    velocity_window: int = _DEFAULT_VELOCITY_WINDOW,
    as_of: "pd.Timestamp | None" = None,
) -> pd.DataFrame:
    """Compute per-symbol news features from classified event records.

    Args:
        events: DataFrame with columns:
            - event_date: date of the event (PIT: use disclosure date if available)
            - symbol: affected ticker(s) — one row per (event, symbol)
            - direction: "bullish" / "bearish" / "neutral"
            - confidence: float in [0, 1]
        prices_dates: Optional array of trading dates to align features to.
            If provided, features are forward-filled to cover all trading dates.
        lookback_days: Rolling window for event count and sentiment (calendar days).
        velocity_window: Longer window used to compute velocity baseline.
        as_of: PIT cutoff — events after this date are excluded.

    Returns:
        DataFrame with columns: timestamp, symbol, news_sentiment,
        news_event_count, news_velocity, news_confidence.
        Sorted by symbol, then timestamp.
    """
    required_cols = {"event_date", "symbol", "direction", "confidence"}
    missing = required_cols - set(events.columns)
    if missing:
        raise ValueError(f"events missing required columns: {missing}")

    if events.empty:
        return pd.DataFrame(
            columns=["timestamp", "symbol", "news_sentiment",
                     "news_event_count", "news_velocity", "news_confidence"]
        )

    events = events.copy()
    events["event_date"] = pd.to_datetime(events["event_date"], utc=True)

    if as_of is not None:
        as_of_ts = pd.Timestamp(as_of).tz_localize("UTC") if pd.Timestamp(as_of).tzinfo is None else pd.Timestamp(as_of)
        events = events[events["event_date"] <= as_of_ts]

    # Map direction to numeric score
    direction_map = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}
    events["_score"] = events["direction"].map(direction_map).fillna(0.0)
    events["_weighted_score"] = events["_score"] * events["confidence"].clip(0.0, 1.0)

    all_rows: list[pd.DataFrame] = []

    for symbol, grp in events.groupby("symbol"):
        grp = grp.sort_values("event_date")

        # Daily aggregation
        daily = (
            grp.groupby(grp["event_date"].dt.date)
            .agg(
                raw_sentiment=("_weighted_score", "sum"),
                event_count=("_weighted_score", "count"),
                mean_confidence=("confidence", "mean"),
            )
            .reset_index()
        )
        daily["event_date"] = pd.to_datetime(daily["event_date"], utc=True)
        daily = daily.set_index("event_date").sort_index()

        # Extend to all trading dates if provided
        if prices_dates is not None:
            _raw_idx = pd.DatetimeIndex(prices_dates)
            if _raw_idx.tz is None:
                trade_idx = _raw_idx.tz_localize("UTC", nonexistent="shift_forward").normalize()
            else:
                trade_idx = _raw_idx.tz_convert("UTC").normalize()
            trade_idx = pd.DatetimeIndex(sorted(set(trade_idx)))
            combined = daily.index.union(trade_idx)
            combined.name = "event_date"  # preserve index name after union
            daily = daily.reindex(combined)

        # Fill gaps: count / confidence → 0 on days with no events
        daily["event_count"] = daily["event_count"].fillna(0.0)
        daily["mean_confidence"] = daily["mean_confidence"].fillna(0.0)
        daily["raw_sentiment"] = daily["raw_sentiment"].fillna(0.0)

        # Smoothed sentiment: EWM to decay old news
        daily["news_sentiment"] = (
            daily["raw_sentiment"]
            .ewm(halflife=_EWM_HALFLIFE_DAYS, min_periods=1)
            .mean()
            .clip(-1.0, 1.0)
        )

        # Rolling event count
        daily["news_event_count"] = (
            daily["event_count"]
            .rolling(lookback_days, min_periods=1)
            .sum()
        )

        # Velocity: short / long window ratio
        short_avg = daily["event_count"].rolling(lookback_days, min_periods=1).mean()
        long_avg = daily["event_count"].rolling(velocity_window, min_periods=1).mean()
        with np.errstate(invalid="ignore", divide="ignore"):
            raw_vel = np.where(long_avg > 0, short_avg / long_avg - 1.0, 0.0)
        daily["news_velocity"] = pd.Series(raw_vel, index=daily.index).clip(-2.0, 2.0)

        # Smoothed confidence
        daily["news_confidence"] = (
            daily["mean_confidence"]
            .rolling(lookback_days, min_periods=1)
            .mean()
        )

        sym_df = daily[
            ["news_sentiment", "news_event_count", "news_velocity", "news_confidence"]
        ].reset_index().rename(columns={"event_date": "timestamp"})
        sym_df["symbol"] = symbol
        all_rows.append(sym_df)

    if not all_rows:
        return pd.DataFrame(
            columns=["timestamp", "symbol", "news_sentiment",
                     "news_event_count", "news_velocity", "news_confidence"]
        )

    result = pd.concat(all_rows, ignore_index=True)
    result = result.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    logger.debug(
        "[news_features] computed %d rows for %d symbols",
        len(result), result["symbol"].nunique(),
    )

    return result


def add_news_features(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    short_window: int = 7,
    long_window: int = 30,
) -> pd.DataFrame:
    """Add rolling news features to a prices DataFrame (Phase 6 compatibility API).

    Accepts events in the raw news format (with ``sentiment_score`` column) or the
    classified format (with ``direction`` and ``confidence`` columns) and adds
    per-symbol rolling features joined onto the prices index.

    Args:
        prices: DataFrame with columns: timestamp, symbol (and any other columns).
        events: DataFrame with per-event news data.  Accepted formats:
            - Raw: timestamp, symbol, sentiment_score (numeric, -1 to 1)
            - Classified: event_date, symbol, direction (bullish/bearish/neutral), confidence
        short_window: Rolling window in calendar days for short-term features (default 7).
        long_window: Rolling window in calendar days for long-term features (default 30).

    Returns:
        prices DataFrame with added columns:
            news_sentiment_{short_window}d, news_sentiment_{long_window}d,
            news_count_{short_window}d, news_count_{long_window}d
        All new columns are float; missing values filled with 0.0.
    """
    prices = prices.copy()

    if events is None or events.empty:
        prices[f"news_sentiment_{short_window}d"] = 0.0
        prices[f"news_sentiment_{long_window}d"] = 0.0
        prices[f"news_count_{short_window}d"] = 0.0
        prices[f"news_count_{long_window}d"] = 0.0
        return prices

    # Normalise events to classified format
    if "sentiment_score" in events.columns and "direction" not in events.columns:
        evts = events.copy()
        evts["event_date"] = pd.to_datetime(evts.get("timestamp", evts.get("event_date")), utc=True)
        evts["direction"] = np.where(
            evts["sentiment_score"] > 0.1, "bullish",
            np.where(evts["sentiment_score"] < -0.1, "bearish", "neutral")
        )
        evts["confidence"] = evts["sentiment_score"].abs().clip(0.0, 1.0)
    else:
        evts = events.copy()
        if "event_date" not in evts.columns and "timestamp" in evts.columns:
            evts["event_date"] = evts["timestamp"]

    prices_ts = pd.to_datetime(prices["timestamp"], utc=True)

    short_feats = compute_news_features(evts, prices_dates=prices_ts, lookback_days=short_window)
    long_feats = compute_news_features(evts, prices_dates=prices_ts, lookback_days=long_window)

    prices["_ts_key"] = pd.to_datetime(prices["timestamp"], utc=True).dt.normalize()

    def _merge_feature(df: pd.DataFrame, feat_col: str, out_col: str) -> None:
        feat = df[["timestamp", "symbol", feat_col]].copy()
        feat["timestamp"] = pd.to_datetime(feat["timestamp"], utc=True).dt.normalize()
        merged = prices.merge(feat.rename(columns={"timestamp": "_ts_key"}),
                              on=["_ts_key", "symbol"], how="left")
        prices[out_col] = merged[feat_col].fillna(0.0).values

    _merge_feature(short_feats, "news_sentiment", f"news_sentiment_{short_window}d")
    _merge_feature(long_feats, "news_sentiment", f"news_sentiment_{long_window}d")
    _merge_feature(short_feats, "news_event_count", f"news_count_{short_window}d")
    _merge_feature(long_feats, "news_event_count", f"news_count_{long_window}d")

    prices = prices.drop(columns=["_ts_key"])
    return prices
