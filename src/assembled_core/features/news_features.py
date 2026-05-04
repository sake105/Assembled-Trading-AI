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
            columns=[
                "timestamp",
                "symbol",
                "news_sentiment",
                "news_event_count",
                "news_velocity",
                "news_confidence",
            ]
        )

    events = events.copy()
    events["event_date"] = pd.to_datetime(events["event_date"], utc=True)

    if as_of is not None:
        as_of_ts = (
            pd.Timestamp(as_of).tz_localize("UTC")
            if pd.Timestamp(as_of).tzinfo is None
            else pd.Timestamp(as_of)
        )
        events = events[events["event_date"] <= as_of_ts].copy()

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
                trade_idx = _raw_idx.tz_localize(
                    "UTC", nonexistent="shift_forward"
                ).normalize()
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
            daily["event_count"].rolling(lookback_days, min_periods=1).sum()
        )

        # Velocity: short / long window ratio
        short_avg = daily["event_count"].rolling(lookback_days, min_periods=1).mean()
        long_avg = daily["event_count"].rolling(velocity_window, min_periods=1).mean()
        with np.errstate(invalid="ignore", divide="ignore"):
            raw_vel = np.where(long_avg > 0, short_avg / long_avg - 1.0, 0.0)
        daily["news_velocity"] = pd.Series(raw_vel, index=daily.index).clip(-2.0, 2.0)

        # Smoothed confidence
        daily["news_confidence"] = (
            daily["mean_confidence"].rolling(lookback_days, min_periods=1).mean()
        )

        sym_df = (
            daily[
                [
                    "news_sentiment",
                    "news_event_count",
                    "news_velocity",
                    "news_confidence",
                ]
            ]
            .reset_index()
            .rename(columns={"event_date": "timestamp"})
        )
        sym_df["symbol"] = symbol
        all_rows.append(sym_df)

    if not all_rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "symbol",
                "news_sentiment",
                "news_event_count",
                "news_velocity",
                "news_confidence",
            ]
        )

    result = pd.concat(all_rows, ignore_index=True)
    result = result.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    logger.debug(
        "[news_features] computed %d rows for %d symbols",
        len(result),
        result["symbol"].nunique(),
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
        evts["event_date"] = pd.to_datetime(
            evts.get("timestamp", evts.get("event_date")), utc=True
        )
        evts["direction"] = np.where(
            evts["sentiment_score"] > 0.1,
            "bullish",
            np.where(evts["sentiment_score"] < -0.1, "bearish", "neutral"),
        )
        evts["confidence"] = evts["sentiment_score"].abs().clip(0.0, 1.0)
    else:
        evts = events.copy()
        if "event_date" not in evts.columns and "timestamp" in evts.columns:
            evts["event_date"] = evts["timestamp"]

    prices_ts = pd.to_datetime(prices["timestamp"], utc=True)

    short_feats = compute_news_features(
        evts, prices_dates=prices_ts, lookback_days=short_window
    )
    long_feats = compute_news_features(
        evts, prices_dates=prices_ts, lookback_days=long_window
    )

    prices["_ts_key"] = pd.to_datetime(prices["timestamp"], utc=True).dt.normalize()

    def _merge_feature(df: pd.DataFrame, feat_col: str, out_col: str) -> None:
        feat = df[["timestamp", "symbol", feat_col]].copy()
        feat["timestamp"] = pd.to_datetime(feat["timestamp"], utc=True).dt.normalize()
        merged = prices.merge(
            feat.rename(columns={"timestamp": "_ts_key"}),
            on=["_ts_key", "symbol"],
            how="left",
        )
        prices[out_col] = merged[feat_col].fillna(0.0).values

    _merge_feature(short_feats, "news_sentiment", f"news_sentiment_{short_window}d")
    _merge_feature(long_feats, "news_sentiment", f"news_sentiment_{long_window}d")
    _merge_feature(short_feats, "news_event_count", f"news_count_{short_window}d")
    _merge_feature(long_feats, "news_event_count", f"news_count_{long_window}d")

    prices = prices.drop(columns=["_ts_key"])
    return prices


def compute_sector_rotation_signal(
    events_df: pd.DataFrame,
    window_hours: float = 24.0,
    min_events: int = 2,
) -> dict[str, float]:
    """Compute a sector rotation signal from news momentum (Point 36).

    Score = (severity_weighted_momentum) / sqrt(total_events + 1)
    Normalised to [-1, +1] across all sectors.
    """
    if events_df.empty:
        return {}

    import math

    df = events_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    now = df["timestamp"].max()
    window_td = pd.Timedelta(hours=window_hours)
    prior_td = pd.Timedelta(hours=window_hours * 2)

    current = df[df["timestamp"] >= now - window_td]
    prior = df[
        (df["timestamp"] >= now - prior_td) & (df["timestamp"] < now - window_td)
    ]

    def _sector_scores(frame: pd.DataFrame) -> dict[str, dict]:
        if frame.empty or "affected_sectors" not in frame.columns:
            return {}
        tmp = (
            frame[["affected_sectors", "severity", "market_direction"]].copy()
            if "severity" in frame.columns and "market_direction" in frame.columns
            else frame[["affected_sectors"]].copy()
        )
        tmp["affected_sectors"] = tmp["affected_sectors"].apply(
            lambda x: [x] if isinstance(x, str) else (x if isinstance(x, list) else [])
        )
        tmp = tmp.explode("affected_sectors")
        tmp = tmp[tmp["affected_sectors"].notna() & (tmp["affected_sectors"] != "")]
        if tmp.empty:
            return {}
        sev = (
            pd.to_numeric(tmp.get("severity", 1.0), errors="coerce").fillna(1.0)
            if "severity" in tmp.columns
            else pd.Series(1.0, index=tmp.index)
        )
        dir_raw = (
            tmp["market_direction"].str.lower()
            if "market_direction" in tmp.columns
            else pd.Series("neutral", index=tmp.index)
        )
        dir_sign = dir_raw.map({"bullish": 1.0, "bearish": -1.0}).fillna(0.0)
        tmp = tmp.copy()
        tmp["_sev"] = sev.values
        tmp["_w"] = (sev * dir_sign.where(dir_sign != 0, 1.0)).values
        agg = tmp.groupby("affected_sectors").agg(
            count=("_sev", "count"), weighted=("_w", "sum")
        )
        return agg.rename_axis(None).to_dict("index")

    curr_scores = _sector_scores(current)
    prior_scores = _sector_scores(prior)

    rotation: dict[str, float] = {}
    for sector in set(curr_scores) | set(prior_scores):
        curr = curr_scores.get(sector, {"count": 0, "weighted": 0.0})
        if curr["count"] < min_events:
            continue
        prev = prior_scores.get(sector, {"count": 0, "weighted": 0.0})
        momentum = curr["weighted"] - prev["weighted"]
        rotation[sector] = round(momentum / math.sqrt(curr["count"] + 1), 4)

    if not rotation:
        return {}

    max_abs = max(abs(v) for v in rotation.values())
    if max_abs > 0:
        rotation = {k: round(v / max_abs, 4) for k, v in rotation.items()}
    return rotation


def compute_earnings_proximity_boost(
    events_df: pd.DataFrame,
    quarter_end_months: tuple[int, ...] = (3, 6, 9, 12),
    proximity_days: int = 14,
) -> pd.DataFrame:
    """Add earnings_proximity_boost column to events near quarter-end (Point 32).

    Events within proximity_days of a quarter-end get a boost 1.0–1.5.
    """
    if events_df.empty:
        return events_df.copy()

    import calendar as _cal

    df = events_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    boosts = []
    for ts in df["timestamp"]:
        month = ts.month
        day = ts.day
        boost = 1.0
        for qe_month in quarter_end_months:
            last_day = _cal.monthrange(ts.year, qe_month)[1]
            if month == qe_month and day >= (last_day - proximity_days):
                days_remaining = last_day - day
                boost = max(boost, 1.0 + 0.5 * (1.0 - days_remaining / proximity_days))
            elif month == qe_month - 1 and qe_month > 1:
                last_day_this = _cal.monthrange(ts.year, month)[1]
                days_until_qe_start = last_day_this - day
                total_proximity = proximity_days - days_until_qe_start
                if total_proximity > 0:
                    boost = max(boost, 1.0 + 0.3 * (total_proximity / proximity_days))
        boosts.append(round(boost, 3))

    df["earnings_proximity_boost"] = boosts
    return df
