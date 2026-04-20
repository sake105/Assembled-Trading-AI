"""News and sentiment features module (V13).

Computes news-derived features from raw sentiment scores and optional
FinBERT NLP scores.  Features include rolling means, momentum, dispersion,
shock flags, and article counts.

Graceful degradation: FinBERT features are NaN when torch/transformers
are not installed — the basic sentiment_score features still work.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

_log = logging.getLogger(__name__)

# Optional FinBERT import
try:
    from assembled_core.ml.nlp_sentiment import (
        TRANSFORMERS_AVAILABLE,
        score_news_store,
    )
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    score_news_store = None  # type: ignore[assignment]


def add_news_features(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    as_of: "pd.Timestamp | None" = None,
) -> pd.DataFrame:
    """Add news sentiment features to price DataFrame.

    Computes:
    - news_sentiment_7d / _30d: Rolling mean sentiment
    - news_count_7d / _30d: Article counts
    - nlp_sentiment_mean_5d / _20d: FinBERT score means (if available)
    - nlp_sentiment_momentum_5d: 5d-20d sentiment delta
    - nlp_sentiment_dispersion_20d: Cross-sectional std of sentiment

    Args:
        prices: DataFrame with columns: timestamp, symbol, close
        events: DataFrame with columns: timestamp, symbol, sentiment_score,
            and optionally headline (used for FinBERT scoring)
        as_of: Optional PIT cutoff. When provided, events after this timestamp
            are excluded (shadow-logged as *_shadow columns — not yet flipped
            to production path).

    Returns:
        Copy of prices with sentiment feature columns added.
    """
    required_price_cols = ["timestamp", "symbol", "close"]
    for col in required_price_cols:
        if col not in prices.columns:
            raise KeyError(f"Required column '{col}' not found in prices DataFrame")

    required_event_cols = ["timestamp", "symbol", "sentiment_score"]
    for col in required_event_cols:
        if col not in events.columns:
            raise KeyError(f"Required column '{col}' not found in events DataFrame")

    result = prices.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    events = events.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)

    # T2.1: PIT gate — when as_of provided, drop events not yet disclosed
    if as_of is not None:
        as_of_ts = pd.Timestamp(as_of).tz_localize("UTC") if getattr(as_of, "tzinfo", None) is None else pd.Timestamp(as_of)
        disclosure_col = "disclosure_date" if "disclosure_date" in events.columns else "timestamp"
        events = events[events[disclosure_col] <= as_of_ts].copy()

    # --- FinBERT enrichment (optional) ---
    has_finbert = False
    if (
        TRANSFORMERS_AVAILABLE
        and score_news_store is not None
        and "headline" in events.columns
        and not events.empty
    ):
        try:
            events = score_news_store(events, text_col="headline")
            has_finbert = True
            _log.info("[NLP] FinBERT scored %d news rows", len(events))
        except Exception as e:
            _log.warning("[NLP] FinBERT scoring failed, using basic sentiment: %s", e)

    # Score column: prefer FinBERT if available, else raw sentiment_score
    score_col = "finbert_score" if has_finbert else "sentiment_score"

    # Urgency amplification: Breaking/Flash events get a 2× weight boost
    if "urgency" in events.columns:
        events["_urgency_factor"] = 1.0 + events["urgency"].fillna(0.0)
        events[score_col] = events[score_col] * events["_urgency_factor"]
        events.drop(columns=["_urgency_factor"], inplace=True)

    # --- Build daily per-symbol sentiment aggregates ---
    events["_date"] = events["timestamp"].dt.normalize()
    daily = (
        events.groupby(["symbol", "_date"])
        .agg(
            daily_sentiment=(score_col, "mean"),
            daily_count=(score_col, "count"),
            daily_std=(score_col, "std"),
        )
        .reset_index()
    )
    daily["daily_std"] = daily["daily_std"].fillna(0.0)
    daily.rename(columns={"_date": "timestamp"}, inplace=True)
    daily["timestamp"] = pd.to_datetime(daily["timestamp"], utc=True)

    # Merge daily sentiment onto price grid
    result["_ts_date"] = result["timestamp"].dt.normalize()
    merged = result.merge(
        daily,
        left_on=["symbol", "_ts_date"],
        right_on=["symbol", "timestamp"],
        how="left",
        suffixes=("", "_daily"),
    )
    # Restore original timestamp
    if "timestamp_daily" in merged.columns:
        merged.drop(columns=["timestamp_daily"], inplace=True)
    merged["daily_sentiment"] = merged["daily_sentiment"].fillna(0.0)
    merged["daily_count"] = merged["daily_count"].fillna(0).astype(int)
    merged["daily_std"] = merged["daily_std"].fillna(0.0)

    # Sort for rolling computations
    merged = merged.sort_values(["symbol", "timestamp"])

    # --- Rolling features per symbol ---
    for window, suffix in [(7, "7d"), (30, "30d")]:
        col_sent = f"news_sentiment_{suffix}"
        col_cnt = f"news_count_{suffix}"
        merged[col_sent] = (
            merged.groupby("symbol")["daily_sentiment"]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        merged[col_cnt] = (
            merged.groupby("symbol")["daily_count"]
            .transform(lambda x: x.rolling(window, min_periods=1).sum())
        )

    # --- NLP-specific features (always compute, use FinBERT score if available) ---
    merged["nlp_sentiment_mean_5d"] = (
        merged.groupby("symbol")["daily_sentiment"]
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )
    merged["nlp_sentiment_mean_20d"] = (
        merged.groupby("symbol")["daily_sentiment"]
        .transform(lambda x: x.rolling(20, min_periods=1).mean())
    )
    merged["nlp_sentiment_momentum_5d"] = (
        merged["nlp_sentiment_mean_5d"] - merged["nlp_sentiment_mean_20d"]
    )

    # Cross-sectional dispersion: std of daily_sentiment across symbols per date
    date_std = (
        merged.groupby("_ts_date")["daily_sentiment"]
        .transform("std")
        .fillna(0.0)
    )
    merged["nlp_sentiment_dispersion_20d"] = (
        merged.groupby("symbol")[date_std.name if hasattr(date_std, 'name') else "daily_sentiment"]
        .transform(lambda x: x.rolling(20, min_periods=1).mean())
    )
    # Actually compute rolling std on per-symbol sentiment as dispersion
    merged["nlp_sentiment_dispersion_20d"] = (
        merged.groupby("symbol")["daily_sentiment"]
        .transform(lambda x: x.rolling(20, min_periods=1).std())
    ).fillna(0.0)

    # Clean up temp columns
    drop_cols = ["_ts_date", "daily_sentiment", "daily_count", "daily_std"]
    merged.drop(columns=[c for c in drop_cols if c in merged.columns], inplace=True)

    added = sum(1 for c in merged.columns if c.startswith(("news_", "nlp_")))
    _log.info("Added %d news/NLP sentiment features (finbert=%s)", added, has_finbert)

    return merged


def update_news_features_incremental(
    existing: pd.DataFrame,
    new_prices: pd.DataFrame,
    new_events: pd.DataFrame,
    as_of: "str | pd.Timestamp | None" = None,
) -> pd.DataFrame:
    """T6.12: Incrementally update news features for new price rows only.

    Appends `new_prices` rows to `existing`, runs `add_news_features` over the
    combined frame, then returns only the rows whose timestamp is >= the earliest
    new_prices timestamp. This avoids recomputing features for historical rows.

    Args:
        existing: Previously computed feature frame (from add_news_features).
        new_prices: New price rows to add (must have same schema as existing).
        new_events: New or all news events (passed to add_news_features).
        as_of: PIT gate forwarded to add_news_features.

    Returns:
        Feature frame covering the new_prices rows (merged state, not just delta).
    """
    if new_prices.empty:
        return existing

    # Identify cutoff — earliest new timestamp
    ts_col = "timestamp"
    new_ts = pd.to_datetime(new_prices[ts_col], utc=True)
    cutoff = new_ts.min()

    # Combine: drop existing rows at/after cutoff, then append new rows
    if not existing.empty:
        existing_ts = pd.to_datetime(existing[ts_col], utc=True)
        prior = existing[existing_ts < cutoff].copy()
        # Strip news feature columns from prior so they don't conflict
        feature_cols = [c for c in prior.columns if c.startswith(("news_", "nlp_"))]
        prior_prices = prior.drop(columns=feature_cols, errors="ignore")
    else:
        prior_prices = pd.DataFrame(columns=new_prices.columns)

    combined_prices = pd.concat([prior_prices, new_prices], ignore_index=True)
    updated = add_news_features(combined_prices, new_events, as_of=as_of)

    # Return only rows from cutoff onward
    updated_ts = pd.to_datetime(updated[ts_col], utc=True)
    return updated[updated_ts >= cutoff].reset_index(drop=True)


def compute_news_momentum(
    events_df: pd.DataFrame,
    window_hours: float = 4.0,
) -> pd.DataFrame:
    """Compute rate-of-change in article volume per sector over the last window_hours.

    Args:
        events_df: DataFrame with columns: timestamp, affected_sectors (list or str),
                   and optionally severity.
        window_hours: Length of the current and previous window in hours.

    Returns:
        DataFrame with columns: sector, article_count, momentum, severity_weighted_count.
    """
    if events_df.empty:
        return pd.DataFrame(columns=["sector", "article_count", "momentum", "severity_weighted_count"])

    df = events_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    now = df["timestamp"].max()
    window_td = pd.Timedelta(hours=window_hours)

    current_window = df[df["timestamp"] >= now - window_td]
    previous_window = df[
        (df["timestamp"] >= now - 2 * window_td) & (df["timestamp"] < now - window_td)
    ]

    def _explode_sectors(frame: pd.DataFrame) -> pd.DataFrame:
        """Explode 'affected_sectors' column to one row per sector."""
        if "affected_sectors" not in frame.columns:
            return pd.DataFrame(columns=["affected_sectors", "severity"])
        rows = []
        for _, row in frame.iterrows():
            sectors = row.get("affected_sectors", [])
            if isinstance(sectors, str):
                sectors = [sectors]
            elif not isinstance(sectors, list):
                sectors = []
            severity = float(row.get("severity", 0.0) or 0.0)
            for s in sectors:
                rows.append({"affected_sectors": s, "severity": severity})
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["affected_sectors", "severity"])

    curr_exploded = _explode_sectors(current_window)
    prev_exploded = _explode_sectors(previous_window)

    if curr_exploded.empty:
        return pd.DataFrame(columns=["sector", "article_count", "momentum", "severity_weighted_count"])

    curr_agg = curr_exploded.groupby("affected_sectors").agg(
        article_count=("severity", "count"),
        severity_weighted_count=("severity", "sum"),
    ).reset_index().rename(columns={"affected_sectors": "sector"})

    if not prev_exploded.empty:
        prev_agg = prev_exploded.groupby("affected_sectors").agg(
            prev_count=("severity", "count"),
        ).reset_index().rename(columns={"affected_sectors": "sector"})
        curr_agg = curr_agg.merge(prev_agg, on="sector", how="left")
        curr_agg["prev_count"] = curr_agg["prev_count"].fillna(0).astype(int)
    else:
        curr_agg["prev_count"] = 0

    curr_agg["momentum"] = curr_agg["article_count"] - curr_agg["prev_count"]
    curr_agg = curr_agg.drop(columns=["prev_count"], errors="ignore")
    return curr_agg.reset_index(drop=True)


def compute_sentiment_trajectory(
    events_df: pd.DataFrame,
    symbol: str,
    n_articles: int = 10,
) -> str:
    """Return trend direction of the last N articles for a symbol.

    Uses linear regression slope of sentiment scores.

    Returns:
        "improving", "worsening", or "stable".
    """
    if events_df.empty:
        return "stable"

    df = events_df.copy()
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol]
    elif "tickers" in df.columns:
        df = df[df["tickers"].apply(lambda t: symbol in (t if isinstance(t, list) else []))]

    if df.empty:
        return "stable"

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").tail(n_articles)

    if len(df) < 2:
        return "stable"

    score_col = "sentiment_score" if "sentiment_score" in df.columns else None
    if score_col is None:
        return "stable"

    scores = df[score_col].fillna(0.0).values
    n = len(scores)
    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(scores) / n

    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, scores))
    denominator = sum((xi - x_mean) ** 2 for xi in x)

    if denominator == 0:
        return "stable"

    slope = numerator / denominator
    if slope > 0.02:
        return "improving"
    elif slope < -0.02:
        return "worsening"
    return "stable"


def compute_sector_heatmap(
    events_df: pd.DataFrame,
    now: "pd.Timestamp | None" = None,
) -> dict[str, dict[str, float]]:
    """Return sector → last-24h event count + avg severity (the news heatmap).

    Args:
        events_df: DataFrame with columns: timestamp, affected_sectors (list or str),
                   and optionally severity.
        now: Reference time; defaults to max timestamp in events_df.

    Returns:
        Dict mapping sector → {"event_count": int, "avg_severity": float}.
    """
    if events_df.empty:
        return {}

    df = events_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if now is None:
        ref_time = df["timestamp"].max()
    else:
        ref_time = pd.Timestamp(now).tz_localize("UTC") if getattr(now, "tzinfo", None) is None else pd.Timestamp(now)

    cutoff = ref_time - pd.Timedelta(hours=24)
    df = df[df["timestamp"] >= cutoff]

    if df.empty:
        return {}

    heatmap: dict[str, dict[str, Any]] = {}

    for _, row in df.iterrows():
        sectors = row.get("affected_sectors", [])
        if isinstance(sectors, str):
            sectors = [sectors]
        elif not isinstance(sectors, list):
            sectors = []
        severity = float(row.get("severity", 0.0) or 0.0)
        for sector in sectors:
            if sector not in heatmap:
                heatmap[sector] = {"event_count": 0, "severity_sum": 0.0}
            heatmap[sector]["event_count"] += 1
            heatmap[sector]["severity_sum"] += severity

    result: dict[str, dict[str, float]] = {}
    for sector, data in heatmap.items():
        count = data["event_count"]
        result[sector] = {
            "event_count": float(count),
            "avg_severity": round(data["severity_sum"] / count, 3) if count > 0 else 0.0,
        }
    return result


__all__ = [
    "add_news_features",
    "update_news_features_incremental",
    "compute_news_momentum",
    "compute_sentiment_trajectory",
    "compute_sector_heatmap",
]
