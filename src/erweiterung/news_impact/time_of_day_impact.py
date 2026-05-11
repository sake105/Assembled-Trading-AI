"""Time-of-Day News-Impact Modeling.

Empirisch
---------
News during market hours wirkt anders als pre-market oder after-hours:
1. **Pre-Market News** (4:00-9:30 ET): Oft kompletter Gap; Pricing via Auktion bei Open.
2. **Intra-Day News** (9:30-16:00): Sofortige Reaktion + Continuation oder Fade.
3. **After-Hours News** (16:00-20:00): Earnings, Material-Events;
   Reaktion erfolgt am nächsten Morgen-Open.

Anwendung
---------
- Separat trainierte Decay-Modelle je Time-of-Day-Bucket.
- Open-vs-Close-Return-Decomposition (Open-Gap = Pre-Market-News-Reaction).
"""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd


def classify_time_of_day(
    timestamps: pd.Series,
    market_open: time = time(9, 30),
    market_close: time = time(16, 0),
    market_tz: str = "US/Eastern",
) -> pd.Series:
    """Classify timestamps as 'pre_market', 'intraday', 'after_hours'.

    Args:
        timestamps: Series of datetimes. tz-aware ⇒ wird zu ``market_tz`` konvertiert
            vor Time-of-Day-Extraktion. Naive ⇒ wird als bereits in ``market_tz``
            angenommen (kein silent shift).
        market_open / market_close: Öffnungs-/Schlusszeit in ``market_tz``.
        market_tz: Börsen-Zeitzone (default US/Eastern für NYSE/Nasdaq).
    """
    ts = pd.to_datetime(timestamps)
    # tz-aware → convert to market timezone; tz-naive → leave as-is
    accessor = ts.dt if hasattr(ts, "dt") else ts
    tz = getattr(accessor, "tz", None) if not hasattr(ts, "dt") else ts.dt.tz
    if tz is not None:
        ts = (
            ts.dt.tz_convert(market_tz)
            if hasattr(ts, "dt")
            else ts.tz_convert(market_tz)
        )
    out = pd.Series("intraday", index=ts.index, dtype=object)
    times_of_day = ts.dt.time
    out[times_of_day < market_open] = "pre_market"
    out[times_of_day >= market_close] = "after_hours"
    return out


def split_returns_by_session(
    ohlc_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Decompose daily returns into overnight (close→open) and intraday (open→close).

    Args:
        ohlc_panel: DataFrame [date, symbol, open, close].

    Returns:
        DataFrame with added ``overnight_return`` and ``intraday_return``.
    """
    df = ohlc_panel.copy().sort_values(["symbol", "date"])
    df["prev_close"] = df.groupby("symbol")["close"].shift(1)
    df["overnight_return"] = df["open"] / df["prev_close"] - 1
    df["intraday_return"] = df["close"] / df["open"] - 1
    return df


def news_impact_by_session(
    news_df: pd.DataFrame,
    ohlc_panel: pd.DataFrame,
    sentiment_col: str = "sentiment",
    timestamp_col: str = "timestamp",
) -> dict:
    """Average return-impact per news-session-bucket.

    Args:
        news_df: with ``timestamp`` (US-Eastern), ``symbol``, ``sentiment``.
        ohlc_panel: OHLC daily.

    Returns:
        dict mit per-session impact-statistics.
    """
    news = news_df.copy()
    news[timestamp_col] = pd.to_datetime(news[timestamp_col])
    news["session"] = classify_time_of_day(news[timestamp_col])
    news["date"] = news[timestamp_col].dt.normalize()

    ret_df = split_returns_by_session(ohlc_panel)
    ret_df["date"] = pd.to_datetime(ret_df["date"]).dt.normalize()

    merged = news.merge(
        ret_df[["date", "symbol", "overnight_return", "intraday_return"]],
        on=["date", "symbol"],
        how="left",
    )

    results: dict = {}
    for session_label, group in merged.groupby("session"):
        s = group.dropna(subset=[sentiment_col])
        if s.empty:
            continue
        # Pre-market news → next overnight return effect (gap-up/down)
        # Intraday news → same-day intraday return
        # After-hours news → next overnight
        if session_label == "intraday":
            target = s["intraday_return"]
        else:
            # overnight following the news
            target = s["overnight_return"]
        df = pd.concat(
            [s[sentiment_col].rename("s"), target.rename("r")], axis=1
        ).dropna()
        if df.empty:
            continue
        # OLS: r = α + β · s
        X = np.column_stack([np.ones(len(df)), df["s"].values])
        y = df["r"].values
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            corr = float(df["s"].corr(df["r"])) if df["s"].std() > 0 else float("nan")
            results[session_label] = {
                "alpha": float(beta[0]),
                "beta": float(beta[1]),
                "correlation": corr,
                "n_obs": int(len(df)),
                "mean_return": float(df["r"].mean()),
                "mean_sentiment": float(df["s"].mean()),
            }
        except np.linalg.LinAlgError:
            continue
    return results


__all__ = [
    "classify_time_of_day",
    "split_returns_by_session",
    "news_impact_by_session",
]
