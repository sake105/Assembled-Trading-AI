"""Alt-Data Feature Builder: Wikipedia Pageview Attention Features.

Transforms Wikipedia pageview data (from wikipedia_views_source) into
per-symbol ML-ready attention features.

Based on Moat et al. (2013, Nature Scientific Reports): Wikipedia pageviews
with a 1-day lag predict price drawdowns. Particularly useful for
small/mid-cap stocks with limited news coverage.

Wikipedia source (fetch_article_views) returns:
    DataFrame with datetime index, article names as columns, view counts as values.

This builder expects a tidy long-format DataFrame with columns:
    symbol (or article), date (or timestamp), views (int/float)

Audit: C2-059
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "build_wikipedia_attention_features",
]

# Attention regime thresholds (z-score based)
_HIGH_ATTENTION_ZSCORE = 2.0
_LOW_ATTENTION_ZSCORE = -1.0

# Rolling windows for feature computation
_SHORT_WINDOW = 7  # days for recent mean
_LONG_WINDOW = 30  # days for baseline mean (used for z-score reference)

_OUTPUT_COLS = [
    "symbol",
    "pageview_zscore",
    "pageview_7d_change",
    "attention_spike",
    "attention_regime",
]


def build_wikipedia_attention_features(
    wiki_df: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Build Wikipedia attention features per symbol.

    PIT-safe: only uses pageview data up to and including as_of.

    Accepts two input formats:
    1. Wide format (from fetch_article_views): DatetimeIndex, one column per article/symbol.
    2. Long format: columns ``symbol`` (or ``article``), ``date`` (or ``timestamp``), ``views``.

    For each symbol, computes:
    - ``pageview_zscore``: (7d mean − 30d mean) / 30d std, measuring recent attention vs baseline.
    - ``pageview_7d_change``: relative change = (7d mean / 30d mean) − 1.
    - ``attention_spike``: True when |pageview_zscore| > 2σ.
    - ``attention_regime``: "high" / "normal" / "low" based on zscore.

    Args:
        wiki_df: Pageview data in wide or long format (see above).
        as_of: PIT cutoff — data after this date is excluded.

    Returns:
        DataFrame with columns:
            symbol, pageview_zscore (float), pageview_7d_change (float),
            attention_spike (bool), attention_regime (str: high/normal/low).
        Empty DataFrame with correct columns when input is empty or
        insufficient data for computation.
    """
    empty = pd.DataFrame(columns=_OUTPUT_COLS)
    empty["attention_spike"] = empty["attention_spike"].astype(bool)

    if wiki_df is None or wiki_df.empty:
        return empty.copy()

    as_of_ts = pd.Timestamp(as_of)

    # --- Normalise to long format: symbol, date, views ---
    df_long = _to_long_format(wiki_df, as_of_ts)

    if df_long.empty:
        return empty.copy()

    # PIT filter
    df_long = df_long[df_long["date"] <= as_of_ts.normalize()].copy()

    if df_long.empty:
        return empty.copy()

    rows: list[dict] = []

    for symbol, sym_df in df_long.groupby("symbol", sort=False):
        sym_df = sym_df.sort_values("date")
        views = sym_df.set_index("date")["views"].astype(float)

        # Need enough data for the long window
        if len(views) < _SHORT_WINDOW + 1:
            rows.append(_null_row(symbol))
            continue

        recent = views.iloc[-_SHORT_WINDOW:]
        baseline = views.iloc[-_LONG_WINDOW:] if len(views) >= _LONG_WINDOW else views

        recent_mean = float(recent.mean())
        long_mean = float(baseline.mean())
        long_std = float(baseline.std(ddof=1)) if len(baseline) > 1 else 0.0

        # pageview_zscore
        if long_std > 0:
            zscore = (recent_mean - long_mean) / long_std
        else:
            zscore = 0.0

        # pageview_7d_change: relative change vs 30d baseline
        if long_mean > 0:
            change_7d = recent_mean / long_mean - 1.0
        else:
            change_7d = 0.0

        # attention_spike: |zscore| > 2
        spike = abs(zscore) >= _HIGH_ATTENTION_ZSCORE

        # attention_regime
        if zscore >= _HIGH_ATTENTION_ZSCORE:
            regime = "high"
        elif zscore <= _LOW_ATTENTION_ZSCORE:
            regime = "low"
        else:
            regime = "normal"

        rows.append(
            {
                "symbol": symbol,
                "pageview_zscore": zscore,
                "pageview_7d_change": change_7d,
                "attention_spike": spike,
                "attention_regime": regime,
            }
        )

    if not rows:
        return empty.copy()

    result = pd.DataFrame(rows)[_OUTPUT_COLS].copy()
    result["attention_spike"] = result["attention_spike"].astype(bool)
    return result.reset_index(drop=True)


def _to_long_format(wiki_df: pd.DataFrame, as_of_ts: pd.Timestamp) -> pd.DataFrame:
    """Coerce wide or long Wikipedia pageview DataFrame to tidy long format.

    Returns DataFrame with columns: symbol (str), date (date, tz-naive), views (float).
    Returns empty DataFrame on failure.
    """
    try:
        # Long format: must have symbol/article and date/timestamp and views columns
        has_symbol = "symbol" in wiki_df.columns or "article" in wiki_df.columns
        has_date = "date" in wiki_df.columns or "timestamp" in wiki_df.columns
        has_views = "views" in wiki_df.columns

        if has_symbol and has_date and has_views:
            df = wiki_df.copy()
            if "article" in df.columns and "symbol" not in df.columns:
                df = df.rename(columns={"article": "symbol"})
            if "timestamp" in df.columns and "date" not in df.columns:
                df = df.rename(columns={"timestamp": "date"})
            _dt = pd.to_datetime(df["date"])
            if _dt.dt.tz is not None:
                _dt = _dt.dt.tz_convert(None)
            df["date"] = _dt.dt.normalize()
            df["views"] = pd.to_numeric(df["views"], errors="coerce").fillna(0.0)
            return df[["symbol", "date", "views"]].copy()

        # Wide format: DatetimeIndex (or date index), columns = symbols/articles
        # Check if index is datetime-like
        if isinstance(wiki_df.index, pd.DatetimeIndex) or _index_is_datetime(wiki_df):
            df = wiki_df.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            _idx = df.index.normalize()
            if _idx.tz is not None:
                _idx = _idx.tz_convert(None)
            df.index = _idx
            df.index.name = "date"

            # Melt to long format
            long = df.reset_index().melt(
                id_vars="date", var_name="symbol", value_name="views"
            )
            long["views"] = pd.to_numeric(long["views"], errors="coerce").fillna(0.0)
            return long[["symbol", "date", "views"]].copy()

        logger.warning(
            "[WIKI] Cannot parse DataFrame shape — expected wide (DatetimeIndex) "
            "or long (symbol/date/views columns)."
        )
        return pd.DataFrame(columns=["symbol", "date", "views"])

    except Exception as exc:  # noqa: BLE001
        logger.warning("[WIKI] _to_long_format error: %s", exc)
        return pd.DataFrame(columns=["symbol", "date", "views"])


def _index_is_datetime(df: pd.DataFrame) -> bool:
    """Return True if the DataFrame index can be coerced to DatetimeIndex."""
    try:
        pd.to_datetime(df.index[:3])
        return True
    except Exception:  # noqa: BLE001
        return False


def _null_row(symbol: object) -> dict:
    """Return a row with NaN features for a symbol with insufficient data."""
    return {
        "symbol": symbol,
        "pageview_zscore": float("nan"),
        "pageview_7d_change": float("nan"),
        "attention_spike": False,
        "attention_regime": "normal",
    }
