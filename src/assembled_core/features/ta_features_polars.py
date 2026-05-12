"""Polars-backed TA features (audit B-001 / §8.2).

Drop-in alternatives to the pandas implementations in
:mod:`features.ta_features`. Each function accepts and returns a
**pandas** DataFrame so call-sites do not need to change — the Polars
LazyFrame is constructed internally, executed, and materialized back
to pandas before return.

Why a parallel module and not in-place replacement:
    * Existing pandas path is well-tested and well-known; numerical
      differences in floating-point order of operations can shift
      downstream backtest equity by basis points.
    * Some callers parse the resulting DataFrame's metadata in
      pandas-specific ways. Two paths let consumers opt in one at a
      time.

Tests in ``tests/test_ta_features_polars_equivalence.py`` pin
numerical equivalence with the pandas versions to within 1e-9
tolerance.

When to use:
    * Large universes (~500+ symbols × multi-year history). The Polars
      group-over-symbol expressions dispatch into Rust and parallelize
      across cores automatically.
    * Memory-constrained runs: LazyFrame avoids the intermediate
      groupby-rolling-reset cycles that pandas materializes.

Reference: Polars 1.x stable, MIT license. No new optional dep
contract — polars is now a required test dep (see pyproject.toml).
"""

from __future__ import annotations

import logging

import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


def add_log_returns_polars(
    df: pd.DataFrame,
    price_col: str = "close",
    out_col: str | None = None,
    use_namespace: bool = True,
) -> pd.DataFrame:
    """Polars-backed equivalent of :func:`ta_features.add_log_returns`."""
    if "symbol" not in df.columns:
        raise KeyError("symbol")
    if price_col not in df.columns:
        raise KeyError(
            f"Price column '{price_col}' not found. Available columns: {list(df.columns)}"
        )
    if out_col is None:
        out_col = "ta_log_return_v1" if use_namespace else "log_return"

    # Preserve the caller's row order via a positional index column.
    df_in = df.reset_index(drop=False).rename(columns={"index": "__row_idx__"})
    sort_cols = ["symbol"]
    if "timestamp" in df.columns:
        sort_cols.append("timestamp")

    lf = pl.from_pandas(df_in).lazy()
    lf = lf.sort(sort_cols)
    lf = lf.with_columns(
        (pl.col(price_col).cast(pl.Float64).clip(lower_bound=1e-10).log())
        .diff()
        .over("symbol")
        .alias(out_col)
    )
    out = lf.collect().to_pandas()
    out = (
        out.sort_values("__row_idx__")
        .drop(columns="__row_idx__")
        .reset_index(drop=True)
    )
    # Legacy mirror.
    if (
        use_namespace
        and out_col == "ta_log_return_v1"
        and "log_return" not in out.columns
    ):
        out["log_return"] = out[out_col]
    return out


def add_moving_averages_polars(
    df: pd.DataFrame,
    windows: tuple[int, ...] = (20, 50, 200),
    price_col: str = "close",
    use_namespace: bool = True,
) -> pd.DataFrame:
    """Polars-backed equivalent of :func:`ta_features.add_moving_averages`.

    Returns the DataFrame sorted by (symbol, timestamp) — same behavior
    as the pandas path which calls ``sort_values(...)`` before computing.
    """
    if price_col not in df.columns:
        raise KeyError(
            f"Price column '{price_col}' not found. Available columns: {list(df.columns)}"
        )

    has_symbol = "symbol" in df.columns
    lf = pl.from_pandas(df).lazy()
    sort_cols = ["symbol", "timestamp"] if has_symbol else ["timestamp"]
    sort_cols = [c for c in sort_cols if c in df.columns]
    if sort_cols:
        lf = lf.sort(sort_cols)

    exprs = []
    for window in windows:
        col_name = f"ta_ma_{window}_v1" if use_namespace else f"ma_{window}"
        roll_expr = (
            pl.col(price_col)
            .cast(pl.Float64)
            .rolling_mean(window_size=window, min_samples=1)
        )
        if has_symbol:
            roll_expr = roll_expr.over("symbol")
        exprs.append(roll_expr.alias(col_name))
    lf = lf.with_columns(exprs)
    out = lf.collect().to_pandas().reset_index(drop=True)

    # Legacy mirrors to match the pandas function's compatibility behavior.
    if use_namespace:
        for window in windows:
            ns = f"ta_ma_{window}_v1"
            legacy = f"ma_{window}"
            if ns in out.columns and legacy not in out.columns:
                out[legacy] = out[ns]
    return out


def add_atr_polars(
    df: pd.DataFrame,
    window: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> pd.DataFrame:
    """Polars-backed equivalent of :func:`ta_features.add_atr`.

    Returns the DataFrame in the caller's original row order (the
    pandas path also does so via ``reindex(result.index)``).
    """
    required = ["symbol", high_col, low_col, close_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")

    df_in = df.reset_index(drop=False).rename(columns={"index": "__row_idx__"})
    sort_cols = ["symbol"]
    if "timestamp" in df.columns:
        sort_cols.append("timestamp")

    lf = pl.from_pandas(df_in).lazy().sort(sort_cols)

    high = pl.col(high_col).cast(pl.Float64)
    low = pl.col(low_col).cast(pl.Float64)
    close = pl.col(close_col).cast(pl.Float64)
    prev_close = close.shift(1).over("symbol")

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    # Polars equivalent of pd.concat([t1, t2, t3], axis=1).max(axis=1):
    true_range = pl.max_horizontal(tr1, tr2, tr3)

    atr = true_range.rolling_mean(window_size=window, min_samples=1).over("symbol")

    ns_col = f"ta_atr_{window}_v1"
    legacy_col = f"atr_{window}"

    lf = lf.with_columns(atr.alias(ns_col))
    out = lf.collect().to_pandas()
    out = (
        out.sort_values("__row_idx__")
        .drop(columns="__row_idx__")
        .reset_index(drop=True)
    )
    if legacy_col not in out.columns:
        out[legacy_col] = out[ns_col]
    return out


def add_rsi_polars(
    df: pd.DataFrame,
    window: int = 14,
    price_col: str = "close",
) -> pd.DataFrame:
    """Polars-backed equivalent of :func:`ta_features.add_rsi`.

    Uses the simple-moving-average Wilder variant
    (``avg_gain / avg_loss``) — matches the pandas implementation.
    """
    if "symbol" not in df.columns:
        raise KeyError("symbol")
    if price_col not in df.columns:
        raise KeyError(f"Price column '{price_col}' not found")

    df_in = df.reset_index(drop=False).rename(columns={"index": "__row_idx__"})
    sort_cols = ["symbol"]
    if "timestamp" in df.columns:
        sort_cols.append("timestamp")

    lf = pl.from_pandas(df_in).lazy().sort(sort_cols)

    delta = pl.col(price_col).cast(pl.Float64).diff().over("symbol")
    # Match pandas: gain = delta.clip(lower=0); loss = -delta.clip(upper=0).
    # Preserve NaN in delta so the first-per-symbol row stays NaN and is NOT
    # counted toward the rolling-window sample count (pandas treats it the
    # same way via diff()).
    gain = (
        pl.when(delta.is_null()).then(None).when(delta > 0).then(delta).otherwise(0.0)
    )
    loss = (
        pl.when(delta.is_null()).then(None).when(delta < 0).then(-delta).otherwise(0.0)
    )
    # Match pandas: min_periods=window (NaN until the window is full),
    # and NO 1e-12 fallback on zero avg_loss — keep the division to
    # propagate inf / NaN exactly like pandas does.
    avg_gain = gain.rolling_mean(window_size=window, min_samples=window).over("symbol")
    avg_loss = loss.rolling_mean(window_size=window, min_samples=window).over("symbol")

    rs = avg_gain / avg_loss
    rsi_expr = 100.0 - (100.0 / (1.0 + rs))

    ns_col = f"ta_rsi_{window}_v1"
    legacy_col = f"rsi_{window}"

    lf = lf.with_columns(rsi_expr.alias(ns_col))
    out = lf.collect().to_pandas()
    out = (
        out.sort_values("__row_idx__")
        .drop(columns="__row_idx__")
        .reset_index(drop=True)
    )
    if legacy_col not in out.columns:
        out[legacy_col] = out[ns_col]
    return out


def is_polars_available() -> bool:
    """Diagnostic helper — used by tests and benchmark scripts."""
    try:
        _ = pl.__version__  # noqa: F841
        return True
    except Exception:  # pragma: no cover
        return False


__all__ = [
    "add_log_returns_polars",
    "add_moving_averages_polars",
    "add_atr_polars",
    "add_rsi_polars",
    "is_polars_available",
]
