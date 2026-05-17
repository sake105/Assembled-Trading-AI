"""Event Study Engine for analyzing price reactions to events.

This module provides functions for event study analysis:
- Extracting price windows around events
- Computing normal and abnormal returns (Mean-Adjusted)
- Market-Model abnormal returns (OLS regression-based) — C4-081
- BMP (Boehmer-Musumeci-Poulsen 1991) standardised cross-sectional t-statistic
- BHAR (Buy-and-Hold Abnormal Returns) for long-horizon studies
- Aggregating results across events

Part of Phase C3: Event Study Framework. Audit C4-081 closure (2026-05-17).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_event_window_prices(
    prices_df: pd.DataFrame,
    events_df: pd.DataFrame,
    window_before: int = 20,
    window_after: int = 40,
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
    price_col: str = "close",
    event_id_col: str | None = None,
    event_type_col: str = "event_type",
) -> pd.DataFrame:
    """Extract price windows around events.

    For each event, extracts prices from `window_before` days before to `window_after` days after
    the event. Returns a "stacked" DataFrame with one row per (event, relative_day).

    Args:
        prices_df: Panel DataFrame with at least timestamp, symbol, close (and optionally
            other columns like factors). Must be sorted by symbol, then timestamp.
        events_df: DataFrame with timestamp, symbol, event_type (and optionally event_id).
            If event_id is not present, it will be generated.
        window_before: Number of days before event to include (default: 20)
        window_after: Number of days after event to include (default: 40)
        group_col: Column name for grouping (default: "symbol")
        timestamp_col: Column name for timestamp (default: "timestamp")
        price_col: Column name for price (default: "close")
        event_id_col: Column name for event ID (default: None, will generate if missing)
        event_type_col: Column name for event type (default: "event_type")

    Returns:
        DataFrame with columns:
        - event_id: Unique event identifier
        - symbol: Symbol
        - event_type: Event type
        - event_timestamp: Event timestamp
        - rel_day: Relative day (-window_before to +window_after, 0 = event day)
        - timestamp: Actual timestamp for this day
        - close: Price at this timestamp (or price_col value)
        - Additional columns from prices_df (e.g., open, high, low, volume, factors)

    Raises:
        KeyError: If required columns are missing
        ValueError: If timestamps are not UTC-aware

    Example:
        >>> events = pd.DataFrame({
        ...     "timestamp": [pd.Timestamp("2024-01-15", tz="UTC")],
        ...     "symbol": ["AAPL"],
        ...     "event_type": ["earnings"]
        ... })
        >>> prices = load_prices(...)  # Panel with timestamp, symbol, close
        >>> windows = build_event_window_prices(prices, events, window_before=5, window_after=5)
        >>> # Result has rows for rel_day = -5, -4, ..., 0, ..., +5
    """
    # Validate inputs
    required_price_cols = [timestamp_col, group_col, price_col]
    for col in required_price_cols:
        if col not in prices_df.columns:
            raise KeyError(f"Required column '{col}' not found in prices_df")

    required_event_cols = [timestamp_col, group_col, event_type_col]
    for col in required_event_cols:
        if col not in events_df.columns:
            raise KeyError(f"Required column '{col}' not found in events_df")

    # Ensure timestamps are UTC-aware
    if prices_df[timestamp_col].dtype != "datetime64[ns, UTC]":
        prices_df = prices_df.copy()
        prices_df[timestamp_col] = pd.to_datetime(prices_df[timestamp_col], utc=True)

    if events_df[timestamp_col].dtype != "datetime64[ns, UTC]":
        events_df = events_df.copy()
        events_df[timestamp_col] = pd.to_datetime(events_df[timestamp_col], utc=True)

    # Generate event_id if not present
    events_work = events_df.copy()
    if event_id_col is None:
        # Check if "event_id" column exists
        if "event_id" in events_work.columns:
            event_id_col = "event_id"
        else:
            # Generate event_id
            events_work["event_id"] = (
                events_work[event_type_col].astype(str)
                + "_"
                + events_work[group_col].astype(str)
                + "_"
                + events_work[timestamp_col].dt.strftime("%Y%m%d")
            )
            # Make unique by adding index if duplicates
            if events_work["event_id"].duplicated().any():
                events_work["event_id"] = (
                    events_work["event_id"] + "_" + events_work.index.astype(str)
                )
            event_id_col = "event_id"
    elif event_id_col not in events_work.columns:
        # event_id_col specified but not found - generate it
        events_work["event_id"] = (
            events_work[event_type_col].astype(str)
            + "_"
            + events_work[group_col].astype(str)
            + "_"
            + events_work[timestamp_col].dt.strftime("%Y%m%d")
        )
        # Make unique by adding index if duplicates
        if events_work["event_id"].duplicated().any():
            events_work["event_id"] = (
                events_work["event_id"] + "_" + events_work.index.astype(str)
            )
        event_id_col = "event_id"

    # Sort prices by symbol, then timestamp
    prices_sorted = prices_df.sort_values([group_col, timestamp_col]).reset_index(
        drop=True
    )

    # Pre-group prices by symbol; reset index so label == positional offset
    _prices_by_sym = {
        sym: grp.reset_index(drop=True)
        for sym, grp in prices_sorted.groupby(group_col, sort=False)
    }

    # Build event windows
    all_windows = []

    for event_row in events_work.itertuples(index=False):
        event_symbol = getattr(event_row, group_col)
        event_timestamp = getattr(event_row, timestamp_col)
        event_type = getattr(event_row, event_type_col)
        event_id = getattr(event_row, event_id_col)

        # Filter prices for this symbol
        symbol_prices = _prices_by_sym.get(event_symbol)

        if symbol_prices is None or symbol_prices.empty:
            continue

        # Find event day index using pure-pandas comparison (avoids ns vs us unit mismatch
        # on pandas 2.2+ where DatetimeArray.astype("int64") may return microseconds while
        # pd.Timestamp.value always returns nanoseconds)
        sym_ts_idx = pd.DatetimeIndex(symbol_prices[timestamp_col])
        event_ts = pd.Timestamp(event_timestamp)
        if event_ts.tzinfo is None:
            event_ts = event_ts.tz_localize("UTC")
        exact_mask = sym_ts_idx == event_ts  # numpy bool array
        if exact_mask.any():
            event_row_idx = int(np.argmax(exact_mask))
        else:
            diffs = (sym_ts_idx - event_ts).abs()
            closest_pos = int(diffs.argmin())
            if diffs[closest_pos] > pd.Timedelta(days=1):
                continue
            event_row_idx = closest_pos

        # Extract window: from (event_row_idx - window_before) to (event_row_idx + window_after)
        start_idx = max(0, event_row_idx - window_before)
        end_idx = min(len(symbol_prices), event_row_idx + window_after + 1)

        window_prices = symbol_prices.iloc[start_idx:end_idx].copy()

        if window_prices.empty:
            continue

        # Calculate relative day
        event_day_timestamp = symbol_prices.iloc[event_row_idx][timestamp_col]
        window_prices["rel_day"] = (
            window_prices[timestamp_col] - event_day_timestamp
        ).dt.days

        # Add event metadata
        window_prices["event_id"] = event_id
        window_prices["event_type"] = event_type
        window_prices["event_timestamp"] = event_timestamp
        # Ensure group_col exists (it should already, but make sure)
        if group_col not in window_prices.columns:
            window_prices[group_col] = event_symbol

        # Reorder columns: event metadata first, then price data
        cols_order = [
            "event_id",
            group_col,
            "event_type",
            "event_timestamp",
            "rel_day",
            timestamp_col,
        ]
        # Add price_col and other columns from prices_df
        other_cols = [c for c in window_prices.columns if c not in cols_order]
        cols_order.extend(other_cols)

        # Keep only existing columns
        cols_order = [c for c in cols_order if c in window_prices.columns]
        window_prices = window_prices[cols_order]

        all_windows.append(window_prices)

    if not all_windows:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(
            columns=[
                "event_id",
                group_col,
                "event_type",
                "event_timestamp",
                "rel_day",
                timestamp_col,
                price_col,
            ]
        )

    result = pd.concat(all_windows, ignore_index=True)

    # Ensure rel_day is integer
    result["rel_day"] = result["rel_day"].astype(int)

    return result.sort_values(["event_id", "rel_day"]).reset_index(drop=True)


def compute_event_returns(
    event_windows_df: pd.DataFrame,
    price_col: str = "close",
    benchmark_col: str | None = None,
    return_type: str = "log",
    rel_day_col: str = "rel_day",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Compute normal and abnormal returns for event windows.

    Computes returns for each event and relative day. If benchmark_col is provided,
    also computes abnormal returns (return - benchmark_return).

    Args:
        event_windows_df: Output from build_event_window_prices()
        price_col: Column name for price (default: "close")
        benchmark_col: Column name for benchmark price (default: None).
            If provided, abnormal returns will be computed.
        return_type: "log" for log returns, "simple" for simple returns (default: "log")
        rel_day_col: Column name for relative day (default: "rel_day")
        timestamp_col: Column name for timestamp (default: "timestamp")

    Returns:
        DataFrame with columns:
        - event_id, symbol, event_type, event_timestamp, rel_day (from input)
        - timestamp: Actual timestamp
        - event_return: Return for this relative day
        - abnormal_return: Abnormal return (if benchmark_col provided)
        - Additional columns from input (e.g., close, factors)

    Raises:
        KeyError: If required columns are missing
        ValueError: If return_type is not "log" or "simple"

    Example:
        >>> windows = build_event_window_prices(prices, events)
        >>> returns = compute_event_returns(windows, price_col="close")
        >>> # Returns has event_return column with log returns
        >>>
        >>> # With benchmark
        >>> returns = compute_event_returns(windows, price_col="close", benchmark_col="spy_close")
        >>> # Returns has both event_return and abnormal_return columns
    """
    if return_type not in ["log", "simple"]:
        raise ValueError(f"return_type must be 'log' or 'simple', got '{return_type}'")

    # Validate inputs
    required_cols = [price_col, rel_day_col, "event_id"]
    for col in required_cols:
        if col not in event_windows_df.columns:
            raise KeyError(f"Required column '{col}' not found in event_windows_df")

    if benchmark_col is not None and benchmark_col not in event_windows_df.columns:
        raise KeyError(f"benchmark_col '{benchmark_col}' not found in event_windows_df")

    result = event_windows_df.copy()

    # Compute returns per event
    result["event_return"] = np.nan

    for event_id, event_grp in result.groupby("event_id", sort=False):
        event_idx = event_grp.index
        event_data = event_grp.sort_values(rel_day_col)

        if len(event_data) < 2:
            continue

        prices = event_data[price_col].values

        if return_type == "log":
            # Log returns: ln(price[t] / price[t-1])
            returns = np.diff(np.log(np.clip(prices, 1e-10, None)))
            # First day has no return (NaN)
            returns = np.concatenate([[np.nan], returns])
        else:
            # Simple returns: (price[t] / price[t-1]) - 1
            returns = np.diff(prices) / prices[:-1]
            # First day has no return (NaN)
            returns = np.concatenate([[np.nan], returns])

        result.loc[event_idx, "event_return"] = returns

    # Compute abnormal returns if benchmark provided
    if benchmark_col is not None:
        result["abnormal_return"] = np.nan

        for event_id, event_grp in result.groupby("event_id", sort=False):
            event_idx = event_grp.index
            event_data = event_grp.sort_values(rel_day_col)

            if len(event_data) < 2:
                continue

            event_returns = event_data["event_return"].values
            benchmark_prices = event_data[benchmark_col].values

            # Compute benchmark returns
            if return_type == "log":
                benchmark_returns = np.diff(
                    np.log(np.clip(benchmark_prices, 1e-10, None))
                )
                benchmark_returns = np.concatenate([[np.nan], benchmark_returns])
            else:
                benchmark_returns = np.diff(benchmark_prices) / benchmark_prices[:-1]
                benchmark_returns = np.concatenate([[np.nan], benchmark_returns])

            # Abnormal return = event_return - benchmark_return
            abnormal_returns = event_returns - benchmark_returns
            result.loc[event_idx, "abnormal_return"] = abnormal_returns

    return result


def aggregate_event_study(
    returns_df: pd.DataFrame,
    use_abnormal: bool = True,
    return_col: str | None = None,
    rel_day_col: str = "rel_day",
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Aggregate event returns across events.

    Computes average returns, cumulative returns, and confidence intervals
    for each relative day across all events.

    Args:
        returns_df: Output from compute_event_returns()
        use_abnormal: If True, use abnormal_return; if False, use event_return (default: True)
        return_col: Explicit column name for returns (default: None, auto-detect)
        rel_day_col: Column name for relative day (default: "rel_day")
        confidence_level: Confidence level for intervals (default: 0.95)

    Returns:
        DataFrame with columns:
        - rel_day: Relative day
        - avg_ret: Average return across events
        - std_ret: Standard deviation of returns
        - cum_ret: Cumulative return (sum from first day to this day)
        - n_events: Number of events with valid data for this day
        - ci_lower: Lower bound of confidence interval
        - ci_upper: Upper bound of confidence interval
        - se: Standard error (std / sqrt(n))

    Raises:
        KeyError: If required columns are missing
        ValueError: If no valid return column found

    Example:
        >>> returns = compute_event_returns(windows, price_col="close", benchmark_col="spy_close")
        >>> aggregated = aggregate_event_study(returns, use_abnormal=True)
        >>> # Result has avg_ret, cum_ret, n_events, ci_lower, ci_upper for each rel_day
    """
    # Determine return column
    if return_col is None:
        if use_abnormal and "abnormal_return" in returns_df.columns:
            return_col = "abnormal_return"
        elif "event_return" in returns_df.columns:
            return_col = "event_return"
        else:
            raise ValueError(
                "No return column found. Expected 'abnormal_return' or 'event_return'. "
                "Set use_abnormal=False or provide return_col explicitly."
            )

    if return_col not in returns_df.columns:
        raise KeyError(f"Return column '{return_col}' not found in returns_df")

    if rel_day_col not in returns_df.columns:
        raise KeyError(f"Relative day column '{rel_day_col}' not found in returns_df")

    # Group by relative day and aggregate
    grouped = returns_df.groupby(rel_day_col)[return_col]

    # Compute statistics
    result = pd.DataFrame(
        {
            "rel_day": grouped.mean().index,
            "avg_ret": grouped.mean().values,
            "std_ret": grouped.std().values,
            "n_events": grouped.count().values,
        }
    )

    # Compute standard error
    result["se"] = result["std_ret"] / np.sqrt(result["n_events"])

    # Compute confidence intervals (using z-score for normal distribution)
    # For large n, use z-score; for small n, could use t-distribution
    # Using z-score for simplicity (1.96 for 95% CI)
    try:
        from scipy import stats

        z_score = stats.norm.ppf((1 + confidence_level) / 2)
    except ImportError:
        # Fallback: use approximate z-scores for common confidence levels
        z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576,
        }
        z_score = z_scores.get(confidence_level, 1.96)  # Default to 95% CI
    result["ci_lower"] = result["avg_ret"] - z_score * result["se"]
    result["ci_upper"] = result["avg_ret"] + z_score * result["se"]

    # Compute cumulative return
    result = result.sort_values("rel_day")
    result["cum_ret"] = result["avg_ret"].cumsum()

    # Reset index
    result = result.reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# C4-081 (KNOWN_ISSUES §8.13) — Market-Model + BMP-t-stat + BHAR
#
# References:
# - MacKinlay (1997), "Event Studies in Economics and Finance", JEL 35(1).
# - Boehmer, Musumeci, Poulsen (1991), "Event-study methodology under
#   conditions of event-induced variance", JFE 30(2).
# - Barber & Lyon (1997), "Detecting long-run abnormal stock returns: The
#   empirical power and specification of test statistics", JFE 43(3) — BHAR.
# ---------------------------------------------------------------------------


@dataclass
class MarketModelResult:
    """Result of estimating the market model on one event's estimation window.

    Attributes:
        alpha: OLS intercept (Jensen's alpha).
        beta: OLS slope (market beta).
        sigma_resid: residual standard deviation (used for BMP standardisation).
        n_estimation_obs: number of obs in the estimation window.
        r_squared: regression R^2.
    """

    alpha: float
    beta: float
    sigma_resid: float
    n_estimation_obs: int
    r_squared: float


def estimate_market_model(
    asset_returns: pd.Series | np.ndarray,
    market_returns: pd.Series | np.ndarray,
) -> MarketModelResult:
    """OLS market-model regression: r_i = α + β · r_m + ε.

    Args:
        asset_returns: Series/array of asset returns over the estimation window.
        market_returns: Series/array of market returns over the same window.

    Returns:
        MarketModelResult with α, β, residual std, n_obs, R².

    Raises:
        ValueError: If inputs have <30 finite-aligned observations or are
            length-mismatched.
    """
    a = pd.Series(asset_returns, dtype=float).reset_index(drop=True)
    m = pd.Series(market_returns, dtype=float).reset_index(drop=True)
    if len(a) != len(m):
        raise ValueError(
            f"estimate_market_model: length mismatch asset={len(a)}, market={len(m)}"
        )
    # Align finite-only
    mask = a.notna() & m.notna() & np.isfinite(a) & np.isfinite(m)
    a_f = a[mask].to_numpy()
    m_f = m[mask].to_numpy()
    if len(a_f) < 30:
        raise ValueError(
            f"estimate_market_model: need ≥30 finite-aligned obs, got {len(a_f)}"
        )

    # OLS: design matrix [1, m_f]
    design = np.column_stack([np.ones(len(m_f)), m_f])
    coef, *_ = np.linalg.lstsq(design, a_f, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    fitted = alpha + beta * m_f
    resid = a_f - fitted
    n = len(a_f)
    # Sample residual std with (n-2) ddof (OLS with intercept + 1 regressor)
    sigma_resid = float(np.sqrt(np.sum(resid**2) / max(n - 2, 1)))
    # R^2 = 1 - SS_resid / SS_total
    ss_total = float(np.sum((a_f - a_f.mean()) ** 2))
    r_squared = 1.0 - (np.sum(resid**2) / ss_total) if ss_total > 0 else 0.0
    return MarketModelResult(
        alpha=alpha,
        beta=beta,
        sigma_resid=sigma_resid,
        n_estimation_obs=n,
        r_squared=float(r_squared),
    )


def compute_market_model_abnormal_returns(
    event_returns: pd.DataFrame,
    market_return_col: str = "market_return",
    rel_day_col: str = "rel_day",
    return_col: str = "event_return",
    estimation_window: tuple[int, int] = (-250, -10),
    event_id_col: str = "event_id",
) -> pd.DataFrame:
    """Compute Market-Model abnormal returns and per-event sigma_resid for BMP.

    For each event:
      1. Fit OLS on the estimation_window (default −250..−10 rel days) to get α, β.
      2. For ALL rel days in the input, compute AR = r_asset − (α + β · r_market).
      3. Attach `sigma_resid` (from estimation window) — needed for BMP-t standardisation.

    Args:
        event_returns: Output from `compute_event_returns` PLUS a market return
            column. Must contain `event_id_col`, `rel_day_col`, `return_col`,
            and `market_return_col`.
        market_return_col: Column with the market (benchmark) return.
        rel_day_col: Column with the relative day (negative=pre, 0=event).
        return_col: Column with the per-asset event return.
        estimation_window: (start_rel_day, end_rel_day) for OLS fitting. Default
            (−250, −10) matches MacKinlay (1997) convention.
        event_id_col: Column with the event identifier.

    Returns:
        Copy of `event_returns` with two new columns:
        - `mm_abnormal_return`: r_asset − (α + β · r_market) per row
        - `sigma_resid`: per-event residual std from the estimation window
        Events with <30 valid estimation obs have `mm_abnormal_return=NaN` and
        `sigma_resid=NaN` and are logged at DEBUG.

    Raises:
        KeyError: If required columns are missing.
    """
    for col in (event_id_col, rel_day_col, return_col, market_return_col):
        if col not in event_returns.columns:
            raise KeyError(
                f"compute_market_model_abnormal_returns: missing column '{col}'"
            )

    result = event_returns.copy()
    result["mm_abnormal_return"] = np.nan
    result["sigma_resid"] = np.nan
    est_start, est_end = estimation_window

    for event_id, grp in result.groupby(event_id_col, sort=False):
        est_mask = (grp[rel_day_col] >= est_start) & (grp[rel_day_col] <= est_end)
        est = grp[est_mask]
        if est_mask.sum() < 30:
            logger.debug(
                "Event %s: only %d obs in estimation window — AR=NaN",
                event_id,
                int(est_mask.sum()),
            )
            continue
        try:
            mm = estimate_market_model(
                est[return_col].to_numpy(),
                est[market_return_col].to_numpy(),
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            logger.debug("Event %s: market-model fit failed: %s", event_id, exc)
            continue

        # AR for ALL rel days (including estimation window and event window)
        ar = grp[return_col].to_numpy() - (
            mm.alpha + mm.beta * grp[market_return_col].to_numpy()
        )
        result.loc[grp.index, "mm_abnormal_return"] = ar
        result.loc[grp.index, "sigma_resid"] = mm.sigma_resid

    return result


def bmp_t_statistic(
    abnormal_returns: pd.DataFrame,
    event_window: tuple[int, int] = (-5, 5),
    rel_day_col: str = "rel_day",
    ar_col: str = "mm_abnormal_return",
    sigma_col: str = "sigma_resid",
    event_id_col: str = "event_id",
) -> dict:
    """Boehmer-Musumeci-Poulsen (1991) standardised cross-sectional t-stat.

    Procedure (BMP §3):
      1. Standardise AR per event: SAR_it = AR_it / sigma_i (estimation window std).
      2. For each event, sum SAR across event window: CSAR_i = Σ_t SAR_it.
      3. Cross-sectional test stat: t = mean(CSAR) / (std(CSAR) / √N).

    BMP standardisation is robust to event-induced variance (each event's
    abnormal return is scaled by its own estimation-window noise level).

    Args:
        abnormal_returns: Output from `compute_market_model_abnormal_returns`.
            Must contain `event_id_col`, `rel_day_col`, `ar_col`, `sigma_col`.
        event_window: (start_rel_day, end_rel_day) for the CAR calculation.
            Default (−5, +5) is the common short-window choice.
        rel_day_col: Column with the relative day.
        ar_col: Column with the abnormal return (default `mm_abnormal_return`).
        sigma_col: Column with the per-event estimation-window residual std.
        event_id_col: Column with the event identifier.

    Returns:
        Dict with keys:
        - `car_mean`: mean cumulative abnormal return across events
        - `n_events`: number of events with non-NaN CSAR
        - `t_statistic`: BMP t-stat
        - `pvalue`: two-sided p-value (normal approx)
        - `event_window`: echo of input
        - `is_significant_at_5pct`: convenience bool

    Raises:
        KeyError: If required columns are missing.
        ValueError: If no events have valid (AR, sigma_resid) pairs.
    """
    for col in (event_id_col, rel_day_col, ar_col, sigma_col):
        if col not in abnormal_returns.columns:
            raise KeyError(f"bmp_t_statistic: missing column '{col}'")

    win_start, win_end = event_window
    in_window = (abnormal_returns[rel_day_col] >= win_start) & (
        abnormal_returns[rel_day_col] <= win_end
    )
    window_df = abnormal_returns[in_window].copy()

    csars = []
    for event_id, grp in window_df.groupby(event_id_col, sort=False):
        sigma = grp[sigma_col].dropna()
        if sigma.empty or float(sigma.iloc[0]) <= 0:
            continue
        sigma_val = float(sigma.iloc[0])
        ar_vals = grp[ar_col].dropna().to_numpy()
        if len(ar_vals) == 0:
            continue
        # SAR_it = AR_it / sigma_i; CSAR_i = sum_t SAR_it
        sar = ar_vals / sigma_val
        csars.append(float(np.sum(sar)))

    if not csars:
        raise ValueError(
            "bmp_t_statistic: no events with valid (AR, sigma_resid) pairs"
        )

    csar_arr = np.asarray(csars, dtype=float)
    n = len(csar_arr)
    mean_csar = float(np.mean(csar_arr))
    std_csar = float(np.std(csar_arr, ddof=1)) if n > 1 else float("nan")

    if not (n > 1) or std_csar == 0 or not np.isfinite(std_csar):
        t_stat = float("nan")
        pvalue = float("nan")
    else:
        t_stat = mean_csar / (std_csar / np.sqrt(n))
        # Two-sided p-value via normal approximation (BMP §3 — large-N)
        from math import erfc, sqrt

        pvalue = float(erfc(abs(t_stat) / sqrt(2.0)))

    # CAR = average raw AR within window (not standardised — for reporting)
    # car_mean here is the average across events of the per-event sum AR
    car_per_event = (
        window_df.groupby(event_id_col)[ar_col].sum(min_count=1).dropna().to_numpy()
    )
    car_mean = float(np.mean(car_per_event)) if len(car_per_event) > 0 else float("nan")

    return {
        "car_mean": car_mean,
        "n_events": n,
        "t_statistic": float(t_stat) if np.isfinite(t_stat) else float("nan"),
        "pvalue": pvalue if np.isfinite(pvalue) else float("nan"),
        "event_window": event_window,
        "is_significant_at_5pct": bool(np.isfinite(pvalue) and pvalue < 0.05),
    }


def compute_bhar(
    event_returns: pd.DataFrame,
    market_return_col: str = "market_return",
    horizon_days: int = 250,
    rel_day_col: str = "rel_day",
    return_col: str = "event_return",
    event_id_col: str = "event_id",
) -> pd.DataFrame:
    """Buy-and-Hold Abnormal Return (BHAR) for long-horizon event studies.

    Per Barber & Lyon (1997): BHAR_i = ∏(1+r_asset_t) − ∏(1+r_market_t) over
    [event_day, event_day + horizon_days]. Compounding-based — appropriate for
    long horizons where simple summed CARs accumulate bias.

    Args:
        event_returns: Per-event/rel-day returns plus market column.
        market_return_col: Market return column.
        horizon_days: Holding-period horizon (days from event=0).
        rel_day_col: Column with rel day.
        return_col: Per-asset return column.
        event_id_col: Event identifier column.

    Returns:
        DataFrame with one row per event:
        - `event_id`
        - `bhar`: Buy-and-Hold Abnormal Return
        - `n_obs_in_window`: number of valid rel-day observations used

    Raises:
        KeyError: If required columns are missing.
        ValueError: If horizon_days < 1.
    """
    if horizon_days < 1:
        raise ValueError(f"compute_bhar: horizon_days must be ≥1, got {horizon_days}")
    for col in (event_id_col, rel_day_col, return_col, market_return_col):
        if col not in event_returns.columns:
            raise KeyError(f"compute_bhar: missing column '{col}'")

    in_window = (event_returns[rel_day_col] >= 0) & (
        event_returns[rel_day_col] <= horizon_days
    )
    win_df = event_returns[in_window]
    rows = []
    for event_id, grp in win_df.groupby(event_id_col, sort=False):
        sorted_grp = grp.sort_values(rel_day_col)
        ar = sorted_grp[return_col].dropna().to_numpy()
        mr = sorted_grp[market_return_col].dropna().to_numpy()
        if len(ar) == 0 or len(mr) == 0:
            continue
        # Use the shorter length (defensive)
        n = min(len(ar), len(mr))
        asset_compound = float(np.prod(1.0 + ar[:n]))
        market_compound = float(np.prod(1.0 + mr[:n]))
        rows.append(
            {
                "event_id": event_id,
                "bhar": asset_compound - market_compound,
                "n_obs_in_window": n,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["event_id", "bhar", "n_obs_in_window"])
    return pd.DataFrame(rows)


__all__ = [
    # Existing
    "build_event_window_prices",
    "compute_event_returns",
    "aggregate_event_study",
    # C4-081 — Market-Model + BMP-t + BHAR
    "MarketModelResult",
    "estimate_market_model",
    "compute_market_model_abnormal_returns",
    "bmp_t_statistic",
    "compute_bhar",
]
