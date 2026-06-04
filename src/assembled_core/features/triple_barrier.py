"""Triple-Barrier Labeling and CUSUM Event Filter.

From 11_FREE_MODELLE.md §11.4 — mlfinpy (López de Prado AFML).
Uses mlfinpy as primary, falls back to clean numpy/pandas implementation.

Key functions:
- cusum_filter: Symmetric CUSUM filter for event detection
- triple_barrier_labels: López de Prado triple-barrier labeling
- fractional_diff: Fractional differentiation for stationary features
- meta_label: Generate 0/1 meta-labels for a primary signal

Install: pip install mlfinpy==0.1.2
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _try_mlfinpy():
    try:
        import mlfinpy

        return mlfinpy
    except ImportError:
        logger.warning("mlfinpy not installed — pip install mlfinpy==0.1.2")
        return None


# ---------------------------------------------------------------------------
# CUSUM Filter
# ---------------------------------------------------------------------------


def cusum_filter(
    prices: pd.Series,
    threshold: float,
    use_mlfinpy: bool = True,
) -> pd.DatetimeIndex:
    """Symmetric CUSUM filter — returns timestamps where a threshold is crossed.

    Args:
        prices: Price series (or log-returns).
        threshold: CUSUM crossing threshold h.
        use_mlfinpy: Try mlfinpy first, fall back to numpy.

    Returns:
        DatetimeIndex of event timestamps.
    """
    mfp = _try_mlfinpy()
    if use_mlfinpy and mfp is not None:
        try:
            from mlfinpy.filters.filters import cusum_filter as mfp_cusum

            events = mfp_cusum(prices, threshold)
            return events
        except Exception as exc:
            logger.debug("mlfinpy CUSUM failed, using numpy fallback: %s", exc)

    # Numpy fallback
    return _cusum_filter_numpy(prices, threshold)


def _cusum_filter_numpy(prices: pd.Series, threshold: float) -> pd.DatetimeIndex:
    """Pure numpy/pandas CUSUM event filter."""
    log_ret = np.log(prices.clip(lower=1e-10)).diff().dropna()
    s_pos = 0.0
    s_neg = 0.0
    events: list = []

    for t, ret in log_ret.items():
        s_pos = max(0.0, s_pos + ret)
        s_neg = min(0.0, s_neg + ret)

        if s_pos > threshold:
            s_pos = 0.0
            events.append(t)
        elif s_neg < -threshold:
            s_neg = 0.0
            events.append(t)

    return pd.DatetimeIndex(events)


# ---------------------------------------------------------------------------
# Triple-Barrier Labeling
# ---------------------------------------------------------------------------


def triple_barrier_labels(
    prices: pd.Series,
    events: pd.DatetimeIndex,
    pt_sl: tuple[float, float] = (2.0, 1.0),
    vol: pd.Series | None = None,
    min_ret: float = 0.0,
    vertical_barrier_days: int = 20,
) -> pd.DataFrame:
    """Triple-Barrier Labeling (López de Prado, AFML Ch. 3).

    Labels: +1 = profit-take hit first, -1 = stop-loss hit first, 0 = vertical barrier.

    Args:
        prices: Close prices.
        events: Event timestamps from cusum_filter.
        pt_sl: (profit_take, stop_loss) multipliers for vol-based barriers.
        vol: Rolling volatility series (default: 20-day std of log-returns).
        min_ret: Minimum absolute return for non-zero label.
        vertical_barrier_days: Max holding period in days.

    Returns:
        DataFrame with columns: [t1 (exit_time), ret (realized return), bin (label)].
    """
    mfp = _try_mlfinpy()
    if mfp is not None:
        try:
            from mlfinpy.labeling.labeling import (
                add_vertical_barrier,
                get_bins,
                get_events,
            )

            if vol is None:
                log_ret = np.log(prices.clip(lower=1e-10)).diff()
                vol = log_ret.rolling(20).std().dropna()

            t1 = add_vertical_barrier(events, prices, num_days=vertical_barrier_days)
            events_df = get_events(prices, events, pt_sl, vol, min_ret, cpus=1, t1=t1)
            labels = get_bins(events_df, prices)
            return labels
        except Exception as exc:
            logger.debug("mlfinpy triple-barrier failed, using numpy fallback: %s", exc)

    return _triple_barrier_numpy(prices, events, pt_sl, vol, vertical_barrier_days)


def _triple_barrier_numpy(
    prices: pd.Series,
    events: pd.DatetimeIndex,
    pt_sl: tuple[float, float],
    vol: pd.Series | None,
    max_days: int,
) -> pd.DataFrame:
    """Numpy fallback triple-barrier implementation."""
    if vol is None:
        log_ret = np.log(prices.clip(lower=1e-10)).diff()
        vol = log_ret.rolling(20, min_periods=20).std()
        # No forward-fill — first 20 rows stay NaN to avoid look-ahead leakage in ML labels.

    pt_mult, sl_mult = pt_sl
    rows = []

    for t0 in events:
        if t0 not in prices.index:
            continue

        p0 = prices.loc[t0]
        v_raw = (
            vol.asof(t0)
            if hasattr(vol, "asof")
            else (vol.loc[t0] if t0 in vol.index else None)
        )
        if v_raw is None or pd.isna(v_raw):
            continue  # Skip events before vol window is fully populated (avoids look-ahead)
        v = float(v_raw)

        pt_barrier = p0 * (1 + pt_mult * v)
        sl_barrier = p0 * (1 - sl_mult * v)

        # Lookahead window
        future = prices.loc[t0:]
        if len(future) > 1:
            window = future.iloc[1 : max_days + 1]
        else:
            window = pd.Series(dtype=float)

        t1 = future.index[-1] if len(future) > 0 else t0
        label = 0
        exit_price = p0

        for t, p in window.items():
            if p >= pt_barrier:
                label = 1
                t1 = t
                exit_price = p
                break
            elif p <= sl_barrier:
                label = -1
                t1 = t
                exit_price = p
                break
        else:
            if len(window) > 0:
                t1 = window.index[-1]
                exit_price = window.iloc[-1]

        ret = (exit_price - p0) / p0
        rows.append({"t0": t0, "t1": t1, "ret": ret, "bin": label})

    if not rows:
        return pd.DataFrame(columns=["t1", "ret", "bin"])

    df = pd.DataFrame(rows).set_index("t0")
    df.index.name = None
    return df


# ---------------------------------------------------------------------------
# Fractional Differentiation
# ---------------------------------------------------------------------------


def fractional_diff(
    series: pd.Series,
    d: float = 0.4,
    threshold: float = 1e-4,
) -> pd.Series:
    """Fractional differentiation for memory-preserving stationarity.

    Args:
        series: Price or feature series.
        d: Differentiation order in (0, 1). 0.4 preserves ~75% memory.
        threshold: Minimum weight (truncation threshold).

    Returns:
        Fractionally differenced series, same index.
    """
    mfp = _try_mlfinpy()
    if mfp is not None:
        try:
            from mlfinpy.features.fracdiff import frac_diff_ffd

            df = pd.DataFrame({"val": series})
            result = frac_diff_ffd(df, d, threshold)
            return result["val"].rename(series.name)
        except Exception as exc:
            logger.debug("mlfinpy fracdiff failed: %s", exc)

    return _fracdiff_numpy(series, d, threshold)


def _fracdiff_numpy(series: pd.Series, d: float, threshold: float) -> pd.Series:
    """Pure numpy fractional differentiation (FFD fixed-width window).

    Note: Weight vector for small d decays slowly as k^(d-1). We cap the
    window at min(threshold_width, len(series)) to avoid empty output.
    """
    arr = series.values.astype(float)
    max_width = max(1, len(arr) - 1)  # cap at data length

    # Compute weights until threshold or max_width
    w = [1.0]
    k = 1
    while k <= max_width:
        w_next = -w[-1] * (d - k + 1) / k
        if abs(w_next) < threshold:
            break
        w.append(w_next)
        k += 1

    w_arr = np.array(w[::-1])
    width = len(w_arr)

    result_values = np.full(len(arr), np.nan)
    if len(arr) >= width:
        # np.convolve(arr, w, 'full')[k] = sum(arr[k-j]*w[j]) = dot(w, window) at position k
        result_values[width - 1 :] = np.convolve(arr, w_arr, mode="full")[
            width - 1 : len(arr)
        ]
    return pd.Series(result_values, index=series.index, name=series.name)


# ---------------------------------------------------------------------------
# Meta-Labeling
# ---------------------------------------------------------------------------


def meta_label(
    primary_signal: pd.Series,
    triple_barrier_df: pd.DataFrame,
) -> pd.Series:
    """Generate meta-labels: 1 if primary signal direction matched outcome, else 0.

    Args:
        primary_signal: Series of +1/-1 signals indexed by event timestamps.
        triple_barrier_df: Output of triple_barrier_labels (has 'bin' column).

    Returns:
        Binary Series: 1 = bet on primary signal, 0 = skip.
    """
    common = primary_signal.index.intersection(triple_barrier_df.index)
    if len(common) == 0:
        return pd.Series(dtype=int)

    ps = primary_signal.loc[common]
    tb = triple_barrier_df.loc[common, "bin"]

    # Meta-label = 1 if primary direction == actual direction
    meta = ((ps * tb) > 0).astype(int)
    return meta.rename("meta_label")


def compute_sample_weights(
    events: pd.DataFrame,
    prices: pd.Series,
    t_in_col: str = "t_in",
    t_out_col: str = "t_out",
) -> pd.Series:
    """López de Prado sample weights for overlapping triple-barrier labels.

    Weight_i = 1 / mean(count of concurrent events during event i's hold period).
    Prevents double-counting when multiple events overlap in time.

    Parameters
    ----------
    events:
        DataFrame with columns ``t_in`` and ``t_out`` (datetime index or columns).
    prices:
        Price series whose index defines the time grid for overlap counting.
    t_in_col, t_out_col:
        Column names for entry/exit timestamps.

    Returns
    -------
    pd.Series of float weights aligned to events.index.
    """
    prices_idx = prices.index
    t_in_arr = events[t_in_col].values
    t_out_arr = events[t_out_col].values

    # O(N_events) count accumulation via searchsorted + cumsum difference encoding
    left_idxs = np.searchsorted(prices_idx, t_in_arr, side="left")
    right_idxs = np.searchsorted(prices_idx, t_out_arr, side="right")
    delta = np.zeros(len(prices_idx) + 1)
    np.add.at(delta, left_idxs, 1)
    np.add.at(delta, right_idxs, -1)
    counts_arr = np.cumsum(delta[:-1])

    weights_arr = np.empty(len(events))
    for i, (li, ri) in enumerate(zip(left_idxs, right_idxs)):
        overlap = counts_arr[li:ri]
        if len(overlap) == 0 or overlap.sum() == 0:
            weights_arr[i] = 1.0
        else:
            weights_arr[i] = float((1.0 / overlap).mean())
    weights = pd.Series(weights_arr, index=events.index)

    total = weights.sum()
    if total > 0:
        weights = weights / total * len(weights)
    return weights


def find_min_d_for_stationarity(
    series: pd.Series,
    d_grid: list[float] | None = None,
    pvalue_threshold: float = 0.05,
    threshold: float = 1e-4,
) -> dict:
    """Find the minimum d ∈ (0, 1) that makes the fractionally-differenced series stationary.

    Implements López de Prado AFML §5.5 procedure: search a grid of d values
    and return the smallest one for which the Augmented Dickey-Fuller test
    rejects the unit-root null at ``pvalue_threshold``. This is the "minimal
    memory loss while achieving stationarity" point, the standard recipe for
    calibrating fractional-difference orders on price-like series.

    Closes KNOWN_ISSUES §8.13 C4-076 — "Default-d-Param verifizierbar".

    Args:
        series: Input series (typically log-prices for financial use).
        d_grid: List of d values to search. Default: ``np.linspace(0.05, 0.95, 19)``
            (steps of 0.05). Caller can pass a finer grid for precision.
        pvalue_threshold: ADF p-value below which we accept stationarity (default 0.05).
        threshold: FFD weight cutoff passed to :func:`fractional_diff`.

    Returns:
        Dict with keys:
            ``d`` (float | None): the chosen minimum d, or ``None`` if no d in
                the grid achieves stationarity.
            ``adf_statistic`` (float | None): ADF statistic at the chosen d.
            ``pvalue`` (float | None): ADF p-value at the chosen d.
            ``is_stationary`` (bool): True iff a stationarising d was found.
            ``correlation_with_original`` (float | None): Pearson correlation
                between the diffed series and the original (memory-preservation
                metric; AFML §5.5).
            ``grid_tested`` (list[float]): the actual d-grid that was tried.

    Raises:
        ValueError: If series has fewer than 30 non-NaN observations.
        ImportError: If statsmodels is not installed.
    """
    import numpy as np
    from statsmodels.tsa.stattools import adfuller

    s = pd.Series(series, dtype=float).dropna()
    if len(s) < 30:
        raise ValueError(f"find_min_d_for_stationarity: need ≥30 obs, got {len(s)}")

    if d_grid is None:
        d_grid = list(np.linspace(0.05, 0.95, 19))

    for d in d_grid:
        try:
            diffed = fractional_diff(s, d=float(d), threshold=threshold)
            diffed_clean = diffed.dropna()
            if len(diffed_clean) < 30:
                continue
            adf_stat, pvalue, *_ = adfuller(diffed_clean.to_numpy(), autolag="AIC")
            if pvalue < pvalue_threshold:
                # Pearson correlation on the overlapping (non-NaN) tail
                aligned_original = s.loc[diffed_clean.index]
                if aligned_original.std(ddof=1) == 0 or diffed_clean.std(ddof=1) == 0:
                    corr = float("nan")
                else:
                    corr = float(aligned_original.corr(diffed_clean))
                return {
                    "d": float(d),
                    "adf_statistic": float(adf_stat),
                    "pvalue": float(pvalue),
                    "is_stationary": True,
                    "correlation_with_original": corr,
                    "grid_tested": list(d_grid),
                }
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
            # F-senior-1 (2026-05-17): narrow from `except Exception`.
            # ADF/fracdiff failure modes are bounded: ValueError (constant
            # series after diff), LinAlgError (singular matrix), RuntimeError
            # (statsmodels internal). Other exceptions propagate.
            logger.debug("ADF failed at d=%s: %s", d, exc)
            continue

    return {
        "d": None,
        "adf_statistic": None,
        "pvalue": None,
        "is_stationary": False,
        "correlation_with_original": None,
        "grid_tested": list(d_grid),
    }


__all__ = [
    "cusum_filter",
    "triple_barrier_labels",
    "fractional_diff",
    "find_min_d_for_stationarity",
    "meta_label",
    "compute_sample_weights",
]
