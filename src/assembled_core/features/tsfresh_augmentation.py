"""tsfresh feature augmentation with manual fallback.

Augments OHLCV time series with tsfresh-style features.
Uses tsfresh if installed; falls back to manual numpy/pandas extraction.

Install: pip install tsfresh>=0.20.0
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _tsfresh_available() -> bool:
    try:
        import tsfresh  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_features(
    prices: pd.DataFrame,
    column_id: str = "symbol",
    column_sort: str = "date",
    target_columns: Sequence[str] | None = None,
    use_tsfresh: bool = True,
) -> pd.DataFrame:
    """Extract time-series features from a multi-symbol price DataFrame.

    Parameters
    ----------
    prices:
        Long-format DataFrame with at minimum [symbol, date, close] columns.
    column_id:
        Column identifying each time series (default ``"symbol"``).
    column_sort:
        Timestamp column (default ``"date"``).
    target_columns:
        Numeric columns to extract features from. Defaults to all numeric
        columns except ``column_id`` and ``column_sort``.
    use_tsfresh:
        Attempt tsfresh first; fall back to manual extraction if unavailable.

    Returns
    -------
    DataFrame indexed by (symbol, window_end_date) — or just symbol if tsfresh
    is used, with one row per series.
    """
    if target_columns is None:
        exclude = {column_id, column_sort}
        target_columns = [c for c in prices.select_dtypes(include=[np.number]).columns if c not in exclude]

    if use_tsfresh and _tsfresh_available():
        return _extract_tsfresh(prices, column_id, column_sort, list(target_columns))
    return _extract_manual(prices, column_id, column_sort, list(target_columns))


def extract_rolling_features(
    series: pd.Series,
    window: int = 20,
    feature_set: str = "minimal",
) -> pd.DataFrame:
    """Compute rolling window features for a single price series.

    Parameters
    ----------
    series:
        Univariate numeric Series (e.g. close prices).
    window:
        Rolling window size in bars.
    feature_set:
        ``"minimal"`` (fast, ~8 features) or ``"full"`` (~20 features).

    Returns
    -------
    DataFrame with one column per feature, same index as input.
    """
    return _rolling_feature_set(series, window=window, full=(feature_set == "full"))


# ---------------------------------------------------------------------------
# tsfresh backend
# ---------------------------------------------------------------------------


def _extract_tsfresh(
    prices: pd.DataFrame,
    column_id: str,
    column_sort: str,
    target_columns: list[str],
) -> pd.DataFrame:
    from tsfresh import extract_features as ts_extract  # noqa: PLC0415
    from tsfresh.feature_extraction import MinimalFCParameters  # noqa: PLC0415

    settings = MinimalFCParameters()
    try:
        result = ts_extract(
            prices[[column_id, column_sort] + target_columns],
            column_id=column_id,
            column_sort=column_sort,
            default_fc_parameters=settings,
            disable_progressbar=True,
        )
        logger.info("[tsfresh] Extracted %d features for %d series", result.shape[1], result.shape[0])
        return result
    except Exception as exc:
        logger.warning("[tsfresh] Extraction failed (%s) — falling back to manual", exc)
        return _extract_manual(prices, column_id, column_sort, target_columns)


# ---------------------------------------------------------------------------
# Manual fallback
# ---------------------------------------------------------------------------


def _extract_manual(
    prices: pd.DataFrame,
    column_id: str,
    column_sort: str,
    target_columns: list[str],
) -> pd.DataFrame:
    """Pure pandas/numpy feature extraction grouped by symbol."""
    rows = []
    for sym, grp in prices.groupby(column_id, sort=False):
        grp = grp.sort_values(column_sort)
        row: dict = {column_id: sym}
        for col in target_columns:
            s = grp[col].dropna()
            if len(s) == 0:
                continue
            row.update(_manual_features(s, prefix=col))
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).set_index(column_id)
    return result


def _manual_features(s: pd.Series, prefix: str) -> dict[str, float]:
    """~20 statistical features for a single series."""
    arr = s.values.astype(float)
    n = len(arr)
    ret = np.diff(np.log(np.clip(arr, 1e-9, None))) if n > 1 else np.array([0.0])

    features: dict[str, float] = {}
    p = prefix + "__"

    features[p + "mean"] = float(arr.mean())
    features[p + "std"] = float(arr.std(ddof=1)) if n > 1 else 0.0
    features[p + "min"] = float(arr.min())
    features[p + "max"] = float(arr.max())
    features[p + "range"] = float(arr.max() - arr.min())
    features[p + "skew"] = float(pd.Series(arr).skew())
    features[p + "kurtosis"] = float(pd.Series(arr).kurtosis())
    features[p + "last_minus_first"] = float(arr[-1] - arr[0])
    features[p + "pct_change_total"] = float((arr[-1] / arr[0] - 1)) if arr[0] != 0 else 0.0
    features[p + "mean_abs_change"] = float(np.abs(np.diff(arr)).mean()) if n > 1 else 0.0
    features[p + "num_peaks"] = _count_peaks(arr)
    features[p + "autocorr_lag1"] = float(pd.Series(arr).autocorr(lag=1)) if n > 2 else 0.0
    features[p + "autocorr_lag5"] = float(pd.Series(arr).autocorr(lag=5)) if n > 6 else 0.0
    features[p + "log_ret_mean"] = float(ret.mean())
    features[p + "log_ret_std"] = float(ret.std(ddof=1)) if len(ret) > 1 else 0.0
    features[p + "log_ret_skew"] = float(pd.Series(ret).skew()) if len(ret) > 2 else 0.0
    features[p + "above_mean_frac"] = float((arr > arr.mean()).mean())
    features[p + "trend_slope"] = _linear_slope(arr)
    features[p + "hurst_exponent"] = _hurst(arr) if n >= 20 else 0.5
    features[p + "sample_entropy"] = _sample_entropy(ret, m=2) if len(ret) >= 10 else 0.0

    return features


def _rolling_feature_set(series: pd.Series, window: int, full: bool) -> pd.DataFrame:
    arr = series.values.astype(float)
    idx = series.index

    out: dict[str, np.ndarray] = {}
    roll = pd.Series(arr, index=idx).rolling(window, min_periods=max(2, window // 2))

    out["mean"] = roll.mean().values
    out["std"] = roll.std(ddof=1).values
    out["min"] = roll.min().values
    out["max"] = roll.max().values
    out["range"] = (roll.max() - roll.min()).values
    out["skew"] = roll.skew().values
    out["autocorr_lag1"] = _rolling_autocorr(arr, window, lag=1)
    out["mean_abs_change"] = _rolling_mean_abs_change(arr, window)

    if full:
        out["kurtosis"] = roll.kurt().values
        out["above_mean_frac"] = _rolling_above_mean_frac(arr, window)
        out["trend_slope"] = _rolling_slope(arr, window)
        out["last_minus_first"] = _rolling_last_minus_first(arr, window)
        out["pct_change_total"] = _rolling_pct_change_total(arr, window)
        out["log_ret_mean"] = _rolling_log_ret_stat(arr, window, "mean")
        out["log_ret_std"] = _rolling_log_ret_stat(arr, window, "std")
        out["num_peaks"] = _rolling_peaks(arr, window)
        out["autocorr_lag5"] = _rolling_autocorr(arr, window, lag=5)
        out["hurst_exponent"] = _rolling_hurst(arr, window)
        out["sample_entropy"] = _rolling_sample_entropy(arr, window)

    return pd.DataFrame(out, index=idx)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def _count_peaks(arr: np.ndarray) -> float:
    if len(arr) < 3:
        return 0.0
    peaks = ((arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:])).sum()
    return float(peaks)


def _linear_slope(arr: np.ndarray) -> float:
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x -= x.mean()
    y = arr - arr.mean()
    denom = (x * x).sum()
    return float((x * y).sum() / denom) if denom != 0 else 0.0


def _hurst(arr: np.ndarray, max_lag: int = 20) -> float:
    """Simplified Hurst exponent via R/S analysis."""
    lags = range(2, min(max_lag, len(arr) // 2))
    rs_vals = []
    for lag in lags:
        sub = arr[:lag]
        if len(sub) < 2:
            continue
        mean = sub.mean()
        dev = np.cumsum(sub - mean)
        r = dev.max() - dev.min()
        s = sub.std(ddof=1)
        if s > 0:
            rs_vals.append((lag, r / s))
    if len(rs_vals) < 2:
        return 0.5
    log_lags = np.log([x[0] for x in rs_vals])
    log_rs = np.log([x[1] for x in rs_vals])
    slope = _linear_slope(log_rs / log_lags) if len(log_lags) else 0.5
    return float(np.clip(slope, 0.0, 1.0)) if not np.isnan(slope) else 0.5


def _sample_entropy(arr: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Approximate sample entropy (SampEn)."""
    n = len(arr)
    if n < m + 2:
        return 0.0
    std = arr.std()
    if std == 0:
        return 0.0
    tol = r * std

    def _count(length: int) -> int:
        count = 0
        for i in range(n - length):
            template = arr[i:i + length]
            for j in range(i + 1, n - length):
                if np.max(np.abs(arr[j:j + length] - template)) <= tol:
                    count += 1
        return count

    A = _count(m + 1)
    B = _count(m)
    if B == 0:
        return 0.0
    return float(-np.log(A / B)) if A > 0 else 0.0


# ---------------------------------------------------------------------------
# Rolling helpers
# ---------------------------------------------------------------------------


def _rolling_autocorr(arr: np.ndarray, window: int, lag: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        w = arr[i - window + 1: i + 1]
        if len(w) > lag + 1:
            out[i] = float(pd.Series(w).autocorr(lag=lag))
    return out


def _rolling_mean_abs_change(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        w = arr[i - window + 1: i + 1]
        out[i] = float(np.abs(np.diff(w)).mean()) if len(w) > 1 else 0.0
    return out


def _rolling_above_mean_frac(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        w = arr[i - window + 1: i + 1]
        out[i] = float((w > w.mean()).mean())
    return out


def _rolling_slope(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        w = arr[i - window + 1: i + 1]
        out[i] = _linear_slope(w)
    return out


def _rolling_last_minus_first(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        w = arr[i - window + 1: i + 1]
        out[i] = w[-1] - w[0]
    return out


def _rolling_pct_change_total(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        w = arr[i - window + 1: i + 1]
        out[i] = (w[-1] / w[0] - 1) if w[0] != 0 else 0.0
    return out


def _rolling_log_ret_stat(arr: np.ndarray, window: int, stat: str) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        w = arr[i - window + 1: i + 1]
        ret = np.diff(np.log(np.clip(w, 1e-9, None)))
        if len(ret) > 0:
            out[i] = float(ret.mean() if stat == "mean" else ret.std(ddof=1))
    return out


def _rolling_peaks(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        w = arr[i - window + 1: i + 1]
        out[i] = _count_peaks(w)
    return out


def _rolling_hurst(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        w = arr[i - window + 1: i + 1]
        if len(w) >= 20:
            out[i] = _hurst(w)
    return out


def _rolling_sample_entropy(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(window - 1, len(arr)):
        w = arr[i - window + 1: i + 1]
        ret = np.diff(np.log(np.clip(w, 1e-9, None)))
        if len(ret) >= 10:
            out[i] = _sample_entropy(ret)
    return out
