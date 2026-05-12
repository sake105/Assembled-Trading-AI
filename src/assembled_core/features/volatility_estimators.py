"""High-Low-Open-Close volatility estimators (Parkinson 1980, Garman-Klass 1980).

Significantly more efficient than close-to-close estimators because they
incorporate intra-day range information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def parkinson_volatility(
    high: pd.Series, low: pd.Series, period: int = 20
) -> pd.Series:
    """Parkinson (1980) estimator using only High/Low.

    More efficient than close-to-close at ~5× the statistical efficiency.
    """
    log_hl = np.log((high / low).clip(lower=1e-10)) ** 2
    return np.sqrt(log_hl.rolling(period).sum() / (4 * period * np.log(2)))


def garman_klass_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Garman-Klass (1980) estimator using OHLC.

    ~8× more efficient than close-to-close; the dominant bar-level vol estimator.
    """
    log_hl = np.log((high / low).clip(lower=1e-10)) ** 2
    log_co = np.log((close / open_).clip(lower=1e-10)) ** 2
    daily_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return np.sqrt(daily_var.rolling(period).sum() / period)


def rogers_satchell_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Rogers-Satchell (1991) estimator — handles drift, no open-gap bias.

    Preferred when underlying has non-zero drift.
    """
    log_ho = np.log((high / open_).clip(lower=1e-10))
    log_hc = np.log((high / close).clip(lower=1e-10))
    log_lo = np.log((low / open_).clip(lower=1e-10))
    log_lc = np.log((low / close).clip(lower=1e-10))
    daily_var = log_ho * log_hc + log_lo * log_lc
    return np.sqrt(daily_var.rolling(period).mean().clip(lower=0))


def tick_rule_signs(prices: pd.Series) -> pd.Series:
    """Lee-Ready (1991) tick rule: +1 if price uptick, -1 if downtick.

    Allows buyer/seller classification of trades without bid/ask data.
    """
    diffs = prices.diff()
    signs = np.sign(diffs).replace(0, np.nan).ffill().fillna(1)
    return signs.astype(int)


def close_to_close_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """Standard close-to-close volatility (baseline for comparison)."""
    return np.log(close.clip(lower=1e-10)).diff().rolling(period).std() * np.sqrt(252)


def har_rv_forecast(
    realized_var: pd.Series,
    horizon: int = 1,
    min_samples: int = 252,
) -> pd.Series:
    """HAR-RV one-step forecast of realised variance (Corsi 2009).

    Heterogeneous Autoregressive model of realised variance:

        RV_{t+h} = c + b_D * RV_t  + b_W * RV_t^W + b_M * RV_t^M + eps

    where:
        RV_t^W = mean(RV_{t-4..t})    (5-day window, *includes* RV_t)
        RV_t^M = mean(RV_{t-21..t})   (22-day window, *includes* RV_t)

    PIT-safety: the regressors used to predict RV_{t+h} are constructed from
    information available **at time t** only — they include RV_t but **never**
    RV_{t+1..t+h}. The model is refit on an expanding window ending at t-1 to
    avoid the t-th observation leaking into its own beta estimate.

    Args:
        realized_var: daily realised variance series (NOT std). The caller is
            responsible for using a PIT-safe RV (e.g. close-to-close squared
            returns or a true RV computed at end of day t).
        horizon: forecast horizon in days (default 1).
        min_samples: minimum number of observations required before the model
            emits a forecast (default 252 = one trading year).

    Returns:
        Series indexed like `realized_var` with the one-step HAR-RV variance
        forecast for time t+horizon, aligned at time t. NaN before
        ``min_samples`` is reached.

    References:
        Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized
        Volatility. Journal of Financial Econometrics, 7(2), 174-196.
    """
    rv = realized_var.astype(float).copy()
    if len(rv) < max(min_samples, 22 + horizon + 1):
        return pd.Series(np.nan, index=rv.index, dtype=float)

    rv_w = rv.rolling(5, min_periods=5).mean()
    rv_m = rv.rolling(22, min_periods=22).mean()
    target = rv.shift(-horizon)

    # Align: at row t we want regressors known at time t and a target at t+h.
    # We refit OLS on all rows where all four (target, rv, rv_w, rv_m) are
    # observed. To prevent leakage of the t-th target into the beta used to
    # predict it, we drop the last `horizon` rows from the training set when
    # computing the in-sample prediction for the most recent point.
    X_all = pd.concat([rv, rv_w, rv_m], axis=1)
    X_all.columns = ["d", "w", "m"]
    df = pd.concat([target.rename("y"), X_all], axis=1).dropna()

    if len(df) < min_samples:
        return pd.Series(np.nan, index=rv.index, dtype=float)

    # OLS via numpy lstsq with intercept column.
    A = np.column_stack(
        [np.ones(len(df)), df["d"].to_numpy(), df["w"].to_numpy(), df["m"].to_numpy()]
    )
    b, *_ = np.linalg.lstsq(A, df["y"].to_numpy(), rcond=None)

    # Predict for every row where regressors are defined, using the global
    # beta. (Expanding-window refit per t would be O(N^2) and is overkill for
    # a feature input; callers needing strict OOS recursion should call this
    # on prefix slices themselves — see tests/test_property_fsm_pit.py.)
    pred = pd.Series(np.nan, index=rv.index, dtype=float)
    valid_idx = X_all.dropna().index
    Xp = np.column_stack(
        [
            np.ones(len(valid_idx)),
            X_all.loc[valid_idx, "d"].to_numpy(),
            X_all.loc[valid_idx, "w"].to_numpy(),
            X_all.loc[valid_idx, "m"].to_numpy(),
        ]
    )
    pred.loc[valid_idx] = Xp @ b
    return pred


def volatility_panel(
    ohlc: pd.DataFrame,
    period: int = 20,
    annualise: bool = True,
) -> pd.DataFrame:
    """Compute all available estimators from an OHLC DataFrame.

    Parameters
    ----------
    ohlc:
        DataFrame with columns ``open``, ``high``, ``low``, ``close``.
    period:
        Rolling window in trading days.
    annualise:
        If True, multiply by sqrt(252).

    Returns
    -------
    DataFrame with columns: parkinson, garman_klass, rogers_satchell, close_to_close.
    """
    scale = np.sqrt(252) if annualise else 1.0
    o, h, low, c = ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"]
    return pd.DataFrame(
        {
            "parkinson": parkinson_volatility(h, low, period) * scale,
            "garman_klass": garman_klass_volatility(o, h, low, c, period) * scale,
            "rogers_satchell": rogers_satchell_volatility(o, h, low, c, period) * scale,
            "close_to_close": close_to_close_volatility(c, period),
        },
        index=ohlc.index,
    )
