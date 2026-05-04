"""GJR-GARCH(1,1) vol forecast for position sizing. From 13_FREE_MODULE.md §13.3.

Replaces simple realized-vol with a proper conditional-variance forecast when
the `arch` library is available. Falls back to rolling-window realized vol if
arch is not installed or the fit fails.

Usage::

    from assembled_core.risk.garch_vol import forecast_vol, size_vol_target

    sigma_annual = forecast_vol(returns_series, horizon=5)
    size = size_vol_target(sigma_annual, target_vol=0.15)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ARCH_AVAILABLE: bool
try:
    from arch import arch_model as _arch_model  # type: ignore  # noqa: F401

    _ARCH_AVAILABLE = True
except ImportError:
    _ARCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Vol forecast
# ---------------------------------------------------------------------------


def forecast_vol(
    returns: pd.Series | np.ndarray,
    horizon: int = 5,
    min_obs: int = 60,
    annualize_factor: float = 252.0,
    fallback_window: int = 20,
) -> float:
    """Forecast annualised volatility over `horizon` trading days.

    Uses GJR-GARCH(1,1,1) with skewed-t distribution when arch is available.
    Falls back to rolling-window std if arch is absent or the model fails to
    converge.

    Args:
        returns: Daily (or intraday) period returns as a fraction, not %.
                 Should have at least `min_obs` non-NaN observations.
        horizon: Forecast horizon in bars (default 5 = one trading week).
        min_obs: Minimum non-NaN observations needed for GARCH; below this,
                 returns fallback.
        annualize_factor: 252 for daily, 52 for weekly, 12 for monthly.
        fallback_window: Lookback window for the rolling-std fallback.

    Returns:
        Annualised volatility forecast (float). Returns NaN if insufficient data.
    """
    if isinstance(returns, pd.Series):
        r = returns.dropna()
    else:
        r = pd.Series(returns).dropna()

    if len(r) < min_obs:
        return _fallback_vol(r, fallback_window, annualize_factor)

    if not _ARCH_AVAILABLE:
        logger.debug("arch not installed; using rolling-std vol fallback")
        return _fallback_vol(r, fallback_window, annualize_factor)

    try:
        return _garch_forecast(r, horizon=horizon, annualize_factor=annualize_factor)
    except Exception as exc:
        logger.warning("GARCH fit failed (%s); using rolling-std fallback", exc)
        return _fallback_vol(r, fallback_window, annualize_factor)


def _garch_forecast(r: pd.Series, horizon: int, annualize_factor: float) -> float:
    # arch_model expects returns in percent (it rescales internally)
    from arch import arch_model  # type: ignore

    model = arch_model(
        r * 100,
        mean="Constant",
        vol="GARCH",
        p=1,
        o=1,  # GJR term
        q=1,
        dist="skewt",
    )
    res = model.fit(disp="off", show_warning=False)
    fc = res.forecast(horizon=horizon, reindex=False)
    # fc.variance.iloc[-1] has shape (horizon,), in (%^2)
    mean_var_pct2 = float(fc.variance.iloc[-1].mean())
    # Convert back: std in fraction = sqrt(var_pct2) / 100
    sigma_per_bar = np.sqrt(mean_var_pct2) / 100.0
    return sigma_per_bar * np.sqrt(annualize_factor)


def _fallback_vol(r: pd.Series, window: int, annualize_factor: float) -> float:
    tail = r.tail(window)
    if len(tail) < 2:
        return float("nan")
    return float(tail.std(ddof=1)) * np.sqrt(annualize_factor)


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


def size_vol_target(
    asset_vol_forecast: float,
    target_vol: float = 0.15,
    max_leverage: float = 1.5,
    min_size: float = 0.0,
) -> float:
    """Return position-size multiplier that targets `target_vol` annualised.

    Formula:  size = clip(target_vol / asset_vol_forecast, min_size, max_leverage)

    Args:
        asset_vol_forecast: Annualised vol forecast (from forecast_vol).
        target_vol: Target annualised portfolio volatility (default 15%).
        max_leverage: Hard cap (default 1.5 — no leverage beyond 150%).
        min_size: Hard floor (default 0.0 — no short-vol position).

    Returns:
        Scalar multiplier in [min_size, max_leverage].
    """
    if np.isnan(asset_vol_forecast):
        return 1.0  # neutral when forecast unavailable
    if asset_vol_forecast <= 0:
        return 1.0
    if np.isposinf(asset_vol_forecast):
        return float(min_size)  # infinite vol → zero position

    raw = target_vol / asset_vol_forecast
    return float(np.clip(raw, min_size, max_leverage))


# ---------------------------------------------------------------------------
# Batch helper: per-ticker vol forecast
# ---------------------------------------------------------------------------


def compute_vol_forecasts(
    prices_df: pd.DataFrame,
    price_col: str = "close",
    ticker_col: str = "ticker",
    timestamp_col: str = "timestamp",
    horizon: int = 5,
    target_vol: float = 0.15,
    max_leverage: float = 1.5,
) -> pd.DataFrame:
    """Compute vol forecasts + size multipliers for each ticker.

    Args:
        prices_df: OHLCV-style DataFrame with ticker + price columns.
        price_col: Column to use for return calculation.
        ticker_col / timestamp_col: Identifier columns.
        horizon: GARCH forecast horizon.
        target_vol: Annualised vol target.
        max_leverage: Max size multiplier.

    Returns:
        DataFrame with columns [ticker, vol_forecast_annual, size_multiplier].
    """
    records: list[dict[str, Any]] = []
    for ticker, group in prices_df.groupby(ticker_col, sort=False):
        group = group.sort_values(timestamp_col)
        returns = group[price_col].pct_change(fill_method=None).dropna()
        vol = forecast_vol(returns, horizon=horizon)
        size = size_vol_target(vol, target_vol=target_vol, max_leverage=max_leverage)
        records.append(
            {
                "ticker": ticker,
                "vol_forecast_annual": vol,
                "size_multiplier": size,
            }
        )
    return pd.DataFrame(records)
