"""GARCH-based volatility forecasting via the `arch` library.

.. deprecated:: 2026-05-17
   This module is DEPRECATED in favor of ``src.assembled_core.risk.garch_vol``
   which provides equivalent functionality plus:

   - Rolling-window realized-vol FALLBACK when ``arch`` is unavailable or fit
     fails (this module returns ``NaN`` instead — production hazard).
   - Batch helper ``compute_vol_forecasts(prices_df)`` for per-ticker forecasts.
   - More defensive ``size_vol_target`` (handles inf/NaN explicitly).

   See `KNOWN_ISSUES.md` §6.5.2 for the consolidation status and migration plan.

   Migration path:
       Old: ``from src.assembled_core.risk.garch_vol_forecast import forecast_garch_vol``
       New: ``from src.assembled_core.risk.garch_vol import forecast_vol``

   Configurable parameters (vol_model / p / o / q / dist) are NOT yet exposed by
   ``garch_vol``'s public API — if you need them, file a follow-up to expand
   the canonical module rather than continuing to depend on this deprecated one.

From 11_FREE_MODELLE.md §11.8 and 13_FREE_MODULE.md §13.3.

De-facto standard: GJR-GARCH(1,1,1) with skew-t innovations for US equity daily.
Feeds into vol-targeting (replaces simple rolling std when arch is available).

Install: pip install arch==8.0.0
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

warnings.warn(
    "src.assembled_core.risk.garch_vol_forecast is deprecated since 2026-05-17. "
    "Use src.assembled_core.risk.garch_vol instead (provides rolling-window "
    "fallback + batch helper + defensive sizing). See KNOWN_ISSUES.md §6.5.2.",
    DeprecationWarning,
    stacklevel=2,
)


def _try_import_arch():
    try:
        from arch import arch_model

        return arch_model
    except ImportError:
        logger.warning("arch not installed — install with: pip install arch==8.0.0")
        return None


def forecast_garch_vol(
    returns: pd.Series,
    horizon: int = 5,
    vol_model: str = "GARCH",
    p: int = 1,
    o: int = 1,
    q: int = 1,
    dist: str = "skewt",
    annualize: bool = True,
    annualize_factor: float = 252.0,
) -> float:
    """Forecast annualized volatility using GJR-GARCH.

    Args:
        returns: Return series (daily). Will be multiplied by 100 internally.
        horizon: Forecast horizon in bars (default 5 = 1 week).
        vol_model: 'GARCH', 'EGARCH', 'APARCH' (default 'GARCH' for GJR).
        p: ARCH order (default 1)
        o: GJR asymmetry order (default 1)
        q: GARCH order (default 1)
        dist: Innovation distribution — 'skewt' | 'normal' | 't'
        annualize: Whether to annualize the output (default True)
        annualize_factor: 252 for daily data.

    Returns:
        Annualized volatility forecast (float). Returns NaN if arch unavailable
        or insufficient data.
    """
    arch_model = _try_import_arch()
    if arch_model is None:
        return float("nan")

    clean = returns.dropna()
    if len(clean) < 60:
        return float("nan")

    try:
        model = arch_model(clean * 100, vol=vol_model, p=p, o=o, q=q, dist=dist)
        res = model.fit(disp="off", show_warning=False)
        forecasts = res.forecast(horizon=horizon)
        sigma_pct = float(np.sqrt(forecasts.variance.iloc[-1].mean()))
        sigma = sigma_pct / 100.0
        if annualize:
            sigma = sigma * np.sqrt(annualize_factor)
        return sigma
    except Exception as exc:
        logger.debug("GARCH fit failed: %s", exc)
        return float("nan")


def garch_vol_target_size(
    returns: pd.Series,
    target_vol: float = 0.15,
    max_size: float = 1.5,
    horizon: int = 5,
) -> float:
    """Size factor for vol-targeting using GARCH forecast.

    Returns target_vol / garch_forecast, clamped to [0, max_size].
    Falls back to 1.0 if GARCH unavailable or fails.
    """
    forecast = forecast_garch_vol(returns, horizon=horizon)
    if not np.isfinite(forecast) or forecast <= 0:
        return 1.0
    raw = target_vol / forecast
    return float(max(0.0, min(max_size, raw)))


__all__ = [
    "forecast_garch_vol",
    "garch_vol_target_size",
]
