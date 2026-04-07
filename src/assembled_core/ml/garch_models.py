"""GARCH family: conditional volatility modelling.

Implements GARCH(1,1), EGARCH, and GJR-GARCH for forward-looking volatility
forecasting.  Unlike realized-vol (backward-looking rolling std), GARCH
models volatility clustering and mean-reversion to produce 1-step and
multi-step-ahead variance forecasts.

Use-cases:
  - Position sizing (vol-targeting with GARCH instead of realized vol)
  - Risk limits (GARCH-VaR instead of historical VaR)
  - Features (garch_vol_1d, garch_asymmetry, garch_persistence)

Requires the ``arch`` Python package (``pip install arch``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from arch import arch_model  # type: ignore[import-untyped]

    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

# Annualization factor
_TRADING_DAYS = 252


@dataclass
class GARCHResult:
    """Result of fitting a single GARCH model."""

    symbol: str
    model_type: str
    vol_1d: float  # 1-step-ahead annualized vol
    vol_5d: float  # 5-step-ahead annualized vol (sqrt-of-time approx)
    persistence: float  # alpha + beta (close to 1 = long memory)
    asymmetry: float  # leverage/asymmetry parameter (EGARCH gamma, GJR gamma)
    bic: float  # Bayesian Information Criterion (lower = better)
    params: dict[str, float] = field(default_factory=dict)
    converged: bool = True
    fit_date: str = ""


def fit_garch(
    returns: pd.Series | np.ndarray,
    symbol: str = "",
    *,
    model_type: Literal["garch", "egarch", "gjr"] = "garch",
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
) -> GARCHResult | None:
    """Fit a single GARCH-family model to a return series.

    Args:
        returns: Daily log-returns (or simple returns). NaN rows are dropped.
        symbol: Ticker symbol for labeling.
        model_type: ``"garch"`` (standard), ``"egarch"`` (Nelson 1991),
            or ``"gjr"`` (Glosten-Jagannathan-Runkle 1993).
        p: GARCH order (default 1).
        q: ARCH order (default 1).
        dist: Error distribution — ``"normal"`` or ``"t"`` (Student-t).

    Returns:
        :class:`GARCHResult` or ``None`` if the ``arch`` package is
        unavailable or the fit fails.
    """
    if not ARCH_AVAILABLE:
        logger.debug("[GARCH] arch package not installed — skipping %s", symbol)
        return None

    # Clean input
    if isinstance(returns, pd.Series):
        returns = returns.dropna().values
    else:
        returns = returns[~np.isnan(returns)]

    if len(returns) < 60:
        logger.debug("[GARCH] %s: insufficient data (%d < 60)", symbol, len(returns))
        return None

    # Scale to percentage for numerical stability (arch convention)
    ret_pct = returns * 100.0

    vol_type_map = {"garch": "GARCH", "egarch": "EGARCH", "gjr": "GARCH"}

    try:
        if model_type == "gjr":
            am = arch_model(
                ret_pct, mean="Zero", vol="GARCH", p=p, o=1, q=q, dist=dist,
            )
        elif model_type == "egarch":
            am = arch_model(
                ret_pct, mean="Zero", vol="EGARCH", p=p, q=q, dist=dist,
            )
        else:
            am = arch_model(
                ret_pct, mean="Zero", vol="GARCH", p=p, q=q, dist=dist,
            )

        res = am.fit(disp="off", show_warning=False)
    except Exception as exc:
        logger.debug("[GARCH] %s fit failed (%s): %s", symbol, model_type, exc)
        return None

    if not res.convergence_flag == 0:
        logger.debug("[GARCH] %s: %s did not converge", symbol, model_type)

    # Extract parameters
    params_dict = dict(res.params)

    # 1-step-ahead conditional variance (in pct² space)
    try:
        forecast = res.forecast(horizon=1)
        var_1d_pct2 = float(forecast.variance.values[-1, 0])
    except Exception:
        var_1d_pct2 = float(res.conditional_volatility.iloc[-1] ** 2)

    # Convert back from pct to decimal, then annualize
    var_1d = var_1d_pct2 / 1e4  # pct² → decimal²
    vol_1d_ann = np.sqrt(var_1d * _TRADING_DAYS)

    # 5-step approximation (sqrt-of-time for simplicity)
    vol_5d_ann = np.sqrt(var_1d * 5) * np.sqrt(_TRADING_DAYS / 5)
    # Simpler: vol_5d_ann ≈ vol_1d_ann (same annualized vol, different horizon)
    # But for reporting the 5-day realized vol forecast:
    vol_5d_total = np.sqrt(var_1d * 5)  # total vol over 5 days

    # Persistence & asymmetry
    alpha = float(params_dict.get("alpha[1]", 0.0))
    beta = float(params_dict.get("beta[1]", 0.0))
    gamma = float(params_dict.get("gamma[1]", 0.0))  # GJR/EGARCH asymmetry

    persistence = alpha + beta
    if model_type == "gjr":
        # GJR persistence includes half the asymmetry term
        persistence = alpha + beta + gamma / 2.0

    return GARCHResult(
        symbol=symbol,
        model_type=model_type,
        vol_1d=round(float(vol_1d_ann), 6),
        vol_5d=round(float(vol_5d_ann), 6),
        persistence=round(float(persistence), 4),
        asymmetry=round(float(gamma), 4),
        bic=round(float(res.bic), 2),
        params={k: round(float(v), 6) for k, v in params_dict.items()},
        converged=res.convergence_flag == 0,
    )


def fit_best_garch(
    returns: pd.Series | np.ndarray,
    symbol: str = "",
    *,
    candidates: list[str] | None = None,
    dist: str = "normal",
) -> GARCHResult | None:
    """Fit multiple GARCH variants and return the best by BIC.

    Args:
        returns: Daily returns.
        symbol: Ticker symbol.
        candidates: Model types to try (default: all three).
        dist: Error distribution.

    Returns:
        Best :class:`GARCHResult` by BIC, or ``None`` if all fail.
    """
    if candidates is None:
        candidates = ["garch", "egarch", "gjr"]

    results: list[GARCHResult] = []
    for mt in candidates:
        r = fit_garch(returns, symbol, model_type=mt, dist=dist)  # type: ignore[arg-type]
        if r is not None and r.converged:
            results.append(r)

    if not results:
        # Try again without convergence requirement
        for mt in candidates:
            r = fit_garch(returns, symbol, model_type=mt, dist=dist)  # type: ignore[arg-type]
            if r is not None:
                results.append(r)

    if not results:
        return None

    # Select by lowest BIC
    best = min(results, key=lambda r: r.bic)
    logger.debug(
        "[GARCH] %s: best model=%s (BIC=%.1f, vol_1d=%.4f, persistence=%.3f)",
        symbol, best.model_type, best.bic, best.vol_1d, best.persistence,
    )
    return best


def fit_panel_garch(
    prices: pd.DataFrame,
    symbols: list[str] | None = None,
    *,
    lookback_days: int = 252,
    candidates: list[str] | None = None,
    dist: str = "normal",
) -> dict[str, GARCHResult]:
    """Fit best GARCH model for each symbol in a price panel.

    Args:
        prices: DataFrame with columns ``timestamp``, ``symbol``, ``close``.
        symbols: Symbols to fit (default: all unique symbols in prices).
        lookback_days: Number of recent trading days to use for fitting.
        candidates: GARCH variants to try.
        dist: Error distribution.

    Returns:
        Dict mapping symbol → :class:`GARCHResult`.
    """
    if prices is None or prices.empty:
        return {}

    required = {"timestamp", "symbol", "close"}
    if not required.issubset(prices.columns):
        logger.warning("[GARCH] prices missing columns %s", required - set(prices.columns))
        return {}

    if symbols is None:
        symbols = list(prices["symbol"].unique())

    results: dict[str, GARCHResult] = {}

    for sym in symbols:
        sym_data = prices[prices["symbol"] == sym].sort_values("timestamp")
        if len(sym_data) < 60:
            continue

        closes = sym_data["close"].astype(float).values
        # Use last lookback_days
        closes = closes[-lookback_days:] if len(closes) > lookback_days else closes

        # Log returns
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret = np.diff(np.log(closes))
        log_ret = log_ret[np.isfinite(log_ret)]

        if len(log_ret) < 60:
            continue

        r = fit_best_garch(log_ret, sym, candidates=candidates, dist=dist)
        if r is not None:
            results[sym] = r

    logger.info("[GARCH] Fitted %d/%d symbols", len(results), len(symbols))
    return results


def garch_vol_forecast_series(
    prices: pd.DataFrame,
    symbol: str,
    *,
    model_type: Literal["garch", "egarch", "gjr"] = "garch",
    fit_window: int = 252,
    refit_every: int = 5,
) -> pd.DataFrame:
    """Compute rolling GARCH vol forecast for a single symbol.

    Fits the model every ``refit_every`` days on a rolling window
    of ``fit_window`` days, then produces 1-step-ahead forecasts.

    Args:
        prices: DataFrame with ``timestamp``, ``symbol``, ``close``.
        symbol: Symbol to process.
        model_type: GARCH variant.
        fit_window: Fitting window in trading days.
        refit_every: Re-estimate every N days (default 5).

    Returns:
        DataFrame with columns ``timestamp``, ``garch_vol_1d``,
        ``garch_asymmetry``, ``garch_persistence``.
    """
    if not ARCH_AVAILABLE:
        return pd.DataFrame(columns=["timestamp", "garch_vol_1d", "garch_asymmetry", "garch_persistence"])

    sym_data = prices[prices["symbol"] == symbol].sort_values("timestamp").reset_index(drop=True)
    if len(sym_data) < fit_window + 10:
        return pd.DataFrame(columns=["timestamp", "garch_vol_1d", "garch_asymmetry", "garch_persistence"])

    closes = sym_data["close"].astype(float).values
    timestamps = sym_data["timestamp"].values

    # Compute log returns
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.diff(np.log(closes))

    rows = []
    last_result: GARCHResult | None = None

    for i in range(fit_window, len(log_ret)):
        day_idx = i  # index into log_ret

        # Refit periodically
        if last_result is None or (day_idx - fit_window) % refit_every == 0:
            window = log_ret[max(0, day_idx - fit_window):day_idx]
            window = window[np.isfinite(window)]
            if len(window) >= 60:
                result = fit_garch(window, symbol, model_type=model_type)
                if result is not None:
                    last_result = result

        if last_result is not None:
            rows.append({
                "timestamp": timestamps[day_idx + 1],  # +1 because log_ret[i] uses close[i+1]
                "garch_vol_1d": last_result.vol_1d,
                "garch_asymmetry": last_result.asymmetry,
                "garch_persistence": last_result.persistence,
            })

    if not rows:
        return pd.DataFrame(columns=["timestamp", "garch_vol_1d", "garch_asymmetry", "garch_persistence"])

    return pd.DataFrame(rows)


__all__ = [
    "ARCH_AVAILABLE",
    "GARCHResult",
    "fit_best_garch",
    "fit_garch",
    "fit_panel_garch",
    "garch_vol_forecast_series",
]
