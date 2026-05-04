"""Market confirmation signals for crisis state transitions.

Computes real-time market-based confirmation signals (oil move, gold move,
VIX spike) to validate geopolitical intel before escalating crisis state.

Without market confirmation, the crisis state machine cannot transition
from WATCH → ACTIVE, preventing intel signals from operating in a vacuum.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Proxy tickers for each signal
_OIL_TICKER = "CL=F"  # WTI Crude futures
_GOLD_TICKER = "GC=F"  # Gold futures
_VIX_TICKER = "^VIX"  # CBOE VIX

# Fallback ETF tickers if futures fail
_OIL_FALLBACK = "USO"
_GOLD_FALLBACK = "GLD"

# Thresholds (used for VIX spike detection)
_VIX_SPIKE_LEVEL = 25.0  # absolute level
_VIX_SPIKE_CHANGE_PCT = 15.0  # 1-day % change


def compute_market_confirmation(
    *,
    lookback_days: int = 5,
    cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute market confirmation dict for crisis state machine.

    Returns dict with:
        oil_move   (float): oil price % change over lookback window
        gold_move  (float): gold price % change over lookback window
        vix_spike  (bool):  True if VIX > level threshold OR VIX 1d change > change threshold
        vix_level  (float): current VIX level
        computed_utc (str): ISO timestamp of computation

    On any fetch failure, returns conservative zeros (no false confirmation).
    Uses optional cache dict to avoid re-fetching within same cycle.
    """
    cache = cache if cache is not None else {}

    result: dict[str, Any] = {
        "oil_move": 0.0,
        "gold_move": 0.0,
        "vix_spike": False,
        "vix_level": 0.0,
        "computed_utc": datetime.now(tz=timezone.utc).isoformat(),
    }

    try:
        import yfinance as yf  # noqa: F401 — availability check
    except ImportError:
        logger.warning("[WARN] yfinance not installed — market confirmation disabled")
        return result

    period = f"{lookback_days + 2}d"  # extra buffer for weekends

    # --- Oil ---
    oil_pct = _fetch_pct_change(_OIL_TICKER, period, cache)
    if oil_pct is None:
        oil_pct = _fetch_pct_change(_OIL_FALLBACK, period, cache)
    if oil_pct is not None:
        result["oil_move"] = round(oil_pct, 4)

    # --- Gold ---
    gold_pct = _fetch_pct_change(_GOLD_TICKER, period, cache)
    if gold_pct is None:
        gold_pct = _fetch_pct_change(_GOLD_FALLBACK, period, cache)
    if gold_pct is not None:
        result["gold_move"] = round(gold_pct, 4)

    # --- VIX ---
    vix_data = _fetch_vix(period, cache)
    if vix_data is not None:
        vix_level, vix_1d_change = vix_data
        result["vix_level"] = round(vix_level, 2)
        result["vix_spike"] = (
            vix_level > _VIX_SPIKE_LEVEL or abs(vix_1d_change) > _VIX_SPIKE_CHANGE_PCT
        )

    logger.info(
        "[MarketConfirm] oil=%.2f%% gold=%.2f%% vix=%.1f spike=%s",
        result["oil_move"],
        result["gold_move"],
        result["vix_level"],
        result["vix_spike"],
    )
    return result


def _fetch_pct_change(
    ticker: str,
    period: str,
    cache: dict[str, Any],
) -> float | None:
    """Fetch % change for a ticker over the period. Returns None on failure."""
    cache_key = f"pct_{ticker}"
    if cache_key in cache:
        return cache[cache_key]

    try:
        import yfinance as yf

        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if data is None or len(data) < 2:
            return None
        closes = data["Close"].dropna()
        if len(closes) < 2:
            return None
        _last = (
            float(closes.iloc[-1].iloc[0])
            if hasattr(closes.iloc[-1], "iloc")
            else float(closes.iloc[-1])
        )
        _first = (
            float(closes.iloc[0].iloc[0])
            if hasattr(closes.iloc[0], "iloc")
            else float(closes.iloc[0])
        )
        if _first == 0.0:
            return None
        pct = (_last - _first) / _first * 100
        cache[cache_key] = pct
        return pct
    except Exception as exc:
        logger.debug("[MarketConfirm] Failed to fetch %s: %s", ticker, exc)
        return None


def _fetch_vix(
    period: str,
    cache: dict[str, Any],
) -> tuple[float, float] | None:
    """Fetch VIX level and 1-day % change. Returns (level, 1d_change_pct) or None."""
    cache_key = "vix_data"
    if cache_key in cache:
        return cache[cache_key]

    try:
        import yfinance as yf

        data = yf.download(_VIX_TICKER, period=period, progress=False, auto_adjust=True)
        if data is None or len(data) < 2:
            return None
        closes = data["Close"].dropna()
        if len(closes) < 2:
            return None
        _v_last = (
            float(closes.iloc[-1].iloc[0])
            if hasattr(closes.iloc[-1], "iloc")
            else float(closes.iloc[-1])
        )
        _v_prev = (
            float(closes.iloc[-2].iloc[0])
            if hasattr(closes.iloc[-2], "iloc")
            else float(closes.iloc[-2])
        )
        level = _v_last
        if _v_prev == 0.0:
            return None
        change_1d = (_v_last - _v_prev) / _v_prev * 100
        result = (level, change_1d)
        cache[cache_key] = result
        return result
    except Exception as exc:
        logger.debug("[MarketConfirm] Failed to fetch VIX: %s", exc)
        return None
