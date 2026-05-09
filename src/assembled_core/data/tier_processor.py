"""Tier-based universe processor with async concurrency control.

From 14_FREE_UNIVERSUM.md §14.3 (Tier-3 on-demand) and §14.6 (FastAPI concurrency).

Tier-1: S&P 500 + EURO STOXX 50 + ETF — processed every minute during session.
Tier-2: Expansion tickers — processed every 5 min or EOD.
Tier-3: On-demand at news/volume/gap trigger — cached with 7-day TTL.

Uses asyncio.Semaphore to cap concurrent yfinance/Alpaca calls.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# On-demand analysis (Tier-3) — §14.3
# ---------------------------------------------------------------------------


def compute_basic_features(data: pd.DataFrame) -> dict[str, float]:
    """Minimal feature extraction for on-demand analysis (no NLP, no GARCH)."""
    if data.empty or "Close" not in data.columns:
        return {}

    close = data["Close"].dropna()
    if len(close) < 5:
        return {}

    returns = close.pct_change().dropna()
    features: dict[str, float] = {
        "ret_1d": float(returns.iloc[-1]) if len(returns) > 0 else 0.0,
        "ret_5d": (
            float(close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 5 else 0.0
        ),
        "ret_20d": (
            float(close.iloc[-1] / close.iloc[-20] - 1) if len(close) >= 20 else 0.0
        ),
        "vol_20d": (
            float(returns.tail(20).std() * (252**0.5)) if len(returns) >= 20 else 0.0
        ),
        "price": float(close.iloc[-1]),
    }

    if "Volume" in data.columns:
        vol = data["Volume"].dropna()
        mean_vol_20d = vol.tail(20).mean()
        features["volume_ratio_20d"] = (
            float(vol.iloc[-1] / mean_vol_20d)
            if len(vol) >= 20 and mean_vol_20d > 0
            else 1.0
        )

    return features


def lightweight_composite(features: dict[str, float]) -> float:
    """Simplified composite score without NLP/GARCH for fast on-demand analysis.

    Returns score in [-1, +1].
    """
    if not features:
        return 0.0

    ret_5d = features.get("ret_5d", 0.0)
    vol_20d = features.get("vol_20d", 0.20)
    vol_ratio = features.get("volume_ratio_20d", 1.0)

    # Sharpe-like: return / vol
    sharpe_proxy = ret_5d / (vol_20d + 1e-6)
    volume_signal = 1.0 if vol_ratio > 2.0 else 0.0

    raw = 0.7 * sharpe_proxy + 0.3 * volume_signal
    return max(-1.0, min(1.0, raw))


async def on_demand_analysis(
    ticker: str,
    lookback_days: int = 60,
    redis_client: Any | None = None,
    ttl_seconds: int = 86400 * 7,
) -> dict[str, Any]:
    """Tier-3 on-demand analysis: fetch, compute, cache.

    Args:
        ticker: Stock ticker to analyze.
        lookback_days: History to fetch (default 60 days).
        redis_client: Optional async Redis client for TTL caching.
        ttl_seconds: Cache TTL in seconds (default 7 days).

    Returns:
        Dict with features and composite score.
    """
    cache_key = f"ondemand:{ticker}"

    # Check cache first
    if redis_client is not None:
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                import json

                return json.loads(cached)
        except Exception as _exc:
            logger.debug("[tier_processor] Redis cache GET failed: %s", _exc)

    # Fetch data
    try:
        import yfinance as yf

        data = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: yf.Ticker(ticker).history(period=f"{lookback_days}d"),
        )
    except ImportError:
        logger.warning("yfinance not installed")
        return {"ticker": ticker, "error": "yfinance unavailable"}
    except Exception as exc:
        logger.debug("on_demand_analysis fetch failed for %s: %s", ticker, exc)
        return {"ticker": ticker, "error": str(exc)}

    features = compute_basic_features(data)
    score = lightweight_composite(features)

    result = {
        "ticker": ticker,
        "features": features,
        "composite_score": score,
    }

    # Cache result
    if redis_client is not None:
        try:
            import json

            await redis_client.setex(cache_key, ttl_seconds, json.dumps(result))
        except Exception as _exc:
            logger.debug("[tier_processor] Redis cache SET failed: %s", _exc)

    return result


# ---------------------------------------------------------------------------
# TierProcessor — §14.6 FastAPI concurrency pattern
# ---------------------------------------------------------------------------


@dataclass
class TierConfig:
    tier1_concurrency: int = 50
    tier2_concurrency: int = 20
    tier3_concurrency: int = 5
    alpaca_symbols_per_request: int = 200


class TierProcessor:
    """Async tier processor with semaphore-based concurrency control.

    Tier-1: up to 50 concurrent, 1-min polling.
    Tier-2: up to 20 concurrent, 5-min polling.
    Tier-3: up to 5 concurrent, on-demand.
    """

    def __init__(
        self,
        config: TierConfig | None = None,
        redis_client: Any | None = None,
    ):
        cfg = config or TierConfig()
        self._tier1_sem = asyncio.Semaphore(cfg.tier1_concurrency)
        self._tier2_sem = asyncio.Semaphore(cfg.tier2_concurrency)
        self._tier3_sem = asyncio.Semaphore(cfg.tier3_concurrency)
        self._alpaca_batch = cfg.alpaca_symbols_per_request
        self._redis = redis_client

    async def process_tier1(
        self,
        tickers: list[str],
        analyze_fn: Callable[[str], Any] | None = None,
    ) -> list[Any]:
        """Process Tier-1 tickers concurrently (up to 50 parallel)."""
        fn = analyze_fn or (lambda t: {"ticker": t})

        async def _bounded(ticker: str) -> Any:
            async with self._tier1_sem:
                if asyncio.iscoroutinefunction(fn):
                    return await fn(ticker)
                return await asyncio.get_running_loop().run_in_executor(
                    None, fn, ticker
                )

        return await asyncio.gather(*[_bounded(t) for t in tickers])

    async def process_tier2(
        self,
        tickers: list[str],
        analyze_fn: Callable[[str], Any] | None = None,
    ) -> list[Any]:
        """Process Tier-2 tickers concurrently (up to 20 parallel)."""
        fn = analyze_fn or (lambda t: {"ticker": t})

        async def _bounded(ticker: str) -> Any:
            async with self._tier2_sem:
                if asyncio.iscoroutinefunction(fn):
                    return await fn(ticker)
                return await asyncio.get_running_loop().run_in_executor(
                    None, fn, ticker
                )

        return await asyncio.gather(*[_bounded(t) for t in tickers])

    async def process_tier3_ondemand(
        self,
        ticker: str,
        lookback_days: int = 60,
    ) -> dict[str, Any]:
        """Process a single Tier-3 on-demand request."""
        async with self._tier3_sem:
            return await on_demand_analysis(
                ticker,
                lookback_days=lookback_days,
                redis_client=self._redis,
            )

    def alpaca_batches(self, tickers: list[str]) -> list[list[str]]:
        """Split ticker list into Alpaca multi-symbol request batches.

        Alpaca allows up to 200 symbols per /v2/stocks/bars/latest call.
        3 requests for 585 Tier-1 tickers = 1.5% of 200 req/min limit.
        """
        n = self._alpaca_batch
        return [tickers[i : i + n] for i in range(0, len(tickers), n)]


# ---------------------------------------------------------------------------
# Tier-3 trigger conditions
# ---------------------------------------------------------------------------


def should_trigger_on_demand(
    ticker: str,
    news_velocity: float = 0.0,
    volume_ratio: float = 1.0,
    gap_pct: float = 0.0,
    has_earnings: bool = False,
) -> bool:
    """Determine if a Tier-3 on-demand analysis should be triggered.

    Args:
        ticker: Ticker symbol.
        news_velocity: Recent news articles per hour.
        volume_ratio: Current volume / 20d average.
        gap_pct: Gap open percentage (absolute).
        has_earnings: True if earnings announcement today.

    Returns:
        True if on-demand analysis should be triggered.
    """
    # Triggers from 14_FREE_UNIVERSUM.md §14.3
    if has_earnings:
        return True
    if news_velocity > 3.0:  # >3 articles/hour
        return True
    if volume_ratio > 3.0:  # >3× average volume
        return True
    if abs(gap_pct) > 0.03:  # >3% gap open
        return True
    return False


__all__ = [
    "TierConfig",
    "TierProcessor",
    "compute_basic_features",
    "lightweight_composite",
    "on_demand_analysis",
    "should_trigger_on_demand",
]
