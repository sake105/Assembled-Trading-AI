# src/assembled_core/utils/async_fetch.py
"""Async I/O helper for parallel external fetches (audit B-004).

Wraps ``httpx.AsyncClient`` with a per-host semaphore, exponential
backoff, and the project retry-convention. Designed for the case the
audit calls out: yfinance / Polygon / Alpaca etc. with N symbols where
sequential blocking takes 10-30 s but parallel async takes 1-3 s.

The deliberate constraints:

- **httpx only** (already in venv) — no aiohttp dependency.
- **Semaphore-bounded** — concurrent requests capped per-call, default 10.
  The audit explicitly warns: "ohne Rate-Limit-Awareness wirst du gesperrt".
- **Retry with jitter** — uses ``utils.retry.retry`` for the sync path,
  open-coded for the async one (tenacity is also absent).
- **Per-request timeout** — 10 s default; never block forever.

Example::

    from src.assembled_core.utils.async_fetch import fetch_many

    async def _one(client, symbol):
        r = await client.get(
            "https://query1.finance.yahoo.com/v7/finance/download/" + symbol,
            params={"interval": "1d"},
        )
        r.raise_for_status()
        return symbol, r.text

    results = await fetch_many(symbols, _one, max_concurrency=10)
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Awaitable, Callable, Sequence

logger = logging.getLogger(__name__)


async def _async_retry(
    coro_factory: Callable[[], Awaitable[Any]],
    *,
    attempts: int = 5,
    base: float = 0.5,
    cap: float = 8.0,
    jitter: float = 0.1,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Any:
    """Async version of utils/retry.retry — no tenacity dependency."""
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await coro_factory()
        except exceptions as exc:
            last_exc = exc
            if attempt >= attempts:
                break
            wait = min(base * (2 ** (attempt - 1)), cap)
            if jitter > 0:
                wait *= 1.0 + random.uniform(-jitter, jitter)
            logger.info(
                "[async_retry] attempt %d/%d failed (%s); sleeping %.2fs",
                attempt,
                attempts,
                exc.__class__.__name__,
                wait,
            )
            await asyncio.sleep(max(0.0, wait))
    if last_exc is None:  # pragma: no cover — loop invariant
        raise RuntimeError("async retry exhausted with no captured exception")
    raise last_exc


async def fetch_many(
    items: Sequence[Any],
    fetch_one: Callable[[Any, Any], Awaitable[Any]],
    *,
    max_concurrency: int = 10,
    request_timeout: float = 10.0,
    retry_attempts: int = 3,
    http2: bool = True,
    return_exceptions: bool = False,
) -> list[Any]:
    """Apply ``fetch_one`` to every item, in parallel with bounded concurrency.

    ``fetch_one`` MUST accept ``(client, item)`` and return whatever the
    caller wants (a tuple, a dict, anything). The helper owns the single
    shared ``httpx.AsyncClient`` so connection pooling actually amortises.

    Args:
        items: arbitrary iterable of fetch keys (e.g. ticker strings).
        fetch_one: async callable ``async def fn(client, item) -> result``.
        max_concurrency: semaphore size (default 10). The audit warned
            that "asyncio + Semaphore(50) ohne Rate-Limit-Awareness" gets
            you blocked; 10 is a conservative default that respects
            yfinance / Polygon / Alpaca free-tier limits.
        request_timeout: per-request timeout in seconds.
        retry_attempts: retries per item.
        http2: enable HTTP/2 in the shared client.
        return_exceptions: if True, exceptions are surfaced as elements
            in the result list (asyncio.gather semantics). If False,
            the first unhandled exception propagates.

    Returns:
        List of results in input order.
    """
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover — httpx is in deps
        raise ImportError("async_fetch requires httpx (already in deps)") from exc

    sem = asyncio.Semaphore(max_concurrency)

    async with httpx.AsyncClient(http2=http2, timeout=request_timeout) as client:

        async def _bounded(item: Any) -> Any:
            async with sem:
                return await _async_retry(
                    lambda: fetch_one(client, item),
                    attempts=retry_attempts,
                    exceptions=(
                        httpx.HTTPError,
                        TimeoutError,
                        asyncio.TimeoutError,
                    ),
                )

        results = await asyncio.gather(
            *[_bounded(item) for item in items],
            return_exceptions=return_exceptions,
        )
    return list(results)


__all__ = ["fetch_many"]
