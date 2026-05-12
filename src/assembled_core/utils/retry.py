# src/assembled_core/utils/retry.py
"""Project-default retry convention (audit C4-022).

A minimal, stdlib-only decorator for the project's standard
exponential-backoff retry policy. Mirrors the tenacity API surface enough
that callers can swap to tenacity later without churn.

Default policy:
    - 5 attempts total (1 try + 4 retries)
    - exponential backoff with multiplier=0.5, max wait=8s, jitter=0.1
    - retries on a configurable exception tuple (default: ``Exception``,
      but call-sites should narrow this to e.g.
      ``src.assembled_core.errors.RecoverableError``)

Usage::

    from src.assembled_core.utils.retry import retry

    @retry()
    def fetch():
        return broker.get_positions()

    # narrow exception class:
    @retry(exceptions=(ConnectionError, TimeoutError))
    def call():
        ...

The decorator preserves __name__ / __doc__ via functools.wraps.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

DEFAULT_ATTEMPTS = 5
DEFAULT_BASE = 0.5
DEFAULT_MAX = 8.0
DEFAULT_JITTER = 0.1


def _backoff_seconds(attempt: int, base: float, cap: float, jitter: float) -> float:
    """Exponential backoff with proportional jitter.

    ``attempt`` is 1-indexed. Wait = min(base * 2**(attempt-1), cap)
    multiplied by (1 + uniform(-jitter, +jitter)).
    """
    wait: float = min(base * (2 ** (attempt - 1)), cap)
    if jitter > 0:
        wait *= 1.0 + random.uniform(-jitter, jitter)
    return float(max(0.0, wait))


def retry(
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    base: float = DEFAULT_BASE,
    cap: float = DEFAULT_MAX,
    jitter: float = DEFAULT_JITTER,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Decorator factory implementing the project retry convention.

    Args:
        attempts: total attempts including the initial call (default 5).
        base: first-retry wait in seconds; doubled each retry.
        cap: maximum single-retry wait (caps the exponential growth).
        jitter: fractional jitter applied to each wait (0 = none).
        exceptions: tuple of exceptions that trigger a retry; everything
            else propagates immediately.
        on_retry: optional callback ``(attempt_index, exc)`` invoked
            before each backoff sleep (use for logging / metrics).

    Returns:
        A decorator that wraps a callable with the retry behaviour.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")

    def _decorator(fn: Callable[..., _T]) -> Callable[..., _T]:
        @functools.wraps(fn)
        def _wrapped(*args: Any, **kwargs: Any) -> _T:
            last_exc: BaseException | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt >= attempts:
                        break
                    if on_retry is not None:
                        try:
                            on_retry(attempt, exc)
                        except Exception as cb_exc:  # noqa: BLE001
                            logger.debug("[retry] on_retry callback raised: %s", cb_exc)
                    wait = _backoff_seconds(attempt, base, cap, jitter)
                    logger.info(
                        "[retry] %s attempt %d/%d failed (%s); sleeping %.2fs",
                        fn.__name__,
                        attempt,
                        attempts,
                        exc.__class__.__name__,
                        wait,
                    )
                    time.sleep(wait)
            # Exhausted — re-raise the last exception. ``last_exc`` is
            # always set inside the loop on the failure path; this guard
            # appeases type checkers without a bare assert.
            if last_exc is None:  # pragma: no cover — loop invariant
                raise RuntimeError("retry exhausted with no captured exception")
            raise last_exc

        return _wrapped

    return _decorator
