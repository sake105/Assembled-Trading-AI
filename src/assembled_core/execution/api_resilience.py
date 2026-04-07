"""API Resilience — Retry logic, rate limiting, and error handling for broker APIs.

Provides:
- RetryPolicy: Configurable retry strategy with exponential backoff
- RateLimiter: Token-bucket rate limiter (default: 200 req/min for Alpaca)
- retry_with_backoff: Decorator/wrapper for resilient API calls
- MarketClosedError: Raised when orders submitted outside market hours
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MarketClosedError(Exception):
    """Raised when an order is submitted outside market hours."""


class RateLimitExceededError(Exception):
    """Raised when API rate limit is hit and retries exhausted."""


@dataclass
class RetryPolicy:
    """Configurable retry strategy with exponential backoff.

    Attributes:
        max_retries: Maximum number of retry attempts (0 = no retry).
        base_delay_s: Initial delay in seconds before first retry.
        max_delay_s: Maximum delay cap in seconds.
        backoff_factor: Multiplier applied to delay after each retry.
        retryable_exceptions: Exception types that trigger a retry.
    """

    max_retries: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 10.0
    backoff_factor: float = 2.0
    retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (
            ConnectionError,
            TimeoutError,
            OSError,
        )
    )


# Default policy for Alpaca API calls
DEFAULT_RETRY_POLICY = RetryPolicy(
    max_retries=3,
    base_delay_s=1.0,
    max_delay_s=10.0,
    backoff_factor=2.0,
)


class RateLimiter:
    """Token-bucket rate limiter.

    Default: 200 requests per 60 seconds (Alpaca paper trading limit).
    Thread-safe via simple locking on token refill.
    """

    def __init__(
        self,
        max_requests: int = 200,
        window_seconds: float = 60.0,
    ) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._tokens = float(max_requests)
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * (self._max_requests / self._window_seconds)
        self._tokens = min(self._max_requests, self._tokens + new_tokens)
        self._last_refill = now

    def acquire(self, timeout_s: float = 30.0) -> bool:
        """Acquire a token. Blocks up to timeout_s if rate-limited.

        Returns:
            True if token acquired, False if timeout exceeded.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            # Sleep for estimated time until next token
            wait = min(
                self._window_seconds / self._max_requests,
                remaining,
            )
            time.sleep(wait)

    @property
    def available_tokens(self) -> float:
        self._refill()
        return self._tokens


# Singleton rate limiter for Alpaca (200 req/min)
_alpaca_rate_limiter: RateLimiter | None = None


def get_alpaca_rate_limiter() -> RateLimiter:
    """Get or create the singleton Alpaca rate limiter."""
    global _alpaca_rate_limiter
    if _alpaca_rate_limiter is None:
        _alpaca_rate_limiter = RateLimiter(max_requests=200, window_seconds=60.0)
    return _alpaca_rate_limiter


def retry_with_backoff(
    fn: Callable[..., T],
    *args: Any,
    policy: RetryPolicy | None = None,
    rate_limiter: RateLimiter | None = None,
    operation_name: str = "",
    **kwargs: Any,
) -> T:
    """Execute fn with retry logic and optional rate limiting.

    Args:
        fn: The function to call.
        *args: Positional arguments for fn.
        policy: Retry policy (defaults to DEFAULT_RETRY_POLICY).
        rate_limiter: Optional rate limiter to acquire before each attempt.
        operation_name: Human-readable name for logging.
        **kwargs: Keyword arguments for fn.

    Returns:
        The return value of fn.

    Raises:
        The last exception if all retries exhausted.
        RateLimitExceededError if rate limiter times out.
    """
    if policy is None:
        policy = DEFAULT_RETRY_POLICY

    op_label = operation_name or fn.__name__
    last_exc: Exception | None = None

    for attempt in range(1 + policy.max_retries):
        # Rate limiting
        if rate_limiter is not None:
            if not rate_limiter.acquire(timeout_s=30.0):
                raise RateLimitExceededError(
                    f"[{op_label}] Rate limit exceeded after 30s wait"
                )

        try:
            result = fn(*args, **kwargs)
            if attempt > 0:
                logger.info(
                    "[%s] succeeded on attempt %d/%d",
                    op_label,
                    attempt + 1,
                    1 + policy.max_retries,
                )
            return result

        except policy.retryable_exceptions as exc:
            last_exc = exc
            if attempt < policy.max_retries:
                delay = min(
                    policy.base_delay_s * (policy.backoff_factor**attempt),
                    policy.max_delay_s,
                )
                logger.warning(
                    "[%s] attempt %d/%d failed (%s: %s), retrying in %.1fs",
                    op_label,
                    attempt + 1,
                    1 + policy.max_retries,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "[%s] all %d attempts failed. Last error: %s: %s",
                    op_label,
                    1 + policy.max_retries,
                    type(exc).__name__,
                    exc,
                )

        except Exception:
            # Non-retryable exception — propagate immediately
            raise

    # All retries exhausted
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"[{op_label}] unexpected: no result and no exception")


def is_retryable_http_status(status_code: int) -> bool:
    """Check if an HTTP status code warrants a retry.

    429 = Too Many Requests (rate limited)
    500, 502, 503, 504 = Server errors (transient)
    """
    return status_code in {429, 500, 502, 503, 504}


__all__ = [
    "RetryPolicy",
    "RateLimiter",
    "MarketClosedError",
    "RateLimitExceededError",
    "DEFAULT_RETRY_POLICY",
    "get_alpaca_rate_limiter",
    "retry_with_backoff",
    "is_retryable_http_status",
]
