"""Sentry error tracking integration.

From 12_FREE_INFRASTRUKTUR.md §12.13.
Free tier: 5k errors/month — sufficient for solo trading system.
Critical: a single unhandled exception in the order pipeline can cause 4-figure losses.

Install: pip install sentry-sdk[fastapi]

Usage:
    from src.assembled_core.ops.error_tracking import init_sentry, capture_exception
    init_sentry()  # reads SENTRY_DSN from environment
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

_sentry_initialized = False


def _try_sentry():
    try:
        import sentry_sdk
        return sentry_sdk
    except ImportError:
        logger.debug("sentry-sdk not installed — pip install sentry-sdk[fastapi]")
        return None


def init_sentry(
    dsn: str | None = None,
    environment: str | None = None,
    traces_sample_rate: float = 0.1,
    profiles_sample_rate: float = 0.1,
    release: str | None = None,
) -> bool:
    """Initialize Sentry SDK.

    Args:
        dsn: Sentry DSN. Reads SENTRY_DSN env var if not provided.
        environment: deployment environment ('production', 'paper', 'dev').
        traces_sample_rate: Fraction of transactions to trace (0.0-1.0).
        profiles_sample_rate: Fraction of profiles to capture (0.0-1.0).
        release: App release identifier (e.g. git SHA).

    Returns:
        True if Sentry initialized, False if not configured or not installed.
    """
    global _sentry_initialized
    sentry_sdk = _try_sentry()
    if sentry_sdk is None:
        return False

    effective_dsn = dsn or os.environ.get("SENTRY_DSN", "")
    if not effective_dsn:
        logger.debug("SENTRY_DSN not set — Sentry disabled")
        return False

    sentry_sdk.init(
        dsn=effective_dsn,
        environment=environment or os.environ.get("ENVIRONMENT", "development"),
        traces_sample_rate=traces_sample_rate,
        profiles_sample_rate=profiles_sample_rate,
        release=release,
        # Don't send PII
        send_default_pii=False,
    )
    _sentry_initialized = True
    logger.info("Sentry initialized (env=%s, traces=%.0f%%)", environment, traces_sample_rate * 100)
    return True


def capture_exception(exc: Exception, context: dict[str, Any] | None = None) -> None:
    """Capture an exception to Sentry.

    Args:
        exc: Exception to capture.
        context: Optional dict of additional context tags/data.
    """
    sentry_sdk = _try_sentry()
    if sentry_sdk is None or not _sentry_initialized:
        logger.error("Exception (Sentry not active): %s", exc, exc_info=exc)
        return

    with sentry_sdk.push_scope() as scope:
        if context:
            for key, value in context.items():
                scope.set_tag(key, str(value))
        sentry_sdk.capture_exception(exc)


def capture_message(message: str, level: str = "warning", context: dict[str, Any] | None = None) -> None:
    """Capture a message to Sentry.

    Args:
        message: Message string.
        level: Sentry level — 'debug', 'info', 'warning', 'error', 'fatal'.
        context: Optional additional context.
    """
    sentry_sdk = _try_sentry()
    if sentry_sdk is None or not _sentry_initialized:
        logger.log(
            {"debug": 10, "info": 20, "warning": 30, "error": 40, "fatal": 50}.get(level, 30),
            "Sentry msg (not active): %s",
            message,
        )
        return

    with sentry_sdk.push_scope() as scope:
        if context:
            for key, value in context.items():
                scope.set_tag(key, str(value))
        sentry_sdk.capture_message(message, level=level)


@contextmanager
def sentry_transaction(name: str, op: str = "task"):
    """Context manager for Sentry performance transactions.

    Usage:
        with sentry_transaction("eod_pipeline", op="pipeline"):
            run_eod_pipeline()
    """
    sentry_sdk = _try_sentry()
    if sentry_sdk is None or not _sentry_initialized:
        yield
        return

    with sentry_sdk.start_transaction(name=name, op=op):
        yield


def set_user_context(user_id: str) -> None:
    """Set the current user in Sentry scope."""
    sentry_sdk = _try_sentry()
    if sentry_sdk is not None and _sentry_initialized:
        sentry_sdk.set_user({"id": user_id})


def init_sentry_fastapi(app: Any) -> bool:
    """Initialize Sentry with FastAPI integration.

    Call this in the FastAPI lifespan or during app startup.

    Args:
        app: FastAPI app instance.

    Returns:
        True if initialized.
    """
    try:
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
        sentry_sdk = _try_sentry()
        if sentry_sdk is None:
            return False

        dsn = os.environ.get("SENTRY_DSN", "")
        if not dsn:
            return False

        sentry_sdk.init(
            dsn=dsn,
            integrations=[
                StarletteIntegration(transaction_style="endpoint"),
                FastApiIntegration(transaction_style="endpoint"),
            ],
            traces_sample_rate=0.1,
            profiles_sample_rate=0.1,
            send_default_pii=False,
        )
        return True
    except Exception as exc:
        logger.warning("Sentry FastAPI init failed: %s", exc)
        return False


__all__ = [
    "init_sentry",
    "init_sentry_fastapi",
    "capture_exception",
    "capture_message",
    "sentry_transaction",
    "set_user_context",
]
