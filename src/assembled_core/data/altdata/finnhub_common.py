"""Common utilities for Finnhub API clients.

This module provides shared functions for Finnhub API access,
avoiding code duplication between finnhub_events.py and finnhub_news_macro.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.assembled_core.config.settings import Settings

logger = logging.getLogger(__name__)

# Finnhub API base URL
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

# Rate limit: 60 calls/minute for free tier
RATE_LIMIT_DELAY_SECONDS = 1.0


def get_finnhub_session(settings: Settings) -> tuple:
    """Get Finnhub API session and validate API key.

    Args:
        settings: Application settings (must contain finnhub_api_key)

    Returns:
        Tuple of (session, api_key) where session is a requests.Session configured
        with base URL, and api_key is the validated API key string

    Raises:
        RuntimeError: If finnhub_api_key is not set or empty
        ImportError: If requests is not installed
    """
    try:
        import requests
    except ImportError:
        raise ImportError(
            "requests is required for Finnhub API. "
            "Install with: pip install requests"
        )

    # Multi-key rotation (2026-05-22): prefer rotator pool, fallback to
    # settings.finnhub_api_key for backward compat with the canonical
    # ASSEMBLED_FINNHUB_API_KEY env var. When the user adds FINNHUB_API_KEY_2
    # or comma-separated FINNHUB_API_KEYS, those join the pool automatically.
    api_key: str | None = None
    try:
        from src.assembled_core.utils.api_key_rotator import get_rotator

        api_key = get_rotator().get_key("finnhub")
    except Exception:  # noqa: BLE001 — defensive
        api_key = None
    if not api_key or not api_key.strip():
        api_key = settings.finnhub_api_key
    if not api_key or not api_key.strip():
        raise RuntimeError(
            "FINNHUB_API_KEY not set. "
            "Set via ASSEMBLED_FINNHUB_API_KEY environment variable or in settings."
        )

    # Create session with base URL
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Assembled-Trading-AI/1.0",
        }
    )

    return session, api_key.strip()


def mark_finnhub_rate_limited(key: str | None, exc_or_response: object) -> None:
    """Cool down `key` in the rotator pool if exc_or_response signals 429.

    Call from each finnhub fetch site's except branch (or on 429 status):
        try:
            response = session.get(...)
            response.raise_for_status()
        except Exception as exc:
            mark_finnhub_rate_limited(api_key, exc)
            ...

    Finnhub free tier: 60 calls/minute → short cooldown. Best-effort:
    silent no-op if rotator unavailable or signal is not rate-limit.
    """
    if not key:
        return
    try:
        from src.assembled_core.utils.api_key_rotator import (
            get_rotator,
            is_rate_limit_signal,
        )

        if is_rate_limit_signal(exc_or_response):
            get_rotator().mark_rate_limited("finnhub", key, cooldown_seconds=70.0)
    except Exception:  # noqa: BLE001
        pass
