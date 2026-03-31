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
    
    api_key = settings.finnhub_api_key
    if not api_key or not api_key.strip():
        raise RuntimeError(
            "FINNHUB_API_KEY not set. "
            "Set via ASSEMBLED_FINNHUB_API_KEY environment variable or in settings."
        )
    
    # Create session with base URL
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Assembled-Trading-AI/1.0",
    })
    
    return session, api_key.strip()

