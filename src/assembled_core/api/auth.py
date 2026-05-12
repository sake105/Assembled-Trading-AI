# src/assembled_core/api/auth.py
"""Shared authentication dependency for command-style API endpoints.

The harness uses a single ``ASSEMBLED_API_KEY`` environment variable for
"command" endpoints — anything that mutates state (kill-switch, paper orders,
paper reset). Read endpoints (``GET /...``) remain open.

When ``ASSEMBLED_API_KEY`` is unset the dependency falls open with a loud
warning so local development and tests keep working. In production the env var
MUST be set; ``/ready`` reports the auth posture so ops can verify.

Uses ``hmac.compare_digest`` for constant-time comparison (defense in depth
against timing side-channels — practically irrelevant for a single-tenant
backend but cheap and correct).
"""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import Header, HTTPException

logger = logging.getLogger(__name__)

_ENV_VAR = "ASSEMBLED_API_KEY"


def _expected_key() -> str | None:
    raw = os.environ.get(_ENV_VAR, "")
    return raw or None


def require_api_key(x_api_key: str = Header(default="")) -> None:
    """FastAPI dependency: enforce X-API-Key header for command endpoints.

    Behaviour:
    - ``ASSEMBLED_API_KEY`` unset → warn-once, allow (dev/test default).
    - ``ASSEMBLED_API_KEY`` set + header matches → allow.
    - ``ASSEMBLED_API_KEY`` set + header missing/mismatch → 401.
    """
    expected = _expected_key()
    if expected is None:
        logger.warning("[API] %s not set — command endpoints are UNPROTECTED", _ENV_VAR)
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def auth_is_configured() -> bool:
    """Return True iff a real API key is configured (used by /ready)."""
    return _expected_key() is not None
