# src/assembled_core/api/auth.py
"""Shared authentication dependency for command-style API endpoints.

The harness uses a single ``ASSEMBLED_API_KEY`` environment variable for
"command" endpoints — anything that mutates state (kill-switch, paper orders,
paper reset). Read endpoints (``GET /...``) remain open.

Fail-open vs. fail-closed (audit SEC-1): when ``ASSEMBLED_API_KEY`` is unset
the dependency historically fell open with a warning so local development and
tests keep working. That silent fail-open is dangerous in production. The key
is now treated as MANDATORY (fail-closed → HTTP 503) whenever either:

- ``ASSEMBLED_API_REQUIRE_AUTH`` is truthy (explicit opt-in), or
- ``ASSEMBLED_RUNTIME_PROFILE`` names a production-like profile
  (``production`` / ``prod`` / ``live``).

Otherwise (dev / test / CI, key unset) it still falls open with a loud warning
so existing workflows are unchanged. ``/ready`` reports the resulting auth
posture so ops can verify a deployment is actually protected.

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
_REQUIRE_AUTH_VAR = "ASSEMBLED_API_REQUIRE_AUTH"
_PROFILE_VAR = "ASSEMBLED_RUNTIME_PROFILE"
_PRODUCTION_PROFILES = frozenset({"production", "prod", "live"})


def _expected_key() -> str | None:
    raw = os.environ.get(_ENV_VAR, "")
    return raw or None


def _truthy(raw: str | None) -> bool:
    return (raw or "").strip().lower() in {"1", "true", "yes", "on"}


def auth_required_when_unset() -> bool:
    """Whether a missing API key must fail closed (True) or warn-and-allow (False).

    Fails closed when auth is explicitly required (``ASSEMBLED_API_REQUIRE_AUTH``)
    or the runtime profile is production-like, so a prod deploy that forgets the
    key is still protected rather than silently serving open command endpoints.
    """
    if _truthy(os.environ.get(_REQUIRE_AUTH_VAR)):
        return True
    profile = os.environ.get(_PROFILE_VAR, "").strip().lower()
    return profile in _PRODUCTION_PROFILES


def require_api_key(x_api_key: str = Header(default="")) -> None:
    """FastAPI dependency: enforce X-API-Key header for command endpoints.

    Behaviour:
    - ``ASSEMBLED_API_KEY`` set + header matches → allow.
    - ``ASSEMBLED_API_KEY`` set + header missing/mismatch → 401.
    - ``ASSEMBLED_API_KEY`` unset + auth required (prod / opt-in) → 503
      (server misconfiguration: no key to authenticate against).
    - ``ASSEMBLED_API_KEY`` unset + auth not required → warn, allow (dev/test).
    """
    expected = _expected_key()
    if expected is None:
        if auth_required_when_unset():
            logger.error(
                "[API] %s not set but auth is required (%s=%r, %s=%r) "
                "— refusing command request",
                _ENV_VAR,
                _REQUIRE_AUTH_VAR,
                os.environ.get(_REQUIRE_AUTH_VAR, ""),
                _PROFILE_VAR,
                os.environ.get(_PROFILE_VAR, ""),
            )
            raise HTTPException(
                status_code=503,
                detail="API authentication is required but not configured",
            )
        logger.warning("[API] %s not set — command endpoints are UNPROTECTED", _ENV_VAR)
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def auth_is_configured() -> bool:
    """Return True iff a real API key is configured (used by /ready)."""
    return _expected_key() is not None
