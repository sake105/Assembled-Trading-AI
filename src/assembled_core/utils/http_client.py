"""Thin HTTP helper with enforced request timeouts (Item 164).

Every external API call in this codebase should use `get()` / `post()` from this
module instead of `requests.get()` / `requests.post()` directly, so that:

  * a sensible timeout is always present (never hangs indefinitely)
  * a structured error is raised on timeout with the URL logged
  * the default can be overridden per-call or globally via env var

Usage::

    from src.assembled_core.utils.http_client import get, post

    data = get("https://api.example.com/v1/prices", timeout=5.0)
    data.raise_for_status()

The env var ``HTTP_DEFAULT_TIMEOUT_SECONDS`` controls the module-level default.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT: float = float(os.environ.get("HTTP_DEFAULT_TIMEOUT_SECONDS", "10.0"))


def get(
    url: str,
    *,
    timeout: float | None = None,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    **kwargs: Any,
) -> requests.Response:
    """GET with an enforced timeout.

    Args:
        url: Full URL.
        timeout: Request timeout in seconds. Defaults to _DEFAULT_TIMEOUT.
        headers: Optional HTTP headers.
        params: Optional query parameters.
        **kwargs: Additional kwargs forwarded to ``requests.get``.

    Returns:
        ``requests.Response`` — caller must call ``.raise_for_status()`` if needed.

    Raises:
        requests.Timeout: when the server does not respond within *timeout* seconds.
        requests.RequestException: for all other network errors.
    """
    t = timeout if timeout is not None else _DEFAULT_TIMEOUT
    try:
        return requests.get(url, timeout=t, headers=headers, params=params, **kwargs)
    except requests.Timeout:
        log.warning("[http_client] GET timed out after %.1fs: %s", t, url)
        raise


def post(
    url: str,
    *,
    timeout: float | None = None,
    headers: dict[str, str] | None = None,
    json: Any = None,
    data: Any = None,
    **kwargs: Any,
) -> requests.Response:
    """POST with an enforced timeout.

    Args:
        url: Full URL.
        timeout: Request timeout in seconds. Defaults to _DEFAULT_TIMEOUT.
        headers: Optional HTTP headers.
        json: JSON-serialisable body.
        data: Raw body (used when *json* is None).
        **kwargs: Additional kwargs forwarded to ``requests.post``.

    Returns:
        ``requests.Response``.

    Raises:
        requests.Timeout: when the server does not respond within *timeout* seconds.
        requests.RequestException: for all other network errors.
    """
    t = timeout if timeout is not None else _DEFAULT_TIMEOUT
    try:
        return requests.post(
            url, timeout=t, headers=headers, json=json, data=data, **kwargs
        )
    except requests.Timeout:
        log.warning("[http_client] POST timed out after %.1fs: %s", t, url)
        raise


__all__ = ["get", "post", "_DEFAULT_TIMEOUT"]
