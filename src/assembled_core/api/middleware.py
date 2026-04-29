"""FastAPI middleware: Request-ID tracing + process-time headers.

Usage in app.py:
    from assembled_core.api.middleware import add_middleware
    add_middleware(app)

Or manually:
    from assembled_core.api.middleware import request_id_middleware
    app.middleware("http")(request_id_middleware)
"""
from __future__ import annotations

import time
import uuid
from contextvars import ContextVar

from fastapi import FastAPI, Request

# ContextVar so log statements in any coroutine/thread can pick up the current ID
request_id_var: ContextVar[str] = ContextVar("request_id", default="")


async def request_id_middleware(request: Request, call_next):
    """Attach X-Request-ID and X-Process-Time-Ms to every response.

    If the client sends X-Request-ID, echo it back (idempotent tracing).
    Otherwise generate a new UUID4.
    """
    rid = request.headers.get("X-Request-ID", "") or str(uuid.uuid4())
    token = request_id_var.set(rid)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)

    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = rid
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


def get_request_id() -> str:
    """Return the current request ID (empty string outside a request context)."""
    return request_id_var.get("")


def add_middleware(app: FastAPI) -> None:
    """Register all API middleware on a FastAPI app instance."""
    app.middleware("http")(request_id_middleware)
