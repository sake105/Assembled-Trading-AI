"""FastAPI middleware: Request-ID tracing + process-time headers + audit log + rate limit.

Usage in app.py:
    from assembled_core.api.middleware import add_middleware
    add_middleware(app)

Or manually:
    from assembled_core.api.middleware import request_id_middleware
    app.middleware("http")(request_id_middleware)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI, Request

logger = logging.getLogger(__name__)

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


def _audit_log_path() -> Path:
    override = os.environ.get("ASSEMBLED_API_AUDIT_PATH", "")
    return Path(override) if override else Path("output/ops/api_audit.jsonl")


def _hash_short(s: str) -> str:
    if not s:
        return ""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


async def api_audit_middleware(request: Request, call_next):
    """Append an audit record per request to output/ops/api_audit.jsonl.

    Audit C3-073: every authenticated command (POST/DELETE/PUT/PATCH) is
    logged with (ts, method, path, ip, actor_hash, payload_hash, status).
    GET requests are deliberately skipped to keep the log focused on
    state-mutating events.

    Opt-in via env: set ``ASSEMBLED_API_AUDIT=1`` to enable. When disabled
    the middleware is a no-op pass-through.

    Hashing rationale: payload + API key are sha256-truncated so the log
    is safe to share but still correlates uniquely to the request/actor.
    """
    enabled = os.environ.get("ASSEMBLED_API_AUDIT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled or request.method.upper() == "GET":
        return await call_next(request)

    # Capture request fingerprint BEFORE the handler runs — at this point
    # call_next has not yet consumed the body, so it stays available to the
    # endpoint. We only sniff the body for sha256 if it's small (< 64 KiB).
    body_hash = ""
    try:
        body = await request.body()
        if body and len(body) <= 64 * 1024:
            body_hash = _hash_short(body.decode("utf-8", errors="ignore"))
        # Re-inject so the endpoint can still consume the body.
        # Starlette caches the body after first await, so this is OK.
    except Exception as exc:  # noqa: BLE001
        logger.debug("[api-audit] body read failed: %s", exc)

    actor_hash = _hash_short(request.headers.get("X-API-Key", ""))
    client_ip = request.client.host if request.client else ""
    method = request.method
    path = str(request.url.path)

    response = await call_next(request)

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "path": path,
        "ip": client_ip,
        "actor_hash": actor_hash,
        "body_hash": body_hash,
        "status": int(getattr(response, "status_code", 0)),
        "request_id": response.headers.get("X-Request-ID", ""),
    }
    try:
        p = _audit_log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError as exc:
                logger.debug("[api-audit] fsync failed: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.error("[api-audit] failed to write %s: %s", p, exc)

    return response


# ---------------------------------------------------------------------------
# Rate limiting (audit C3-114)
# ---------------------------------------------------------------------------

_RL_STATE: dict[str, tuple[float, float]] = {}  # ip -> (tokens, last_refill_ts)


def _rate_limit_config() -> tuple[float, float] | None:
    """Return (capacity, refill_per_second) from env, or None if disabled.

    Env vars:
        ASSEMBLED_API_RATE_LIMIT — capacity per IP (integer). 0 / unset = disabled.
        ASSEMBLED_API_RATE_WINDOW_SEC — refill window in seconds (default 60).
    """
    capacity_raw = os.environ.get("ASSEMBLED_API_RATE_LIMIT", "").strip()
    if not capacity_raw:
        return None
    try:
        capacity = float(capacity_raw)
    except ValueError:
        return None
    if capacity <= 0:
        return None
    window = float(os.environ.get("ASSEMBLED_API_RATE_WINDOW_SEC", "60") or 60)
    if window <= 0:
        window = 60.0
    return capacity, capacity / window


async def rate_limit_middleware(request: Request, call_next):
    """Token-bucket per-IP rate limit (audit C3-114).

    Stateless across restarts — buckets live in-process and reset when the
    app restarts. For a single-tenant solo-quant backend this is enough;
    multi-process / horizontal deployment would need Redis (out of scope).

    Disabled by default; enable by setting ``ASSEMBLED_API_RATE_LIMIT`` to a
    positive integer (capacity per window). Window length defaults to 60 s
    and is overridable via ``ASSEMBLED_API_RATE_WINDOW_SEC``.
    """
    cfg = _rate_limit_config()
    if cfg is None:
        return await call_next(request)
    capacity, refill_rate = cfg

    ip = request.client.host if request.client else "anonymous"
    now = time.monotonic()
    tokens, last = _RL_STATE.get(ip, (capacity, now))
    # Refill since last check.
    tokens = min(capacity, tokens + refill_rate * (now - last))
    if tokens < 1.0:
        # FastAPI is imported here — lazy to keep the top-level light.
        from fastapi.responses import JSONResponse

        retry_after = max(1, int((1.0 - tokens) / max(refill_rate, 1e-9)))
        _RL_STATE[ip] = (tokens, now)
        return JSONResponse(
            content={
                "detail": "Rate limit exceeded",
                "retry_after_seconds": retry_after,
            },
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )
    _RL_STATE[ip] = (tokens - 1.0, now)
    return await call_next(request)


def reset_rate_limit_state() -> None:
    """Test helper — wipes the in-memory bucket state."""
    _RL_STATE.clear()


# ---------------------------------------------------------------------------
# Chaos engineering — staging-only middleware (audit C2-010)
# ---------------------------------------------------------------------------


async def chaos_middleware(request: Request, call_next):
    """Inject latency / drop / 5xx based on ``ASSEMBLED_CHAOS_MODE`` (audit C2-010).

    Off by default. Enable via env (``ASSEMBLED_CHAOS_MODE=1``) on staging
    only; production deployments MUST keep this disabled.

    Knobs (env-driven, optional):
        ASSEMBLED_CHAOS_LATENCY_MS     — fixed extra delay per request
        ASSEMBLED_CHAOS_DROP_PROB      — probability of returning 503
        ASSEMBLED_CHAOS_5XX_PROB       — probability of returning 500

    A real chaos stack (LitmusChaos, audit C2-012) is the production
    answer; this middleware is the dev-loop version that lets retry /
    circuit-breaker / bulkhead logic exercise locally.
    """
    import asyncio
    import random

    if os.environ.get("ASSEMBLED_CHAOS_MODE", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return await call_next(request)

    drop = float(os.environ.get("ASSEMBLED_CHAOS_DROP_PROB", "0") or 0)
    five = float(os.environ.get("ASSEMBLED_CHAOS_5XX_PROB", "0") or 0)
    lat = float(os.environ.get("ASSEMBLED_CHAOS_LATENCY_MS", "0") or 0)

    if lat > 0:
        await asyncio.sleep(lat / 1000.0)
    if drop > 0 and random.random() < drop:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            content={"detail": "chaos: simulated 503"},
            status_code=503,
            headers={"X-Chaos": "drop"},
        )
    if five > 0 and random.random() < five:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            content={"detail": "chaos: simulated 500"},
            status_code=500,
            headers={"X-Chaos": "5xx"},
        )

    return await call_next(request)


def add_middleware(app: FastAPI) -> None:
    """Register all API middleware on a FastAPI app instance."""
    # Order: chaos first (fail-loud before any other layer runs), then
    # rate-limit (cheap reject), request-id (trace surviving requests),
    # audit-log (record passed-through requests).
    app.middleware("http")(chaos_middleware)
    app.middleware("http")(rate_limit_middleware)
    app.middleware("http")(request_id_middleware)
    app.middleware("http")(api_audit_middleware)
