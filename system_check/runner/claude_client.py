"""Async Anthropic client wrapper with retries, rate-limit awareness and
a dry-run mode.

The class is the only place in ``system_check`` that talks to the Anthropic
API. Keeping it narrow makes it easy to mock in tests and guarantees
redaction of secrets.

The SDK is imported lazily so that `pip install -e ".[system_check]"`
becomes a runtime requirement only when the client is actually used
(tests can mock the class without importing the SDK).
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Data containers
# -------------------------------------------------------------------------


@dataclass
class CallResult:
    """Normalised response from a single Claude call."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stop_reason: str | None = None
    attempts: int = 1
    # Error message from the SDK when the call ultimately failed — None on success.
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class RetryConfig:
    max_attempts: int = 3
    initial_backoff_seconds: float = 2.0
    backoff_multiplier: float = 2.0
    retry_on_status: tuple[int, ...] = (408, 429, 500, 502, 503, 504)
    per_call_timeout_seconds: float = 120.0


@dataclass
class ClaudeClientConfig:
    """Runtime configuration for :class:`ClaudeClient`."""

    api_key: str | None = None
    dry_run: bool = False
    retry: RetryConfig = field(default_factory=RetryConfig)


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------


class ClaudeClient:
    """Thin async wrapper around ``anthropic.AsyncAnthropic``.

    In ``dry_run`` mode, no SDK is imported and calls return a deterministic
    placeholder string so the tournament orchestrator can be exercised end
    to end without API credits.
    """

    def __init__(self, config: ClaudeClientConfig | None = None) -> None:
        self.config = config or ClaudeClientConfig()
        self._sdk_client: Any | None = None
        self._lock = asyncio.Lock()

    # ---------------------------------------------------------------
    # Setup helpers
    # ---------------------------------------------------------------

    async def _ensure_sdk(self) -> Any:
        if self.config.dry_run:
            return None
        if self._sdk_client is None:
            async with self._lock:
                if self._sdk_client is None:
                    # Key check before SDK import so a missing-key error is
                    # always the more actionable message.
                    api_key = self.config.api_key or os.environ.get(
                        "ANTHROPIC_API_KEY"
                    )
                    if not api_key:
                        raise RuntimeError(
                            "ANTHROPIC_API_KEY not set. "
                            "Add it to .env (never commit) and re-run."
                        )
                    try:
                        from anthropic import AsyncAnthropic
                    except ImportError as exc:  # pragma: no cover - env-specific
                        raise RuntimeError(
                            "anthropic SDK not installed — run "
                            "`pip install -e \".[system_check]\"`"
                        ) from exc
                    self._sdk_client = AsyncAnthropic(api_key=api_key)
        return self._sdk_client

    # ---------------------------------------------------------------
    # Core call
    # ---------------------------------------------------------------

    async def call(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
    ) -> CallResult:
        """Issue a single Messages call with retry and return a normalised result."""
        if self.config.dry_run:
            return _fake_result(model=model, max_tokens=max_tokens,
                                user_prompt=user_prompt)

        sdk = await self._ensure_sdk()
        retry = self.config.retry

        attempt = 0
        backoff = retry.initial_backoff_seconds
        last_error: Exception | None = None

        while attempt < retry.max_attempts:
            attempt += 1
            try:
                response = await asyncio.wait_for(
                    sdk.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                    ),
                    timeout=retry.per_call_timeout_seconds,
                )
                content = _extract_text(response)
                usage = getattr(response, "usage", None)
                return CallResult(
                    content=content,
                    model=model,
                    prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "output_tokens", 0) or 0,
                    stop_reason=getattr(response, "stop_reason", None),
                    attempts=attempt,
                )
            except asyncio.TimeoutError as exc:
                last_error = exc
                logger.warning(
                    "[claude] timeout model=%s attempt=%s/%s",
                    model, attempt, retry.max_attempts,
                )
            except Exception as exc:  # pragma: no cover - SDK specific
                last_error = exc
                status = _extract_status(exc)
                retriable = status is None or status in retry.retry_on_status
                logger.warning(
                    "[claude] error model=%s attempt=%s/%s status=%s retriable=%s: %s",
                    model, attempt, retry.max_attempts, status, retriable,
                    _redact(str(exc)),
                )
                if not retriable:
                    break

            if attempt < retry.max_attempts:
                sleep_for = backoff * (1.0 + random.random() * 0.25)  # nosec - not security
                await asyncio.sleep(sleep_for)
                backoff *= retry.backoff_multiplier

        return CallResult(
            content="",
            model=model,
            error=_redact(str(last_error)) if last_error else "unknown error",
            attempts=attempt,
        )

    # ---------------------------------------------------------------
    # Convenience batching
    # ---------------------------------------------------------------

    async def call_many(
        self,
        jobs: list[dict[str, Any]],
        *,
        max_parallel: int = 8,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[CallResult]:
        """Run many :meth:`call` invocations with bounded parallelism.

        Each entry in *jobs* is a kwargs dict accepted by :meth:`call`.
        Order of results matches order of *jobs*.
        """
        sem = asyncio.Semaphore(max_parallel)
        total = len(jobs)
        done = 0

        async def _run(i: int, kwargs: dict[str, Any]) -> CallResult:
            nonlocal done
            async with sem:
                result = await self.call(**kwargs)
            done += 1
            if progress is not None:
                try:
                    progress(done, total)
                except Exception:  # pragma: no cover - progress is best effort
                    pass
            return result

        coros = [_run(i, j) for i, j in enumerate(jobs)]
        return await asyncio.gather(*coros)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _extract_text(response: Any) -> str:
    """Concatenate text blocks from an Anthropic Messages response."""
    try:
        parts = []
        for block in getattr(response, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts).strip()
    except Exception:  # pragma: no cover - defensive
        return ""


def _extract_status(exc: Exception) -> int | None:
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


_SECRET_PATTERNS = ("sk-ant-", "sk-", "Bearer ")


def _redact(text: str) -> str:
    """Strip anything resembling an API key before logging."""
    if not text:
        return text
    redacted = text
    for pattern in _SECRET_PATTERNS:
        idx = redacted.find(pattern)
        while idx != -1:
            end = idx + len(pattern)
            # Redact the following alnum/hyphen run.
            j = end
            while j < len(redacted) and (redacted[j].isalnum() or redacted[j] in "-_"):
                j += 1
            redacted = redacted[:end] + "***redacted***" + redacted[j:]
            idx = redacted.find(pattern, end + len("***redacted***"))
    return redacted


def _fake_result(*, model: str, max_tokens: int, user_prompt: str) -> CallResult:
    """Deterministic placeholder response for dry-run mode."""
    preview = user_prompt.splitlines()[0] if user_prompt else ""
    preview = preview[:120]
    content = (
        f"[dry-run placeholder — model={model}, max_tokens={max_tokens}]\n"
        f"prompt preview: {preview}\n"
        "This response exists so the tournament pipeline can run without API cost."
    )
    return CallResult(
        content=content,
        model=model,
        prompt_tokens=len(user_prompt) // 4,
        completion_tokens=len(content) // 4,
        stop_reason="end_turn",
        attempts=1,
    )
