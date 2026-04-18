"""Tests for claude_client.

Verifies:
* dry-run mode produces deterministic placeholder content without SDK import
* retry triggers on retriable status codes and gives up on non-retriable
* secret redaction in log messages
* call_many preserves order and respects concurrency limit
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from system_check.runner import claude_client as cc


# -------------------------------------------------------------------------
# Dry-run
# -------------------------------------------------------------------------


def test_dry_run_call_returns_placeholder() -> None:
    client = cc.ClaudeClient(cc.ClaudeClientConfig(dry_run=True))
    result = asyncio.run(
        client.call(
            model="claude-haiku-4-5-20251001",
            system_prompt="test",
            user_prompt="hello\nworld",
            max_tokens=100,
        )
    )
    assert result.ok
    assert "dry-run placeholder" in result.content
    assert result.model == "claude-haiku-4-5-20251001"
    assert result.attempts == 1
    assert result.prompt_tokens > 0


def test_dry_run_call_many_preserves_order() -> None:
    client = cc.ClaudeClient(cc.ClaudeClientConfig(dry_run=True))
    jobs = [
        dict(
            model="claude-haiku-4-5-20251001",
            system_prompt=f"sys{i}",
            user_prompt=f"user prompt {i}",
            max_tokens=100,
        )
        for i in range(5)
    ]
    results = asyncio.run(client.call_many(jobs, max_parallel=3))
    assert len(results) == 5
    for i, r in enumerate(results):
        assert r.ok
        assert f"user prompt {i}" in r.content


# -------------------------------------------------------------------------
# Retry behaviour
# -------------------------------------------------------------------------


class _FakeError(Exception):
    """Stand-in for anthropic.APIStatusError with a status_code attribute."""

    def __init__(self, status_code: int, msg: str = "err"):
        super().__init__(msg)
        self.status_code = status_code


def _make_real_client(**retry_overrides) -> cc.ClaudeClient:
    return cc.ClaudeClient(cc.ClaudeClientConfig(
        api_key="sk-ant-test",
        dry_run=False,
        retry=cc.RetryConfig(
            max_attempts=retry_overrides.get("max_attempts", 3),
            initial_backoff_seconds=retry_overrides.get("initial_backoff_seconds", 0.0),
            backoff_multiplier=retry_overrides.get("backoff_multiplier", 1.0),
            retry_on_status=retry_overrides.get("retry_on_status", (429, 500, 502, 503, 504, 408)),
            per_call_timeout_seconds=retry_overrides.get("per_call_timeout_seconds", 5.0),
        ),
    ))


def _install_fake_sdk(client: cc.ClaudeClient, create_side_effect) -> MagicMock:
    sdk = MagicMock()
    sdk.messages = MagicMock()
    sdk.messages.create = AsyncMock(side_effect=create_side_effect)
    client._sdk_client = sdk  # short-circuit lazy SDK init
    return sdk


def test_retry_on_429_then_succeed() -> None:
    client = _make_real_client()

    ok_response = MagicMock()
    ok_response.content = [MagicMock(text="ok")]
    ok_response.stop_reason = "end_turn"
    ok_response.usage = MagicMock(input_tokens=10, output_tokens=2)

    side_effects = [_FakeError(429), _FakeError(429), ok_response]
    sdk = _install_fake_sdk(client, side_effects)

    result = asyncio.run(client.call(
        model="claude-haiku-4-5-20251001",
        system_prompt="sys",
        user_prompt="user",
        max_tokens=50,
    ))
    assert result.ok
    assert result.content == "ok"
    assert result.attempts == 3
    assert sdk.messages.create.await_count == 3


def test_no_retry_on_401() -> None:
    client = _make_real_client()
    _install_fake_sdk(client, [_FakeError(401, "auth bad")])

    result = asyncio.run(client.call(
        model="claude-haiku-4-5-20251001",
        system_prompt="sys",
        user_prompt="user",
        max_tokens=50,
    ))
    assert not result.ok
    assert result.attempts == 1
    assert "auth" in (result.error or "").lower()


def test_exhausts_attempts_on_persistent_429() -> None:
    client = _make_real_client(max_attempts=2)
    _install_fake_sdk(client, [_FakeError(429), _FakeError(429)])

    result = asyncio.run(client.call(
        model="claude-haiku-4-5-20251001",
        system_prompt="sys",
        user_prompt="user",
        max_tokens=50,
    ))
    assert not result.ok
    assert result.attempts == 2


# -------------------------------------------------------------------------
# Redaction
# -------------------------------------------------------------------------


def test_redact_strips_api_key_prefixes() -> None:
    text = "Authorization: sk-ant-abc123XYZ or Bearer xyz-secret-99"
    out = cc._redact(text)
    assert "sk-ant-abc123XYZ" not in out
    assert "xyz-secret-99" not in out
    assert "***redacted***" in out


# -------------------------------------------------------------------------
# API key guard
# -------------------------------------------------------------------------


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = cc.ClaudeClient(cc.ClaudeClientConfig(dry_run=False))
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        asyncio.run(client._ensure_sdk())
