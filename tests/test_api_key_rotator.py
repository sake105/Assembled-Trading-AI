"""Tests for src.assembled_core.utils.api_key_rotator.

Covers:
  - Single-key env discovery (backward compat with existing call sites)
  - Numbered multi-key env discovery (KEY, KEY_2, KEY_3, ...)
  - Comma-separated plural env discovery (KEY_KEYS="a,b,c")
  - Discovery dedup + whitespace handling
  - Round-robin rotation
  - mark_rate_limited puts a key into cooldown; rotator skips it
  - All-cooled-down returns None + logs WARN
  - State persistence (cooldown survives rotator reset)
  - State persistence file does NOT leak the actual key (only last-4 suffix)
"""

from __future__ import annotations

import json
import time

import pytest

pytestmark = pytest.mark.fast

from src.assembled_core.utils.api_key_rotator import (
    ApiKeyRotator,
    _discover_keys_for_provider,
    is_rate_limit_signal,
    known_providers,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Drop any inherited provider env vars (canonical + aliases) per test."""
    for prefix in ("ALPHAVANTAGE", "POLYGON", "NEWSAPI", "FINNHUB", "FRED"):
        for var in (f"{prefix}_API_KEY", f"{prefix}_API_KEYS"):
            monkeypatch.delenv(var, raising=False)
        for i in range(2, 32):
            monkeypatch.delenv(f"{prefix}_API_KEY_{i}", raising=False)
    # Legacy aliases documented in .env.example
    for alias in (
        "ALPHAVANTAGE_KEY",
        "NEWSAPI_KEY",
        "ASSEMBLED_FINNHUB_API_KEY",
        "FINNHUB_KEY",
    ):
        monkeypatch.delenv(alias, raising=False)


def test_known_providers_includes_all_five():
    providers = set(known_providers())
    assert providers == {"alphavantage", "polygon", "newsapi", "finnhub", "fred"}


def test_discover_keys_single_env_var(monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "key-A")
    assert _discover_keys_for_provider("alphavantage") == ["key-A"]


def test_discover_keys_numbered_form(monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "key-A")
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY_2", "key-B")
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY_3", "key-C")
    keys = _discover_keys_for_provider("alphavantage")
    assert keys == ["key-A", "key-B", "key-C"]


def test_discover_keys_plural_form(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEYS", "p-1, p-2 , p-3")
    keys = _discover_keys_for_provider("polygon")
    # Whitespace-trimmed
    assert keys == ["p-1", "p-2", "p-3"]


def test_discover_keys_combines_single_numbered_and_plural(monkeypatch):
    monkeypatch.setenv("NEWSAPI_API_KEY", "alpha")
    monkeypatch.setenv("NEWSAPI_API_KEY_2", "beta")
    monkeypatch.setenv("NEWSAPI_API_KEYS", "gamma,delta")
    keys = _discover_keys_for_provider("newsapi")
    # Order preserved: single → numbered → plural
    assert keys == ["alpha", "beta", "gamma", "delta"]


def test_discover_keys_dedupes(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "same-key")
    monkeypatch.setenv("FINNHUB_API_KEY_2", "same-key")  # duplicate
    monkeypatch.setenv("FINNHUB_API_KEYS", "same-key, other-key")
    keys = _discover_keys_for_provider("finnhub")
    assert keys == ["same-key", "other-key"]


def test_discover_keys_skips_whitespace_only(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "  ")  # whitespace only
    monkeypatch.setenv("POLYGON_API_KEY_2", "real-key")
    keys = _discover_keys_for_provider("polygon")
    assert keys == ["real-key"]


def test_alphavantage_legacy_alias_picked_up(monkeypatch):
    """scripts/fetch_news_alphavantage.py:48 reads ALPHAVANTAGE_KEY as fallback."""
    monkeypatch.setenv("ALPHAVANTAGE_KEY", "legacy-alias")
    keys = _discover_keys_for_provider("alphavantage")
    assert "legacy-alias" in keys


def test_newsapi_legacy_alias_picked_up(monkeypatch):
    """.env.example uses NEWSAPI_KEY (no _API_) for backward compat."""
    monkeypatch.setenv("NEWSAPI_KEY", "newsapi-legacy")
    keys = _discover_keys_for_provider("newsapi")
    assert "newsapi-legacy" in keys


def test_finnhub_assembled_alias_picked_up(monkeypatch):
    """ASSEMBLED_FINNHUB_API_KEY is the Pydantic-Settings env-prefix form."""
    monkeypatch.setenv("ASSEMBLED_FINNHUB_API_KEY", "finnhub-pydantic")
    keys = _discover_keys_for_provider("finnhub")
    assert "finnhub-pydantic" in keys


def test_unknown_provider_returns_empty():
    assert _discover_keys_for_provider("does_not_exist") == []


def test_get_key_returns_first_when_single(monkeypatch, tmp_path):
    monkeypatch.setenv("POLYGON_API_KEY", "only-key")
    rot = ApiKeyRotator(state_path=tmp_path / "state.json")
    assert rot.get_key("polygon") == "only-key"


def test_get_key_round_robin_across_pool(monkeypatch, tmp_path):
    monkeypatch.setenv("POLYGON_API_KEY", "a")
    monkeypatch.setenv("POLYGON_API_KEY_2", "b")
    monkeypatch.setenv("POLYGON_API_KEY_3", "c")
    rot = ApiKeyRotator(state_path=tmp_path / "state.json")
    seen = [rot.get_key("polygon") for _ in range(6)]
    # Two full cycles
    assert seen == ["a", "b", "c", "a", "b", "c"]


def test_get_key_returns_none_when_no_keys(monkeypatch, tmp_path):
    rot = ApiKeyRotator(state_path=tmp_path / "state.json")
    assert rot.get_key("alphavantage") is None


def test_mark_rate_limited_skips_cooled_down_key(monkeypatch, tmp_path):
    monkeypatch.setenv("FINNHUB_API_KEY", "k1")
    monkeypatch.setenv("FINNHUB_API_KEY_2", "k2")
    rot = ApiKeyRotator(state_path=tmp_path / "state.json")

    first = rot.get_key("finnhub")
    rot.mark_rate_limited("finnhub", first, cooldown_seconds=3600)

    # Next call must skip the cooled-down key
    next_key = rot.get_key("finnhub")
    assert next_key != first
    assert next_key in {"k1", "k2"}


def test_all_cooled_down_returns_none(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("POLYGON_API_KEY", "alpha")
    monkeypatch.setenv("POLYGON_API_KEY_2", "beta")
    rot = ApiKeyRotator(state_path=tmp_path / "state.json")

    rot.mark_rate_limited("polygon", "alpha", cooldown_seconds=3600)
    rot.mark_rate_limited("polygon", "beta", cooldown_seconds=3600)

    with caplog.at_level("WARNING", logger="src.assembled_core.utils.api_key_rotator"):
        assert rot.get_key("polygon") is None
    assert any(
        "all" in rec.message and "cooled down" in rec.message
        for rec in caplog.records
        if rec.levelname == "WARNING"
    )


def test_cooldown_expires_and_key_returns_to_pool(monkeypatch, tmp_path):
    monkeypatch.setenv("POLYGON_API_KEY", "transient")
    rot = ApiKeyRotator(state_path=tmp_path / "state.json")

    # Cooldown of 0.05s — will have elapsed by the second call
    rot.mark_rate_limited("polygon", "transient", cooldown_seconds=0.05)
    assert rot.get_key("polygon") is None  # immediately cooled
    time.sleep(0.1)
    assert rot.get_key("polygon") == "transient"


def test_state_persistence_survives_rotator_restart(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("FINNHUB_API_KEY", "persist-1")
    monkeypatch.setenv("FINNHUB_API_KEY_2", "persist-2")

    rot1 = ApiKeyRotator(state_path=state_path)
    rot1.mark_rate_limited("finnhub", "persist-1", cooldown_seconds=3600)

    # State written
    assert state_path.exists()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert "finnhub" in payload

    # New rotator instance — should rehydrate cooldown
    rot2 = ApiKeyRotator(state_path=state_path)
    # First call should skip persist-1
    assert rot2.get_key("finnhub") == "persist-2"


def test_state_file_does_not_contain_raw_keys(monkeypatch, tmp_path):
    """Persisted state must not leak full secret keys — only last-4 suffix."""
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "SECRET_KEY_DO_NOT_LEAK")

    rot = ApiKeyRotator(state_path=state_path)
    rot.mark_rate_limited("alphavantage", "SECRET_KEY_DO_NOT_LEAK")

    raw = state_path.read_text(encoding="utf-8")
    assert "SECRET_KEY_DO_NOT_LEAK" not in raw
    assert "EAK" in raw  # last-3 of "_LEAK" lands in suffix (4 chars: LEAK)


def test_pool_size_and_available_count(monkeypatch, tmp_path):
    monkeypatch.setenv("POLYGON_API_KEY", "a")
    monkeypatch.setenv("POLYGON_API_KEY_2", "b")
    monkeypatch.setenv("POLYGON_API_KEY_3", "c")
    rot = ApiKeyRotator(state_path=tmp_path / "state.json")

    assert rot.pool_size("polygon") == 3
    assert rot.available_count("polygon") == 3

    rot.mark_rate_limited("polygon", "b", cooldown_seconds=3600)
    assert rot.pool_size("polygon") == 3  # total unchanged
    assert rot.available_count("polygon") == 2  # one cooled down


def test_mark_rate_limited_unknown_key_is_noop(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("POLYGON_API_KEY", "real")
    rot = ApiKeyRotator(state_path=tmp_path / "state.json")
    with caplog.at_level("WARNING", logger="src.assembled_core.utils.api_key_rotator"):
        rot.mark_rate_limited("polygon", "not-in-pool")
    assert any("unknown key" in rec.message for rec in caplog.records)
    # Real key still available
    assert rot.get_key("polygon") == "real"


# ---------------------------------------------------------------------------
# FRED provider — added 2026-05-22 to close the rotation-coverage gap.
# ---------------------------------------------------------------------------


def test_discover_keys_fred_single(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "fred-primary")
    assert _discover_keys_for_provider("fred") == ["fred-primary"]


def test_discover_keys_fred_numbered(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "fred-primary")
    monkeypatch.setenv("FRED_API_KEY_2", "fred-secondary")
    monkeypatch.setenv("FRED_API_KEY_3", "fred-tertiary")
    assert _discover_keys_for_provider("fred") == [
        "fred-primary",
        "fred-secondary",
        "fred-tertiary",
    ]


def test_fred_round_robin(monkeypatch, tmp_path):
    monkeypatch.setenv("FRED_API_KEY", "a")
    monkeypatch.setenv("FRED_API_KEY_2", "b")
    rot = ApiKeyRotator(state_path=tmp_path / "state.json")
    seen = [rot.get_key("fred") for _ in range(4)]
    assert seen == ["a", "b", "a", "b"]


def test_fred_mark_rate_limited_skips(monkeypatch, tmp_path):
    monkeypatch.setenv("FRED_API_KEY", "fred-a")
    monkeypatch.setenv("FRED_API_KEY_2", "fred-b")
    rot = ApiKeyRotator(state_path=tmp_path / "state.json")
    first = rot.get_key("fred")
    rot.mark_rate_limited("fred", first, cooldown_seconds=3600)
    second = rot.get_key("fred")
    assert second != first
    assert second in {"fred-a", "fred-b"}


def test_finnhub_legacy_alias_finnhub_key(monkeypatch):
    """FINNHUB_KEY (no _API_) appears in some live .env files."""
    monkeypatch.setenv("FINNHUB_KEY", "finnhub-no-api")
    keys = _discover_keys_for_provider("finnhub")
    assert "finnhub-no-api" in keys


# ---------------------------------------------------------------------------
# is_rate_limit_signal helper — used by client wrappers to decide cooldown.
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


class _FakeHTTPError(Exception):
    def __init__(self, response):
        super().__init__("HTTP error")
        self.response = response


def test_is_rate_limit_signal_http_429():
    assert is_rate_limit_signal(_FakeResponse(429)) is True
    assert is_rate_limit_signal(_FakeResponse(200)) is False
    assert is_rate_limit_signal(_FakeResponse(500)) is False


def test_is_rate_limit_signal_wrapped_exception():
    inner = _FakeResponse(429)
    exc = _FakeHTTPError(inner)
    assert is_rate_limit_signal(exc) is True


def test_is_rate_limit_signal_alphavantage_note_text():
    """AV free tier returns 200 + 'Note' body — must be detected by text."""
    note = (
        "Thank you for using Alpha Vantage! Our standard API call frequency is "
        "5 calls per minute and 25 calls per day."
    )
    assert is_rate_limit_signal(note) is True


def test_is_rate_limit_signal_too_many_requests_text():
    assert is_rate_limit_signal(Exception("429 Too Many Requests")) is True
    assert is_rate_limit_signal("HTTP 429: rate limit exceeded") is True


def test_is_rate_limit_signal_quota_exceeded():
    assert is_rate_limit_signal(Exception("quota exceeded for today")) is True


def test_is_rate_limit_signal_none_is_false():
    assert is_rate_limit_signal(None) is False


def test_is_rate_limit_signal_unrelated_error_is_false():
    assert is_rate_limit_signal(Exception("connection refused")) is False
    assert is_rate_limit_signal(Exception("invalid JSON response")) is False
    assert is_rate_limit_signal(_FakeResponse(404)) is False


# ---------------------------------------------------------------------------
# False-positive regression — post-review F-AKR2-1/2 hardening: vendor
# greetings and non-throttle "Information" replies must NOT trigger 429
# detection. Locks the contract from F-AKR2-1.
# ---------------------------------------------------------------------------


def test_is_rate_limit_signal_av_premium_required_is_false():
    """AV 'premium endpoint required' response shares the welcome preamble
    but is NOT a rate-limit — must return False to avoid needless cooldown."""
    msg = (
        "Thank you for using Alpha Vantage! This is a premium endpoint and "
        "is available exclusively through any of the premium membership plans."
    )
    assert is_rate_limit_signal(msg) is False


def test_is_rate_limit_signal_av_invalid_api_key_is_false():
    """AV invalid-key response also uses the welcome preamble."""
    msg = (
        "Thank you for using Alpha Vantage! Please consider supporting us "
        "with a premium membership. Invalid API call."
    )
    assert is_rate_limit_signal(msg) is False


def test_is_rate_limit_signal_word_rate_alone_is_false():
    """Single token 'rate' (without 'rate limit') must not trigger."""
    assert is_rate_limit_signal(Exception("exchange rate not available")) is False


def test_is_rate_limit_signal_word_limit_alone_is_false():
    assert is_rate_limit_signal(Exception("max recursion limit reached")) is False


def test_is_rate_limit_signal_urllib_httperror_code_429():
    """urllib.error.HTTPError uses .code (not .status_code)."""
    import urllib.error

    exc = urllib.error.HTTPError(
        "http://x",
        429,
        "Too Many Requests",
        {},
        None,  # type: ignore[arg-type]
    )
    assert is_rate_limit_signal(exc) is True


def test_is_rate_limit_signal_urllib_httperror_code_500_is_false():
    import urllib.error

    exc = urllib.error.HTTPError(
        "http://x",
        500,
        "Internal Server Error",
        {},
        None,  # type: ignore[arg-type]
    )
    assert is_rate_limit_signal(exc) is False


# ---------------------------------------------------------------------------
# Integration: a client-shaped fetch loop that uses mark_rate_limited on 429
# actually rotates to the other key. This is the User-Intent integration
# test: "wenn einer erschöpft, zum nächsten wechseln" must work end-to-end.
# ---------------------------------------------------------------------------


def test_429_loop_rotates_to_alternate_key(monkeypatch, tmp_path):
    monkeypatch.setenv("POLYGON_API_KEY", "exhausted")
    monkeypatch.setenv("POLYGON_API_KEY_2", "fresh")
    rot = ApiKeyRotator(state_path=tmp_path / "state.json")

    # First call: pool returns "exhausted"; simulated 429.
    k1 = rot.get_key("polygon")
    fake_429 = _FakeResponse(429)
    assert is_rate_limit_signal(fake_429)
    rot.mark_rate_limited("polygon", k1, cooldown_seconds=3600)

    # Second call: must rotate to "fresh".
    k2 = rot.get_key("polygon")
    assert k2 != k1
    assert k2 == "fresh"

    # Third call: "fresh" round-robins back; "exhausted" still cooled.
    # Cursor advanced past "fresh", wraps to index 0 which is still cooled.
    k3 = rot.get_key("polygon")
    assert k3 == "fresh"  # only available key


def test_429_loop_all_exhausted_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv("FINNHUB_API_KEY", "x")
    monkeypatch.setenv("FINNHUB_API_KEY_2", "y")
    rot = ApiKeyRotator(state_path=tmp_path / "state.json")
    rot.mark_rate_limited("finnhub", "x", cooldown_seconds=3600)
    rot.mark_rate_limited("finnhub", "y", cooldown_seconds=3600)
    assert rot.get_key("finnhub") is None
