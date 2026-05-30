"""SEC-1: API command-endpoint auth must fail closed in production.

Historically ``require_api_key`` fell open (warn-and-allow) whenever
``ASSEMBLED_API_KEY`` was unset — including in production. These tests lock in
the fail-closed contract: a production profile or an explicit opt-in turns a
missing key into HTTP 503 instead of silently serving open command endpoints,
while dev/test keep the fail-open convenience. ``/ready`` must surface the
resulting auth posture so a misconfigured deploy is visible to ops.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.api.app import create_app  # noqa: E402
from src.assembled_core.api.auth import auth_required_when_unset  # noqa: E402

pytestmark = pytest.mark.fast

_ACTIVATE_URL = "/api/v1/kill-switch/activate"


@pytest.fixture()
def env(monkeypatch, tmp_path):
    """Isolate kill-switch state and start from a clean auth env."""
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_AUDIT", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_SENTINEL", str(tmp_path / ".sentinel"))
    monkeypatch.delenv("ASSEMBLED_KILL_SWITCH", raising=False)
    monkeypatch.delenv("ASSEMBLED_API_KEY", raising=False)
    monkeypatch.delenv("ASSEMBLED_API_REQUIRE_AUTH", raising=False)
    monkeypatch.delenv("ASSEMBLED_RUNTIME_PROFILE", raising=False)
    return monkeypatch


def _client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# auth_required_when_unset() unit behaviour
# ---------------------------------------------------------------------------


def test_not_required_in_clean_env(env):
    assert auth_required_when_unset() is False


@pytest.mark.parametrize("val", ["1", "true", "YES", "on", "True"])
def test_required_when_explicit_optin(env, val):
    env.setenv("ASSEMBLED_API_REQUIRE_AUTH", val)
    assert auth_required_when_unset() is True


@pytest.mark.parametrize("profile", ["production", "prod", "live", "PRODUCTION"])
def test_required_in_production_profile(env, profile):
    env.setenv("ASSEMBLED_RUNTIME_PROFILE", profile)
    assert auth_required_when_unset() is True


@pytest.mark.parametrize("profile", ["development", "dev", "ci", "paper", ""])
def test_not_required_in_nonprod_profile(env, profile):
    env.setenv("ASSEMBLED_RUNTIME_PROFILE", profile)
    assert auth_required_when_unset() is False


# ---------------------------------------------------------------------------
# require_api_key via a command endpoint (kill-switch activate)
# ---------------------------------------------------------------------------


def test_command_open_when_key_unset_and_not_required(env):
    """Dev/test default: missing key falls open → command succeeds."""
    r = _client().post(_ACTIVATE_URL, params={"reason": "test"})
    assert r.status_code == 200, r.text


def test_command_fails_closed_when_required_but_key_unset(env):
    env.setenv("ASSEMBLED_API_REQUIRE_AUTH", "1")
    r = _client().post(_ACTIVATE_URL, params={"reason": "test"})
    assert r.status_code == 503, r.text


def test_command_fails_closed_in_production_profile(env):
    env.setenv("ASSEMBLED_RUNTIME_PROFILE", "production")
    r = _client().post(_ACTIVATE_URL, params={"reason": "test"})
    assert r.status_code == 503, r.text


def test_command_allows_with_correct_key(env):
    env.setenv("ASSEMBLED_API_KEY", "s3cret")
    env.setenv("ASSEMBLED_RUNTIME_PROFILE", "production")  # required, but key set
    r = _client().post(
        _ACTIVATE_URL, params={"reason": "test"}, headers={"X-API-Key": "s3cret"}
    )
    assert r.status_code == 200, r.text


def test_command_rejects_wrong_key(env):
    env.setenv("ASSEMBLED_API_KEY", "s3cret")
    r = _client().post(
        _ACTIVATE_URL, params={"reason": "test"}, headers={"X-API-Key": "nope"}
    )
    assert r.status_code == 401, r.text


def test_command_rejects_missing_header_when_key_set(env):
    env.setenv("ASSEMBLED_API_KEY", "s3cret")
    r = _client().post(_ACTIVATE_URL, params={"reason": "test"})
    assert r.status_code == 401, r.text


# ---------------------------------------------------------------------------
# /ready surfaces the auth posture
# ---------------------------------------------------------------------------


def test_ready_auth_posture_ok_when_not_required(env):
    body = _client().get("/ready").json()
    assert body["checks"]["auth_posture"] is True
    assert body["details"]["auth_required"] is False
    assert body["details"]["auth_configured"] is False


def test_ready_not_ready_when_auth_required_but_unconfigured(env):
    env.setenv("ASSEMBLED_RUNTIME_PROFILE", "production")
    r = _client().get("/ready")
    body = r.json()
    assert body["checks"]["auth_posture"] is False
    assert body["details"]["auth_required"] is True
    assert body["details"]["auth_configured"] is False
    assert r.status_code == 503, r.text


def test_ready_auth_posture_ok_when_required_and_configured(env):
    env.setenv("ASSEMBLED_RUNTIME_PROFILE", "production")
    env.setenv("ASSEMBLED_API_KEY", "s3cret")
    body = _client().get("/ready").json()
    assert body["checks"]["auth_posture"] is True
    assert body["details"]["auth_required"] is True
    assert body["details"]["auth_configured"] is True
