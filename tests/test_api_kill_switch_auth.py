"""HTTP-layer tests for the kill-switch deactivate endpoint (Paket 4b / GO_LIVE C2).

Verifies that the X-Operator-Token requirement is enforced at the HTTP boundary:
missing / wrong token → 403, correct token → 200.
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

pytestmark = pytest.mark.fast

_API_URL = "/api/v1/kill-switch/deactivate"
_ACTIVATE_URL = "/api/v1/kill-switch/activate"


@pytest.fixture()
def client(monkeypatch, tmp_path):
    """TestClient with isolated kill-switch state."""
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_AUDIT", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_SENTINEL", str(tmp_path / ".sentinel"))
    monkeypatch.delenv("ASSEMBLED_KILL_SWITCH", raising=False)
    monkeypatch.delenv(
        "ASSEMBLED_API_KEY", raising=False
    )  # fail-open: no API key needed in tests
    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


def _activate(client: TestClient) -> None:
    r = client.post(_ACTIVATE_URL, params={"reason": "test", "throttle_pct": 0.0})
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Case 1: missing X-Operator-Token → 403
# ---------------------------------------------------------------------------


def test_deactivate_endpoint_missing_token_returns_403(client, monkeypatch):
    monkeypatch.setenv("OPERATOR_KILL_TOKEN", "secret-token")
    _activate(client)
    r = client.post(_API_URL, params={"reason": "test", "actor": "test"})
    assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.text}"


# ---------------------------------------------------------------------------
# Case 2: wrong X-Operator-Token → 403
# ---------------------------------------------------------------------------


def test_deactivate_endpoint_wrong_token_returns_403(client, monkeypatch):
    monkeypatch.setenv("OPERATOR_KILL_TOKEN", "correct-token")
    _activate(client)
    r = client.post(
        _API_URL,
        params={"reason": "test", "actor": "test"},
        headers={"X-Operator-Token": "wrong-token"},
    )
    assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.text}"


# ---------------------------------------------------------------------------
# Case 3: correct X-Operator-Token → 200
# ---------------------------------------------------------------------------


def test_deactivate_endpoint_correct_token_returns_200(client, monkeypatch):
    _TOKEN = "correct-token"
    monkeypatch.setenv("OPERATOR_KILL_TOKEN", _TOKEN)
    _activate(client)
    r = client.post(
        _API_URL,
        params={"reason": "test", "actor": "test"},
        headers={"X-Operator-Token": _TOKEN},
    )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    assert r.json()["action"] == "deactivated"


# ---------------------------------------------------------------------------
# Case 4: OPERATOR_KILL_TOKEN not set → 403 (fail-closed at API layer)
# ---------------------------------------------------------------------------


def test_deactivate_endpoint_env_absent_returns_403(client, monkeypatch):
    monkeypatch.delenv("OPERATOR_KILL_TOKEN", raising=False)
    _activate(client)
    r = client.post(
        _API_URL,
        params={"reason": "test", "actor": "test"},
        headers={"X-Operator-Token": "any-token"},
    )
    assert r.status_code == 403, f"Expected 403, got {r.status_code}: {r.text}"
