"""Regression tests for Diagnostik Batch 3a (ledger read-path security).

A6  /ledger and /performance/{freq}/live-curve accepted an arbitrary `ledger_path`
    query param with NO safe-roots check, allowing an unauthenticated caller to point
    the loader at any file on disk. Now both reject paths outside OUTPUT_DIR / temp.
A7  /live-curve loaded the ledger without the corruption sentinel and
A27 raised HTTP 500 (leaking the exception text) on a loader failure. Now it loads
    with start_capital=-1.0, detects the silent fallback, and fails closed to an empty
    curve (its documented "never 404/500" contract).

Each test fails on the pre-fix code and passes after the fix.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from src.assembled_core.api.app import create_app
from src.assembled_core.api.routers.health import _is_safe_output_dir
from src.assembled_core.config import OUTPUT_DIR

pytestmark = pytest.mark.fast


@pytest.fixture
def client():
    return TestClient(create_app())


_RICH_STATE = {
    "cash": 99999.0,
    "equity": 99999.0,
    "updated_utc": "2026-01-01T00:00:00+00:00",
    "positions": {},
    "equity_curve": [],
}


# --- guard unit ------------------------------------------------------------
def test_is_safe_output_dir_unit():
    assert _is_safe_output_dir((OUTPUT_DIR / "runs" / "x.json").resolve()) is True
    # repo root / pyproject.toml is outside OUTPUT_DIR and temp -> unsafe
    assert (
        _is_safe_output_dir((OUTPUT_DIR.parent / "pyproject.toml").resolve()) is False
    )


# --- A6: path-traversal blocked BEFORE the loader runs ---------------------
def test_ledger_blocks_out_of_bounds_even_when_loadable(client, monkeypatch):
    """Out-of-bounds ledger_path -> no_ledger, even though the loader *would* return data."""
    monkeypatch.setattr(
        "src.assembled_core.ops.paper_ledger.load_ledger_state",
        lambda *a, **k: dict(_RICH_STATE),
    )
    oob = str(OUTPUT_DIR.parent / "pyproject.toml")  # exists, but outside safe roots
    assert Path(oob).exists()
    r = client.get("/api/v1/ledger", params={"ledger_path": oob})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "no_ledger"
    assert body["cash"] == 0.0  # the rich loader result was NOT used


def test_live_curve_blocks_out_of_bounds_even_when_loadable(client, monkeypatch):
    monkeypatch.setattr(
        "src.assembled_core.ops.paper_ledger.load_ledger_state",
        lambda *a, **k: {
            "updated_utc": "x",
            "cash": 1.0,
            "equity_curve": [{"utc": "2026-01-01", "equity": 5.0}],
        },
    )
    oob = str(OUTPUT_DIR.parent / "pyproject.toml")
    r = client.get("/api/v1/performance/1d/live-curve", params={"ledger_path": oob})
    assert r.status_code == 200
    assert r.json()["count"] == 0  # blocked before the loadable curve was read


# --- A7 / A27: in-bounds loader failure fails closed to empty, not HTTP 500 -
def test_live_curve_load_failure_returns_empty_not_500(client, monkeypatch, tmp_path):
    """An in-bounds but unreadable ledger must yield an empty curve, not a 500 leak."""

    def _boom(*a, **k):
        raise RuntimeError("corrupt ledger internals")

    monkeypatch.setattr("src.assembled_core.ops.paper_ledger.load_ledger_state", _boom)
    # tmp_path is under the system temp dir == a safe root; create the file so .exists() passes
    p = tmp_path / "ledger_state.json"
    p.write_text("{ not valid json", encoding="utf-8")
    r = client.get("/api/v1/performance/1d/live-curve", params={"ledger_path": str(p)})
    assert r.status_code == 200  # pre-fix raised 500
    assert r.json()["count"] == 0


def test_live_curve_corruption_sentinel_returns_empty(client, monkeypatch, tmp_path):
    """A silent loader fallback (sentinel: updated_utc=None, cash<0) -> empty, not a fake curve."""
    monkeypatch.setattr(
        "src.assembled_core.ops.paper_ledger.load_ledger_state",
        lambda *a, **k: {
            "updated_utc": None,
            "cash": -1.0,
            "equity_curve": [{"utc": "2026-01-01", "equity": 5.0}],
        },
    )
    p = tmp_path / "ledger_state.json"
    p.write_text("{}", encoding="utf-8")
    r = client.get("/api/v1/performance/1d/live-curve", params={"ledger_path": str(p)})
    assert r.status_code == 200
    assert r.json()["count"] == 0  # corrupt-fallback curve suppressed
