# tests/test_api_monitoring_path_safety.py
"""Path-traversal safety tests for the monitoring router.

These endpoints are unauthenticated GETs that accept a caller-supplied
``db_path`` / ``output_dir`` and open/glob/read it. A guard
(``_is_safe_monitoring_path``) confines every read to a small set of legitimate
roots (``output``, ``src/output``, ``data``, tempdir). An out-of-bounds path
must be rejected to the endpoint's benign cold-start response — never leaking
file contents or existence (Diagnostik: monitoring sibling of A6).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

# Add repo root to path (mirrors tests/test_api_monitoring.py).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.api.app import create_app
from src.assembled_core.api.routers import monitoring as mon
from src.assembled_core.config import get_base_dir

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Unit: _is_safe_monitoring_path  (locks the safe-roots reconciliation)
# ---------------------------------------------------------------------------


class TestIsSafeMonitoringPath:
    """Direct unit tests of the safe-roots predicate."""

    def test_src_output_is_in_bounds(self):
        p = (get_base_dir() / "src" / "output").resolve()
        assert mon._is_safe_monitoring_path(p) is True

    def test_src_output_child_is_in_bounds(self):
        p = (get_base_dir() / "src" / "output" / "regime_state_x.json").resolve()
        assert mon._is_safe_monitoring_path(p) is True

    def test_data_dir_is_in_bounds(self):
        p = (get_base_dir() / "data").resolve()
        assert mon._is_safe_monitoring_path(p) is True

    def test_data_ledger_default_is_in_bounds(self):
        # The documented db_path default must NOT be wrongly rejected.
        p = Path("data/paper_ledger.db").resolve()
        assert mon._is_safe_monitoring_path(p) is True

    def test_output_dir_default_is_in_bounds(self):
        # The config canonical OUTPUT_DIR root.
        from src.assembled_core.config import OUTPUT_DIR

        assert mon._is_safe_monitoring_path(OUTPUT_DIR.resolve()) is True

    def test_repo_root_pyproject_is_out_of_bounds(self):
        # A real, sensitive file at the repo root — must be rejected.
        p = (get_base_dir() / "pyproject.toml").resolve()
        assert mon._is_safe_monitoring_path(p) is False

    def test_repo_root_itself_is_out_of_bounds(self):
        assert mon._is_safe_monitoring_path(get_base_dir().resolve()) is False

    def test_arbitrary_system_path_is_out_of_bounds(self):
        # An absolute path outside every safe root.
        p = Path(get_base_dir().anchor) / "etc" / "passwd"
        assert mon._is_safe_monitoring_path(p.resolve()) is False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


# An out-of-bounds db_path: a sensitive file outside every safe root.
OUT_OF_BOUNDS_DB = str(get_base_dir() / "pyproject.toml")


@pytest.fixture
def oob_dir(tmp_path, monkeypatch):
    """A genuinely out-of-bounds directory that EXISTS and is writable.

    ``tmp_path`` lives under the system temp dir, which is itself a safe root.
    We narrow ``_MON_SAFE_ROOTS`` to drop ONLY the temp root (keeping the
    legitimate output / src/output / data roots intact), so ``tmp_path`` becomes
    out-of-bounds while the endpoints' legitimate defaults stay in-bounds. This
    lets us PLANT a file the endpoint would read if the guard were absent —
    making the suppression test discriminating (it goes red if the guard is
    removed). pytest auto-cleans ``tmp_path``.
    """
    base = get_base_dir().resolve()
    roots = (
        mon.OUTPUT_DIR.resolve(),
        (base / "src" / "output").resolve(),
        (base / "data").resolve(),
    )
    monkeypatch.setattr(mon, "_MON_SAFE_ROOTS", roots)
    # tmp_path is now out of bounds, but the legitimate defaults are NOT.
    assert mon._is_safe_monitoring_path(tmp_path.resolve()) is False
    assert mon._is_safe_monitoring_path(Path("data/paper_ledger.db").resolve()) is True
    return tmp_path


# ---------------------------------------------------------------------------
# /monitoring/portfolio (db_path) — guard short-circuits before LedgerStore
# ---------------------------------------------------------------------------


class TestPortfolioPathSafety:
    def test_out_of_bounds_db_path_short_circuits_before_ledgerstore(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        """If the guard failed, LedgerStore would be constructed on the
        out-of-bounds path; assert we instead get no_ledger — proving the guard
        runs BEFORE LedgerStore. Pre-fix this reaches ``LedgerStore("pyproject.toml")``
        and yields a 500 ("file is not a database")."""

        class _RichLedger:
            def __init__(self, *a, **k):
                raise AssertionError(
                    "LedgerStore must not be constructed for an out-of-bounds db_path"
                )

        monkeypatch.setattr(
            "src.assembled_core.data.ledger_store.LedgerStore", _RichLedger
        )
        resp = client.get(
            "/api/v1/monitoring/portfolio", params={"db_path": OUT_OF_BOUNDS_DB}
        )
        assert resp.status_code == 200
        assert resp.json() == {
            "status": "no_ledger",
            "cash": 0.0,
            "positions": [],
            "equity": 0.0,
            "n_positions": 0,
        }
        assert "pyproject" not in resp.text.lower()


# ---------------------------------------------------------------------------
# /monitoring/regime (output_dir) — guard suppresses a planted read
# ---------------------------------------------------------------------------


class TestRegimePathSafety:
    def test_guard_suppresses_out_of_bounds_regime_file(self, client, oob_dir):
        """Plant a regime_state file in an out-of-bounds dir; the guard must NOT
        read it. Pre-fix, the endpoint globs the dir and returns regime='crisis'."""
        (oob_dir / "regime_state_2026.json").write_text(
            '{"regime": "crisis", "regime_score": 9.9}', encoding="utf-8"
        )
        resp = client.get(
            "/api/v1/monitoring/regime", params={"output_dir": str(oob_dir)}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["regime"] == "unknown"  # planted "crisis" was NOT read
        assert body["status"] == "stale"
        assert "crisis" not in resp.text


# ---------------------------------------------------------------------------
# /monitoring/alerts (output_dir) — guard suppresses a planted read
# ---------------------------------------------------------------------------


class TestAlertsPathSafety:
    def test_guard_suppresses_out_of_bounds_zombie_alert(self, client, oob_dir):
        """Plant a zombie_report file; pre-fix the endpoint surfaces a zombie
        alert (n_alerts>=1, symbols echoed). The guard must suppress it."""
        (oob_dir / "zombie_report_2026.json").write_text(
            '{"zombie_symbols": ["AAAA", "BBBB"]}', encoding="utf-8"
        )
        resp = client.get(
            "/api/v1/monitoring/alerts", params={"output_dir": str(oob_dir)}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_alerts"] == 0  # planted zombie alert NOT surfaced
        assert "AAAA" not in resp.text


# ---------------------------------------------------------------------------
# /monitoring/signals + /monitoring/data-quality (output_dir)
# ---------------------------------------------------------------------------
# Both apply the IDENTICAL `_is_safe_monitoring_path(output_dir)` guard, in the
# same pre-glob position, as /monitoring/regime and /monitoring/alerts above.
# That guard logic is locked non-tautologically by TestIsSafeMonitoringPath (8
# tests, incl. the legitimate src/output + data defaults), and its behavioural
# read-suppression is proven by the regime + alerts planted-file tests above.
# A dedicated planted-parquet test for each would be redundant; the guard
# placement in both handlers was confirmed by code review (security sweep).
