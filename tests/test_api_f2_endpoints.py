"""Tests for Paket 5 — GO_LIVE F2 API endpoints.

Covers:
  /health                              — 200 happy-path, 503 on critical failure
  /api/v1/ledger                       — no-ledger empty, with data, ?date= filter
  /api/v1/performance/{freq}/live-curve — no pilot → empty valid, with data → correct schema
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from src.assembled_core.api.app import create_app

pytestmark = pytest.mark.fast


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_200_valid_structure(self, client: TestClient, tmp_path: Path):
        """200 with {status, timestamp_utc, checks} when output_dir is writable."""
        response = client.get(f"/health?output_dir={tmp_path}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp_utc" in data
        assert isinstance(data["checks"], dict)

    def test_health_checks_contains_output_dir_key(
        self, client: TestClient, tmp_path: Path
    ):
        """checks dict must contain output_dir key."""
        response = client.get(f"/health?output_dir={tmp_path}")
        assert response.status_code == 200
        checks = response.json()["checks"]
        assert "output_dir" in checks
        assert checks["output_dir"]["ok"] is True

    def test_health_503_when_dir_unwritable(self, client: TestClient, tmp_path: Path):
        """503 when .health_check_probe exists as a directory (write_text fails)."""
        probe = tmp_path / ".health_check_probe"
        probe.mkdir()  # make it a directory so write_text raises
        response = client.get(f"/health?output_dir={tmp_path}")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["checks"]["output_dir"]["ok"] is False

    def test_health_returns_all_expected_checks(
        self, client: TestClient, tmp_path: Path
    ):
        """Response includes data_freshness, broker, kill_switch keys."""
        response = client.get(f"/health?output_dir={tmp_path}")
        checks = response.json()["checks"]
        assert "data_freshness" in checks
        assert "broker" in checks
        assert "kill_switch" in checks


# ---------------------------------------------------------------------------
# /api/v1/ledger
# ---------------------------------------------------------------------------


class TestLedgerEndpoint:
    def test_ledger_no_file_returns_empty_valid(
        self, client: TestClient, tmp_path: Path
    ):
        """When ledger file absent: status=no_ledger, cash=0, positions=[] — never 500."""
        missing = tmp_path / "nonexistent.json"
        response = client.get(f"/api/v1/ledger?ledger_path={missing}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_ledger"
        assert data["cash"] == pytest.approx(0.0)
        assert data["equity"] == pytest.approx(0.0)
        assert data["positions"] == []
        assert data["n_positions"] == 0

    def test_ledger_with_data_returns_correct_values(
        self, client: TestClient, tmp_path: Path
    ):
        """Ledger with one AAPL position returns correct cash, equity, position."""
        ledger = tmp_path / "ledger_state.json"
        state = {
            "schema_version": "paper.ledger_state.v1",
            "updated_utc": "2026-05-28T21:30:00+00:00",
            "cash": 8000.0,
            "positions": {"AAPL": {"qty": 10.0, "avg_price": 150.0, "hwm": 155.0}},
            "equity_curve": [{"utc": "2026-05-28T21:30:00+00:00", "equity": 9500.0}],
        }
        ledger.write_text(json.dumps(state), encoding="utf-8")
        response = client.get(f"/api/v1/ledger?ledger_path={ledger}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["cash"] == pytest.approx(8000.0)
        assert data["equity"] == pytest.approx(9500.0)
        assert data["n_positions"] == 1
        pos = data["positions"][0]
        assert pos["symbol"] == "AAPL"
        assert pos["qty"] == pytest.approx(10.0)
        assert pos["avg_price"] == pytest.approx(150.0)
        assert pos["cost_basis"] == pytest.approx(1500.0)
        # unrealized_pnl_approx = 9500 - 8000 - 1500 = 0
        assert data["unrealized_pnl_approx"] == pytest.approx(0.0)

    def test_ledger_with_date_param_filters_equity(
        self, client: TestClient, tmp_path: Path
    ):
        """?date=YYYY-MM-DD returns equity from that day's equity_curve entry."""
        ledger = tmp_path / "ledger_state.json"
        state = {
            "schema_version": "paper.ledger_state.v1",
            "updated_utc": "2026-05-28T21:30:00+00:00",
            "cash": 8000.0,
            "positions": {},
            "equity_curve": [
                {"utc": "2026-05-20T21:30:00+00:00", "equity": 10200.0},
                {"utc": "2026-05-21T21:30:00+00:00", "equity": 10350.0},
                {"utc": "2026-05-28T21:30:00+00:00", "equity": 11000.0},
            ],
        }
        ledger.write_text(json.dumps(state), encoding="utf-8")
        response = client.get(f"/api/v1/ledger?ledger_path={ledger}&date=2026-05-20")
        assert response.status_code == 200
        data = response.json()
        assert data["equity"] == pytest.approx(10200.0)
        assert data["date_requested"] == "2026-05-20"

    def test_ledger_date_no_match_returns_zero_equity_gracefully(
        self, client: TestClient, tmp_path: Path
    ):
        """?date= not in equity_curve → equity=0, status=ok, no 500."""
        ledger = tmp_path / "ledger_state.json"
        state = {
            "schema_version": "paper.ledger_state.v1",
            "cash": 8000.0,
            "positions": {},
            "equity_curve": [{"utc": "2026-05-28T21:30:00+00:00", "equity": 9000.0}],
        }
        ledger.write_text(json.dumps(state), encoding="utf-8")
        response = client.get(f"/api/v1/ledger?ledger_path={ledger}&date=2025-01-01")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["equity"] == pytest.approx(0.0)
        assert data["cash"] == pytest.approx(0.0)
        assert data["positions"] == []
        assert data["n_positions"] == 0


# ---------------------------------------------------------------------------
# /api/v1/performance/{freq}/live-curve
# ---------------------------------------------------------------------------


class TestLiveCurveEndpoint:
    def test_live_curve_no_pilot_data_returns_empty_valid(
        self, client: TestClient, tmp_path: Path
    ):
        """No ledger file → empty but valid EquityCurveResponse — no 404 or 500."""
        missing = tmp_path / "no_ledger.json"
        response = client.get(
            f"/api/v1/performance/1d/live-curve?ledger_path={missing}"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["frequency"] == "1d"
        assert data["count"] == 0
        assert data["points"] == []
        assert data["start_equity"] == pytest.approx(0.0)
        assert data["end_equity"] == pytest.approx(0.0)

    def test_live_curve_same_schema_as_backtest_curve(
        self, client: TestClient, tmp_path: Path
    ):
        """live-curve has identical top-level fields to backtest-curve schema."""
        ledger = tmp_path / "ledger_state.json"
        state = {
            "schema_version": "paper.ledger_state.v1",
            "cash": 8000.0,
            "positions": {},
            "equity_curve": [
                {"utc": "2026-05-27T21:30:00+00:00", "equity": 10100.0},
                {"utc": "2026-05-28T21:30:00+00:00", "equity": 10250.0},
            ],
        }
        ledger.write_text(json.dumps(state), encoding="utf-8")
        response = client.get(f"/api/v1/performance/1d/live-curve?ledger_path={ledger}")
        assert response.status_code == 200
        data = response.json()
        # Schema identical to EquityCurveResponse
        assert data["frequency"] == "1d"
        assert data["count"] == 2
        assert len(data["points"]) == 2
        assert data["start_equity"] == pytest.approx(10100.0)
        assert data["end_equity"] == pytest.approx(10250.0)
        # Each point has timestamp + equity
        pt = data["points"][0]
        assert "timestamp" in pt
        assert "equity" in pt
        assert pt["equity"] == pytest.approx(10100.0)

    def test_live_curve_invalid_freq_rejected(self, client: TestClient, tmp_path: Path):
        """Unsupported frequency → 422 from FastAPI path param validation."""
        missing = tmp_path / "no_ledger.json"
        response = client.get(
            f"/api/v1/performance/9d/live-curve?ledger_path={missing}"
        )
        assert response.status_code == 422
