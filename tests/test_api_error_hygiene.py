# tests/test_api_error_hygiene.py
"""API error-hygiene tests (A27 info-disclosure + A28 OMS concurrency lock).

A27: unauthenticated GET handlers must NOT reflect internal exception text
(or absolute paths) back to anonymous callers via a 500 ``detail``. The real
exception is logged server-side; the caller sees a generic message. These
tests assert the sensitive marker string is ABSENT from the response body.

A28: ``oms.py`` reads the shared paper-trading engine under ``_engine_lock``
(the RLock paper_trading uses for all engine access). A light smoke test
confirms ``/oms/blotter`` still returns 200 — the RLock is re-entrant, so a
single acquire around the read cannot deadlock.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

# Add repo root to path (mirrors the other API test modules).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.api.app import create_app
from src.assembled_core.api.routers import paper_trading as pt

pytestmark = pytest.mark.fast


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# A27 — OMS blotter / executions: engine failure must not leak exception text
# ---------------------------------------------------------------------------

# A distinctive marker that would only appear in the response if the handler
# reflected the raw exception message back to the caller.
_OMS_SECRET = "SENSITIVE-INTERNAL-abc"  # pragma: allowlist secret


class TestOmsErrorHygiene:
    def test_blotter_500_does_not_leak_exception_text(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        """Force the engine read to raise; assert the 500 body is generic."""

        def _boom(*a, **k):
            raise RuntimeError(_OMS_SECRET)

        monkeypatch.setattr(pt._engine, "list_orders", _boom)

        resp = client.get("/api/v1/oms/blotter")
        assert resp.status_code == 500
        # The discriminating assertion: the secret must be ABSENT from the body.
        assert _OMS_SECRET not in resp.text
        assert resp.json()["detail"] == "internal error retrieving blotter"

    def test_executions_500_does_not_leak_exception_text(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ):
        """Same for /oms/executions."""

        def _boom(*a, **k):
            raise RuntimeError(_OMS_SECRET)

        monkeypatch.setattr(pt._engine, "list_orders", _boom)

        resp = client.get("/api/v1/oms/executions")
        assert resp.status_code == 500
        assert _OMS_SECRET not in resp.text
        assert resp.json()["detail"] == "internal error retrieving executions"


# ---------------------------------------------------------------------------
# A27 — monitoring/portfolio: ledger failure must not leak exception text
# ---------------------------------------------------------------------------

_MON_SECRET = "SENSITIVE-INTERNAL-xyz"  # pragma: allowlist secret


class TestMonitoringErrorHygiene:
    def test_portfolio_500_does_not_leak_exception_text(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        """LedgerStore raising on an IN-BOUNDS db_path must yield a generic 500.

        The default db_path ``data/paper_ledger.db`` is an in-bounds safe root,
        so the path-traversal guard does NOT short-circuit; the handler reaches
        LedgerStore, which we force to raise. The marker must not be reflected.
        """
        # Plant an in-bounds db file so the .exists() check passes and the
        # handler proceeds to construct LedgerStore (where we inject the failure).
        db_dir = ROOT / "data"
        db_dir.mkdir(parents=True, exist_ok=True)
        db_file = db_dir / "test_error_hygiene_ledger.db"
        db_file.write_bytes(b"not-a-real-db")
        try:

            class _BoomLedger:
                def __init__(self, *a, **k):
                    raise RuntimeError(_MON_SECRET)

            monkeypatch.setattr(
                "src.assembled_core.data.ledger_store.LedgerStore", _BoomLedger
            )

            resp = client.get(
                "/api/v1/monitoring/portfolio",
                params={"db_path": str(db_file)},
            )
            assert resp.status_code == 500
            # The discriminating assertion: secret ABSENT from the body.
            assert _MON_SECRET not in resp.text
            assert resp.json()["detail"] == "internal error fetching portfolio status"
        finally:
            db_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# A28 — OMS blotter still returns 200 normally (RLock is re-entrant: no deadlock)
# ---------------------------------------------------------------------------


class TestOmsLockNoDeadlock:
    def test_blotter_returns_200_under_lock(self, client: TestClient):
        """A single acquire of the re-entrant _engine_lock must not deadlock.

        GET /oms/blotter itself acquires _engine_lock; a mis-implemented
        (non-reentrant) lock would hang here. We deliberately avoid the
        auth-gated /paper/reset so this smoke is independent of the API-key
        config (F-senior-5 / F-auditor-2).
        """
        resp = client.get("/api/v1/oms/blotter")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ---------------------------------------------------------------------------
# A27-adjacent — regime intentional 503 survives the broad except (guard fix)
# ---------------------------------------------------------------------------


class TestRegime503Survives:
    def test_missing_in_bounds_output_dir_returns_503_not_500(self, client: TestClient):
        """An in-bounds but ABSENT output_dir must yield the intentional 503,
        not a broad-except 500. Pre-fix, get_regime_status lacked
        ``except HTTPException: raise``, so its 503 was reconverted to a generic
        500 by the broad ``except Exception``."""
        from src.assembled_core.config import OUTPUT_DIR

        missing = OUTPUT_DIR / "does_not_exist_regime_dir_xyz"
        resp = client.get(
            "/api/v1/monitoring/regime", params={"output_dir": str(missing)}
        )
        assert resp.status_code == 503
        assert "regime pipeline has not run" in resp.json()["detail"]
