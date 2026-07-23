"""Regression guards for scripts/ops/book_dividends.py (W15, GESAMTBEWERTUNG).

The paper ledger never booked dividends while the broker credits real cash
payouts — a slow ledger<broker drift source. Pins: booking credits cash,
idempotency via the booked-ids log, API failure is non-fatal (backstop =
reconcile halt), dry-run writes nothing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "ops" / "book_dividends.py"


@pytest.fixture()
def mod():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("book_div_w15", SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write_ledger(path: Path, cash: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "paper_ledger.v1",
                "cash": cash,
                "positions": {},
                "equity_curve": [],
            }
        ),
        encoding="utf-8",
    )


def _activities():
    return [
        {"id": "div-1", "symbol": "TLT", "net_amount": "25.31", "date": "2026-07-01"},
        {"id": "div-2", "symbol": "GLD", "net_amount": "3.10", "date": "2026-07-15"},
    ]


def test_w15_booking_credits_cash_and_logs(mod, tmp_path):
    ledger = tmp_path / "ledger_state.json"
    booked_log = tmp_path / "booked.jsonl"
    _write_ledger(ledger, 1000.0)

    n = mod.book_pending_dividends(
        ledger_path=ledger, booked_log=booked_log, fetch=_activities
    )
    assert n == 2
    state = json.loads(ledger.read_text(encoding="utf-8"))
    assert state["cash"] == pytest.approx(1028.41)
    ids = [json.loads(x)["activity_id"] for x in booked_log.read_text().splitlines()]
    assert ids == ["div-1", "div-2"]


def test_w15_idempotent_second_run_books_nothing(mod, tmp_path):
    ledger = tmp_path / "ledger_state.json"
    booked_log = tmp_path / "booked.jsonl"
    _write_ledger(ledger, 1000.0)

    assert (
        mod.book_pending_dividends(
            ledger_path=ledger, booked_log=booked_log, fetch=_activities
        )
        == 2
    )
    assert (
        mod.book_pending_dividends(
            ledger_path=ledger, booked_log=booked_log, fetch=_activities
        )
        == 0
    )
    state = json.loads(ledger.read_text(encoding="utf-8"))
    assert state["cash"] == pytest.approx(1028.41)  # credited exactly once


def test_w15_api_failure_is_nonfatal(mod, tmp_path):
    ledger = tmp_path / "ledger_state.json"
    _write_ledger(ledger, 1000.0)

    def _boom():
        raise RuntimeError("simulated Alpaca outage")

    n = mod.book_pending_dividends(
        ledger_path=ledger, booked_log=tmp_path / "booked.jsonl", fetch=_boom
    )
    assert n == 0
    state = json.loads(ledger.read_text(encoding="utf-8"))
    assert state["cash"] == pytest.approx(1000.0)


def test_w15_malformed_net_amount_is_skipped_not_booked(mod, tmp_path):
    # Stage-2 F-senior-3 / Stage-3 F-auditor-1: one malformed net_amount
    # must (a) not crash, (b) not block the valid neighbour, (c) NOT be
    # marked booked — so it is retried on the next run.
    ledger = tmp_path / "ledger_state.json"
    booked_log = tmp_path / "booked.jsonl"
    _write_ledger(ledger, 1000.0)

    def _mixed():
        return [
            {"id": "bad-1", "symbol": "XXX", "net_amount": "N/A", "date": "2026-07-01"},
            {
                "id": "div-1",
                "symbol": "TLT",
                "net_amount": "25.31",
                "date": "2026-07-01",
            },
        ]

    n = mod.book_pending_dividends(
        ledger_path=ledger, booked_log=booked_log, fetch=_mixed
    )
    assert n == 1  # only the valid one counted... see note below

    state = json.loads(ledger.read_text(encoding="utf-8"))
    assert state["cash"] == pytest.approx(1025.31)  # only TLT credited
    ids = [json.loads(x)["activity_id"] for x in booked_log.read_text().splitlines()]
    assert "div-1" in ids
    assert "bad-1" not in ids  # NOT marked booked -> retried next run

    # Second run: bad-1 is retried (still malformed -> still skipped),
    # div-1 is idempotent — cash unchanged.
    mod.book_pending_dividends(ledger_path=ledger, booked_log=booked_log, fetch=_mixed)
    state2 = json.loads(ledger.read_text(encoding="utf-8"))
    assert state2["cash"] == pytest.approx(1025.31)


def test_w15_dry_run_writes_nothing(mod, tmp_path):
    ledger = tmp_path / "ledger_state.json"
    booked_log = tmp_path / "booked.jsonl"
    _write_ledger(ledger, 1000.0)

    n = mod.book_pending_dividends(
        ledger_path=ledger, booked_log=booked_log, fetch=_activities, dry_run=True
    )
    assert n == 2
    state = json.loads(ledger.read_text(encoding="utf-8"))
    assert state["cash"] == pytest.approx(1000.0)
    assert not booked_log.exists()
