"""Regression guards for the W4 QA-block flag bridge (2026-07-24, GESAMTBEWERTUNG
Schritt 8): a BLOCK verdict from the QA gates must be able to stop the live
pilot — previously "QA sagt BLOCK" reached no order path (qa_status=None at
both call sites; declared-open since the audit).

Semantics under test:
  - write_qa_block_flag persists ONLY on BLOCK (OK/WARNING -> no flag).
  - read_qa_block_flag: absent -> None; corrupt -> "unreadable" sentinel
    (positive-but-corrupt evidence must not be silently ignored).
  - preflight: flag present -> BLOCK (return False); flag absent -> no
    QA-related block (absence is NOT "QA passed", but must not dead-lock
    the pilot); corrupt flag -> BLOCK.
  - end-to-end: a synthetic BLOCK verdict written via the real writer stops
    the real preflight (the auditor's acceptance criterion).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.assembled_core.qa.qa_gates import (
    QAGateResult,
    QAGatesSummary,
    QAResult,
    read_qa_block_flag,
    write_qa_block_flag,
)

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parents[1]
RLP_PATH = REPO_ROOT / "scripts" / "run_live_paper.py"


def _summary(overall: QAResult) -> QAGatesSummary:
    gate = QAGateResult(
        gate_name="sharpe_ratio",
        result=overall,
        reason=f"synthetic {overall.value} for W4 test",
        details={},
    )
    return QAGatesSummary(
        overall_result=overall,
        passed_gates=0 if overall == QAResult.BLOCK else 1,
        warning_gates=0,
        blocked_gates=1 if overall == QAResult.BLOCK else 0,
        gate_results=[gate],
    )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def test_w4_block_verdict_writes_flag(tmp_path):
    flag = tmp_path / "qa_block.json"
    out = write_qa_block_flag(_summary(QAResult.BLOCK), source="test", flag_path=flag)
    assert out == flag and flag.exists()
    data = json.loads(flag.read_text(encoding="utf-8"))
    assert data["schema"] == "qa_block.v1"
    assert data["blocked_gates"][0]["gate"] == "sharpe_ratio"


@pytest.mark.parametrize("overall", [QAResult.OK, QAResult.WARNING])
def test_w4_non_block_writes_nothing(tmp_path, overall):
    flag = tmp_path / "qa_block.json"
    assert write_qa_block_flag(_summary(overall), source="test", flag_path=flag) is None
    assert not flag.exists()


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def test_w4_reader_absent_is_none(tmp_path):
    assert read_qa_block_flag(tmp_path / "missing.json") is None


def test_w4_reader_corrupt_is_unreadable_sentinel(tmp_path):
    flag = tmp_path / "qa_block.json"
    flag.write_text("{not valid json", encoding="utf-8")
    got = read_qa_block_flag(flag)
    assert got == {"schema": "unreadable"}


# ---------------------------------------------------------------------------
# Preflight wiring (end-to-end against the real script)
# ---------------------------------------------------------------------------


class _CleanAdapter:
    def get_account(self):
        return {"equity": 999_999.0}

    def get_open_orders(self):
        return []

    def cancel_all_orders(self) -> int:
        return 0


@pytest.fixture()
def rlp():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("rlp_w4_mod", RLP_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _preflight(rlp, monkeypatch, tmp_path, qa_flag_return):
    monkeypatch.setattr(rlp, "HALT_FLAG_PATH", tmp_path / "halt.json")
    import src.assembled_core.execution.intent_store as intent_store
    import src.assembled_core.execution.kill_switch as ks
    import src.assembled_core.qa.qa_gates as qg

    monkeypatch.setattr(ks, "is_kill_switch_engaged", lambda: False)
    monkeypatch.setattr(intent_store, "find_pending_order_intents", lambda: [])
    monkeypatch.setattr(qg, "read_qa_block_flag", lambda *a, **k: qa_flag_return)
    app_cfg = {"paper_runner": {"start_capital": 100.0, "dd_stop_pct": 0.99}}
    return rlp._preflight_checks(_CleanAdapter(), app_cfg)


def test_w4_preflight_blocks_on_flag(rlp, monkeypatch, tmp_path):
    flag = {
        "schema": "qa_block.v1",
        "source": "orchestrator/eod_pipeline freq=1d",
        "blocked_gates": [{"gate": "sharpe_ratio", "reason": "synthetic"}],
    }
    assert _preflight(rlp, monkeypatch, tmp_path, flag) is False


def test_w4_preflight_blocks_on_corrupt_flag(rlp, monkeypatch, tmp_path):
    assert _preflight(rlp, monkeypatch, tmp_path, {"schema": "unreadable"}) is False


def test_w4_preflight_passes_without_flag(rlp, monkeypatch, tmp_path):
    assert _preflight(rlp, monkeypatch, tmp_path, None) is True


def test_w4_preflight_fails_closed_when_reader_raises(rlp, monkeypatch, tmp_path):
    # Stage-1 H1: a failing flag READER is "evidence check not performable",
    # not "no evidence" — must block, like the intent-store check.
    monkeypatch.setattr(rlp, "HALT_FLAG_PATH", tmp_path / "halt.json")
    import src.assembled_core.execution.intent_store as intent_store
    import src.assembled_core.execution.kill_switch as ks
    import src.assembled_core.qa.qa_gates as qg

    monkeypatch.setattr(ks, "is_kill_switch_engaged", lambda: False)
    monkeypatch.setattr(intent_store, "find_pending_order_intents", lambda: [])

    def _boom(*a, **k):
        raise RuntimeError("simulated flag-reader infra failure")

    monkeypatch.setattr(qg, "read_qa_block_flag", _boom)
    app_cfg = {"paper_runner": {"start_capital": 100.0, "dd_stop_pct": 0.99}}
    assert rlp._preflight_checks(_CleanAdapter(), app_cfg) is False


def test_w4_ack_script_archives_and_ledgers(tmp_path, monkeypatch):
    # Stage-1 H2: clearing is audited — reason-gated, ledger-appended,
    # flag ARCHIVED (not deleted).
    import importlib.util as _ilu

    ack_path = REPO_ROOT / "scripts" / "ops" / "ack_qa_block.py"
    spec = _ilu.spec_from_file_location("ack_qa_block_w4", ack_path)
    assert spec is not None and spec.loader is not None
    ack = _ilu.module_from_spec(spec)
    spec.loader.exec_module(ack)

    import src.assembled_core.qa.qa_gates as qg

    flag_path = tmp_path / "qa_block.json"
    monkeypatch.setattr(qg, "QA_BLOCK_FLAG_PATH", flag_path)
    monkeypatch.setattr(ack, "ACK_LEDGER_PATH", tmp_path / "ack_ledger.jsonl")
    write_qa_block_flag(_summary(QAResult.BLOCK), source="h2-test", flag_path=flag_path)

    monkeypatch.setattr(
        sys, "argv", ["ack_qa_block.py", "--reason", "reviewed for the H2 test"]
    )
    assert ack.main() == 0

    assert not flag_path.exists()  # cleared for the preflight
    archived = list(tmp_path.glob("qa_block.acked_*.json"))
    assert len(archived) == 1  # evidence archived, not deleted
    ledger_lines = (tmp_path / "ack_ledger.jsonl").read_text().splitlines()
    entry = json.loads(ledger_lines[-1])
    assert entry["reason"] == "reviewed for the H2 test"
    assert entry["cleared_flag"]["source"] == "h2-test"


def test_w4_ack_script_rejects_short_reason(tmp_path, monkeypatch):
    import importlib.util as _ilu

    ack_path = REPO_ROOT / "scripts" / "ops" / "ack_qa_block.py"
    spec = _ilu.spec_from_file_location("ack_qa_block_w4b", ack_path)
    assert spec is not None and spec.loader is not None
    ack = _ilu.module_from_spec(spec)
    spec.loader.exec_module(ack)

    import src.assembled_core.qa.qa_gates as qg

    flag_path = tmp_path / "qa_block.json"
    monkeypatch.setattr(qg, "QA_BLOCK_FLAG_PATH", flag_path)
    write_qa_block_flag(_summary(QAResult.BLOCK), source="short", flag_path=flag_path)
    monkeypatch.setattr(sys, "argv", ["ack_qa_block.py", "--reason", "kurz"])
    assert ack.main() == 1
    assert flag_path.exists()  # NOT cleared


def test_w4_end_to_end_synthetic_block_stops_preflight(rlp, monkeypatch, tmp_path):
    """The auditor's acceptance criterion: a synthetic QA-BLOCK, written via
    the REAL writer to the REAL default path (redirected to tmp), verifiably
    prevents the trading preflight from passing."""
    import src.assembled_core.qa.qa_gates as qg

    flag_path = tmp_path / "ops" / "qa_block.json"
    monkeypatch.setattr(qg, "QA_BLOCK_FLAG_PATH", flag_path)

    written = write_qa_block_flag(
        _summary(QAResult.BLOCK), source="w4-e2e-test", flag_path=flag_path
    )
    assert written is not None

    monkeypatch.setattr(rlp, "HALT_FLAG_PATH", tmp_path / "halt.json")
    import src.assembled_core.execution.intent_store as intent_store
    import src.assembled_core.execution.kill_switch as ks

    monkeypatch.setattr(ks, "is_kill_switch_engaged", lambda: False)
    monkeypatch.setattr(intent_store, "find_pending_order_intents", lambda: [])
    app_cfg = {"paper_runner": {"start_capital": 100.0, "dd_stop_pct": 0.99}}
    assert rlp._preflight_checks(_CleanAdapter(), app_cfg) is False

    # Operator clears the flag (explicit ack) -> pilot trades again.
    flag_path.unlink()
    assert rlp._preflight_checks(_CleanAdapter(), app_cfg) is True
