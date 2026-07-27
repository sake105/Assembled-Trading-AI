"""R2-6 / D-02: the UnifiedPaperEngine EOD reconcile must not let a self-compare
noop masquerade as a verified reconcile.

When no shadow broker is configured, ``_run_reconciliation`` compares the engine
state against ITSELF — green by construction. Before R2-6 that produced a
``severity="ok"`` verdict indistinguishable from a real broker reconcile that
passed. These tests pin the new behaviour:

* the noop is explicitly LABELED (``reconcile['independent_source'] is False`` +
  ``reconcile['source'] == 'self_compare_noop'``) and logged at WARNING, while
  the default severity stays "ok" (behaviour-preserving for pure-sim paper);
* with ``reconcile_require_independent_source=True`` a missing independent
  source is escalated to severity "fail" and an alert file is written
  (fail-closed-able, mirroring OPS-07);
* a real shadow-broker snapshot marks the verdict as independent;
* a configured-but-erroring shadow broker falls back to the noop label (it must
  NOT yield a falsely "verified" green).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.assembled_core.execution.unified_paper_engine import (
    UnifiedPaperConfig,
    UnifiedPaperEngine,
)


@pytest.fixture(autouse=True)
def _isolate_reconcile_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the reconcile SLO audit log into tmp_path.

    Every ``_run_reconciliation`` call below funnels through
    ``accounting.reconciliation.evaluate_reconcile_slo``, which appends a real
    ``slo_eval`` line to ``output/ops/reconciliation_audit.jsonl`` unless the
    ``ASSEMBLED_RECONCILE_AUDIT`` override is set. The override is read at
    CALL time (``_recon_audit_path()`` does ``os.environ.get`` per append, no
    import-time constant), so a plain env monkeypatch is sufficient — without
    it, each test run contaminates the repo's production audit log.
    """
    monkeypatch.setenv(
        "ASSEMBLED_RECONCILE_AUDIT", str(tmp_path / "reconciliation_audit.jsonl")
    )


def _make_engine(tmp_path: Path, **cfg_overrides) -> UnifiedPaperEngine:
    cfg = UnifiedPaperConfig(
        seed_capital=100_000.0,
        ledger_dir=tmp_path / "ledger",
        reconcile_alerts_dir=tmp_path / "alerts",
        run_id="r2_6_test",
        **cfg_overrides,
    )
    engine = UnifiedPaperEngine(cfg)
    # Minimal state — the reconcile reads positions/cash off self._state only.
    engine._state = {"positions": {"AAPL": 10.0, "MSFT": 5.0}, "cash": 50_000.0}
    return engine


class _MatchingBroker:
    """Shadow broker whose snapshot matches engine state exactly."""

    def get_snapshot(self) -> dict:
        return {
            "positions": [
                {"symbol": "AAPL", "qty": 10.0},
                {"symbol": "MSFT", "qty": 5.0},
            ],
            "cash": 50_000.0,
        }


class _BoomBroker:
    """Shadow broker whose snapshot call raises."""

    def get_snapshot(self) -> dict:
        raise RuntimeError("broker snapshot boom")


def test_self_compare_noop_is_labeled_not_verified(tmp_path, caplog) -> None:
    engine = _make_engine(tmp_path)  # no shadow broker

    with caplog.at_level(logging.WARNING):
        verdict = engine._run_reconciliation("2024-04-22")

    assert verdict is not None
    recon = verdict["reconcile"]
    assert recon["independent_source"] is False
    assert recon["source"] == "self_compare_noop"
    # default path stays behaviour-preserving: no escalation, no alert file
    assert verdict["severity"] == "ok"
    assert not (tmp_path / "alerts").exists() or not list(
        (tmp_path / "alerts").glob("*.json")
    )
    assert any(
        "self-compare noop" in r.message or "self-compare noop" in str(r.args)
        for r in caplog.records
    )


def test_require_independent_source_escalates_to_fail(tmp_path, caplog) -> None:
    engine = _make_engine(tmp_path, reconcile_require_independent_source=True)

    with caplog.at_level(logging.WARNING):
        verdict = engine._run_reconciliation("2024-04-22")

    assert verdict is not None
    assert verdict["severity"] == "fail"
    assert verdict["reconcile"]["independent_source"] is False
    reasons = [v.get("reason") for v in verdict["violations"] if isinstance(v, dict)]
    assert "no_independent_reconcile_source" in reasons
    # escalation must have written the audit alert
    alerts = list((tmp_path / "alerts").glob("reconcile_alert_*.json"))
    assert len(alerts) == 1


def test_shadow_broker_snapshot_marks_independent(tmp_path) -> None:
    engine = _make_engine(tmp_path, shadow_broker=_MatchingBroker())

    verdict = engine._run_reconciliation("2024-04-22")

    assert verdict is not None
    recon = verdict["reconcile"]
    assert recon["independent_source"] is True
    assert recon["source"] == "shadow_broker"
    # matching snapshot reconciles clean
    assert verdict["severity"] == "ok"


def test_erroring_shadow_broker_falls_back_to_noop_label(tmp_path) -> None:
    engine = _make_engine(tmp_path, shadow_broker=_BoomBroker())

    verdict = engine._run_reconciliation("2024-04-22")

    assert verdict is not None
    recon = verdict["reconcile"]
    # broker was configured but its snapshot raised → NOT an independent reconcile
    assert recon["independent_source"] is False
    assert recon["source"] == "self_compare_noop"
    # fall-back must stay behaviour-preserving: a self-compare still reconciles
    # clean (severity "ok"), it just isn't an independent verification
    assert verdict["severity"] == "ok"
