"""Discriminating tests for ops correctness fixes A12 / A13 / A35.

A12: reconcile-FAIL must escalate via the real delivering AlertManager
     (ops.alerting), not be silently swallowed because nothing writes *.error.
A13: _resolve_cost_cfg must fail-CLOSED to a conservative cost default instead
     of returning {} (which silently models fills at exact close / 0 bps).
A35: factor curation report must label its quality gate ``ic_tstat`` (a raw IC
     t-stat), NOT ``dsr`` (it is not a Deflated Sharpe Ratio).

Each assertion pins the post-fix behaviour so it fails against the pre-fix code.
All marked @pytest.mark.fast.
"""

from __future__ import annotations

import logging

import pytest

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# A12 — reconcile-FAIL escalation: ENTFERNT 2026-08-17 (Audit-Plan 6.5).
# Die 3 Tests + _RecordingFire/_patch_fire testeten _alert_health_worker aus
# ops/daily_scheduler — archiviert nach archive/orphaned_code_2026-08-17/ops/
# (Scheduler-Kette ohne Launcher, toter Zweitpfad). Der LEBENDE Feuerpfad
# (accounting/reconciliation.py:205) bleibt durch
# tests/test_batchB2_accounting_failclosed.py abgedeckt.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# A13 — cost-model fail-CLOSED
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_a13_both_missing_returns_conservative_default(caplog):
    """Empty policy AND empty app_cfg -> conservative default, NOT {}.

    Pre-fix this returned {} (silently 0 bps). Also asserts a loud WARNING.
    """
    from src.assembled_core.ops.paper_runner import _resolve_cost_cfg

    with caplog.at_level(logging.WARNING, logger="src.assembled_core.ops.paper_runner"):
        result = _resolve_cost_cfg({}, {})

    assert result != {}
    assert result["commission_bps"] == pytest.approx(10.0)
    assert result["spread_w"] == pytest.approx(0.25)
    assert result["impact_w"] == pytest.approx(0.5)
    # loud warning emitted
    assert any(
        rec.levelno >= logging.WARNING and "cost_model" in rec.getMessage()
        for rec in caplog.records
    )


@pytest.mark.fast
def test_a13_policy_cost_model_used_unchanged(caplog):
    """Happy path: policy supplies a cost_model -> used as-is, no warning."""
    from src.assembled_core.ops.paper_runner import _resolve_cost_cfg

    policy = {
        "paper_pilot": {
            "cost_model": {"commission_bps": 7.5, "spread_w": 0.1, "impact_w": 0.2}
        }
    }
    with caplog.at_level(logging.WARNING, logger="src.assembled_core.ops.paper_runner"):
        result = _resolve_cost_cfg({}, policy)

    assert result == {"commission_bps": 7.5, "spread_w": 0.1, "impact_w": 0.2}
    assert not any(
        "falling back to conservative default" in rec.getMessage()
        for rec in caplog.records
    )


@pytest.mark.fast
def test_a13_app_cfg_cost_model_used_when_policy_empty():
    """app_cfg supplies a cost_model and policy is empty -> app_cfg used, no default."""
    from src.assembled_core.ops.paper_runner import _resolve_cost_cfg

    app_cfg = {"paper_runner": {"cost_model": {"commission_bps": 3.0}}}
    result = _resolve_cost_cfg(app_cfg, {})
    assert result["commission_bps"] == pytest.approx(3.0)
    # not the conservative default
    assert "spread_w" not in result


# ---------------------------------------------------------------------------
# A35 — factor curation labels: ENTFERNT 2026-08-17 (Audit-Plan 6.5).
# Testete _factor_curation_worker aus ops/daily_scheduler (mit-archiviert).
# ---------------------------------------------------------------------------
