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

import json
import logging

import pandas as pd
import pytest

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# A12 — reconcile-FAIL escalation
# ---------------------------------------------------------------------------


class _RecordingFire:
    """Stand-in for AlertManager.fire that records (rule_name, context) calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, rule_name, context=None):
        # Replaces AlertManager.fire as a plain attribute (not a function), so it
        # is NOT bound — `self`/instance is not passed; first arg is rule_name.
        self.calls.append((rule_name, dict(context or {})))
        return True


def _patch_fire(monkeypatch) -> _RecordingFire:
    from src.assembled_core.ops import alerting as _alerting

    rec = _RecordingFire()
    monkeypatch.setattr(_alerting.AlertManager, "fire", rec, raising=True)
    return rec


@pytest.mark.fast
def test_a12_reconcile_fail_fires_alert(tmp_path, monkeypatch):
    """status=FAIL in reconcile_latest.json -> reconciliation_fail is FIRED.

    Pre-fix the alert path only globs *.error (which nothing writes), so fire()
    is never called -> discriminating.
    """
    from src.assembled_core.ops.daily_scheduler import _alert_health_worker

    (tmp_path / "reconcile_latest.json").write_text(
        json.dumps(
            {
                "schema_version": "run.reconcile.v1",
                "status": "FAIL",
                "cash": {"before": 100.0, "after": -5.0, "delta": -105.0},
                "notes": ["invariant_cash_non_negative_failed"],
            }
        ),
        encoding="utf-8",
    )

    rec = _patch_fire(monkeypatch)
    result = _alert_health_worker("2026-01-02", str(tmp_path), dry_run=False)

    assert result.status == "ok"
    fired_rules = [name for name, _ctx in rec.calls]
    assert "reconciliation_fail" in fired_rules
    # context carries the violation detail so a human can triage
    ctx = next(c for n, c in rec.calls if n == "reconciliation_fail")
    assert ctx.get("violation_count") == 1
    assert ctx.get("first_violation") == "invariant_cash_non_negative_failed"


@pytest.mark.fast
def test_a12_reconcile_warn_fires_warn_rule(tmp_path, monkeypatch):
    """status=WARN -> reconciliation_warn fired (not the critical fail rule)."""
    from src.assembled_core.ops.daily_scheduler import _alert_health_worker

    (tmp_path / "reconcile_latest.json").write_text(
        json.dumps({"status": "WARN", "notes": ["soft_drift"]}),
        encoding="utf-8",
    )

    rec = _patch_fire(monkeypatch)
    _alert_health_worker("2026-01-02", str(tmp_path), dry_run=False)

    fired_rules = [name for name, _ctx in rec.calls]
    assert "reconciliation_warn" in fired_rules
    assert "reconciliation_fail" not in fired_rules


@pytest.mark.fast
def test_a12_reconcile_ok_fires_nothing(tmp_path, monkeypatch):
    """status=OK -> neither reconcile rule fires (no false alarm)."""
    from src.assembled_core.ops.daily_scheduler import _alert_health_worker

    (tmp_path / "reconcile_latest.json").write_text(
        json.dumps({"status": "OK", "notes": []}),
        encoding="utf-8",
    )

    rec = _patch_fire(monkeypatch)
    _alert_health_worker("2026-01-02", str(tmp_path), dry_run=False)

    fired_rules = [name for name, _ctx in rec.calls]
    assert "reconciliation_fail" not in fired_rules
    assert "reconciliation_warn" not in fired_rules


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
# A35 — factor curation labels ic_tstat, not dsr
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_a35_curation_report_key_is_ic_tstat_not_dsr(tmp_path, monkeypatch):
    """Curation report uses ``ic_tstat`` (not ``dsr``); threshold behaviour at 0.5 unchanged.

    Two crafted factors: one with high IC t-stat (active) and one with ~0 IC
    (flagged). Pre-fix the per-factor key was ``dsr`` -> assertion fails.
    """
    import src.assembled_core.ops.daily_scheduler as sched

    # Craft a factor_scores parquet with two float factor columns.
    panel = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "good_factor": [1.0, 2.0, 3.0, 4.0],
            "dead_factor": [0.0, 0.0, 0.0, 0.0],
        }
    )
    panel.to_parquet(str(tmp_path / "factor_scores_2026-01-02.parquet"))

    # Deterministic IC curves: a strong stable IC (t-stat >> 0.5) and a ~0 IC.
    def _fake_ic_curve(_df, factor_col, max_horizon_days=60):
        if factor_col == "good_factor":
            # tightly clustered, high mean -> t-stat (mean/std*sqrt(n)) >> 0.5.
            # Needs non-zero std (ic_std > 1e-9) so the t-stat is computed.
            ic = [0.30, 0.31, 0.29, 0.30, 0.31, 0.29] * 2
        else:
            # symmetric around ~0 -> mean ~0 -> t-stat ~0 -> flagged.
            ic = [0.01, -0.01, 0.01, -0.01, 0.01, -0.01] * 2
        return pd.DataFrame({"horizon": list(range(1, 13)), "ic": ic})

    def _fake_half_life(_curve):
        return 10.0

    # Patch the qa imports the worker pulls in at call time.
    import src.assembled_core.qa.factor_analysis as _fa

    monkeypatch.setattr(_fa, "compute_ic_decay_curve", _fake_ic_curve, raising=True)
    monkeypatch.setattr(_fa, "compute_factor_half_life", _fake_half_life, raising=True)

    # Force the quarterly-window guard to accept this date (Jan 2 is in-window
    # already, but be explicit about running the body).
    result = sched._factor_curation_worker("2026-01-02", str(tmp_path), dry_run=False)
    assert result.status == "ok"

    report = json.loads(
        (tmp_path / "factor_curation_2026-01-02.json").read_text(encoding="utf-8")
    )
    factors = report["factors"]

    # Key renamed: ic_tstat present, dsr absent.
    assert "ic_tstat" in factors["good_factor"]
    assert "dsr" not in factors["good_factor"]
    assert "ic_tstat" in factors["dead_factor"]
    assert "dsr" not in factors["dead_factor"]

    # Threshold behaviour preserved at 0.5: strong factor active, dead one flagged.
    assert factors["good_factor"]["status"] == "active"
    assert factors["dead_factor"]["status"] == "flagged"
    assert "dead_factor" in report["flagged_for_removal"]
    assert "good_factor" not in report["flagged_for_removal"]
