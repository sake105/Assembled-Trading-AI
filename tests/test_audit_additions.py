"""Targeted unit tests for the new functions added during the audit sweep.

Closes the coverage gap reported by the test-runner: probabilistic_sharpe_ratio,
minimum_track_record_length, set_correlation_context + JSONFormatter
correlation-id propagation, and ops/alerting._send_slack credential handling.
"""

from __future__ import annotations

import io
import json
import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# qa/metrics.py — PSR / MinTRL
# ---------------------------------------------------------------------------


def test_probabilistic_sharpe_ratio_positive_sharpe_above_benchmark() -> None:
    """Positive observed SR above 0 benchmark → PSR > 0.5."""
    pytest.importorskip("scipy")
    from src.assembled_core.qa.metrics import probabilistic_sharpe_ratio

    psr = probabilistic_sharpe_ratio(
        sharpe_observed=1.5, n_obs=252, sharpe_benchmark=0.0
    )
    assert 0.0 <= psr <= 1.0
    assert psr > 0.5, f"PSR(1.5, 252, 0) should clearly exceed 0.5; got {psr}"


def test_probabilistic_sharpe_ratio_equal_to_benchmark_is_half() -> None:
    """Observed SR equal to benchmark → PSR ≈ 0.5 (no asymmetric info)."""
    pytest.importorskip("scipy")
    from src.assembled_core.qa.metrics import probabilistic_sharpe_ratio

    psr = probabilistic_sharpe_ratio(
        sharpe_observed=0.7, n_obs=500, sharpe_benchmark=0.7
    )
    assert abs(psr - 0.5) < 1e-9


def test_probabilistic_sharpe_ratio_nan_for_invalid_inputs() -> None:
    pytest.importorskip("scipy")
    from src.assembled_core.qa.metrics import probabilistic_sharpe_ratio

    assert np.isnan(probabilistic_sharpe_ratio(float("nan"), 252))
    assert np.isnan(probabilistic_sharpe_ratio(1.0, 1))


def test_minimum_track_record_length_inf_when_sr_below_benchmark() -> None:
    pytest.importorskip("scipy")
    from src.assembled_core.qa.metrics import minimum_track_record_length

    n = minimum_track_record_length(sharpe_observed=0.3, sharpe_benchmark=0.5)
    assert n == float("inf")


def test_minimum_track_record_length_decreases_with_higher_sharpe() -> None:
    """Higher SR-gap → shorter MinTRL."""
    pytest.importorskip("scipy")
    from src.assembled_core.qa.metrics import minimum_track_record_length

    n_low = minimum_track_record_length(sharpe_observed=0.6, sharpe_benchmark=0.0)
    n_high = minimum_track_record_length(sharpe_observed=2.0, sharpe_benchmark=0.0)
    assert n_high < n_low


# ---------------------------------------------------------------------------
# logging_config.py — correlation IDs
# ---------------------------------------------------------------------------


def test_json_formatter_includes_correlation_ids_when_set() -> None:
    from src.assembled_core.logging_config import (
        JSONFormatter,
        set_correlation_context,
        _CorrelationFilter,
    )

    logger = logging.getLogger("audit-test-corr")
    logger.handlers.clear()
    logger.filters.clear()
    logger.setLevel(logging.INFO)

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.addFilter(_CorrelationFilter())

    set_correlation_context(trace_id="t-1", run_id="r-1")
    logger.info("hello")

    handler.flush()
    line = buf.getvalue().strip()
    payload = json.loads(line)
    assert payload["event"] == "hello"
    assert payload["trace_id"] == "t-1"
    assert payload["run_id"] == "r-1"


def test_json_formatter_omits_unset_correlation_ids() -> None:
    """A pristine logger with no correlation context emits no corr keys."""
    from src.assembled_core.logging_config import JSONFormatter

    rec = logging.LogRecord(
        name="audit-test-bare",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="bare",
        args=None,
        exc_info=None,
    )
    payload = json.loads(JSONFormatter().format(rec))
    assert payload["event"] == "bare"
    # None of the correlation fields should be present
    for k in ("trace_id", "span_id", "run_id", "correlation_id"):
        assert k not in payload


# ---------------------------------------------------------------------------
# ops/alerting.py — Slack channel credential handling
# ---------------------------------------------------------------------------


def test_send_slack_warns_when_webhook_unset(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from src.assembled_core.ops.alerting import AlertManager

    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    mgr = AlertManager()
    with caplog.at_level(logging.WARNING, logger="src.assembled_core.ops.alerting"):
        mgr._send_slack({}, "test")  # type: ignore[attr-defined]

    assert any("slack webhook not set" in r.message for r in caplog.records)


def test_send_slack_posts_to_webhook_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.assembled_core.ops.alerting import AlertManager

    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.example/abc")

    mgr = AlertManager()
    fake_resp = MagicMock()
    fake_resp.status = 200
    fake_resp.__enter__ = MagicMock(return_value=fake_resp)
    fake_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=fake_resp) as mocked_open:
        mgr._send_slack({}, "ping")  # type: ignore[attr-defined]

    mocked_open.assert_called_once()
    req = mocked_open.call_args.args[0]
    body = req.data.decode("utf-8")
    assert "ping" in body
    assert req.headers.get("Content-type", req.headers.get("Content-Type")) == (
        "application/json"
    )


# ---------------------------------------------------------------------------
# features/volatility_estimators.py — HAR-RV forecast
# ---------------------------------------------------------------------------


def test_har_rv_forecast_returns_positive_predictions() -> None:
    import numpy as np
    import pandas as pd

    from src.assembled_core.features.volatility_estimators import har_rv_forecast

    rng = np.random.default_rng(seed=42)
    n = 300
    rv = pd.Series(0.0001 + 0.00005 * np.abs(rng.standard_normal(n).cumsum() / 20))

    pred = har_rv_forecast(rv, horizon=1, min_samples=100)
    non_nan = pred.dropna()
    assert len(non_nan) > 100, "should produce many forecasts past min_samples"
    assert (non_nan > 0).all(), "variance forecasts must be positive"


def test_har_rv_forecast_returns_nan_when_too_short() -> None:
    import pandas as pd

    from src.assembled_core.features.volatility_estimators import har_rv_forecast

    rv = pd.Series([0.0001] * 30)
    pred = har_rv_forecast(rv, horizon=1, min_samples=252)
    assert pred.isna().all()


# ---------------------------------------------------------------------------
# accounting/reconciliation.py — alert dispatch on SLO breach
# ---------------------------------------------------------------------------


def test_evaluate_reconcile_slo_fires_alert_on_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.assembled_core.accounting import reconciliation as recon_mod

    fired: list[tuple] = []

    class _FakeMgr:
        def fire(self, rule_name: str, context: dict) -> None:
            fired.append((rule_name, context))

    monkeypatch.setattr(
        "src.assembled_core.ops.alerting.AlertManager",
        lambda *a, **kw: _FakeMgr(),
    )

    slo = recon_mod.ReconcileSLO()
    # Force a fail-level cash breach (>= 25 bps of broker_cash).
    result = recon_mod.evaluate_reconcile_slo(
        cash_diff=10_000.0,
        broker_cash=1_000_000.0,  # 100 bps diff → fail
        max_qty_diff=0.0,
        fill_rate=None,
        slippage_p99_bps=None,
        slo=slo,
    )
    assert result["severity"] == "fail"
    assert any(name == "reconciliation_fail" for name, _ in fired), fired


def test_evaluate_reconcile_slo_silent_on_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.assembled_core.accounting import reconciliation as recon_mod

    fired: list[tuple] = []

    class _FakeMgr:
        def fire(self, rule_name: str, context: dict) -> None:
            fired.append((rule_name, context))

    monkeypatch.setattr(
        "src.assembled_core.ops.alerting.AlertManager",
        lambda *a, **kw: _FakeMgr(),
    )

    slo = recon_mod.ReconcileSLO()
    result = recon_mod.evaluate_reconcile_slo(
        cash_diff=0.0,
        broker_cash=1_000_000.0,
        max_qty_diff=0.0,
        fill_rate=None,
        slippage_p99_bps=None,
        slo=slo,
    )
    assert result["severity"] == "ok"
    assert fired == []


# ---------------------------------------------------------------------------
# Wave-3 — kill_switch hash-chain audit log (C4-016)
# ---------------------------------------------------------------------------


def test_kill_switch_audit_hash_chain_holds(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_AUDIT", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_STATE", str(tmp_path / "state.json"))
    monkeypatch.delenv("ASSEMBLED_KILL_SWITCH", raising=False)
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_SENTINEL", str(tmp_path / ".sentinel"))

    from src.assembled_core.execution.kill_switch import (
        activate_kill_switch,
        deactivate_kill_switch,
        verify_audit_chain,
    )

    activate_kill_switch(throttle_pct=0.25, reason="prop", actor="t")
    deactivate_kill_switch(reason="prop-done", actor="t")
    ok, n = verify_audit_chain()
    assert ok is True
    assert n >= 2


def test_kill_switch_audit_hash_chain_detects_tamper(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    audit_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_AUDIT", str(audit_path))
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_STATE", str(tmp_path / "state.json"))
    monkeypatch.delenv("ASSEMBLED_KILL_SWITCH", raising=False)
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_SENTINEL", str(tmp_path / ".sentinel"))

    from src.assembled_core.execution.kill_switch import (
        activate_kill_switch,
        deactivate_kill_switch,
        verify_audit_chain,
    )

    activate_kill_switch(throttle_pct=0.0, reason="real", actor="t")
    deactivate_kill_switch(reason="real-done", actor="t")

    # Tamper: edit first record's reason — hash on that record now mismatches.
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["reason"] = "MUTATED"
    lines[0] = json.dumps(first, sort_keys=True)
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, n = verify_audit_chain()
    assert ok is False, "chain MUST flag tampered record"
    assert n >= 1


# ---------------------------------------------------------------------------
# Wave-3 — order_lifecycle SUBMITTED-timeout detector (C4-020)
# ---------------------------------------------------------------------------


def test_find_stuck_orders_returns_only_expired() -> None:
    from datetime import datetime, timedelta, timezone

    from src.assembled_core.execution.order_lifecycle import (
        OrderLifecycleTracker,
        OrderState,
    )

    tracker = OrderLifecycleTracker()
    oid_old = tracker.create(symbol="OLD", side="BUY", quantity=1.0)
    oid_new = tracker.create(symbol="NEW", side="BUY", quantity=1.0)
    tracker.transition(oid_old, OrderState.VALIDATED)
    tracker.transition(oid_old, OrderState.SUBMITTED)
    tracker.transition(oid_new, OrderState.VALIDATED)
    tracker.transition(oid_new, OrderState.SUBMITTED)

    # Simulate clock 5 minutes after submission.
    later = datetime.now(timezone.utc) + timedelta(minutes=5)
    stuck = tracker.find_stuck_orders(max_age_seconds=60.0, now=later)
    ids = {o.order_id for o in stuck}
    assert oid_old in ids and oid_new in ids

    # Threshold larger than elapsed → none stuck
    not_stuck = tracker.find_stuck_orders(max_age_seconds=600.0, now=later)
    assert not_stuck == []


def test_find_stuck_orders_ignores_filled() -> None:
    from datetime import datetime, timedelta, timezone

    from src.assembled_core.execution.order_lifecycle import (
        OrderLifecycleTracker,
        OrderState,
    )

    tracker = OrderLifecycleTracker()
    oid = tracker.create(symbol="DONE", side="BUY", quantity=1.0)
    tracker.transition(oid, OrderState.VALIDATED)
    tracker.transition(oid, OrderState.SUBMITTED)
    tracker.transition(oid, OrderState.FILLED, fill_price=100.0)
    later = datetime.now(timezone.utc) + timedelta(hours=1)
    assert tracker.find_stuck_orders(max_age_seconds=1.0, now=later) == []


# ---------------------------------------------------------------------------
# Wave-3 — /ready disk-quota check (C4-040)
# ---------------------------------------------------------------------------


def test_ready_endpoint_includes_disk_quota_detail(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("ASSEMBLED_DISK_QUOTA_PATH", str(tmp_path))

    from src.assembled_core.api.app import create_app

    client = TestClient(create_app())
    r = client.get("/ready")
    assert r.status_code in (200, 503)
    body = r.json()
    assert "checks" in body and "disk_quota" in body["checks"]
    assert "details" in body and "disk_used_pct" in body["details"]


# ---------------------------------------------------------------------------
# Wave-3 — Gawande pre_trade_gate (C2-070)
# ---------------------------------------------------------------------------


def test_pre_trade_gate_raises_when_killswitch_engaged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    from src.assembled_core.execution.pre_trade_checks import (
        PreTradeGateBlocked,
        pre_trade_gate,
    )

    # Patch the kill-switch read used inside the gate.
    monkeypatch.setattr(
        "src.assembled_core.execution.kill_switch.get_kill_switch_state",
        lambda: {"engaged": True, "throttle_pct": 0.0, "sources": {}, "persistent": {}},
    )

    orders = pd.DataFrame(
        [{"symbol": "AAPL", "side": "BUY", "qty": 1.0, "price": 100.0}]
    )
    with pytest.raises(PreTradeGateBlocked) as ei:
        pre_trade_gate(orders)
    assert ei.value.check == "kill_switch"


def test_pre_trade_gate_passes_when_no_blockers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    from src.assembled_core.execution.pre_trade_checks import pre_trade_gate

    # Stub the kill-switch as disengaged so we don't depend on real state.
    monkeypatch.setattr(
        "src.assembled_core.execution.kill_switch.get_kill_switch_state",
        lambda: {
            "engaged": False,
            "throttle_pct": 1.0,
            "sources": {},
            "persistent": {},
        },
    )

    orders = pd.DataFrame(
        [{"symbol": "AAPL", "side": "BUY", "qty": 1.0, "price": 100.0}]
    )
    filtered = pre_trade_gate(orders)
    assert len(filtered) == 1
