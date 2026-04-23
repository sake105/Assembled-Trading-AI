"""Tests for wave-42 module wiring into trading_cycle.py.

Covers:
  Step 8.36 — ops.certification (build_default_runner / CertificationRunner)
  Step 7.70 — ops.alert_sinks (dispatch_alerts / SlackWebhookSink / EmailSink)
  Step 8.37 — ml.online_gradient_boosting (OnlineAdaptiveLearner)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.assembled_core.ops.certification import (
    build_default_runner,
    check_imports_ok,
    check_numpy_scipy,
    CertificationRunner,
    CertificationReport,
    CheckResult,
)
from src.assembled_core.ops.alert_sinks import (
    dispatch_alerts,
    SlackWebhookSink,
    EmailSink,
)
from src.assembled_core.ops.alert_manager import Alert
from src.assembled_core.ml.online_gradient_boosting import OnlineAdaptiveLearner


# ---------------------------------------------------------------------------
# CertificationRunner (Step 8.36)
# ---------------------------------------------------------------------------

def test_certification_runner_creates():
    runner = CertificationRunner()
    assert isinstance(runner, CertificationRunner)


def test_build_default_runner_returns_runner():
    runner = build_default_runner()
    assert isinstance(runner, CertificationRunner)


def test_certification_runner_run_returns_report():
    runner = build_default_runner()
    report = runner.run()
    assert isinstance(report, CertificationReport)


def test_certification_report_has_counts():
    runner = build_default_runner()
    report = runner.run()
    assert report.total_checks >= 1
    assert report.passed_count + report.failed_count == report.total_checks


def test_certification_pass_rate_in_01():
    runner = build_default_runner()
    report = runner.run()
    assert 0.0 <= report.pass_rate <= 1.0


def test_check_imports_ok_returns_result():
    result = check_imports_ok()
    assert isinstance(result, CheckResult)
    assert isinstance(result.passed, bool)


def test_check_numpy_scipy_returns_result():
    result = check_numpy_scipy()
    assert isinstance(result, CheckResult)
    assert result.passed is True


def test_certification_custom_check():
    runner = CertificationRunner()
    runner.add_check("always_pass", lambda: CheckResult(name="always_pass", passed=True, message="ok"))
    report = runner.run()
    assert report.passed_count == 1
    assert report.all_passed is True


# ---------------------------------------------------------------------------
# dispatch_alerts / alert_sinks (Step 7.70)
# ---------------------------------------------------------------------------

def _make_alert(level: str = "WARNING") -> Alert:
    return Alert(level=level, source="test", message="wave42 test alert")


def test_dispatch_alerts_empty_sinks():
    alert = _make_alert()
    results = dispatch_alerts(alert, sinks=[])
    assert isinstance(results, list)
    assert len(results) == 0


def test_dispatch_alerts_list_empty_sinks():
    alerts = [_make_alert("INFO"), _make_alert("CRITICAL")]
    results = dispatch_alerts(alerts, sinks=[])
    assert isinstance(results, list)
    assert len(results) == 0


def test_slack_sink_creates():
    sink = SlackWebhookSink()
    assert isinstance(sink, SlackWebhookSink)


def test_slack_sink_skips_without_env(monkeypatch):
    monkeypatch.delenv("ASSEMBLED_SLACK_WEBHOOK_URL", raising=False)
    sink = SlackWebhookSink()
    result = sink.send(_make_alert("WARNING"))
    assert result["status"] in {"skipped", "error"}


def test_slack_sink_skips_below_min_severity():
    sink = SlackWebhookSink(min_severity="WARNING")
    result = sink.send(_make_alert("INFO"))
    assert result["status"] == "skipped"
    assert result["reason"] == "below_min_severity"


def test_email_sink_creates():
    sink = EmailSink()
    assert isinstance(sink, EmailSink)


def test_email_sink_skips_without_config(monkeypatch):
    monkeypatch.delenv("ASSEMBLED_EMAIL_HOST", raising=False)
    sink = EmailSink()
    result = sink.send(_make_alert("WARNING"))
    assert result["status"] in {"skipped", "error"}


def test_dispatch_alerts_slack_sink_skips_no_env(monkeypatch):
    monkeypatch.delenv("ASSEMBLED_SLACK_WEBHOOK_URL", raising=False)
    alert = _make_alert("WARNING")
    sink = SlackWebhookSink()
    results = dispatch_alerts(alert, sinks=[sink])
    assert len(results) == 1
    assert results[0]["status"] in {"skipped", "error"}


# ---------------------------------------------------------------------------
# OnlineAdaptiveLearner (Step 8.37)
# ---------------------------------------------------------------------------

def test_online_learner_creates():
    learner = OnlineAdaptiveLearner()
    assert isinstance(learner, OnlineAdaptiveLearner)


def test_online_learner_has_available_attr():
    learner = OnlineAdaptiveLearner()
    assert isinstance(learner.available, bool)


def test_online_learner_model_type():
    learner = OnlineAdaptiveLearner(model_type="adaptive_tree")
    assert learner.model_type == "adaptive_tree"


def test_online_learner_predict_one_returns_float():
    learner = OnlineAdaptiveLearner()
    x = np.array([0.1, 0.2, 0.3])
    result = learner.predict_one(x)
    assert isinstance(result, float)


def test_online_learner_learn_one_returns_float():
    learner = OnlineAdaptiveLearner()
    x = np.array([0.1, 0.2, 0.3])
    error = learner.learn_one(x, y=1.0)
    assert isinstance(error, float)


def test_online_learner_learn_batch():
    rng = np.random.default_rng(0)
    learner = OnlineAdaptiveLearner()
    X = rng.normal(0, 1, (20, 3))
    y = rng.normal(0, 1, 20)
    errors = learner.learn_batch(X, y)
    assert len(errors) == 20


def test_online_learner_predict_batch():
    rng = np.random.default_rng(0)
    learner = OnlineAdaptiveLearner()
    X = rng.normal(0, 1, (10, 3))
    preds = learner.predict_batch(X)
    assert len(preds) == 10


def test_online_learner_forest_type():
    learner = OnlineAdaptiveLearner(model_type="adaptive_forest")
    assert learner.model_type == "adaptive_forest"
    assert isinstance(learner.available, bool)
