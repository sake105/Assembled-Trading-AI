"""Tests for ops/alert_sinks.py (Sprint 4 / Plan C14)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.ops.alert_manager import Alert  # noqa: E402
from src.assembled_core.ops.alert_sinks import (  # noqa: E402
    EmailSink,
    SlackWebhookSink,
    dispatch_alerts,
)


def _alert(level: str = "WARNING", details: dict | None = None) -> Alert:
    return Alert(
        level=level,  # type: ignore[arg-type]
        source="unit_test",
        message="something happened",
        details=details or {"pnl": -123.45},
    )


# ---------- SlackWebhookSink ----------


def test_slack_skipped_below_min_severity() -> None:
    sink = SlackWebhookSink(min_severity="CRITICAL")
    res = sink.send(_alert(level="WARNING"))
    assert res["status"] == "skipped"
    assert res["reason"] == "below_min_severity"


def test_slack_skipped_when_env_unset(monkeypatch) -> None:
    monkeypatch.delenv("ASSEMBLED_SLACK_WEBHOOK_URL", raising=False)
    sink = SlackWebhookSink(min_severity="INFO")
    res = sink.send(_alert(level="CRITICAL"))
    assert res["status"] == "skipped"
    assert "env_var" in res["reason"]


def test_slack_sends_when_env_set(monkeypatch) -> None:
    monkeypatch.setenv("ASSEMBLED_SLACK_WEBHOOK_URL", "https://example.invalid/hook")

    captured: dict = {}

    class _FakeResp:
        status_code = 200

    class _FakeRequests:
        def post(self, url, json, timeout):  # noqa: A002
            captured["url"] = url
            captured["json"] = json
            captured["timeout"] = timeout
            return _FakeResp()

    fake = _FakeRequests()
    monkeypatch.setitem(sys.modules, "requests", fake)

    sink = SlackWebhookSink(min_severity="INFO")
    res = sink.send(_alert(level="WARNING"))
    assert res["status"] == "sent"
    assert res["http_status"] == 200
    assert captured["url"] == "https://example.invalid/hook"
    assert "attachments" in captured["json"]


def test_slack_handles_post_exception(monkeypatch) -> None:
    monkeypatch.setenv("ASSEMBLED_SLACK_WEBHOOK_URL", "https://example.invalid/hook")

    class _FakeRequests:
        def post(self, *a, **kw) -> None:
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "requests", _FakeRequests())
    sink = SlackWebhookSink(min_severity="INFO")
    res = sink.send(_alert(level="WARNING"))
    assert res["status"] == "error"
    assert "post_failed" in res["reason"]


# ---------- EmailSink ----------


def test_email_skipped_below_severity() -> None:
    sink = EmailSink(min_severity="CRITICAL", to_addrs=["ops@example.com"])
    res = sink.send(_alert(level="WARNING"))
    assert res["status"] == "skipped"


def test_email_skipped_without_recipients() -> None:
    sink = EmailSink(min_severity="INFO", to_addrs=[])
    res = sink.send(_alert(level="CRITICAL"))
    assert res["status"] == "skipped"
    assert res["reason"] == "no_recipients"


def test_email_skipped_without_host(monkeypatch) -> None:
    monkeypatch.delenv("ASSEMBLED_SMTP_HOST", raising=False)
    sink = EmailSink(min_severity="INFO", to_addrs=["ops@example.com"])
    res = sink.send(_alert(level="CRITICAL"))
    assert res["status"] == "skipped"
    assert "ASSEMBLED_SMTP_HOST" in res["reason"]


def test_email_sends_via_fake_smtp(monkeypatch) -> None:
    monkeypatch.setenv("ASSEMBLED_SMTP_HOST", "smtp.example.invalid")
    monkeypatch.setenv("ASSEMBLED_SMTP_PORT", "2525")

    calls: dict = {"sendmail": [], "login": False, "starttls": False}

    class _FakeSMTP:
        def __init__(self, host, port, timeout):
            calls["host"] = host
            calls["port"] = port
            calls["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            calls["starttls"] = True

        def login(self, user, password):
            calls["login"] = True

        def sendmail(self, from_addr, to_addrs, msg):
            calls["sendmail"].append((from_addr, tuple(to_addrs), len(msg)))

    import smtplib

    monkeypatch.setattr(smtplib, "SMTP", _FakeSMTP)

    sink = EmailSink(
        min_severity="INFO",
        to_addrs=["ops@example.com", "oncall@example.com"],
        use_tls=True,
    )
    res = sink.send(_alert(level="CRITICAL"))
    assert res["status"] == "sent"
    assert res["recipients"] == 2
    assert calls["starttls"] is True
    assert calls["host"] == "smtp.example.invalid"
    assert calls["port"] == 2525
    assert len(calls["sendmail"]) == 1


# ---------- dispatch_alerts ----------


def test_dispatch_handles_sink_exception(monkeypatch) -> None:
    class _BadSink:
        min_severity = "INFO"

        def send(self, alert):
            raise RuntimeError("nope")

    results = dispatch_alerts(_alert(level="CRITICAL"), [_BadSink()])
    assert len(results) == 1
    assert results[0]["status"] == "error"
    assert "sink_exception" in results[0]["reason"]


def test_dispatch_single_vs_list_equivalence() -> None:
    class _NoopSink:
        min_severity = "INFO"

        def send(self, alert):
            return {"status": "sent"}

    single = dispatch_alerts(_alert(), [_NoopSink()])
    lst = dispatch_alerts([_alert()], [_NoopSink()])
    assert len(single) == len(lst) == 1
    assert single[0]["status"] == "sent"
