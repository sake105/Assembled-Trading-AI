"""Regression tests for Diagnostik A10 (BLOCKER): silent alert-rule drop.

`reconciliation_fail`, `reconciliation_warn` and `circuit_breaker_tripped` were
fired by the code (accounting/reconciliation.py, risk/circuit_breaker.py) but had
NO matching rule in `configs/alerting.yaml`. `AlertManager.fire()` returns False
for an unknown rule and logs only at DEBUG, so those CRITICAL/WARN alerts were
silently dropped and reached no channel.

These tests fail on the pre-fix repo and pass once the rules exist.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.assembled_core.ops.alerting import AlertManager

pytestmark = [pytest.mark.unit, pytest.mark.fast]

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

_CRED_ENV = (
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "ALERT_EMAIL_TO",
    "SMTP_HOST",
    "SMTP_USER",
    "SMTP_PASS",
)

# Rule names + the exact context the call sites pass (regression anchors for A10).
PREVIOUSLY_MISSING = [
    (
        "reconciliation_fail",
        {
            "cash_diff_bps": 12.3,
            "max_qty_diff": 5,
            "violation_count": 2,
            "first_violation": "cash",
        },
    ),
    (
        "reconciliation_warn",
        {
            "cash_diff_bps": 4.0,
            "max_qty_diff": 1,
            "violation_count": 1,
            "first_violation": "qty",
        },
    ),
    (
        "circuit_breaker_tripped",
        {"ratio": 3.2, "threshold": 2.0, "trip_count": 1},
    ),
    # F-senior-11 (2026-08-16): aus scripts/ops_watchdog.py gefeuert — der
    # SRC-Scanner dieses Tests sieht scripts/ nicht, deshalb hier als Anker
    # verankert (Existenz + Message-Rendering).
    (
        "pull_log_errors",
        {
            "log": "pull_log_yfinance_x.json",
            "n_error": 6,
            "requested": 10,
            "ratio": 0.6,
            "threshold": 0.5,
        },
    ),
]


@pytest.mark.parametrize("rule_name, ctx", PREVIOUSLY_MISSING)
def test_fired_rule_is_dispatched_not_dropped(rule_name, ctx, monkeypatch):
    """fire() must return True (rule found + dispatched), not False (silent drop)."""
    # No telegram/email creds -> dispatch is a no-op (no network), fire still returns True.
    for var in _CRED_ENV:
        monkeypatch.delenv(var, raising=False)
    mgr = AlertManager()  # loads the real configs/alerting.yaml (cwd == repo root)
    assert mgr.fire(rule_name, ctx) is True, (
        f"rule '{rule_name}' was dropped — missing from configs/alerting.yaml"
    )


@pytest.mark.parametrize("rule_name, ctx", PREVIOUSLY_MISSING)
def test_fired_rule_message_renders_without_missing_key(rule_name, ctx, monkeypatch):
    """The rule message template must consume exactly the keys the call site passes."""
    for var in _CRED_ENV:
        monkeypatch.delenv(var, raising=False)
    mgr = AlertManager()
    rule = mgr._find_rule(rule_name)
    assert rule is not None, f"rule '{rule_name}' not configured"
    rendered = rule["message"].format(**ctx)
    assert "missing key" not in rendered
    assert "{" not in rendered  # no unsubstituted placeholders


def _fired_rule_names() -> set[str]:
    """Collect every literal rule name passed to AlertManager().fire('<name>', ...) in src/."""
    pat = re.compile(r"""\.fire\(\s*["']([a-zA-Z0-9_]+)["']""")
    names: set[str] = set()
    for py in SRC.rglob("*.py"):
        try:
            text = py.read_text(encoding="utf-8")
        except OSError:
            continue
        if "ops.alerting" not in text:  # only files that use the AlertManager
            continue
        names.update(pat.findall(text))
    return names


def test_every_fired_alert_rule_has_a_config_entry():
    """Coverage guard: no rule fired in src/ may be absent from alerting.yaml."""
    mgr = AlertManager()
    configured = {r.get("name") for r in mgr._rules}
    fired = _fired_rule_names()
    assert fired, (
        "no AlertManager fire('<rule>') call sites found — regex/layout changed?"
    )
    missing = sorted(n for n in fired if n not in configured)
    assert not missing, (
        "alert rules fired in src/ but missing from configs/alerting.yaml "
        f"(would be silently dropped): {missing}"
    )
