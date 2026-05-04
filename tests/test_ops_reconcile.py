"""Tests for OPS-5 reconcile report and invariants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.assembled_core.ops.paper_ledger import apply_fills_to_ledger, simulate_fills
from src.assembled_core.ops.reconcile import (
    build_reconcile_report,
    write_reconcile_artifact,
)

pytestmark = [pytest.mark.unit, pytest.mark.phase6]


def test_reconcile_ok_basic_buy() -> None:
    """Reconcile report status OK for a simple buy: cash delta negative, equity finite, invariants pass."""
    as_of = "2025-01-15T12:00:00+00:00"
    ledger_before: dict[str, Any] = {
        "cash": 10000.0,
        "positions": {},
        "equity_curve": [],
    }
    orders = pd.DataFrame([{"symbol": "A", "side": "BUY", "qty": 10.0, "price": 100.0}])
    prices = pd.DataFrame({"symbol": ["A"], "close": [100.0]})
    fills = simulate_fills(orders, prices, None)
    ledger_after = apply_fills_to_ledger(ledger_before, fills)
    report = build_reconcile_report(
        as_of_utc=as_of,
        ledger_before=ledger_before,
        ledger_after=ledger_after,
        orders=orders,
        fills=fills,
        prices_latest=prices,
        cost_model_cfg={},
    )
    assert report["schema_version"] == "run.reconcile.v1"
    assert report["status"] == "OK"
    assert report["cash"]["before"] == 10000.0
    assert report["cash"]["after"] == 9000.0
    assert report["cash"]["delta"] == -1000.0
    assert report["equity"]["before"] == 10000.0
    assert report["equity"]["after"] == 10000.0  # cash 9k + 10*100
    assert report["trading"]["n_orders"] == 1
    assert report["trading"]["n_fills"] == 1
    inv_names = [i["name"] for i in report["invariants"]]
    assert "cash_non_negative" in inv_names
    assert "equity_finite" in inv_names
    assert "fills_match_orders" in inv_names
    for inv in report["invariants"]:
        assert inv["ok"] is True


def test_reconcile_fail_negative_cash() -> None:
    """When cash_after < -1e-6, status FAIL and cash_non_negative invariant false."""
    as_of = "2025-01-15T12:00:00+00:00"
    ledger_before: dict[str, Any] = {"cash": 100.0, "positions": {}, "equity_curve": []}
    # Force ledger_after to have negative cash (e.g. oversell)
    ledger_after = {"cash": -1.0, "positions": {}, "equity_curve": []}
    orders = pd.DataFrame([{"symbol": "A", "side": "BUY", "qty": 1.0, "price": 50.0}])
    prices = pd.DataFrame({"symbol": ["A"], "close": [50.0]})
    fills = simulate_fills(orders, prices, None)
    report = build_reconcile_report(
        as_of_utc=as_of,
        ledger_before=ledger_before,
        ledger_after=ledger_after,
        orders=orders,
        fills=fills,
        prices_latest=prices,
        cost_model_cfg={},
    )
    assert report["status"] == "FAIL"
    cash_inv = next(i for i in report["invariants"] if i["name"] == "cash_non_negative")
    assert cash_inv["ok"] is False
    assert "invariant_cash_non_negative_failed" in report["notes"]


def test_reconcile_writes_and_alerts_on_fail(tmp_path: Path) -> None:
    """When reconcile status is FAIL, reconcile_latest.json is written and RECONCILE_FAIL alert is added to alerts."""
    from src.assembled_core.ops.alerts import (
        compute_alerts,
        make_reconcile_fail_alert,
        write_alerts_artifact,
    )

    as_of = "2025-01-15T12:00:00+00:00"
    ledger_before: dict[str, Any] = {
        "cash": 10000.0,
        "positions": {},
        "equity_curve": [],
    }
    ledger_after = {"cash": -0.01, "positions": {}, "equity_curve": []}
    orders = pd.DataFrame(columns=["symbol", "side", "qty", "price"])
    prices = pd.DataFrame({"symbol": [], "close": []})
    if prices.empty:
        prices = pd.DataFrame({"symbol": ["A"], "close": [100.0]})
    report = build_reconcile_report(
        as_of_utc=as_of,
        ledger_before=ledger_before,
        ledger_after=ledger_after,
        orders=orders,
        fills=[],
        prices_latest=prices,
        cost_model_cfg={},
    )
    assert report["status"] == "FAIL"
    write_reconcile_artifact(tmp_path, report)
    reconcile_path = tmp_path / "reconcile_latest.json"
    assert reconcile_path.exists()
    data = json.loads(reconcile_path.read_text(encoding="utf-8"))
    assert data["status"] == "FAIL"

    # Simulate alert append when reconcile fails
    run_kpis = {"generated_utc": as_of}
    reasons: dict[str, Any] = {}
    diff: dict[str, Any] = {"notes": [], "summary": {}}
    cfg: dict[str, Any] = {
        "alerts": {
            "enabled": True,
            "severity_map": {"info": 0, "warn": 1, "critical": 2},
        }
    }
    alerts_list = compute_alerts(run_kpis, reasons, diff, cfg)
    alerts_list = list(alerts_list)
    alerts_list.append(make_reconcile_fail_alert(as_of))
    severity_map = cfg.get("alerts", {}).get("severity_map") or {
        "info": 0,
        "warn": 1,
        "critical": 2,
    }
    alerts_list.sort(
        key=lambda a: (-severity_map.get(a["level"], 0), a["kind"], a["alert_id"])
    )
    write_alerts_artifact(tmp_path, alerts_list, as_of, cfg)
    alerts_path = tmp_path / "alerts_latest.json"
    assert alerts_path.exists()
    alerts_data = json.loads(alerts_path.read_text(encoding="utf-8"))
    reconcile_fail = [a for a in alerts_data["items"] if a["kind"] == "RECONCILE_FAIL"]
    assert len(reconcile_fail) == 1
    assert reconcile_fail[0]["level"] == "critical"
