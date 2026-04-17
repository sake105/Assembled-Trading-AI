"""Phase 6 regression tests for hardened reconciliation + SLO alerting.

Covers:

* :class:`ReconcileSLO` default thresholds preserved
* :func:`evaluate_reconcile_slo` classifies ok / warn / fail correctly
* Engine noop reconcile (state-as-broker) → severity=ok, no alert file
* Engine reconcile with diverging shadow broker cash → alert file written
* Alert JSON schema is stable and contains violations
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.assembled_core.accounting.reconciliation import (
    ReconcileSLO,
    evaluate_reconcile_slo,
)
from src.assembled_core.execution.unified_paper_engine import (
    UnifiedPaperConfig,
    UnifiedPaperEngine,
)


def _make_engine(
    tmp_path: Path,
    *,
    shadow_broker=None,
    slo: ReconcileSLO | None = None,
) -> UnifiedPaperEngine:
    cfg = UnifiedPaperConfig(
        seed_capital=1_000_000.0,
        state_dir=tmp_path / "state",
        ledger_dir=tmp_path / "ledger",
        lifecycle_dir=tmp_path / "lifecycle",
        reconcile_alerts_dir=tmp_path / "alerts",
        reconcile_slo=slo,
        shadow_broker=shadow_broker,
        enable_reconciliation=True,
        enable_kill_switch=False,
        enable_fat_finger=False,
        run_id="recon_test",
    )
    eng = UnifiedPaperEngine(cfg)
    eng._state = {"cash": 1_000_000.0, "positions": {}, "cost_basis": {}}
    return eng


# --- ReconcileSLO ------------------------------------------------------------


def test_slo_defaults_preserved() -> None:
    slo = ReconcileSLO()
    assert slo.cash_diff_bps_warn == 5.0
    assert slo.cash_diff_bps_fail == 25.0
    assert slo.position_qty_diff_warn == 1.0
    assert slo.position_qty_diff_fail == 10.0
    assert slo.fill_rate_min_warn == 0.80
    assert slo.fill_rate_min_fail == 0.50
    assert slo.slippage_p99_bps_warn == 30.0
    assert slo.slippage_p99_bps_fail == 100.0


def test_evaluate_slo_clean_is_ok() -> None:
    verdict = evaluate_reconcile_slo(
        cash_diff=0.0,
        broker_cash=1_000_000.0,
        max_qty_diff=0.0,
        fill_rate=0.95,
        slippage_p99_bps=5.0,
        slo=ReconcileSLO(),
    )
    assert verdict["severity"] == "ok"
    assert verdict["violations"] == []


def test_evaluate_slo_warn_only() -> None:
    # 6 bps cash diff → warn (above 5bps warn, below 25bps fail)
    verdict = evaluate_reconcile_slo(
        cash_diff=600.0,
        broker_cash=1_000_000.0,
        max_qty_diff=0.0,
        fill_rate=None,
        slippage_p99_bps=None,
        slo=ReconcileSLO(),
    )
    assert verdict["severity"] == "warn"
    assert any(v["metric"] == "cash_diff_bps" for v in verdict["violations"])
    assert all(v["severity"] == "warn" for v in verdict["violations"])


def test_evaluate_slo_fail_wins() -> None:
    # 30 bps cash diff + 11-share qty diff → both fail
    verdict = evaluate_reconcile_slo(
        cash_diff=3_000.0,
        broker_cash=1_000_000.0,
        max_qty_diff=11.0,
        fill_rate=0.4,
        slippage_p99_bps=150.0,
        slo=ReconcileSLO(),
    )
    assert verdict["severity"] == "fail"
    assert any(v["severity"] == "fail" for v in verdict["violations"])


# --- Engine integration ------------------------------------------------------


def test_engine_reconcile_noop_is_ok(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path)
    eng._state["positions"] = {"AAA": 100.0}
    verdict = eng._run_reconciliation("2025-01-15")
    assert verdict is not None
    assert verdict["severity"] == "ok"
    # No alert file written
    alerts = list((tmp_path / "alerts").glob("*.json")) if (tmp_path / "alerts").exists() else []
    assert alerts == []


class _DivergingBroker:
    """Fake broker whose snapshot diverges from engine state."""

    def __init__(self, cash: float, positions: list[dict]) -> None:
        self._cash = cash
        self._positions = positions

    def get_snapshot(self) -> dict:
        return {"cash": self._cash, "positions": self._positions}


def test_engine_reconcile_cash_fail_writes_alert(tmp_path: Path) -> None:
    # Engine thinks $1M cash; broker says $950_000 → 500 bps diff → fail.
    broker = _DivergingBroker(cash=950_000.0, positions=[])
    eng = _make_engine(tmp_path, shadow_broker=broker)
    verdict = eng._run_reconciliation("2025-01-15")
    assert verdict is not None
    assert verdict["severity"] == "fail"
    alerts = list((tmp_path / "alerts").glob("*.json"))
    assert len(alerts) == 1
    import json
    payload = json.loads(alerts[0].read_text())
    assert payload["severity"] == "fail"
    assert payload["run_id"] == "recon_test"
    assert payload["date"] == "2025-01-15"
    assert any(v["metric"] == "cash_diff_bps" for v in payload["violations"])


def test_engine_reconcile_position_diff_fail_writes_alert(tmp_path: Path) -> None:
    # Engine holds 100 shares; broker says 85 → 15-share diff (fail > 10).
    broker = _DivergingBroker(
        cash=1_000_000.0,
        positions=[{"symbol": "AAA", "qty": 85.0}],
    )
    eng = _make_engine(tmp_path, shadow_broker=broker)
    eng._state["positions"] = {"AAA": 100.0}
    verdict = eng._run_reconciliation("2025-01-15")
    assert verdict is not None
    assert verdict["severity"] == "fail"
    assert verdict["max_qty_diff"] == 15.0
    alerts = list((tmp_path / "alerts").glob("*.json"))
    assert len(alerts) == 1
