"""Tests for KPI artifact writer and shadow-mode behavior (OPS-1)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.assembled_core.ops.kpi_artifacts import maybe_execute_orders, write_run_kpis

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_write_run_kpis_contains_keys(tmp_path) -> None:
    """KPI writer should emit run_kpis.json with expected top-level keys."""
    # Minimal ctx/result/policy objects
    ctx = SimpleNamespace(
        risk_state={"state": "ACTIVE"},
        news_geo={"geo_score": 2, "geo_confidence": 0.9},
        market_stress={"stress_ok": True},
        news_triggers=SimpleNamespace(summary={"n_triggers": 3}),
    )
    meta = {
        "turnover_budget": {"scale_factor": 0.8, "estimated_turnover": 0.2},
        "profit_lock": {"multiplier": 0.9},
    }
    target_positions = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "target_weight": [0.6, 0.4],
            "target_qty": [60.0, 40.0],
        }
    )
    result = SimpleNamespace(meta=meta, target_positions=target_positions)
    policy = {
        "georisk_overlay": {
            "enabled": True,
            "mapping": {"ACTIVE": {"multiplier": 0.7, "hedge": {"enabled": False}}},
            "by_geo_score": {},
            "confidence_floor": 0.6,
        }
    }

    out_dir = tmp_path / "run1"
    path = write_run_kpis(out_dir, ctx=ctx, result=result, policy=policy, mode="shadow")
    assert path.exists()

    data = json.loads(path.read_text(encoding="utf-8"))
    # Basic schema keys
    assert data["schema_version"] == "run.kpis.v1"
    assert data["mode"] == "shadow"
    assert "multipliers" in data
    assert "turnover_budget" in data
    assert "profit_lock" in data
    assert "targets_summary" in data
    assert data["targets_summary"]["n_targets"] == 2
    assert pytest.approx(data["targets_summary"]["sum_target_weight"]) == 1.0


def test_run_kpis_contains_news_and_disclosures_trigger_summaries(tmp_path) -> None:
    """OPS-14: write_run_kpis includes news_triggers_summary and disclosures_triggers_summary from result.meta."""
    ctx = SimpleNamespace(
        risk_state={"state": "WATCH"},
        news_geo=None,
        market_stress=None,
        news_triggers=None,
    )
    meta = {
        "news_triggers_summary": {
            "count": 5,
            "max_severity": 2,
            "count_sev1plus": 3,
            "count_sev2plus": 1,
        },
        "disclosures_triggers_summary": {
            "count": 10,
            "max_severity": 1,
            "count_sev1plus": 10,
            "count_sev2plus": 0,
        },
    }
    result = SimpleNamespace(meta=meta, target_positions=pd.DataFrame())
    out_dir = tmp_path / "run1"
    path = write_run_kpis(out_dir, ctx=ctx, result=result, policy={}, mode="paper")
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("news_triggers_summary") == meta["news_triggers_summary"]
    assert (
        data.get("disclosures_triggers_summary") == meta["disclosures_triggers_summary"]
    )


def test_run_kpis_includes_news_debug_funnel(tmp_path) -> None:
    """NEWS-DEBUG-1: write_run_kpis includes news_debug_funnel from result.meta."""
    ctx = SimpleNamespace(
        risk_state={"state": "WATCH"},
        news_geo=None,
        market_stress=None,
        news_triggers=None,
    )
    funnel_counts = {
        "raw_items_count": 10,
        "normalized_events_count": 9,
        "deduped_events_count": 8,
        "clusters_count": 3,
        "candidate_triggers_count": 1,
        "triggers_count": 1,
    }
    meta = {"news_debug_funnel": funnel_counts}
    result = SimpleNamespace(meta=meta, target_positions=pd.DataFrame())
    path = write_run_kpis(tmp_path, ctx=ctx, result=result, policy={}, mode="paper")
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("news_debug_funnel") == funnel_counts


def test_shadow_mode_does_not_execute_orders() -> None:
    """shadow mode must be a no-op for additional order execution."""
    orders = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2025-01-01", tz="UTC")],
            "symbol": ["A"],
            "side": ["BUY"],
            "qty": [10.0],
            "price": [100.0],
        }
    )

    # maybe_execute_orders currently returns orders unchanged; important part is that
    # shadow mode does not call any external fill/ledger simulation.
    out = maybe_execute_orders("shadow", orders)
    pd.testing.assert_frame_equal(out, orders)


def test_run_kpis_carries_equity_start_of_cycle_and_auto_dd(tmp_path) -> None:
    """2026-08-09: start-of-cycle equity + the auto-DD trace live in
    run_kpis.json (NOT in the run index — its final_equity column has
    post-fill semantics, E-137). 0.0 stays valid (no `or`-swallowing) and
    without a producer the key is explicitly None (E-047 contract)."""

    class _Ctx:
        current_equity = 0.0

    class _Result:
        meta = {"auto_dd_kill_switch": {"level": "soft", "applied": True}}
        target_positions = None

    p = write_run_kpis(
        output_dir=tmp_path, ctx=_Ctx(), result=_Result(), policy={}, mode="paper"
    )
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["equity_start_of_cycle"] == 0.0
    assert d["auto_dd_kill_switch"]["level"] == "soft"
    assert d["auto_dd_kill_switch"]["applied"] is True

    class _CtxOhneEquity:
        pass

    class _ResultLeer:
        meta: dict = {}
        target_positions = None

    p2 = write_run_kpis(
        output_dir=tmp_path / "no_producer",
        ctx=_CtxOhneEquity(),
        result=_ResultLeer(),
        policy={},
        mode="paper",
    )
    d2 = json.loads(p2.read_text(encoding="utf-8"))
    assert d2["equity_start_of_cycle"] is None
    assert d2["auto_dd_kill_switch"] is None
