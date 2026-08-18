# -*- coding: utf-8 -*-
"""Wiring-Pin fuer Audit-Plan 3.1 (S1-M4a, 2026-08-17).

Der Orchestrator-QA-Step muss dem Leakage-Gate (Gate 8) ein ECHTES Frame
uebergeben, wenn output/events_earnings.parquet existiert — bis 3.1 lief das
Gate im Pilot-Pfad dauerhaft SKIPPED (kein einziger laufender Leakage-Check
trotz fertigem Gate). Eine stille Rueck-Regression auf den Vor-3.1-Zustand
macht dieser Spy-Test rot.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

import src.assembled_core.pipeline.orchestrator as orch

pytestmark = pytest.mark.fast


def _write_fixtures(base):
    ts = pd.date_range("2026-01-05", periods=30, freq="B", tz="UTC")
    eq = pd.DataFrame({"timestamp": ts, "equity": 10_000.0 + pd.RangeIndex(30) * 5.0})
    eq.to_csv(base / "portfolio_equity_1d.csv", index=False)
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-10"], utc=True),
            "symbol": ["AAPL"],
            "disclosure_date": pd.to_datetime(["2026-01-10"], utc=True),
            "eps_surprise_pct": [4.2],
        }
    ).to_parquet(base / "events_earnings.parquet", index=False)


def test_qa_step_passes_real_frame_to_gate8(tmp_path, monkeypatch):
    _write_fixtures(tmp_path)
    seen: dict = {}
    real = orch.evaluate_all_gates

    def _spy(metrics, *args, **kwargs):
        seen["feature_df"] = kwargs.get("feature_df")
        seen["feature_col"] = kwargs.get("leakage_feature_col")
        return real(metrics, *args, **kwargs)

    # Top-level from-Import: das gebundene Symbol im Orchestrator patchen.
    monkeypatch.setattr(orch, "evaluate_all_gates", _spy)
    out = orch._eo_step_qa("1d", tmp_path, 10_000.0, None, None, None)
    assert out.get("qa_gate_result") is not None
    assert seen.get("feature_df") is not None, (
        "Gate 8 bekam KEIN Frame — stille Rueck-Regression auf den "
        "Vor-3.1-Zustand (dauerhaft SKIPPED)"
    )
    assert len(seen["feature_df"]) == 1
    assert seen["feature_col"] == "eps_surprise_pct"


def test_qa_step_skips_gate8_honestly_without_parquet(tmp_path, monkeypatch):
    """Ohne events_earnings.parquet: Gate bleibt SKIPPED (kein Spaltenraten,
    kein qa_block aus einer Buchhaltungsluecke)."""
    ts = pd.date_range("2026-01-05", periods=30, freq="B", tz="UTC")
    pd.DataFrame(
        {"timestamp": ts, "equity": 10_000.0 + pd.RangeIndex(30) * 5.0}
    ).to_csv(tmp_path / "portfolio_equity_1d.csv", index=False)
    seen: dict = {}
    real = orch.evaluate_all_gates

    def _spy(metrics, *args, **kwargs):
        seen["called_with_frame"] = kwargs.get("feature_df") is not None
        return real(metrics, *args, **kwargs)

    monkeypatch.setattr(orch, "evaluate_all_gates", _spy)
    logging.getLogger("assembled").setLevel(logging.INFO)
    out = orch._eo_step_qa("1d", tmp_path, 10_000.0, None, None, None)
    assert out.get("qa_gate_result") is not None
    assert seen.get("called_with_frame") is False
