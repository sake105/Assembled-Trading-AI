"""E-059 #2 auditor follow-up: run_paper_daily_one → ctx.execution_mode wiring.

Pins the paper_runner entry normalization (review F-senior-3): the routing-mode
label is validated at the TOP of ``run_paper_daily_one`` — BEFORE it is frozen
into the ``TradingContext`` — so an unknown label can never masquerade as a
real routing mode in the KPI/manifest/journal/heartbeat artifacts, and a valid
"broker" label reaches the ctx verbatim.

E2E approach (variant (a) of the task): the real ``run_trading_cycle`` is
heavyweight (feature build, risk controls, policy loads), so it is replaced by
a capturing stub via the module attribute
``src.assembled_core.pipeline.trading_cycle_v2.run_trading_cycle`` —
``run_paper_daily_one`` imports it function-locally at call time, so patching
the source module is sufficient. ``_load_pilot_policy_fail_fast`` (repo
policy.yaml read), ``_prd_intel_summaries`` and ``_prd_write_artifacts`` are
stubbed to keep the run off real config/intel paths; the artifact stub
additionally records the ``execution_mode`` forwarded downstream. Runs use
mode="shadow" (no ledger) and write only under ``tmp_path``.

Honest coverage note: the broker case asserts LABEL wiring only. With
``broker_adapter=None`` the ctx keeps "broker" by design — the adapter-None →
sim ROUTING fallback lives in ``_prd_paper_fills_and_ledger`` (paper mode) and
is not exercised here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

import src.assembled_core.ops.paper_runner as paper_runner
import src.assembled_core.pipeline.trading_cycle_v2 as trading_cycle_v2
from src.assembled_core.pipeline.trading_cycle_shared import TradingCycleResult

pytestmark = pytest.mark.fast


def _run_one(monkeypatch, tmp_path: Path, execution_mode: str) -> tuple[int, dict]:
    """Drive run_paper_daily_one (mode="shadow") with a stubbed cycle.

    Returns ``(exit_code, captured)`` where ``captured`` holds the ctx seen by
    the stubbed run_trading_cycle and the execution_mode forwarded to
    ``_prd_write_artifacts``.
    """
    captured: dict = {}

    def _fake_cycle(ctx):
        captured["ctx"] = ctx
        return TradingCycleResult(status="success")

    monkeypatch.setattr(trading_cycle_v2, "run_trading_cycle", _fake_cycle)
    monkeypatch.setattr(
        paper_runner, "_load_pilot_policy_fail_fast", lambda context: {}
    )
    monkeypatch.setattr(
        paper_runner, "_prd_intel_summaries", lambda result, paper_cfg, root: None
    )

    def _fake_write_artifacts(**kwargs):
        captured["artifacts_execution_mode"] = kwargs.get("execution_mode")

    monkeypatch.setattr(paper_runner, "_prd_write_artifacts", _fake_write_artifacts)

    as_of = pd.Timestamp("2025-06-26", tz="UTC")
    prices = pd.DataFrame({"timestamp": [as_of], "symbol": ["AAPL"], "close": [100.0]})
    out_dir = tmp_path / "runs" / "2025-06-26"
    out_dir.mkdir(parents=True)
    app_cfg: dict = {"paper_runner": {"strategy": {"name": "none"}}}

    exit_code, _reconcile = paper_runner.run_paper_daily_one(
        as_of,
        out_dir,
        "shadow",
        app_cfg,
        prices,
        root=tmp_path,
        execution_mode=execution_mode,
    )
    return exit_code, captured


# --------------------------------------------------------------------------- #
# (a) unknown label → normalized to "sim" BEFORE ctx construction + log.error
# --------------------------------------------------------------------------- #
def test_unknown_execution_mode_normalized_to_sim(
    monkeypatch, tmp_path, caplog
) -> None:
    with caplog.at_level(logging.ERROR, logger="src.assembled_core.ops.paper_runner"):
        exit_code, cap = _run_one(monkeypatch, tmp_path, "bogus")

    assert exit_code == 0
    assert "ctx" in cap, "stubbed run_trading_cycle must have been called"
    assert cap["ctx"].execution_mode == "sim"
    assert cap["artifacts_execution_mode"] == "sim"
    assert any(
        "Unknown execution_mode" in rec.getMessage() and "bogus" in rec.getMessage()
        for rec in caplog.records
    ), "normalization must be LOUD (log.error), not silent"


# --------------------------------------------------------------------------- #
# (b) valid "broker" label → reaches ctx.execution_mode verbatim, no error log
# --------------------------------------------------------------------------- #
def test_broker_execution_mode_reaches_ctx(monkeypatch, tmp_path, caplog) -> None:
    with caplog.at_level(logging.ERROR, logger="src.assembled_core.ops.paper_runner"):
        exit_code, cap = _run_one(monkeypatch, tmp_path, "broker")

    assert exit_code == 0
    assert cap["ctx"].execution_mode == "broker"
    assert cap["artifacts_execution_mode"] == "broker"
    assert not any(
        "Unknown execution_mode" in rec.getMessage() for rec in caplog.records
    ), "a valid label must not trigger the normalization error"


# --------------------------------------------------------------------------- #
# (c) default path stays "sim" (baseline, guards against default drift)
# --------------------------------------------------------------------------- #
def test_default_execution_mode_is_sim(monkeypatch, tmp_path) -> None:
    exit_code, cap = _run_one(monkeypatch, tmp_path, "sim")

    assert exit_code == 0
    assert cap["ctx"].execution_mode == "sim"
    assert cap["artifacts_execution_mode"] == "sim"
