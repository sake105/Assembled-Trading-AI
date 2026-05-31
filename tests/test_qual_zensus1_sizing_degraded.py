"""QUAL/Zensus-1 sub-commit B: _tc_sizing overlays become OBSERVABLE on failure.

The exposure overlays in ``_tc_sizing.py`` (vol_targeting, trailing_stops,
correlation_guard, crash_prediction_cap, plus the size_positions guards:
policy-load, halt-check, buying-power, pre-earnings) were each wrapped in
``try: <step> ... except Exception: log.debug("... skipped")``. At prod log level
a DEBUG swallow is invisible — a protective overlay that silently no-ops looks
identical to one that ran.

This pins that a real overlay failure routes through ``_record_degraded_step``
(WARN + structured ``meta['degraded_steps']`` trail) instead of a DEBUG swallow,
WITHOUT blocking the sizing path (these are fail-open overlays — making any one
genuinely fail-closed is a separate per-step decision, out of scope here). Two
representative sites are exercised: one module-top-import overlay
(``correlation_guard``) and one ``meta``-param overlay using the module logger
(``crash_prediction_cap``).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd

import src.assembled_core.pipeline._tc_sizing as tc_sizing
from src.assembled_core.pipeline._tc_sizing import (
    _sp_apply_correlation_guard,
    _sp_apply_crash_cap,
)


def _two_row_targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "target_weight": [0.5, 0.5],
            "target_qty": [100.0, 80.0],
        }
    )


def test_correlation_guard_failure_recorded_not_blocking(monkeypatch, caplog) -> None:
    """Force the correlation guard to raise. The failure must land in
    meta['degraded_steps'] + a WARNING — and the positions must survive
    unchanged (soft overlay, not an order-blocking gate)."""

    def _boom(*a, **k):
        raise RuntimeError("corr boom")

    monkeypatch.setattr(tc_sizing, "apply_correlation_guard", _boom)

    meta: dict = {}
    tp = _two_row_targets()
    ctx = SimpleNamespace(as_of=None, capital=100_000.0)

    with caplog.at_level(logging.WARNING):
        out = _sp_apply_correlation_guard(tp, tp.copy(), {}, ctx, meta)

    corr_entries = [
        s for s in meta.get("degraded_steps", []) if s["step"] == "correlation_guard"
    ]
    assert len(corr_entries) == 1
    assert "corr boom" in corr_entries[0]["error"]
    assert any(
        "[DEGRADED]" in r.message and "correlation_guard" in str(r.args)
        for r in caplog.records
    )
    # non-blocking: positions returned unchanged
    assert len(out) == 2
    assert list(out["symbol"]) == ["AAPL", "MSFT"]


def test_crash_cap_failure_recorded_not_blocking(caplog) -> None:
    """A non-numeric crash_probability makes the crash-cap overlay raise inside
    its try. It must be recorded as 'crash_prediction_cap' + WARNING, and the
    positions must survive unchanged."""
    meta: dict = {"crash_prediction": {"crash_probability": "boom"}}
    tp = _two_row_targets()

    with caplog.at_level(logging.WARNING):
        out = _sp_apply_crash_cap(tp, {}, meta, None)

    crash_entries = [
        s for s in meta.get("degraded_steps", []) if s["step"] == "crash_prediction_cap"
    ]
    assert len(crash_entries) == 1
    assert any(
        "[DEGRADED]" in r.message and "crash_prediction_cap" in str(r.args)
        for r in caplog.records
    )
    assert len(out) == 2
    assert list(out["symbol"]) == ["AAPL", "MSFT"]
