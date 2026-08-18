# -*- coding: utf-8 -*-
"""Regressionstests fuer das corr-regime-Gate (Audit-Plan 4.2, 2026-08-17).

Befund (Audit §5, VERDACHT bestaetigt): ``detect_correlation_regime_shift``
skalierte ``target_qty`` UNGATED und ohne Shadow-Aufzeichnung — direkt neben
dem shadow-gegateten Correlation-Guard. Zusaetzlich wurde
``meta["correlation_regime_shift"]`` nie geschrieben, obwohl
``_sp_check_rebalance`` es liest: der corr_spiked-Rebalance-Trigger war
strukturell inert.
"""

from __future__ import annotations

import pandas as pd
import pytest

import src.assembled_core.pipeline._tc_sizing as tc_sizing
from src.assembled_core.pipeline.trading_cycle_shared import TradingContext

pytestmark = pytest.mark.fast


def _setup(monkeypatch, *, detected: bool, scale: float = 0.5):
    ts = pd.Timestamp("2025-06-26", tz="UTC")
    prices = pd.DataFrame(
        {
            "timestamp": [ts] * 2,
            "symbol": ["AAPL", "MSFT"],
            "close": [100.0, 200.0],
        }
    )
    ctx = TradingContext(prices=prices, as_of=ts, write_outputs=False)
    ctx.capital = 10_000.0
    targets = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "target_weight": [0.5, 0.5],
            "target_qty": [100.0, 40.0],
        }
    )
    # Guard selbst neutral halten (keine Anpassungen) — getestet wird NUR
    # der Regime-Shift-Zweig.
    monkeypatch.setattr(
        tc_sizing,
        "apply_correlation_guard",
        lambda w, p, pol: (w, []),
    )
    monkeypatch.setattr(
        tc_sizing,
        "detect_correlation_regime_shift",
        lambda *a, **k: {
            "regime_shift_detected": detected,
            "exposure_scale": scale,
        },
    )
    return targets, prices, ctx


def test_meta_always_written_for_rebalance_consumer(monkeypatch):
    """meta['correlation_regime_shift'] muss IMMER gesetzt werden — der
    _sp_check_rebalance-Consumer las bis zum Fix ein nie geschriebenes Feld."""
    targets, prices, ctx = _setup(monkeypatch, detected=False)
    meta: dict = {}
    tc_sizing._sp_apply_correlation_guard(targets, prices, {}, ctx, meta)
    assert "correlation_regime_shift" in meta
    assert meta["correlation_regime_shift"]["regime_shift_detected"] is False


def test_shadow_default_records_but_does_not_scale(monkeypatch, tmp_path):
    """Part-D-Default: shadow_only=True — Skalierung wird aufgezeichnet,
    aber target_weight/target_qty bleiben unangetastet."""
    targets, prices, ctx = _setup(monkeypatch, detected=True, scale=0.5)
    recorded: list = []
    monkeypatch.setattr(
        tc_sizing, "_record_degraded_step", tc_sizing._record_degraded_step
    )
    import src.assembled_core.ops.shadow_recorder as sr

    monkeypatch.setattr(
        sr, "record_shadow", lambda name, payload, **kw: recorded.append((name, kw))
    )
    meta: dict = {}
    out = tc_sizing._sp_apply_correlation_guard(
        targets, prices, {}, ctx, meta
    )  # policy leer -> shadow_only default True
    row = out.set_index("symbol")
    assert row.loc["AAPL", "target_qty"] == pytest.approx(100.0)  # NICHT skaliert
    assert row.loc["AAPL", "target_weight"] == pytest.approx(0.5)
    assert meta["correlation_regime_shift"]["exposure_scale"] == pytest.approx(0.5)
    assert recorded and recorded[0][0] == "correlation_regime_shift"
    assert recorded[0][1]["meta"]["applied"] is False


def test_policy_flip_applies_scale_to_weight_and_qty(monkeypatch):
    """shadow_only=false => Skalierung wirkt auf BEIDE Spalten (der alte
    Live-Pfad, jetzt explizit opt-in)."""
    targets, prices, ctx = _setup(monkeypatch, detected=True, scale=0.5)
    import src.assembled_core.ops.shadow_recorder as sr

    monkeypatch.setattr(sr, "record_shadow", lambda *a, **k: None)
    policy = {"correlation_regime_shift": {"shadow_only": False}}
    meta: dict = {}
    out = tc_sizing._sp_apply_correlation_guard(targets, prices, policy, ctx, meta)
    row = out.set_index("symbol")
    assert row.loc["AAPL", "target_weight"] == pytest.approx(0.25)
    assert row.loc["AAPL", "target_qty"] == pytest.approx(50.0)
    assert row.loc["MSFT", "target_qty"] == pytest.approx(20.0)
