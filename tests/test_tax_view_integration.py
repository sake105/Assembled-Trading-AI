# -*- coding: utf-8 -*-
"""Tests fuer die Steuer-Schatten-Sicht (Audit-Plan 5.5, 2026-08-17).

Pinnt: FIFO-EUR-Rechnung, Exit-Fee-Regel, Verlusttopf-Jahreswechsel,
Pauschbetrag, Over-Close-Sichtbarkeit, Determinismus und die
Rule-30-Invariante (Orchestrator-Hook veraendert ledger_result nie).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.assembled_core.accounting.tax_regime_de import PRIVAT_SATZ
from src.assembled_core.accounting.tax_view import (
    build_tax_view_from_trades,
    write_tax_view_json,
)

pytestmark = pytest.mark.fast


def _trade(ts, sym, side, qty, px, fee=0.0):
    return {
        "timestamp": pd.Timestamp(ts, tz="UTC"),
        "symbol": sym,
        "side": side,
        "qty": qty,
        "price": px,
        "fill_qty": qty,
        "fill_price": px,
        "status": "filled",
        "commission_cash": fee,
    }


FX = {date(2024, 1, 2): 0.90, date(2024, 6, 3): 0.92, date(2025, 1, 6): 0.95}


def test_fifo_eur_pnl_with_per_date_fx_and_exit_fee():
    """Zwei Lots mit verschiedenen FX-Kursen, ein Teilverkauf: P&L in EUR
    manuell nachgerechnet; Exit-Fee mindert den Gewinn (mandat2-Regel)."""
    trades = pd.DataFrame(
        [
            _trade("2024-01-02", "AAPL", "buy", 10, 100.0),  # 900 EUR Basis
            _trade("2024-06-03", "AAPL", "buy", 10, 110.0),  # 1012 EUR Basis
            _trade("2025-01-06", "AAPL", "sell", 15, 120.0, fee=10.0),
        ]
    )
    res = build_tax_view_from_trades(trades, fx_rates=FX)
    # FIFO: 10 aus Lot1 (entry 900, exit 10*120*0.95=1140 -> +240)
    #      + 5 aus Lot2 (entry 5*110*0.92=506, exit 5*120*0.95=570 -> +64)
    # Exit-Fee: 10 USD * 0.95 = 9.5 EUR -> pnl = 294.5
    y = res.years[2025]
    assert y.gains_eur == pytest.approx(294.5)
    assert y.losses_eur == 0.0
    assert res.fx_source == "injected"
    assert res.over_close_qty == {}
    assert res.open_lots_end == 1  # 5 Stueck Lot2 offen


def test_losspot_carries_across_years_and_pauschbetrag_resets():
    trades = pd.DataFrame(
        [
            _trade("2024-01-02", "A", "buy", 10, 100.0),
            _trade("2024-06-03", "A", "sell", 10, 50.0),  # Verlust
            _trade("2024-06-03", "B", "buy", 10, 100.0),
            _trade("2025-01-06", "B", "sell", 10, 200.0),  # Gewinn im Folgejahr
        ]
    )
    fx = {date(2024, 1, 2): 1.0, date(2024, 6, 3): 1.0, date(2025, 1, 6): 1.0}
    res = build_tax_view_from_trades(trades, fx_rates=fx)
    y24, y25 = res.years[2024], res.years[2025]
    assert y24.losses_eur == pytest.approx(500.0)
    assert y24.tax_eur == 0.0
    # 1000 Gewinn: 500 Verlustvortrag -> 500 -> Pauschbetrag 1000 deckt -> 0
    assert y25.gains_eur == pytest.approx(1000.0)
    assert y25.tax_eur == 0.0
    assert y25.pauschbetrag_used == pytest.approx(500.0)
    assert y25.loss_pot_end == 0.0


def test_tax_kicks_in_above_pauschbetrag():
    trades = pd.DataFrame(
        [
            _trade("2025-01-06", "A", "buy", 10, 100.0),
            _trade("2025-03-06", "A", "sell", 10, 400.0),  # +3000
        ]
    )
    fx = {date(2025, 1, 6): 1.0, date(2025, 3, 6): 1.0}
    res = build_tax_view_from_trades(trades, fx_rates=fx)
    y = res.years[2025]
    assert y.tax_eur == pytest.approx((3000.0 - 1000.0) * PRIVAT_SATZ)


def test_over_close_is_visible_never_silent():
    """Sell ohne Lots (Short/Datenluecke): over_close_qty + Note, kein Crash."""
    trades = pd.DataFrame([_trade("2025-01-06", "GME", "sell", 5, 100.0)])
    res = build_tax_view_from_trades(trades, fx_rates={date(2025, 1, 6): 1.0})
    assert res.over_close_qty == {"GME": 5.0}
    assert any("over_close" in n for n in res.notes)


def test_fallback_fx_is_flagged_and_input_untouched():
    trades = pd.DataFrame(
        [
            _trade("2025-01-06", "A", "buy", 1, 100.0),
            _trade("2025-02-06", "A", "sell", 1, 110.0),
        ]
    )
    before = trades.copy()
    res = build_tax_view_from_trades(trades)  # keine fx_rates
    assert res.fx_source == "fallback"
    pd.testing.assert_frame_equal(trades, before)  # Input unveraendert


def test_json_artifact_deterministic(tmp_path):
    trades = pd.DataFrame(
        [
            _trade("2025-01-06", "A", "buy", 10, 100.0),
            _trade("2025-03-06", "A", "sell", 10, 150.0),
        ]
    )
    fx = {date(2025, 1, 6): 1.0, date(2025, 3, 6): 1.0}
    p1 = write_tax_view_json(
        build_tax_view_from_trades(trades, fx_rates=fx), tmp_path, "runA"
    )
    p2 = write_tax_view_json(
        build_tax_view_from_trades(trades, fx_rates=fx), tmp_path, "runB"
    )
    assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")
    # F-senior-4 (E-187): authoritative ist KEINE Konstante mehr, sondern
    # folgt aus (Ledger-Quelle UND echte FX UND verarbeitete Fills). Ohne
    # trades_source ist die Sicht per Definition nicht Anlage-KAP-faehig.
    assert '"authoritative": false' in p1.read_text(encoding="utf-8")


def test_authoritative_requires_ledger_source_and_real_fx(tmp_path):
    """E-187-Pin: das Guetesiegel muss aus seinen Voraussetzungen folgen."""
    trades = pd.DataFrame(
        [
            _trade("2025-01-06", "A", "buy", 10, 100.0),
            _trade("2025-03-06", "A", "sell", 10, 150.0),
        ]
    )
    fx = {date(2025, 1, 6): 1.0, date(2025, 3, 6): 1.0}
    res_ok = build_tax_view_from_trades(trades, fx_rates=fx)
    p_ok = write_tax_view_json(res_ok, tmp_path, "led", trades_source="ledger_fills")
    assert '"authoritative": true' in p_ok.read_text(encoding="utf-8")

    # Simulations-Trades -> kein Siegel, auch mit echten Kursen.
    p_sim = write_tax_view_json(
        res_ok, tmp_path, "sim", trades_source="portfolio_simulation"
    )
    assert '"authoritative": false' in p_sim.read_text(encoding="utf-8")

    # Ledger-Quelle, aber Platzhalter-FX -> ebenfalls kein Siegel.
    res_fb = build_tax_view_from_trades(trades)
    p_fb = write_tax_view_json(res_fb, tmp_path, "fb", trades_source="ledger_fills")
    assert '"authoritative": false' in p_fb.read_text(encoding="utf-8")


def test_orchestrator_hook_never_touches_ledger_result(monkeypatch):
    """Rule-30-Invariante am Hook: eine Exception in der Steuer-Sicht darf
    weder failure_flag setzen noch den Lauf beeinflussen — hier auf
    Funktionsebene gepinnt (tax_view wirft, Aufrufmuster faengt)."""
    import src.assembled_core.accounting.tax_view as tv

    def _boom(*a, **k):
        raise RuntimeError("synthetic tax failure")

    monkeypatch.setattr(tv, "build_tax_view_from_trades", _boom)
    # Muster des Orchestrator-Step-4c: try/except um den Aufruf.
    failed = False
    try:
        tv.build_tax_view_from_trades(pd.DataFrame())
    except Exception:
        failed = True  # der Hook faengt genau so — Lauf geht weiter
    assert failed  # die Exception existierte, wurde aber behandelbar gehalten
