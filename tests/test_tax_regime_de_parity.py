# -*- coding: utf-8 -*-
"""Paritaetstest: accounting/tax_regime_de-Port vs. research/mandat2-Original.

Die Methodik ist FESTGESCHRIEBEN (Memory: nicht wieder aufmachen) — der Port
muss fuer identische Sequenzen BITGLEICHE Steuerbetraege liefern, inklusive
Jahreswechsel (Pauschbetrag-Reset) und ueberjaehrigem Verlusttopf.
"""

from __future__ import annotations

import pytest

from src.assembled_core.accounting.tax_regime_de import (
    PRIVAT_SATZ,
    SPARERPAUSCHBETRAG,
    AssetClass,
    PrivatDE,
    ZeroTax,
)

pytestmark = pytest.mark.fast

research_regimes = pytest.importorskip(
    "research.mandat2.tax_regimes",
    reason="research/mandat2 nicht importierbar (Layout geaendert?)",
)


SEQUENCE = [
    (2024, -500.0),  # Verlust -> Topf 500
    (2024, 300.0),  # gegen Topf: 0 steuerpflichtig, Topf 200
    (2024, 1500.0),  # Topf 200 -> 1300 taxable -> Pauschbetrag 1000 -> 300 * Satz
    (2025, 800.0),  # neues Jahr: Pauschbetrag frisch -> 0 Steuer
    (2025, 900.0),  # 200 Pauschbetrag-Rest -> 700 * Satz
    (2025, -2000.0),  # Verlust -> Topf 2000
    (2026, 1200.0),  # Topf 2000 -> 0 taxable, Topf 800 (Verlustvortrag!)
    (2026, 2500.0),  # Topf 800 -> 1700 -> Pauschbetrag 1000 -> 700 * Satz
]


def test_privat_de_parity_bitwise():
    port = PrivatDE()
    orig = research_regimes.PrivatDE()
    for year, gain in SEQUENCE:
        port.new_year(year)
        orig.new_year(year)
        t_port = port.on_realized_gain(gain, AssetClass.AKTIE)
        t_orig = orig.on_realized_gain(gain, research_regimes.AssetClass.AKTIE)
        assert t_port == t_orig, f"Divergenz bei ({year}, {gain})"
    assert port.loss_pot == orig.loss_pot
    assert port.pauschbetrag_left == orig.pauschbetrag_left


def test_constants_match_frozen_methodology():
    assert PRIVAT_SATZ == pytest.approx(0.26375)
    assert SPARERPAUSCHBETRAG == 1000.0
    assert PRIVAT_SATZ == research_regimes.PRIVAT_SATZ
    assert SPARERPAUSCHBETRAG == research_regimes.SPARERPAUSCHBETRAG


def test_order_losspot_before_pauschbetrag():
    """Reihenfolge-Pin: Verlusttopf VOR Pauschbetrag VOR Satz."""
    r = PrivatDE()
    r.new_year(2026)
    r.on_realized_gain(-1000.0)
    # 1500 Gewinn: 1000 gegen Topf, 500 gegen Pauschbetrag -> 0 Steuer
    assert r.on_realized_gain(1500.0) == 0.0
    assert r.loss_pot == 0.0
    assert r.pauschbetrag_left == 500.0
    # weitere 1000: 500 Pauschbetrag-Rest -> 500 * Satz
    assert r.on_realized_gain(1000.0) == pytest.approx(500.0 * PRIVAT_SATZ)


def test_fonds_teilfreistellung_and_zerotax():
    r = PrivatDE()
    r.new_year(2026)
    r.pauschbetrag_left = 0.0
    tax = r.on_realized_gain(1000.0, AssetClass.FONDS)
    assert tax == pytest.approx(1000.0 * PRIVAT_SATZ * 0.70)  # 30 % TFS
    z = ZeroTax()
    z.new_year(2026)
    assert z.on_realized_gain(10_000.0) == 0.0
