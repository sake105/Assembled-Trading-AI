# -*- coding: utf-8 -*-
"""Regressionstests fuer den code-versionierten Factor-Store-Cache-Key
(Audit-Plan 4.3, KNOWN_ISSUES §0.05, 2026-08-17).

Belegter Produktionsdefekt: der Key hashte NUR die Symbolliste — ein unter
aelterem Code berechnetes Panel ueberlebte Feature-/Sizing-Aenderungen und
wurde still wiederverwendet (WARM: n_orders=0/56 Spalten vs. COLD:
n_orders=2/55 Spalten bei identischem Code + Input). Seit dem Fix traegt der
Key einen Hash der Feature-Quellen: eine Code-Aenderung erzeugt einen neuen
Key, das alte Panel wird verworfen (COLD) statt verwendet (stale).
"""

from __future__ import annotations

import pandas as pd
import pytest

import src.assembled_core.data.factor_store as fs

pytestmark = pytest.mark.fast


def test_key_is_deterministic_and_symbol_order_independent():
    k1 = fs.compute_universe_key(symbols=["MSFT", "AAPL"])
    k2 = fs.compute_universe_key(symbols=["AAPL", "MSFT"])
    assert k1 == k2
    assert k1.startswith("universe_")
    assert "_code_" in k1  # der 4.3-Kern: Key traegt eine Code-Version


def test_code_change_changes_key(monkeypatch):
    """Simulierte Feature-Code-Aenderung -> anderer Key."""
    monkeypatch.setattr(fs, "_feature_code_version", lambda: "aaaaaaaaaa")
    k_old = fs.compute_universe_key(symbols=["AAPL"])
    monkeypatch.setattr(fs, "_feature_code_version", lambda: "bbbbbbbbbb")
    k_new = fs.compute_universe_key(symbols=["AAPL"])
    assert k_old != k_new
    # Symbol-Anteil bleibt identisch — NUR die Code-Version unterscheidet sich.
    assert k_old.split("_code_")[0] == k_new.split("_code_")[0]


def test_stale_panel_is_discarded_not_reused(tmp_path, monkeypatch):
    """DER §0.05-Pin: ein unter altem Code gespeichertes Panel darf nach
    einer Code-Aenderung NICHT mehr geladen werden (COLD statt stale) —
    genau der Mechanismus, der WARM/COLD-Orderdivergenz erzeugte."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-05", "2026-01-06"], utc=True),
            "symbol": ["AAPL", "AAPL"],
            "mom_20": [0.1, 0.2],
        }
    )
    monkeypatch.setattr(fs, "_feature_code_version", lambda: "aaaaaaaaaa")
    key_old = fs.compute_universe_key(symbols=["AAPL"])
    fs.store_factors(df, "core_ta", "1d", key_old, factors_root=tmp_path)
    # Unter dem alten Code-Stand: Panel ist da (Kontrollprobe).
    assert fs.load_factors("core_ta", "1d", key_old, factors_root=tmp_path) is not None

    # "Code-Aenderung": neuer Hash -> neuer Key -> Cache-Miss.
    monkeypatch.setattr(fs, "_feature_code_version", lambda: "bbbbbbbbbb")
    key_new = fs.compute_universe_key(symbols=["AAPL"])
    assert key_new != key_old
    assert fs.load_factors("core_ta", "1d", key_new, factors_root=tmp_path) is None


def test_real_code_version_is_stable_within_process():
    """Der echte Datei-Hash ist deterministisch (lru_cache + stabile Quelle)."""
    v1 = fs._feature_code_version()
    v2 = fs._feature_code_version()
    assert v1 == v2
    assert len(v1) == 10
