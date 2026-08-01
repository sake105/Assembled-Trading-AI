"""Die Zielfunktion entscheidet ueber jedes Verdikt — also wird sie gepinnt.

Die beiden Eigenschaften, die den Unterschied zwischen einer ehrlichen und
einer geschoenten Auswertung ausmachen: der Median laeuft ueber ALLE
rollierenden Fenster (kein Rosinenpicken), und der DD-Deckel gilt in JEDEM
Fenster (nicht im Mittel).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.mandat2.metrics import (
    DD_DECKEL,
    auswerten,
    max_drawdown,
    rolling_windows,
)

pytestmark = pytest.mark.fast


def _kurve(tage: int, tagesrendite: float, start: str = "1995-01-02") -> pd.Series:
    idx = pd.bdate_range(start, periods=tage)
    return pd.Series(100_000.0 * (1 + tagesrendite) ** np.arange(tage), index=idx)


# --------------------------------------------------------------------- MaxDD
def test_maxdrawdown_misst_peak_to_trough():
    e = pd.Series([100.0, 120.0, 60.0, 90.0])
    assert max_drawdown(e) == pytest.approx(-0.5)  # 120 -> 60


def test_maxdrawdown_ist_null_bei_monotonem_anstieg():
    assert max_drawdown(pd.Series([100.0, 101.0, 102.0])) == 0.0


def test_maxdrawdown_bei_zu_kurzer_reihe():
    assert max_drawdown(pd.Series([100.0])) == 0.0
    assert max_drawdown(pd.Series(dtype=float)) == 0.0


# -------------------------------------------------------------------- Fenster
def test_fenster_sind_zehn_jahre_lang_und_monatlich_versetzt():
    idx = pd.bdate_range("1995-01-02", "2016-12-30")
    w = rolling_windows(idx, jahre=10, schritt_monate=1)
    assert len(w) > 100  # 22 Jahre - 10 Jahre = ~144 Startmonate
    for start, ende in w:
        spanne = (ende - start).days / 365.25
        assert 9.9 < spanne < 10.1
    # aufsteigend und innerhalb der Daten
    assert w[0][0] < w[-1][0]
    assert w[-1][1] <= idx[-1]


def test_kein_fenster_wenn_die_historie_zu_kurz_ist():
    idx = pd.bdate_range("2020-01-01", periods=200)
    assert rolling_windows(idx, jahre=10) == []


def test_leerer_index_gibt_keine_fenster():
    assert rolling_windows(pd.DatetimeIndex([])) == []


# ------------------------------------------------------------------- Verdikt
def test_besserer_kandidat_ohne_drawdown_besteht():
    tage = 22 * 252
    kand = _kurve(tage, 0.0004)
    bench = _kurve(tage, 0.0003)
    a = auswerten(kand, bench, label="test")
    assert a.n_fenster > 100
    assert a.schlaegt_benchmark
    assert a.deckel_eingehalten
    assert a.bestanden
    assert a.anteil_fenster_geschlagen == 1.0


def test_schlechterer_kandidat_faellt_durch():
    tage = 22 * 252
    a = auswerten(_kurve(tage, 0.0002), _kurve(tage, 0.0004), label="test")
    assert not a.schlaegt_benchmark
    assert not a.bestanden
    assert "DURCHGEFALLEN" in a.bericht()


def test_ein_einziges_gerissenes_fenster_reicht_zum_durchfallen():
    """Der DD-Deckel ist bindend, nicht ein Mittelwert.

    Sonst koennte ein Kandidat einen -60-%-Einbruch mit vielen ruhigen
    Fenstern wegmitteln — und genau dann wuerde gehebeltes SPY gewinnen.
    """
    tage = 22 * 252
    bench = _kurve(tage, 0.0003)
    kand = _kurve(tage, 0.0005).copy()
    # Ein kurzer, tiefer Einbruch mitten in der Historie.
    lo, hi = 3000, 3100
    kand.iloc[lo:hi] = kand.iloc[lo] * 0.35
    a = auswerten(kand, bench, label="test")
    assert a.schlaegt_benchmark  # renditeseitig besser
    assert not a.deckel_eingehalten
    assert a.gerissene_fenster
    assert a.schlimmster_maxdd < DD_DECKEL
    assert not a.bestanden  # trotzdem durchgefallen
    assert "DD-Deckel" in a.bericht()


def test_leere_auswertung_besteht_nicht():
    """Fail-closed: kein Urteil ist kein bestandenes Urteil."""
    kurz = _kurve(50, 0.001)
    a = auswerten(kurz, kurz, label="test")
    assert a.n_fenster == 0
    assert not a.bestanden
    assert "KEINE Fenster" in a.bericht()


def test_nur_gemeinsamer_zeitraum_wird_verglichen():
    """Sonst vergliche man verschiedene Zeitraeume und der Median waere Unsinn."""
    tage = 22 * 252
    kand = _kurve(tage, 0.0004)
    bench = _kurve(tage, 0.0004)[500:]  # startet spaeter
    a = auswerten(kand, bench, label="test")
    assert a.fenster
    assert a.fenster[0].start >= bench.index[0]
    # Gleiche Drift auf gleichem Zeitraum -> gleicher Faktor, kein Gewinner.
    assert a.median_kandidat == pytest.approx(a.median_benchmark, rel=1e-9)


def test_median_nicht_bestes_fenster():
    """Rosinenpicken darf nicht funktionieren.

    Der Kandidat gewinnt in einem einzelnen Fenster deutlich, verliert aber in
    der Mehrheit — das Urteil muss dem Median folgen, nicht dem Ausreisser.
    """
    tage = 22 * 252
    bench = _kurve(tage, 0.0004)
    # Kandidat: dauerhaft etwas schwaecher, aber die ersten drei Jahre stark.
    # Fenster, die diese Phase ganz enthalten, gewinnt er; alle spaeteren nicht.
    r = np.full(tage, 0.00030)
    r[: 3 * 252] = 0.0016
    kand = pd.Series(100_000.0 * np.cumprod(1 + r), index=bench.index)
    a = auswerten(kand, bench, label="test")
    quotienten = [f.kandidat_faktor / f.benchmark_faktor for f in a.fenster]
    assert max(quotienten) > 1.0  # es GIBT Fenster, in denen er gewinnt
    assert a.anteil_fenster_geschlagen < 0.5  # aber nicht die Mehrheit
    assert not a.schlaegt_benchmark  # der Median folgt dem Ausreisser nicht
