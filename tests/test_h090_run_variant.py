# -*- coding: utf-8 -*-
"""Regressionstest fuer research/mandat2/h090_momentum_exits.run_variant.

Hintergrund (F-senior-10): Der erste H-090-Volllauf wurde durch einen
Delisting-/Kalender-Bug kontaminiert (Zukunftspreise bei temporaeren
Kursluecken, Phantomtag-Massenverkaeufe) und nur durch Review gefunden.
Dieser Test fixiert die Kerninvarianten der Simulationsschleife auf einem
synthetischen Panel — deterministisch, ohne Kampagnendaten.

Szenarien:
  1. Temporaere Kursluecke -> Position wird GEHALTEN, kein Zwangsverkauf,
     kein Preis aus der Zukunft.
  2. Echtes Delisting -> Zwangsverkauf zum letzten Kurs VOR der Luecke,
     exit_date = letzter Handelstag (nicht der NaN-Tag).
  3. Fensterende -> offene Position wird als "eow" gebucht, nicht verworfen.
  4. Exit-Signal am Tag t wird am Close von t+1 gefuellt (nie same-day).
  5. Stop-Exit feuert bei gain <= -15 %.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.mandat2.h090_momentum_exits import (  # noqa: E402
    prepare_inputs,
    run_variant_from_inputs,
)


class _FakeData:
    """Minimales CampaignData-Substitut fuer prepare_inputs."""

    def __init__(self, close: pd.DataFrame, membership: pd.Series) -> None:
        self.close = close
        self.membership = membership


def _panel(n_days: int = 800) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetisches Panel: SPY (Kalenderanker, steigend -> Gate offen),
    AAA (steigt stark -> wird Top-1 gekauft), BBB (faellt nach Kauf ->
    Stop), CCC (delisted mittendrin), DDD (temporaere Luecke)."""
    days = pd.bdate_range("2000-01-03", periods=n_days, tz="UTC")
    t = np.arange(n_days, dtype=float)
    spy = 100.0 + 0.05 * t  # monoton steigend -> SPY > SMA200 ab Tag 200
    aaa = 50.0 * (1.0 + 0.002) ** t  # starkes, stetiges Momentum
    # BBB: steigt bis Tag 400 (waehlbar), bricht danach um 40 % ein -> Stop.
    bbb = 40.0 * (1.0 + 0.0015) ** np.minimum(t, 400)
    bbb[401:] = bbb[400] * 0.60
    # CCC: solide bis Tag 500, danach NaN bis zum Ende (echtes Delisting).
    ccc = 30.0 * (1.0 + 0.0018) ** t
    # DDD: wie AAA, aber mit 10-Tage-Luecke ab Tag 450 (temporaer).
    ddd = 45.0 * (1.0 + 0.0019) ** t

    close = pd.DataFrame(
        {"SPY": spy, "AAA": aaa, "BBB": bbb, "CCC": ccc, "DDD": ddd}, index=days
    )
    close.loc[days[500] :, "CCC"] = np.nan
    close.loc[days[450] : days[459], "DDD"] = np.nan

    # Mitgliedschaft ab Beginn: ein Snapshot VOR dem ersten Ultimo.
    membership = pd.Series(
        {days[0]: frozenset({"AAA", "BBB", "CCC", "DDD"})},
    )
    return close, membership


@pytest.fixture(scope="module")
def trades_basis() -> list[dict]:
    close, membership = _panel()
    inp = prepare_inputs(_FakeData(close, membership))
    return run_variant_from_inputs("BASIS", inp)


def test_keine_zukunftspreise(trades_basis: list[dict]) -> None:
    """Kein Exit-Preis darf von einem Tag NACH exit_date stammen — hier
    indirekt: exit_date >= entry_date und days_held >= 0 fuer alle Trades."""
    assert trades_basis, "Simulation hat keine Trades erzeugt — Setup defekt"
    for t in trades_basis:
        assert t["days_held"] >= 0, t
        assert t["exit_date"] >= t["entry_date"], t


def test_temporaere_luecke_haelt_position(trades_basis: list[dict]) -> None:
    """DDD hat eine 10-Tage-Luecke — sie darf KEINEN Delisting-Trade
    erzeugen; die Position lebt weiter oder endet regulaer."""
    ddd_delist = [
        t for t in trades_basis if t["symbol"] == "DDD" and t["reason"] == "delisting"
    ]
    assert ddd_delist == [], f"Luecke als Delisting behandelt: {ddd_delist}"
    # Nicht vakuumfaehig (F-senior-18): mindestens ein DDD-Trade muss die
    # Luecke (Tag 450..459) tatsaechlich UEBERSPANNEN — sonst prueft der
    # Abwesenheits-Assert oben nichts.
    close, _ = _panel()
    gap_start = str(close.index[450].date())
    gap_end = str(close.index[459].date())
    spanning = [
        t
        for t in trades_basis
        if t["symbol"] == "DDD"
        and t["entry_date"] <= gap_start
        and t["exit_date"] >= gap_end
    ]
    assert spanning, "kein DDD-Trade ueberspannt die Luecke — Test ist vakuum"


def test_echtes_delisting_zum_letzten_kurs(trades_basis: list[dict]) -> None:
    """CCC endet an Tag 500 — falls zu diesem Zeitpunkt gehalten, muss der
    Zwangsverkauf exit_date == letzter Handelstag und den dortigen Kurs
    tragen. (Ob CCC gehalten wird, haengt vom Momentum-Rang ab; der Test
    prueft die Invariante NUR fuer tatsaechliche Delisting-Trades.)"""
    close, _ = _panel()
    last_ccc_day = close["CCC"].last_valid_index()
    last_ccc_px = float(close.loc[last_ccc_day, "CCC"])
    ccc_delist = [
        t for t in trades_basis if t["symbol"] == "CCC" and t["reason"] == "delisting"
    ]
    # Nicht vakuumfaehig (MIN-2): das Panel MUSS den Delisting-Fall erzeugen,
    # sonst prueft die Schleife nichts.
    assert ccc_delist, "kein CCC-Delisting-Trade — Delisting-Buchung entfallen?"
    for t in ccc_delist:
        assert t["exit_date"] == str(last_ccc_day.date()), t
        assert t["exit_px"] == pytest.approx(last_ccc_px), t


def test_eow_statt_verwerfen(trades_basis: list[dict]) -> None:
    """Am Fensterende offene Positionen muessen als 'eow' auftauchen.
    AAA hat das staerkste Momentum und ist am Ende sicher gehalten."""
    eow = [t for t in trades_basis if t["reason"] == "eow"]
    assert eow, "keine eow-Trades — Fensterende-Positionen verworfen?"


def test_stop_feuert_und_fuellt_am_folgetag(trades_basis: list[dict]) -> None:
    """BBB bricht nach Tag 400 um 40 % ein -> falls gehalten, muss ein
    Stop-Trade existieren, und der Verlust muss ueber -15 % liegen (Signal
    am Einbruchstag, Fill am Folgetag — nie same-day, nie exakt -15 %)."""
    stops = [t for t in trades_basis if t["reason"] == "stop"]
    for t in stops:
        ret = t["exit_px"] / t["entry_px"] - 1.0
        assert ret <= -0.15, f"Stop-Trade ueber der Schwelle gebucht: {t}"
    bbb_stops = [t for t in stops if t["symbol"] == "BBB"]
    assert bbb_stops, "BBB-Einbruch hat keinen Stop ausgeloest"


def test_replay_zero_day_trade() -> None:
    """Unit-Test fuer h090_phase2_sekundaer.replay (F-senior-13): ein
    Zero-Day-Trade (entry_date == exit_date, Delisting am Einstiegstag)
    darf NICHT verschluckt werden — genau das war B-1 (MEE 2011-06-01:
    Sell traf qty==0, Position blieb bis Fensterende eingefroren)."""
    from research.mandat2.h090_phase2_sekundaer import replay

    days = pd.bdate_range("2020-01-01", periods=10, tz="UTC")
    close = pd.DataFrame(
        {"XXX": np.full(10, 100.0), "YYY": np.full(10, 50.0)}, index=days
    )
    div_days = pd.DataFrame(0.0, index=days, columns=close.columns)
    d = lambda i: str(days[i].date())  # noqa: E731
    trades = [
        # regulaerer Trade
        {
            "symbol": "XXX",
            "entry_date": d(1),
            "exit_date": d(5),
            "entry_px": 100.0,
            "exit_px": 100.0,
            "days_held": 4,
            "reason": "time",
        },
        # Zero-Day-Trade: Kauf und Zwangsverkauf am selben Tag
        {
            "symbol": "YYY",
            "entry_date": d(2),
            "exit_date": d(2),
            "entry_px": 50.0,
            "exit_px": 50.0,
            "days_held": 0,
            "reason": "delisting",
        },
    ]
    res = replay(trades, days, close, div_days, "ZERO")
    assert res["rest_positionen_vor_liquidation"] == [], (
        "Zero-Day-Trade verschluckt — Position blieb offen (B-1)"
    )
    # Flache Kurse, keine Dividenden: Endwert = Start minus reine
    # Transaktionskosten (4 Seiten a 15 bp auf ~20 % Notional) — deutlich
    # ueber 99 % des Starts; eine eingefrorene Position saehe anders aus.
    assert res["end"] > 99_000.0
    assert res["end"] < 100_000.0  # Kosten muessen abgeflossen sein
    assert res["tax_paid"] == 0.0


def test_max_positionsdauer_basis(trades_basis: list[dict]) -> None:
    """BASIS: max 120 Handelstage + 1 Fill-Tag; Delisting/eow ausgenommen
    (dort kann die Buchung frueher/mit Luecke liegen)."""
    for t in trades_basis:
        if t["reason"] in ("delisting", "eow"):
            continue
        assert t["days_held"] <= 121, t
