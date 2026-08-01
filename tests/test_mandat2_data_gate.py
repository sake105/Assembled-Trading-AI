"""Die Holdout-Sperre muss WIRKEN, nicht nur gemeint sein (Mandat II, P0).

Vor diesen Tests stand die Regel „Suche nur bis 2016-12-31, ein Schuss aufs
Holdout" ausschliesslich als Prosa in PLAN.md. Bei 1.964 verbrauchten Trials
aus Mandat I ist das kein Schutz, sondern eine spaeter unbeweisbare Behauptung.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from research.mandat2.data_gate import (
    HOLDOUT_START,
    SEARCH_CUTOFF,
    TRIALS_MANDAT_I,
    HoldoutViolation,
    TrialCounter,
    load_holdout,
    load_search,
)

pytestmark = pytest.mark.fast


@pytest.fixture
def panel() -> pd.DataFrame:
    """Long-Format wie research/mandat/data/prices_verdict.parquet."""
    dates = pd.date_range("1995-01-03", "2026-07-06", freq="7D")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["AAA"] * len(dates),
            "close": range(len(dates)),
        }
    )


# --------------------------------------------------------------------------
# Suchpfad: der Holdout ist physisch nicht da
# --------------------------------------------------------------------------
def test_suche_sieht_keinen_tag_nach_dem_cutoff(panel):
    such = load_search(panel, date_col="timestamp")
    assert such["timestamp"].max() <= SEARCH_CUTOFF
    assert len(such) > 0
    # Und zwar nicht, weil jemand daran denkt: die Zeilen fehlen.
    assert not (panel["timestamp"] > SEARCH_CUTOFF).loc[such.index].any()


def test_suche_funktioniert_auch_mit_datum_im_index(panel):
    wide = panel.set_index("timestamp")[["close"]]
    such = load_search(wide)
    assert such.index.max() <= SEARCH_CUTOFF


def test_such_und_holdout_fenster_ueberlappen_nicht(panel):
    such = load_search(panel, date_col="timestamp")
    hold = load_holdout(
        panel,
        candidate_id="TEST-overlap",
        begruendung="Fenstertest",
        date_col="timestamp",
        ledger_path=_tmp_ledger(),
    )
    assert such["timestamp"].max() < hold["timestamp"].min()
    assert hold["timestamp"].min() >= HOLDOUT_START


def _tmp_ledger():
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp()) / "holdout_ledger.jsonl"


# --------------------------------------------------------------------------
# Holdout: ein Kandidat, ein Schuss
# --------------------------------------------------------------------------
def test_zweiter_holdout_schuss_wird_verweigert(panel, tmp_path):
    ledger = tmp_path / "holdout_ledger.jsonl"
    load_holdout(
        panel,
        candidate_id="H-201",
        begruendung="Finaler Kandidat aus P2",
        date_col="timestamp",
        ledger_path=ledger,
    )
    with pytest.raises(HoldoutViolation, match="bereits"):
        load_holdout(
            panel,
            candidate_id="H-201",
            begruendung="nur nochmal kurz schauen",
            date_col="timestamp",
            ledger_path=ledger,
        )


def test_anderer_kandidat_darf_seinen_eigenen_schuss(panel, tmp_path):
    ledger = tmp_path / "holdout_ledger.jsonl"
    load_holdout(
        panel,
        candidate_id="H-201",
        begruendung="Kandidat A",
        date_col="timestamp",
        ledger_path=ledger,
    )
    hold = load_holdout(
        panel,
        candidate_id="H-202",
        begruendung="Kandidat B",
        date_col="timestamp",
        ledger_path=ledger,
    )
    assert len(hold) > 0


def test_force_bricht_die_sperre_aber_protokolliert_den_bruch(panel, tmp_path):
    """Ein bewusster Bruch soll moeglich sein, ein unbemerkter nicht."""
    ledger = tmp_path / "holdout_ledger.jsonl"
    load_holdout(
        panel,
        candidate_id="H-201",
        begruendung="erster Schuss",
        date_col="timestamp",
        ledger_path=ledger,
    )
    load_holdout(
        panel,
        candidate_id="H-201",
        begruendung="Datenfehler im ersten Lauf, dokumentiert in ledger.md",
        date_col="timestamp",
        ledger_path=ledger,
        force=True,
    )
    zeilen = [json.loads(z) for z in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(zeilen) == 2
    assert zeilen[0]["forced"] is False
    assert zeilen[1]["forced"] is True
    assert zeilen[1]["wiederholung_nr"] == 2


def test_leere_begruendung_wird_abgelehnt(panel, tmp_path):
    with pytest.raises(ValueError, match="begruendung"):
        load_holdout(
            panel,
            candidate_id="H-203",
            begruendung="   ",
            date_col="timestamp",
            ledger_path=tmp_path / "l.jsonl",
        )


def test_leere_candidate_id_wird_abgelehnt(panel, tmp_path):
    with pytest.raises(ValueError, match="candidate_id"):
        load_holdout(
            panel,
            candidate_id="",
            begruendung="egal",
            date_col="timestamp",
            ledger_path=tmp_path / "l.jsonl",
        )


def test_abgeschnittene_zeile_des_kandidaten_erzeugt_keinen_freischuss(panel, tmp_path):
    """Der eigentliche fail-open-Fall: die kaputte Zeile IST die des Kandidaten.

    Wuerde sie nur uebersprungen, faellt der Kandidat aus "verbraucht" heraus
    und bekaeme einen zweiten Schuss durch Dateikorruption geschenkt
    (F-auditor-2). Der vorige Test prueft das NICHT — er blockte an einer
    intakten Zweitzeile. Deshalb blockt der Guard, sobald ueberhaupt eine
    Zeile unlesbar ist.
    """
    ledger = tmp_path / "holdout_ledger.jsonl"
    ledger.write_text('{"candidate_id": "H-2', encoding="utf-8")  # abgeschnitten
    with pytest.raises(HoldoutViolation, match="unlesbare"):
        load_holdout(
            panel,
            candidate_id="H-201",
            begruendung="zweiter Versuch",
            date_col="timestamp",
            ledger_path=ledger,
        )


def test_append_klebt_nicht_an_eine_zeile_ohne_newline(panel, tmp_path):
    """Sonst wuerde der neue Eintrag selbst unlesbar — der Schuss waere dann
    unprotokolliert."""
    ledger = tmp_path / "holdout_ledger.jsonl"
    ledger.write_text('{"candidate_id": "ALT"}', encoding="utf-8")  # kein Newline
    load_holdout(
        panel,
        candidate_id="H-301",
        begruendung="nach Ledger-Reparatur",
        date_col="timestamp",
        ledger_path=ledger,
        force=True,
    )
    zeilen = [z for z in ledger.read_text(encoding="utf-8").splitlines() if z.strip()]
    assert len(zeilen) == 2
    assert json.loads(zeilen[1])["candidate_id"] == "H-301"


def test_holdout_hat_eine_obergrenze(panel, tmp_path):
    """Spaeter hinzukommende Daten duerfen nicht still in den einen Schuss
    wandern (F-auditor-5)."""
    from research.mandat2.data_gate import HOLDOUT_END

    erweitert = pd.concat(
        [
            panel,
            pd.DataFrame(
                {
                    "timestamp": pd.date_range("2026-08-01", periods=50, freq="7D"),
                    "symbol": "AAA",
                    "close": 0,
                }
            ),
        ]
    )
    hold = load_holdout(
        erweitert,
        candidate_id="H-401",
        begruendung="Obergrenzentest",
        date_col="timestamp",
        ledger_path=tmp_path / "l.jsonl",
    )
    assert hold["timestamp"].max() <= HOLDOUT_END


# --------------------------------------------------------------------------
# Trial-Zaehler
# --------------------------------------------------------------------------
def test_zaehler_startet_bei_mandat_i_und_nicht_bei_null(tmp_path):
    c = TrialCounter(path=tmp_path / "trials.json")
    assert c.total() == TRIALS_MANDAT_I
    assert TRIALS_MANDAT_I == 1964


def test_zaehler_akkumuliert_ueber_laeufe(tmp_path):
    c = TrialCounter(path=tmp_path / "trials.json")
    c.increment(10, label="P1 Momentum-Familie")
    assert c.total() == TRIALS_MANDAT_I + 10
    # Neue Instanz auf derselben Datei -> Zustand ueberlebt den Prozess.
    c2 = TrialCounter(path=tmp_path / "trials.json")
    c2.increment(5, label="P1 Insider-Familie")
    assert c2.total() == TRIALS_MANDAT_I + 15


def test_zaehler_lehnt_nicht_positive_schritte_ab(tmp_path):
    c = TrialCounter(path=tmp_path / "trials.json")
    with pytest.raises(ValueError):
        c.increment(0)
