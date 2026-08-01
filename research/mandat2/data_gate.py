"""Erzwungene Holdout-Sperre und Trial-Zaehler (Mandat II, Phase 0).

Warum als CODE und nicht als Vorsatz
------------------------------------
Mandat I hat 1.964 Trials verbraucht. Wer danach dieselben Daten erneut
durchsucht, FINDET etwas, das SPY schlaegt — die Frage ist nur, ob es echt ist.
Der einzige Schutz ist ein Zeitfenster, das die Suche nie gesehen hat.

Eine Holdout-Regel, die nur in einem Plandokument steht, ist kein Schutz,
sondern eine spaeter unbeweisbare Behauptung: ``prices_verdict.parquet``
enthaelt 1995-2026 und jedes Ad-hoc-Skript liest es ungefiltert. Deshalb:

* ``load_search()`` schneidet die Daten IM LADER ab. Der Suchpfad kann den
  Holdout gar nicht sehen — nicht, weil man diszipliniert ist, sondern weil
  die Zeilen nicht da sind.
* ``load_holdout()`` verlangt eine Kandidaten-ID und eine Begruendung, schreibt
  beides append-only in ein Ledger und verweigert den ZWEITEN Zugriff auf
  dieselbe ID. Ein Kandidat, ein Schuss.
* ``trials`` fuehrt den Zaehler ueber Mandat I hinaus weiter (N0 = 1.964),
  damit der DSR-Haircut haerter wird statt weicher.

Die Sperre laesst sich mit ``force=True`` brechen — aber nur mit Begruendung,
und der Bruch landet genauso im Ledger. Ein unbemerkter Bruch soll nicht
moeglich sein; ein bewusster schon.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------- Konstanten
#: Letzter Tag, den die SUCHE sehen darf. Gesperrt am 2026-08-01.
SEARCH_CUTOFF = pd.Timestamp("2016-12-31")
#: Erster Tag des Holdouts (9,5 Jahre: COVID-Crash 2020 + Baermarkt 2022).
HOLDOUT_START = pd.Timestamp("2017-01-01")
#: Letzter Tag des Holdouts = Ende des Datenbestands bei der Sperrung.
#: Mit offener Obergrenze wuerden spaeter hinzukommende Daten still in den
#: "einen Schuss" wandern und ihn unbemerkt vergroessern (F-auditor-5).
HOLDOUT_END = pd.Timestamp("2026-07-06")
#: Verbrauchte Trials aus Mandat I (research/mandat/FINAL_REPORT.md).
TRIALS_MANDAT_I = 1964

_ROOT = Path(__file__).resolve().parent
HOLDOUT_LEDGER = _ROOT / "holdout_ledger.jsonl"
TRIALS_STATE = _ROOT / "trials.json"


class HoldoutViolation(RuntimeError):
    """Zweiter Holdout-Zugriff auf dieselbe Kandidaten-ID."""


# ------------------------------------------------------------------- Laden
def _as_ts(col: pd.Series | pd.DatetimeIndex) -> pd.Series | pd.DatetimeIndex:
    return pd.to_datetime(col)


def _split(df: pd.DataFrame, date_col: str | None) -> tuple[pd.Series, pd.DataFrame]:
    """Gibt (Datumsreihe, df) zurueck — akzeptiert Index- ODER Spaltendatum."""
    if date_col is None:
        return pd.Series(_as_ts(df.index), index=df.index), df
    return _as_ts(df[date_col]), df


def load_search(df: pd.DataFrame, date_col: str | None = None) -> pd.DataFrame:
    """Daten fuer die SUCHE — alles nach ``SEARCH_CUTOFF`` wird abgeschnitten.

    Nicht optional und nicht abschaltbar. Wer den Holdout braucht, geht durch
    ``load_holdout`` und hinterlaesst eine Spur.
    """
    dates, df = _split(df, date_col)
    return df.loc[dates <= SEARCH_CUTOFF]


def load_holdout(
    df: pd.DataFrame,
    *,
    candidate_id: str,
    begruendung: str,
    date_col: str | None = None,
    force: bool = False,
    ledger_path: Path | str | None = None,
) -> pd.DataFrame:
    """Daten fuer den EINEN Holdout-Schuss eines Kandidaten.

    Args:
        candidate_id: eindeutige ID des finalen Kandidaten (z. B. "H-201-v3").
        begruendung: warum dieser Kandidat den Schuss verdient. Nicht leer.
        force: einen bereits verbrauchten Schuss wiederholen. Wird als
            ``forced`` im Ledger vermerkt und entwertet den Kandidaten
            statistisch — nur fuer dokumentierte Sonderfaelle.

    Raises:
        HoldoutViolation: zweiter Zugriff ohne ``force``.
        ValueError: leere ID oder leere Begruendung.
    """
    if not candidate_id or not candidate_id.strip():
        raise ValueError("candidate_id darf nicht leer sein")
    if not begruendung or not begruendung.strip():
        raise ValueError(
            "begruendung darf nicht leer sein — der Holdout-Schuss muss "
            "nachvollziehbar bleiben"
        )
    path = Path(ledger_path) if ledger_path is not None else HOLDOUT_LEDGER
    verbraucht, kaputte_zeilen = _read_ledger(path)
    if kaputte_zeilen and not force:
        raise HoldoutViolation(
            f"{path.name} enthaelt {kaputte_zeilen} unlesbare Zeile(n). Eine "
            f"davon koennte der bereits verbrauchte Schuss dieses Kandidaten "
            f"sein — der Guard blockt deshalb fail-closed, statt einen "
            f"Freischuss durch Dateikorruption zu erlauben. Ledger von Hand "
            f"pruefen und reparieren, oder force=True mit Begruendung."
        )
    if candidate_id in verbraucht and not force:
        raise HoldoutViolation(
            f"Kandidat {candidate_id!r} hat seinen Holdout-Schuss bereits "
            f"verbraucht (siehe {path.name}). Ein zweiter Blick macht den "
            f"Holdout zu einem weiteren Suchdatensatz. Bewusst wiederholen: "
            f"force=True — der Bruch wird protokolliert."
        )
    _append_ledger(
        path,
        {
            "candidate_id": candidate_id,
            "begruendung": begruendung,
            "forced": bool(force and (candidate_id in verbraucht or kaputte_zeilen)),
            "wiederholung_nr": sum(1 for c in verbraucht if c == candidate_id) + 1,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    dates, df = _split(df, date_col)
    return df.loc[(dates >= HOLDOUT_START) & (dates <= HOLDOUT_END)]


def _read_ledger(path: Path) -> tuple[list[str], int]:
    """(verbrauchte IDs, Anzahl unlesbarer Zeilen).

    Die unlesbaren Zeilen werden ZURUECKGEGEBEN und nicht verschluckt: eine
    abgeschnittene Zeile koennte genau die des anfragenden Kandidaten sein,
    und ihn stillschweigend aus "verbraucht" herausfallen zu lassen waere ein
    Freischuss durch Dateikorruption (F-auditor-2, fail-open im Guard).
    """
    if not path.exists():
        return [], 0
    ids: list[str] = []
    kaputt = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.append(json.loads(line)["candidate_id"])
        except (json.JSONDecodeError, KeyError):
            kaputt += 1
    return ids, kaputt


def _append_ledger(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Endet die Datei nicht auf einem Newline (abgebrochener Vorschreiber),
    # klebte der neue Eintrag an die kaputte Zeile und wuerde dadurch selbst
    # unlesbar — der Schuss waere dann unprotokolliert (F-auditor-2).
    prefix = ""
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as probe:
            probe.seek(-1, os.SEEK_END)
            if probe.read(1) != b"\n":
                prefix = "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(prefix + json.dumps(entry, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ------------------------------------------------------------------ Trials
@dataclass
class TrialCounter:
    """Kumulierter Trial-Zaehler ueber beide Mandate.

    Der DSR-Haircut skaliert mit der Anzahl der Versuche. Ein Reset auf 0
    wuerde Mandat II genau die Freiheit geben, die Mandat I bereits verbraucht
    hat — deshalb startet der Zaehler bei 1.964.
    """

    path: Path = TRIALS_STATE

    def _read(self) -> dict:
        if not self.path.exists():
            return {"n0_mandat_i": TRIALS_MANDAT_I, "n_mandat_ii": 0}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def total(self) -> int:
        s = self._read()
        return int(s["n0_mandat_i"]) + int(s["n_mandat_ii"])

    def increment(self, n: int = 1, *, label: str = "") -> int:
        if n < 1:
            raise ValueError("increment erwartet n >= 1")
        s = self._read()
        s["n_mandat_ii"] = int(s["n_mandat_ii"]) + n
        s["last_label"] = label
        s["last_ts_utc"] = datetime.now(timezone.utc).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return self.total()
