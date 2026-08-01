"""Der EINZIGE sanktionierte Datenzugang fuer Mandat II (Phase 0).

Ohne diese Schicht war die Holdout-Sperre nur ein Vorsatz: ``data_gate`` hatte
ausser seinen Tests keinen Konsumenten, und jedes Ad-hoc-Skript konnte
``prices_verdict.parquet`` (1995-2026) ungefiltert lesen. Erzwungen ist eine
Sperre erst, wenn sie im WEG liegt.

Regel fuer die gesamte Kampagne
-------------------------------
Jeder Phasen-Code holt seine Daten hier — nicht mit ``pd.read_parquet``.

    from research.mandat2.campaign_data import load_campaign

    d = load_campaign()                      # SUCHE: endet am 2016-12-31
    d.close, d.div_panel, d.membership

    d = load_campaign(holdout=True,          # der EINE Schuss
                      candidate_id="H-201",
                      begruendung="finaler Kandidat aus P2")

``load_campaign()`` ohne Argumente kann den Holdout nicht sehen — die Zeilen
sind nicht da. Wer ihn will, muss eine Kandidaten-ID und eine Begruendung
angeben, und der Zugriff landet append-only im Ledger. Ein zweiter Schuss auf
dieselbe ID wird verweigert.

Hygiene
-------
Die Truncation korrupter Delisting-Serien ist byte-gleich aus Mandat I
uebernommen (``verdict_engine.load_verdict_prices``): EODHD-Serien enden
teils in unmoeglichen Mikro-Preis-Spruengen (+34.000x aus 0,005 USD). Erster
Tag mit |Rendite| > 100 % UND Vortagskurs < 1 USD -> ab da NaN. Konservativ:
der Name war dort ohnehin tot, und der Engine-Zwangsverkauf greift am letzten
sauberen Kurs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from research.mandat2.dividenden import auf_panel_skalieren
from research.mandat2.data_gate import (
    SEARCH_CUTOFF,
    TrialCounter,
    load_holdout,
    load_search,
)

DATA = Path(__file__).resolve().parents[1] / "mandat" / "data"


@dataclass
class CampaignData:
    """Preise, Dividenden, Index-Mitgliedschaft — bereits gefenstert.

    ``fenster`` sagt, WAS man in der Hand haelt. Wer es ausgibt oder
    protokolliert, macht damit sichtbar, auf welchen Daten ein Ergebnis
    entstanden ist.
    """

    close: pd.DataFrame  # trading_day x symbol
    div_panel: pd.DataFrame  # trading_day x symbol, Dividende je Stueck
    membership: pd.Series  # month_end -> frozenset(Mitglieder)
    fenster: str  # "SUCHE" | "HOLDOUT"
    von: pd.Timestamp
    bis: pd.Timestamp

    def __repr__(self) -> str:  # pragma: no cover - Diagnose
        return (
            f"CampaignData(fenster={self.fenster}, {self.von.date()}..{self.bis.date()}, "
            f"{self.close.shape[1]} Symbole, {len(self.close)} Tage)"
        )


def _load_close_raw() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "prices_verdict.parquet")
    close = df.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    r = close.pct_change(fill_method=None)
    bad = (r.abs() > 1.0) & (close.shift(1) < 1.0)
    for sym in close.columns[bad.any()]:
        first_bad = bad.index[bad[sym]][0]
        close.loc[first_bad:, sym] = np.nan
    return close


def _load_div_panel(close_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Ex-Daten auf den naechsten Handelstag schnappen — NUR im Fenster.

    Zwei Fallen, die hier beide zugeschlagen haben (E-070):

    1. **Das Gate galt nur fuer die Kurse.** ``dividends.parquet`` reicht bis
       2027-03-12; 22.507 Zeilen liegen im HOLDOUT. Sie wurden vorher direkt
       gelesen — an genau der Sperre vorbei, deren Modul das hier ist.
    2. **``clip`` staucht den Leak zu einer Randspitze.** Alle Holdout-Zeilen
       landeten gebuendelt auf dem letzten Suchtag: 728 Symbole an einem Tag
       gegen einen Median von 5, SPY 57,93 statt 1,33. Der Leak sah dadurch
       aus wie gueltige Daten statt wie ein Index-Ueberlauf.

    Deshalb: Zeilen ausserhalb des Index werden VERWORFEN, nicht geklemmt.
    """
    d = pd.read_parquet(DATA / "dividends.parquet")
    ex = pd.DatetimeIndex(pd.to_datetime(d["ex_date"]))
    if ex.tz is not None:
        ex = ex.tz_convert("UTC")
    idx = close_index
    if idx.tz is not None and ex.tz is None:
        ex = ex.tz_localize(idx.tz)
    elif idx.tz is None and ex.tz is not None:
        ex = ex.tz_convert("UTC").tz_localize(None)
    pos = idx.searchsorted(ex)
    im_fenster = (pos < len(idx)) & (ex >= idx[0])
    d = d.loc[im_fenster].assign(t=idx[pos[im_fenster]])
    return d.groupby(["t", "symbol"])["dividend"].sum().unstack()


def _load_membership(close_index: pd.DatetimeIndex) -> pd.Series:
    """month_end -> frozenset(Mitglieder), PIT (Snapshot <= as_of).

    Sortierte Tupel statt roher frozensets an den Aufrufer weiterzugeben ist
    NICHT noetig — E-051 (Frozenset-Determinismus) betraf die ITERATION ueber
    ein frozenset, nicht seine Speicherung. Wer hier iteriert, muss selbst
    sortieren.
    """
    snaps: list[tuple[pd.Timestamp, frozenset]] = []
    with open(DATA / "sp500_historical_constituents.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            d = pd.Timestamp(row["date"], tz="UTC")
            snaps.append((d, frozenset(t.strip() for t in row["tickers"].split(","))))
    snaps.sort()
    snap_dates = [d for d, _ in snaps]
    month_ends = close_index.to_series().groupby(close_index.to_period("M")).max()
    out = {}
    for me in month_ends:
        i = np.searchsorted(snap_dates, me, side="right") - 1
        if i >= 0:
            out[me] = snaps[i][1]
    return pd.Series(out)


def load_campaign(
    *,
    holdout: bool = False,
    candidate_id: str | None = None,
    begruendung: str | None = None,
    force: bool = False,
    trial_label: str | None = None,
    trials: int = 0,
) -> CampaignData:
    """Daten fuer einen Kampagnenlauf.

    Args:
        holdout: den EINEN Schuss ziehen. Verlangt ``candidate_id`` und
            ``begruendung``; Zugriff wird append-only protokolliert.
        trials / trial_label: Anzahl der in diesem Lauf verbrauchten Trials.
            Der Zaehler startet bei 1.964 (Mandat I) und geht in den
            DSR-Haircut ein. 0 = nur Daten ansehen, kein Trial.

    Raises:
        ValueError: ``holdout=True`` ohne ID/Begruendung.
        HoldoutViolation: zweiter Schuss auf dieselbe ID (ohne ``force``).
    """
    close_voll = _load_close_raw()
    # Der Skalenfaktor adj/raw MUSS auf der vollen Quelle bestimmt werden.
    # Die Rueckwaerts-Rekursion in dividenden.py verankert am letzten Kurs
    # ("dort ist adj == raw"). Auf einem GEFENSTERTEN Panel liegt dieser Anker
    # daneben — die Adjustierung ist ueber die volle Historie normiert.
    # Gemessen: SPY 1995-01-03 raw = 45,80 auf dem vollen Panel (EODHD-Ist
    # 45,7813) gegen 41,52 auf dem Suchfenster = 9,3 % daneben, und im
    # Holdout-Fenster faellt der Anker zufaellig richtig — Suche und Holdout
    # haetten dann unterschiedlich skalierte Dividenden (E-074).
    #
    # KEIN Holdout-Leck: rekonstruiert wird der ROHKURS von 1995, und der war
    # 1995 bekannt. Die Normierung ist ein Darstellungsartefakt der Quelle,
    # keine Information ueber die Zukunft. Was gefenstert wird, sind die
    # Zeilen — nicht die Skala.
    idx_voll = pd.DatetimeIndex(close_voll.index)
    div_nominal_voll = _load_div_panel(idx_voll).reindex(index=idx_voll)
    div_skaliert_voll = auf_panel_skalieren(close_voll, div_nominal_voll)
    close = close_voll

    if holdout:
        if not candidate_id or not begruendung:
            raise ValueError(
                "Der Holdout-Schuss verlangt candidate_id UND begruendung — "
                "sonst laesst sich hinterher nicht rekonstruieren, wofuer er "
                "verbraucht wurde."
            )
        close = load_holdout(
            close,
            candidate_id=candidate_id,
            begruendung=begruendung,
            force=force,
        )
        fenster = "HOLDOUT"
    else:
        close = load_search(close)
        fenster = "SUCHE"

    if close.empty:
        raise RuntimeError(f"Leeres {fenster}-Fenster — Datenpfad pruefen: {DATA}")

    # Spalten ohne einen einzigen Kurs im Fenster fliegen raus (Namen, die
    # erst spaeter existierten bzw. schon tot waren).
    close = close.dropna(axis=1, how="all")

    idx = pd.DatetimeIndex(close.index)
    div_panel = (
        div_skaliert_voll.reindex(index=idx)
        .reindex(columns=[c for c in div_skaliert_voll.columns if c in close.columns])
        .dropna(axis=1, how="all")
    )
    membership = _load_membership(idx)

    if trials:
        TrialCounter().increment(trials, label=trial_label or fenster)

    return CampaignData(
        close=close,
        div_panel=div_panel,
        membership=membership,
        fenster=fenster,
        von=idx[0],
        bis=idx[-1],
    )


def search_cutoff() -> pd.Timestamp:
    """Fuer Aufrufer, die den Schnitt nur ANZEIGEN wollen."""
    return SEARCH_CUTOFF
