"""Intraday-Panel fuer Mandat II — Rohdaten, bereinigt und gegated.

WARUM DIESE SCHICHT EXISTIERT
-----------------------------
Der EODHD-Intraday-Endpunkt liefert **rohe** Kurse. Ungepruefte Nutzung waere
wertlos gewesen: mehrere Symbole zeigen Stundenspruenge weit ueber 35 %, und das
sind fast keine Marktbewegungen, sondern Kapitalmassnahmen — Splits,
Reverse-Splits, Abspaltungen.

Die konkreten Faelle stehen bewusst NICHT hier, sondern werden von
``_split_diagnose()`` auf dem GEGATETEN Fenster erzeugt und landen als
``split_diagnose`` in ``results/p12_intraday_haltedauer.json``. Zwei Gruende:
ein frueherer Entwurf zitierte hier einen Split von 2018 — der liegt im Holdout
und war nur sichtbar, weil ich beim ersten Sichten das Roh-Parquet direkt
gelesen hatte, vor dem Bau dieses Gates (E-081). Und die von Hand abgeschriebenen
Werte veralteten sofort, als der Sitzungs- und der Abdeckungsfilter dazukamen
(E-085). Eine generierte Tabelle kann beides nicht.

Ein Momentum- oder Reversal-Signal auf unbereinigten Daten misst Splits. Ein
Stop-Loss feuert auf Splits. Jede Kennzahl waere Artefakt.

DIE BEREINIGUNG
---------------
Nicht ueber eine separate Split-Historie (zweite Wahrheit, eigene Luecken),
sondern ueber den Anker, der in dieser Kampagne ohnehin die Wahrheit ist: das
**tagesgenaue, total-return-adjustierte Panel** (`prices_verdict.parquet`).

    faktor(tag) = adj_close(tag) / roh_close(letzte Bar des Tages)
    intraday_adj(bar) = intraday_roh(bar) * faktor(tag(bar))

Der Faktor ist INNERHALB eines Tages konstant. Damit gilt:

* Intraday-Renditen bleiben unveraendert — richtig, denn Splits wirken
  ueber Nacht, nicht innerhalb der Sitzung.
* Uebernacht-Renditen werden korrekt um Split UND Dividende bereinigt.
* Das Intraday-Panel ist per Konstruktion konsistent mit dem Tagespanel,
  auf dem alle uebrigen Phasen der Kampagne rechnen. Keine zweite Wahrheit.

HOLDOUT
-------
Diese Schicht liest den Tagesanker ueber ``campaign_data.load_campaign()`` und
schneidet die Intraday-Bars auf dasselbe Fenster. Der Holdout ist damit nicht
nur „nicht ausgewertet", sondern **nicht vorhanden** — die Zeilen existieren im
zurueckgegebenen Objekt nicht.

HANDELSZEITEN — und warum das UTC-Label dafuer nicht taugt
-----------------------------------------------------------
Gefiltert wird auf die regulaere Sitzung, aber in **Boersenzeit**
(America/New_York, Stundenlabel 9..15), nicht nach UTC-Label. Der erste
Entwurf filterte auf UTC 13..20 und liess damit 12,5 % Extended-Hours-Bars
durch — genau EINEN pro Handelstag, weil die US-Sommerzeit den Versatz
zwischen 4 und 5 Stunden wechseln laesst. Vor- und nachboersliche Bars sind
duenn und tragen oft stehende Kurse; ein Reversal-Signal erzeugt darauf
Scheinsignale. Der Fehler war materiell: der 1-Stunden-Bruttowert verschob
sich dadurch um rund 5 Prozentpunkte.

US-Sitzungen kreuzen nie UTC-Mitternacht, deshalb bleibt das UTC-Datum ein
gueltiger Handelstag-Schluessel.

SURVIVORSHIP — EHRLICH BENANNT
------------------------------
Das Intraday-Universum ist NICHT survivorship-frei. Gezogen wurden Namen, die
2004-2016 durchgehend im Index waren; wer pleiteging oder rausflog, ist nicht
dabei. Ergebnisse hier sind deshalb **nach oben verzerrt** und taugen nur fuer
die RELATIVE Frage („aendert kuerzeres Halten etwas?"), nicht fuer absolute
Renditeaussagen. Die Kontrollen in ``p12`` sind entsprechend gebaut: verglichen
wird gegen dasselbe Universum bei taeglicher Aufloesung, nicht gegen SPY.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from research.mandat2.campaign_data import load_campaign

ROH = Path(__file__).resolve().parents[2] / "data" / "raw" / "intraday_1h"
BOERSE = "America/New_York"
RTH_VON, RTH_BIS = 9, 15  # Stundenlabel BOERSENZEIT (09:30-16:00 ET)
MIN_UEBERLAPPUNG = 250  # < 1 Handelsjahr gemeinsamer Tage -> Symbol verwerfen
MIN_ABDECKUNG = 0.90  # Anteil der Panel-Bars mit Kurs; darunter -> verwerfen
STUFEN_SCHWELLE = 0.005  # fuer den Stufen-Faktor (Robustheitsvariante)


@dataclass
class IntradayData:
    """Bereinigtes Stundenpanel + der Tagesanker, auf dem es beruht."""

    close: pd.DataFrame  # index: UTC-Stunde, columns: Symbol
    tages_close: pd.DataFrame  # derselbe Ausschnitt, taeglich (Anker)
    fenster: str
    roh_spruenge: int  # Stundenspruenge >35% VOR der Bereinigung
    rest_spruenge: int  # ... und danach
    verworfen: dict[str, str]  # Symbol -> Grund; stille Skips sind verboten
    stufig: bool  # True = Robustheitsvariante mit erzwungen stufigem Faktor
    split_diagnose: list[dict]  # groesste Rohspruenge, NUR aus dem Suchfenster

    def __str__(self) -> str:
        return (
            f"IntradayData[{self.fenster}] {self.close.shape[0]:,} Stunden x "
            f"{self.close.shape[1]} Symbole | "
            f"{self.close.index.min():%Y-%m-%d}..{self.close.index.max():%Y-%m-%d} | "
            f"Spruenge >35%: {self.roh_spruenge} roh -> {self.rest_spruenge} bereinigt"
        )


def _lade_roh(symbole: list[str]) -> dict[str, pd.DataFrame]:
    aus = {}
    for s in symbole:
        p = ROH / f"{s}.parquet"
        if p.exists():
            aus[s] = pd.read_parquet(p)
    return aus


def _n_spruenge(close: pd.DataFrame, schwelle: float = 0.35) -> int:
    """Zaehlt Stundenspruenge oberhalb der Schwelle — der Split-Detektor."""
    r = close.pct_change(fill_method=None)
    return int((r.abs() > schwelle).sum().sum())


def _split_diagnose(roh: pd.DataFrame, schwelle: float = 0.35) -> list[dict]:
    """Die groessten Rohspruenge, als versioniertes Artefakt statt als Prosa.

    Jede Zahl, die spaeter in einem Befund steht, muss aus einem
    ``results/*.json`` stammen (E-073/E-076) — deshalb wird die Split-Tabelle
    hier erzeugt und nicht von Hand abgeschrieben.
    """
    r = roh.pct_change(fill_method=None)
    aus = []
    for sym in roh.columns:
        s = r[sym].dropna()
        if s.empty or s.abs().max() <= schwelle:
            continue
        i = s.abs().idxmax()
        aus.append(
            {
                "symbol": sym,
                "zeitpunkt": f"{i:%Y-%m-%d %H:%M}",
                "roher_sprung": float(s.loc[i]),
            }
        )
    return sorted(aus, key=lambda d: abs(d["roher_sprung"]), reverse=True)


def _stufig_machen(faktor: pd.Series) -> pd.Series:
    """Erzwingt eine Treppenfunktion: nur Spruenge > STUFEN_SCHWELLE bleiben.

    Der Tagesfaktor SOLL eine Treppe sein (er korrigiert Kapitalmassnahmen und
    Dividenden, also diskrete Ereignisse). Gemessen ist er es nicht: er
    absorbiert zusaetzlich die Differenz zwischen Vendor-Tagesschluss und
    letzter Stundenbar — ein Rauschen von 11-31 bps pro Tag mit einer
    lag-1-Autokorrelation um -0,4. Das ist REVERSIERENDES Rauschen und damit
    gleichgerichtet mit dem Effekt, den ein Intraday-Test am kurzen Ende sucht.

    Diese Variante entfernt es, indem sie kleine Faktoraenderungen unterdrueckt.
    Sie ist nicht „richtiger" als das Original — sie ist die Gegenprobe. Die
    Differenz beider Laeufe ist die Artefaktschranke des Verfahrens und gehoert
    als solche ausgewiesen (Anti-Pattern E-083).
    """
    f = faktor.astype(float)
    aus = f.copy()
    lauf = float(f.iloc[0])
    for i in range(len(f)):
        if abs(float(f.iloc[i]) / lauf - 1.0) > STUFEN_SCHWELLE:
            lauf = float(f.iloc[i])
        aus.iloc[i] = lauf
    return aus


def load_intraday(
    symbole: list[str] | None = None, *, stufig: bool = False
) -> IntradayData:
    """Bereinigtes Stundenpanel im SUCH-Fenster. Holdout ist nicht enthalten.

    ``stufig=True`` erzwingt einen Treppen-Tagesfaktor (Gegenprobe, s.
    ``_stufig_machen``).
    """
    if symbole is None:
        symbole = sorted(p.stem for p in ROH.glob("*.parquet"))
    roh = _lade_roh(symbole)
    verworfen = {s: "kein Parquet" for s in symbole if s not in roh}
    if not roh:
        raise RuntimeError(f"Keine Intraday-Parquets in {ROH}")

    tag = load_campaign()  # gegated: endet am SEARCH_CUTOFF
    tages_close = tag.close

    adj_spalten: dict[str, pd.Series] = {}
    roh_spalten: dict[str, pd.Series] = {}
    for sym, df in roh.items():
        if sym not in tages_close.columns:
            verworfen[sym] = "nicht im Tagesanker"
            continue
        s = df["close"].copy()
        s.index = pd.to_datetime(s.index, utc=True)
        et = s.index.tz_convert(BOERSE)
        s = s[(et.hour >= RTH_VON) & (et.hour <= RTH_BIS)]
        # Auf das Suchfenster schneiden -> Holdout-Zeilen existieren nicht.
        letzter = pd.Timestamp(tages_close.index.max())
        letzter = letzter.tz_localize("UTC") if letzter.tz is None else letzter
        grenze = letzter + pd.Timedelta(days=1)
        s = s[s.index < grenze]
        if s.empty:
            verworfen[sym] = "keine Bars im Suchfenster"
            continue

        handelstag = pd.Index(s.index.tz_convert("UTC").date, name="tag")
        anker = tages_close[sym].dropna()
        anker.index = pd.Index(pd.to_datetime(anker.index).date, name="tag")
        # Roh-Schlusskurs je Tag = letzte Bar des Tages
        roh_tages = s.groupby(handelstag).last()
        gemeinsam = anker.index.intersection(roh_tages.index)
        if len(gemeinsam) < MIN_UEBERLAPPUNG:
            verworfen[sym] = f"nur {len(gemeinsam)} gemeinsame Tage"
            continue
        faktor = (anker.loc[gemeinsam] / roh_tages.loc[gemeinsam]).replace(
            [float("inf"), float("-inf")], pd.NA
        )
        faktor = faktor.dropna().sort_index()
        if faktor.empty:
            verworfen[sym] = "kein gueltiger Tagesfaktor"
            continue
        if stufig:
            faktor = _stufig_machen(faktor)
        f_bar = pd.Series(faktor.reindex(handelstag).to_numpy(), index=s.index)
        adj = (s * f_bar).dropna()
        if adj.empty:
            verworfen[sym] = "nach Bereinigung leer"
        else:
            adj_spalten[sym] = adj
            roh_spalten[sym] = s.reindex(adj.index)

    if not adj_spalten:
        raise RuntimeError("Kein Symbol ueberlebte die Bereinigung")

    close = pd.DataFrame(adj_spalten).sort_index()
    # Abdeckungsfilter: ein Name, der ueber weite Strecken keinen Kurs hat, ist
    # fuer den KANDIDATEN waehlbar, fuer einen Buy-and-Hold-Benchmark aber nicht
    # kaufbar. Diese Asymmetrie waere ein Benchmark-Bias (E-079), deshalb faellt
    # der Name aus BEIDEN Seiten heraus statt nur aus einer.
    abdeckung = close.notna().mean()
    duenn = abdeckung[abdeckung < MIN_ABDECKUNG].index
    for sym in duenn:
        verworfen[sym] = f"Abdeckung nur {abdeckung[sym]:.1%}"
    close = close.drop(columns=duenn)
    if close.empty:
        raise RuntimeError("Alle Symbole am Abdeckungsfilter gescheitert")
    rohp = pd.DataFrame(roh_spalten).sort_index()[close.columns]
    return IntradayData(
        close=close,
        tages_close=tages_close[close.columns],
        fenster=f"SUCHE bis {tages_close.index.max():%Y-%m-%d}",
        roh_spruenge=_n_spruenge(rohp),
        rest_spruenge=_n_spruenge(close),
        verworfen=verworfen,
        split_diagnose=_split_diagnose(rohp),
        stufig=stufig,
    )


if __name__ == "__main__":
    d = load_intraday()
    print(d)
    print("")
    print("Spalten: " + ", ".join(d.close.columns))
    if d.verworfen:
        print("Verworfen:", d.verworfen)
    print("")
    print("Groesste ROH-Spruenge (Kapitalmassnahmen, nur Suchfenster):")
    for e in d.split_diagnose[:6]:
        print(f"  {e['symbol']:<6}{e['roher_sprung']:>9.1%}  {e['zeitpunkt']}")
    print("")
    print("Groesste verbleibende Spruenge (sollten echte Nachrichten sein):")
    r = d.close.pct_change(fill_method=None)
    for sym in r.abs().max().sort_values(ascending=False).head(5).index:
        wann = r[sym].abs().idxmax()
        print(f"  {sym:<6}{r[sym].loc[wann]:>9.1%}  {wann:%Y-%m-%d %H:%M}")
