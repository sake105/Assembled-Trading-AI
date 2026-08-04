"""P12g — Kann der Intraday-Endpunkt die Survivorship-Lücke überhaupt schließen?

DIE FRAGE
---------
`BEFUND_DATENQUALITAET.md` endet mit dem Satz, der nächste Schritt sei ein
Panel, das Ausscheider mitführt. Der naheliegende Weg dorthin: mehr Symbole
ziehen. Das Intraday-Universum wuchs dabei von 21 auf knapp 300 Namen.

Dieses Skript beantwortet, ob dieser Weg zum Ziel führt. Es ist **Diagnostik,
kein Backtest** — der Trial-Zähler bleibt unberührt.

WARUM DAS DATEISYSTEM DIE FRAGE NICHT BEANTWORTET (E-112)
---------------------------------------------------------
Die erste Fassung schloss aus fehlenden Parquet-Dateien, der Endpunkt liefere
diese Namen nicht, und berechnete daraus einen „Anreicherungsfaktor" von 3,06.
Eine Stichprobe gegen dieselbe API widerlegte das sofort: AMZN, GILD, VRSN und
fünf weitere Namen aus genau dieser Gruppe liefern 7.000–8.000 Bars im
Suchfenster. Sie wurden schlicht **nie angefragt**.

``intraday_pull.py`` schreibt bei Leerergebnis weder Datei noch Protokoll —
damit sind „nie angefragt", „angefragt und leer" und „Anfrage fehlgeschlagen"
auf der Platte ununterscheidbar. Eine Kennzahl über dieses Verzeichnis misst
die **Anfrageliste**, nicht den Anbieter. Und die Anfrageliste ist genau die
Größe, deren Verzerrung untersucht werden soll.

Deshalb steht hier eine **API-Probe** im Mittelpunkt, kein ``ls``.

WARUM UNTER DEM MITGLIEDSCHAFTSSYMBOL (E-113)
---------------------------------------------
Die Q-Ticker (LEHMQ, WAMUQ, EKDKQ, MTLQQ, CCTYQ) entstehen **erst** mit dem
Chapter-11-Handel. Während der Index-Mitgliedschaft hießen die Namen LEH, WM,
EK, GM, CC. Ein Negativbefund auf dem Q-Ticker ist fast garantiert und beweist
nichts über die Historie des Namens — er kann die These nicht widerlegen, also
stützt er sie auch nicht. Geprüft werden deshalb **beide** Symbole.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from research.mandat2.campaign_data import load_campaign  # noqa: E402

HIER = Path(__file__).resolve().parent
OUT = HIER / "results"
INTRADAY = HIER.parents[1] / "data" / "raw" / "intraday_1h"

#: Exakt das P12-Fenster, damit die Zahlen mit P12d/P12e vergleichbar sind.
VON, BIS = "2006-06-01", "2016-12-31"

#: Ausscheider des Suchfensters, die im PIT-Universum stehen — je mit dem
#: Symbol, unter dem sie DAMALS gehandelt wurden, dem Post-Insolvenz-Ticker
#: und einem Probefenster VOR dem Ereignis (Holdout-Disziplin: alles im
#: Suchzeitraum). Klarnamen stehen hier, damit der Befund sie nicht als
#: statische Prosa neben einer generierten Liste führen muss (F-senior-6).
AUSSCHEIDER = (
    ("LEH", "LEHMQ", "Lehman Brothers", 2008, 3),
    ("WM", "WAMUQ", "Washington Mutual", 2008, 3),
    ("EK", "EKDKQ", "Eastman Kodak", 2011, 6),
    ("GM", "MTLQQ", "General Motors", 2008, 6),
    ("CC", "CCTYQ", "Circuit City", 2008, 3),
    ("BSC", None, "Bear Stearns", 2007, 9),
)

#: Überlebende als Kontrollgruppe — dieselbe Abfrage, anderer Ausgang. Ohne sie
#: wäre ein Negativbefund nicht von einem kaputten Aufruf zu unterscheiden.
KONTROLLE = ("AMZN", "GILD", "VRSN", "ADBE", "NVDA", "COST", "ROST", "PAYX")


def vorhandene_symbole(ordner: Path) -> set[str]:
    return {p.stem.upper() for p in ordner.glob("*.parquet")}


def probiere(symbole, tok, *, jahr: int, monat: int, tage: int = 45) -> dict:
    """Fragt den Endpunkt und zählt BARS — keine Kurse.

    Rückgabe je Symbol: Anzahl Bars, oder ein Fehlerstring. Ein Symbol mit
    0 Bars ist die Aussage, um die es geht; ein Fehlerstring ist keine.
    """
    from research.mandat2.intraday_pull import hole_fenster

    von = dt.datetime(jahr, monat, 1, tzinfo=dt.timezone.utc)
    bis = von + dt.timedelta(days=tage)
    aus: dict[str, object] = {}
    for sym in symbole:
        if sym is None:
            continue
        try:
            aus[sym] = len(hole_fenster(sym, von, bis, tok))
        except Exception as ex:  # noqa: BLE001 — Fehlerart gehoert ins Artefakt
            aus[sym] = f"ERR:{type(ex).__name__}"
    return aus


def api_probe() -> dict:
    """Die eigentliche Messung: liefert der Endpunkt Ausscheider oder nicht?"""
    from research.mandat2.intraday_pull import token

    tok = token()
    ausscheider = {}
    for sym, q, name, jahr, monat in AUSSCHEIDER:
        treffer = probiere([sym, q], tok, jahr=jahr, monat=monat)
        ausscheider[sym] = {
            "name": name,
            "q_ticker": q,
            "probefenster": f"{jahr}-{monat:02d}",
            "bars_mitgliedschaftssymbol": treffer.get(sym),
            "bars_q_ticker": treffer.get(q) if q else None,
        }
    kontrolle = probiere(KONTROLLE, tok, jahr=2006, monat=7, tage=30)
    return {"ausscheider": ausscheider, "kontrolle": kontrolle}


def bilanz(membership, vorhanden: set[str]) -> dict:
    """Stand des BISHERIGEN Pulls — ausdrücklich keine Endpunkt-Eigenschaft.

    Die Quoten beschreiben, wie die Anfrageliste zusammengesetzt war, die zu
    den vorhandenen Dateien geführt hat. Da 294 von 294 Namen mit Datei auch
    eine Tagespreisspalte haben (gegen 78,8 % ohne Datei), stammt die Liste
    mit hoher Wahrscheinlichkeit aus dem Tagespanel — dessen eigene
    Abdeckungslücke P12d bereits als survivorship-behaftet ausgewiesen hat.
    Der Faktor misst dann teilweise die Verzerrung seiner eigenen Quelle
    (Stage-2-Finding F-senior-7).
    """
    im_fenster = membership.loc[VON:BIS]
    alle: set[str] = set()
    for s in im_fenster:
        alle |= set(s)
    erste, letzte = set(im_fenster.iloc[0]), set(im_fenster.iloc[-1])
    mit, ohne = erste & vorhanden, erste - vorhanden

    def quote(gruppe: set[str]) -> float | None:
        return len(gruppe & letzte) / len(gruppe) if gruppe else None

    q_mit, q_ohne = quote(mit), quote(ohne)
    faktor = (
        q_mit / q_ohne if (q_mit is not None and q_ohne not in (None, 0.0)) else None
    )
    return {
        "hinweis": (
            "Beschreibt die Zusammensetzung der Anfrageliste des bisherigen "
            "Pulls, NICHT das Verhalten des Endpunkts — dafuer siehe api_probe."
        ),
        "fenster": f"{VON}..{BIS}",
        "referenz_start": f"{im_fenster.index[0]:%Y-%m-%d}",
        "referenz_ende": f"{im_fenster.index[-1]:%Y-%m-%d}",
        "n_dateien": len(vorhanden),
        "n_pit_mitglieder": len(alle),
        "n_mit_datei": len(alle & vorhanden),
        "abdeckung": len(alle & vorhanden) / len(alle) if alle else 0.0,
        "n_referenz_mit": len(mit),
        "n_referenz_ohne": len(ohne),
        "ueberlebensquote_mit_datei": q_mit,
        "ueberlebensquote_ohne_datei": q_ohne,
        "anreicherungsfaktor": faktor,
        # Der Fall q_ohne == 0 ist die STAERKSTE Verzerrung, liefert aber
        # keinen Quotienten. Ohne dieses Flag las der Renderer `None` als
        # „neutral" — ein Fail-Open in Richtung Entwarnung (F-senior-4).
        "maximal_verzerrt": bool(q_ohne == 0.0 and (q_mit or 0.0) > 0.0),
    }


def verdikt(probe: dict | None) -> dict:
    """Was die Probe hergibt — und ob sie ueberhaupt etwas hergibt.

    WARUM DAS EINE EIGENE FUNKTION IST (Stage-3-Finding F-auditor-1)
    ----------------------------------------------------------------
    Skript und Renderer entschieden das getrennt, und beide keyten auf
    „mindestens ein stummer Ausscheider" statt auf „Evidenz vollstaendig".
    Zwei live reproduzierte Folgen:

    * Scheitern ALLE Calls (`ERR:...`), war `stumm` leer — und gemeldet wurde
      „der Endpunkt liefert Ausscheider, der Weg ist gangbar". Eine positive
      Behauptung aus null Messung.
    * War einer von sechs stumm und fuenf lieferten, stand trotzdem „die
      Ausscheider sind bei dieser Quelle nicht zu haben".

    Ein Fehlerstring ist keine Aussage — das stand im Docstring von
    ``probiere`` und wurde von keinem Konsumenten beachtet. Deshalb hier:
    erst Validitaet, dann Einstimmigkeit, dann Verdikt.

    ``status`` ist eines von:
      * ``"keine_probe"``       — nicht gemessen
      * ``"unvollstaendig"``    — gemessen, aber Fehler oder tote Kontrolle
      * ``"weg_zu"``            — alle geprueften Ausscheider stumm
      * ``"teilweise"``         — nur manche stumm; keine Verallgemeinerung
      * ``"weg_offen"``         — kein Ausscheider stumm
    """
    if not probe:
        return {"status": "keine_probe", "n_stumm": 0, "n_ausscheider": 0}
    aus = probe.get("ausscheider") or {}
    kontrolle = probe.get("kontrolle") or {}

    def zahl(x: object) -> bool:
        return isinstance(x, int)

    # Ein einziger Fehler entwertet die Probe: er ist nicht von einem
    # Negativbefund zu unterscheiden.
    fehler = [
        s
        for s, v in aus.items()
        if not zahl(v["bars_mitgliedschaftssymbol"])
        or not (v["bars_q_ticker"] is None or zahl(v["bars_q_ticker"]))
    ] + [s for s, n in kontrolle.items() if not zahl(n)]
    lebt = [s for s, n in kontrolle.items() if zahl(n) and n > 0]
    stumm = [
        v["name"]
        for v in aus.values()
        if v["bars_mitgliedschaftssymbol"] == 0 and v["bars_q_ticker"] in (0, None)
    ]
    basis = {
        "n_stumm": len(stumm),
        "n_ausscheider": len(aus),
        "stumm": stumm,
        "n_kontrolle_lebt": len(lebt),
        "n_kontrolle": len(kontrolle),
        "fehler": sorted(fehler),
    }
    if fehler or not kontrolle or len(lebt) != len(kontrolle):
        return {**basis, "status": "unvollstaendig"}
    if not stumm:
        return {**basis, "status": "weg_offen"}
    if len(stumm) == len(aus):
        return {**basis, "status": "weg_zu"}
    return {**basis, "status": "teilweise"}


def _q(x: object) -> str:
    """Quote als Prozent, None-sicher."""
    return "n/a" if x is None else f"{x:.1%}"


def _f(x: object) -> str:
    """Faktor als Verhaeltnis — NICHT als Prozent."""
    return "n/a" if x is None else f"{x:.2f}x"


def _bars(x: object) -> str:
    """Bar-Zahl. 0 ist eine AUSSAGE, kein fehlender Wert.

    `x or "-"` haette hier aus der 0 ein Strich gemacht — also genau den
    Befund unsichtbar, um den es geht (gleiche Klasse wie E-111).
    """
    return "-" if x is None else str(x)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    if not INTRADAY.exists():
        print(f"[ERROR] {INTRADAY} fehlt — erst den Pull laufen lassen.")
        return 1
    mit_probe = "--probe" in sys.argv
    d = load_campaign()
    print(d, flush=True)
    print("Diagnostik, kein Backtest: Trial-Zaehler bleibt unberuehrt.\n")

    b = bilanz(d.membership, vorhandene_symbole(INTRADAY))
    print("STAND DES BISHERIGEN PULLS (Anfrageliste, nicht Endpunkt!)")
    print(f"  Intraday-Dateien : {b['n_dateien']}")
    print(f"  PIT-Mitglieder   : {b['n_pit_mitglieder']}")
    print(f"  davon mit Datei  : {b['n_mit_datei']} ({b['abdeckung']:.1%})")
    print(
        f"  Ueberleben mit/ohne Datei: {_q(b['ueberlebensquote_mit_datei'])}"
        f" / {_q(b['ueberlebensquote_ohne_datei'])}"
        f"  Faktor {_f(b['anreicherungsfaktor'])}"
    )
    if b["maximal_verzerrt"]:
        print("  ACHTUNG: kein Name ohne Datei ueberlebt — staerkste Verzerrung.")

    ergebnis = {"bilanz": b, "api_probe": None}
    if mit_probe:
        print("\nAPI-PROBE (Bars, keine Kurse; Fenster im Suchzeitraum)")
        p = api_probe()
        ergebnis["api_probe"] = p
        print(f"  {'Name':<20}{'Symbol':>8}{'Bars':>8}{'Q-Ticker':>10}{'Bars':>8}")
        for sym, v in p["ausscheider"].items():
            print(
                f"  {v['name']:<20}{sym:>8}"
                f"{_bars(v['bars_mitgliedschaftssymbol']):>8}"
                f"{(v['q_ticker'] or '-'):>10}{_bars(v['bars_q_ticker']):>8}"
            )
        k = p["kontrolle"]
        geliefert = sum(1 for n in k.values() if isinstance(n, int) and n > 0)
        print(f"  Kontrollgruppe (Ueberlebende): {geliefert}/{len(k)} liefern Bars")
    else:
        print("\n[SKIP] API-Probe (--probe zum Ausfuehren; kostet Calls).")

    ergebnis["verdikt"] = verdikt(ergebnis["api_probe"])
    (OUT / "p12g_pull_bilanz.json").write_text(
        json.dumps(ergebnis, indent=2), encoding="utf-8"
    )
    print(f"\n-> {OUT / 'p12g_pull_bilanz.json'}")

    print(chr(10) + "=" * 72)
    v = verdikt(ergebnis["api_probe"])
    if v["status"] == "keine_probe":
        print("BEFUND: keiner — ohne API-Probe ist ueber den Endpunkt nichts")
        print("        auszusagen. Das Dateisystem beantwortet die Frage nicht.")
    elif v["status"] == "unvollstaendig":
        print("BEFUND: keiner — die Probe ist unvollstaendig.")
        if v["fehler"]:
            print(f"        Fehlgeschlagene Abfragen: {v['fehler'][:8]}")
        if v["n_kontrolle_lebt"] != v["n_kontrolle"]:
            print(
                f"        Kontrollgruppe liefert nur {v['n_kontrolle_lebt']}/"
                f"{v['n_kontrolle']} — ein Negativbefund waere hier nicht vom"
            )
            print("        kaputten Aufruf zu unterscheiden.")
    elif v["status"] == "weg_zu":
        print(
            f"BEFUND: Der Endpunkt fuehrt {v['n_stumm']} von "
            f"{v['n_ausscheider']} geprueften Ausscheidern NICHT —"
        )
        print("        auch nicht unter dem Symbol der Mitgliedschaft. Die")
        print(
            f"        Kontrollgruppe liefert {v['n_kontrolle_lebt']}/"
            f"{v['n_kontrolle']}, der Aufruf ist also in Ordnung."
        )
        print("        Fuer die Survivorship-Korrektur ist dieser Weg zu;")
        print("        es braucht Tagesdaten mit Delisting-Kursen.")
    elif v["status"] == "teilweise":
        print(f"BEFUND: {v['n_stumm']} von {v['n_ausscheider']} geprueften")
        print("        Ausscheidern liefern nichts — die uebrigen schon. Damit")
        print("        ist die Quelle nicht pauschal blind; welche Namen sie")
        print("        fuehrt, ist einzeln zu pruefen.")
    else:
        print("BEFUND: Alle geprueften Ausscheider liefern Bars — der Weg ueber")
        print("        mehr Anfragen ist gangbar.")
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
