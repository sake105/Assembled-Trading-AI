"""P12f — Dreht ein Verdikt, wenn das Panel bereinigt ist?

DIE FRAGE
---------
P12e hat gezeigt, dass Preisfehler in die Ergebnisse eingegangen sind: zwei
Namen wurden über einen Übergangstag gehalten (GPS mit +12,4 % Portfolio-
Tagesrendite, Rang 2 von 5.548), vier weitere wurden mit kontaminierten
Momentum-Beinen gewählt. Umfang: 0,44 % der Auswahlplätze.

Das ist klein — aber „klein" ist keine Antwort auf „dreht es ein Verdikt?".
Diese Frage ist nur durch einen Neulauf zu beantworten, und genau der passiert
hier: **dasselbe Parametergitter wie P2, einmal auf dem Originalpanel und
einmal auf dem gespleißten.**

WAS VERGLICHEN WIRD
-------------------
Nicht einzelne Endwerte — die ändern sich zwangsläufig. Entscheidend sind zwei
Aussagen, die das Verdikt der Kampagne tragen:

1. **Besteht irgendeine Parametrisierung die Zielfunktion?** Also: Median über
   alle rollierenden 10-Jahres-Fenster über dem Benchmark **und** MaxDD
   ≥ −35 % in *jedem* Fenster. Wenn das im Original „nein" ist und bereinigt
   „ja" (oder umgekehrt), ist das Verdikt datenabhängig.
2. **Wandert das Optimum?** P2 schloss aus dem unbewegten Optimum, dass die
   Steuer nicht die bindende Restriktion ist. Wandert es nach der Bereinigung,
   war auch dieser Schluss auf Vendor-Fehlern gebaut.

WARUM NICHT DER ENDWERT
-----------------------
Die erste Fassung verglich Endwerte gegen den Benchmark und hätte „5 von 24
schlagen" als Verdikt ausgewiesen. Das ist nicht das Kriterium der Kampagne:
P2 hielt ausdrücklich fest, dass der beste Kandidat den Index **bei der
Rendite schlägt** und an der Nebenbedingung scheitert. Ein Endwertvergleich
hätte die Frage also ausgetauscht statt sie zu wiederholen — und wäre dabei
gegen eine Aussage getestet worden, die die Kampagne nie gemacht hat.

TRIAL-BUCHHALTUNG
-----------------
Dies ist eine **Wiederholung derselben Hypothesen auf korrigierten Daten**,
keine neue Suche: das Gitter ist identisch mit P2, kein Parameter kommt hinzu.
Der Zähler wird deshalb nicht erhöht (E-090). Die Entscheidung steht hier,
damit sie nachprüfbar ist, statt still zu passieren.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from research.mandat2.campaign_data import load_campaign  # noqa: E402
from research.mandat2.engine import run_buy_and_hold, run_momentum  # noqa: E402
from research.mandat2.metrics import DD_DECKEL, auswerten  # noqa: E402
from research.mandat2.p12d_survivorship_schranke import (  # noqa: E402
    korruptions_spannen,
)
from research.mandat2.panel_bereinigt import bereinige, gegenprobe  # noqa: E402
from research.mandat2.tax_regimes import make_regime  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"

# Identisch zu P2 — sonst waere es keine Wiederholung, sondern eine neue Suche.
WELTEN = [
    ("ZERO", "ZERO", {}),
    ("PRIVAT_DE", "PRIVAT_DE", {}),
    ("GMBH+FK", "GMBH_THESAURIEREND", {"fixkosten_pa": 3500.0}),
]
HALTETAGE = [0, 90, 365, 730]
RANK_OUT = [30, 60, 200]
HEBEL = [1.0, 1.5]


def gitter(data, label: str, name: str, kwargs: dict) -> tuple[list[dict], float]:
    """Dasselbe Gitter wie P2 — und dieselbe ZIELFUNKTION.

    Erste Fassung verglich Endwerte gegen den Benchmark. Das ist nicht das
    Kriterium der Kampagne: die Zielfunktion ist der Median des Endvermoegens
    ueber alle rollierenden 10-Jahres-Fenster **unter der bindenden
    Nebenbedingung** MaxDD >= -35 % in JEDEM Fenster. P2 hielt ausdruecklich
    fest, dass der beste Kandidat den Index bei der Rendite schlaegt und am
    Deckel scheitert — ein Endwertvergleich haette also die Frage
    ausgetauscht, statt sie zu wiederholen, und dabei ein `5/24 schlaegt` als
    Verdikt ausgewiesen, wo die Kampagne `0/24 bestanden` meint.
    """
    bench = run_buy_and_hold(data, make_regime(name, **kwargs))
    b_end = float(bench.equity_netto.iloc[-1])
    zeilen = []
    for haltetage, rank_out, hebel in product(HALTETAGE, RANK_OUT, HEBEL):
        k = run_momentum(
            data,
            make_regime(name, **kwargs),
            top_in=20,
            rank_out=rank_out,
            min_haltetage=haltetage,
            hebel=hebel,
        )
        a = auswerten(
            k.equity_netto,
            bench.equity_netto,
            label=f"{label} hold{haltetage} out{rank_out} x{hebel}",
        )
        zeilen.append(
            {
                "welt": label,
                "haltetage": haltetage,
                "rank_out": rank_out,
                "hebel": hebel,
                "endwert": float(k.equity_netto.iloc[-1]),
                "median_kandidat": a.median_kandidat,
                "median_benchmark": a.median_benchmark,
                "schlimmster_maxdd": a.schlimmster_maxdd,
                "deckel_eingehalten": a.deckel_eingehalten,
                "schlaegt_bench": a.schlaegt_benchmark,
                "bestanden": a.bestanden,
            }
        )
    return zeilen, b_end


def bester_kandidat(zeilen: list[dict]) -> dict:
    """Der beste Kandidat nach der ZIELGROESSE der Kampagne.

    Nicht nach Endwert: die Kampagne optimiert den Median ueber rollierende
    Fenster. Die beiden fallen auseinander — ein Kandidat mit hohem Endwert kann
    einen schlechten Median haben, wenn sein Vorsprung aus wenigen Fenstern
    kommt. Steht hier als eigene Funktion, weil eine Mutation zu `endwert` sonst
    von keinem Test gefangen wird (Stage-1-Finding F6).
    """
    return max(zeilen, key=lambda z: z["median_kandidat"])


def main() -> int:
    OUT.mkdir(exist_ok=True)
    d = load_campaign()
    print(d, flush=True)
    print("Trial-Zaehler NICHT erhoeht: Wiederholung derselben Hypothesen auf")
    print("korrigierten Daten, kein neuer Parameter (E-090).\n", flush=True)

    alle: set[str] = set()
    for m in d.membership:
        alle |= set(m)
    spannen = korruptions_spannen(d.close, alle & set(d.close.columns))
    close_neu, div_neu, protokoll = bereinige(d.close, spannen, d.div_panel)
    n_spannen = sum(len(v) for v in protokoll.values())
    unaufloesbar = sorted(k for k, v in spannen.items() if v["unaufloesbar"])
    gp = gegenprobe(d.close, close_neu)
    # Die Invariante, die beim Mitskalieren der Dividenden erhalten bleiben
    # muss: Dividende JE KURSEINHEIT. Wird nur close geteilt, steigt sie um
    # genau f — bei WIN von 26 % auf 274 % (F-test-3).
    q_alt = (d.div_panel / d.close).replace([np.inf, -np.inf], np.nan)
    q_neu = (div_neu / close_neu).replace([np.inf, -np.inf], np.nan)
    div_abw = float(np.nanmax(np.abs((q_neu - q_alt).to_numpy())))
    print(f"Bereinigt: {len(protokoll)} Symbole, {n_spannen} Spannen gespleisst")
    print(f"Nicht bereinigt (verschraenkte Skalen): {len(unaufloesbar)} Namen")
    print(
        f"Gegenprobe: {gp['auffaellig_original']} auffaellige Tage im Original, "
        f"{gp['beseitigt']} beseitigt, {gp['neu_entstanden']} NEU entstanden, "
        f"{gp['auffaellig_bereinigt']} bleiben"
    )
    print(f"max |Aenderung Dividendenrendite|: {div_abw:.3e} (soll ~0)\n", flush=True)
    d_neu = replace(d, close=close_neu, div_panel=div_neu)

    ergebnis = {
        "dd_deckel": DD_DECKEL,
        "n_symbole_bereinigt": len(protokoll),
        "n_spannen": n_spannen,
        "unaufloesbar": unaufloesbar,
        "unaufloesbar_grund": {
            k: v["unaufloesbar_grund"] for k, v in spannen.items() if v["unaufloesbar"]
        },
        "gegenprobe": {k: v for k, v in gp.items() if k != "wo_neu"},
        "dividendenrendite_max_abweichung": div_abw,
        "protokoll": protokoll,
        "welten": {},
    }
    print(
        f"{'Welt':<11}{'Panel':<12}{'Median K':>10}{'Median B':>10}"
        f"{'schl. DD':>10}{'schlaegt':>10}{'BESTANDEN':>11}"
    )
    print("-" * 72)
    for label, name, kwargs in WELTEN:
        eintrag = {}
        for panel_name, daten in (("original", d), ("bereinigt", d_neu)):
            zeilen, b_end = gitter(daten, label, name, kwargs)
            bester = bester_kandidat(zeilen)
            eintrag[panel_name] = {
                "benchmark": b_end,
                "bester": bester,
                "n_schlagen_bench": sum(1 for z in zeilen if z["schlaegt_bench"]),
                "n_deckel_gehalten": sum(1 for z in zeilen if z["deckel_eingehalten"]),
                "n_bestanden": sum(1 for z in zeilen if z["bestanden"]),
                "zeilen": zeilen,
            }
            print(
                f"{label:<11}{panel_name:<12}{bester['median_kandidat']:>10.3f}"
                f"{bester['median_benchmark']:>10.3f}"
                f"{bester['schlimmster_maxdd']:>9.1%}"
                f"{eintrag[panel_name]['n_schlagen_bench']:>7}/{len(zeilen)}"
                f"{eintrag[panel_name]['n_bestanden']:>8}/{len(zeilen)}"
            )
        # Wandert das Optimum?
        o, b = eintrag["original"]["bester"], eintrag["bereinigt"]["bester"]
        eintrag["optimum_wandert"] = (
            o["haltetage"] != b["haltetage"]
            or o["rank_out"] != b["rank_out"]
            or o["hebel"] != b["hebel"]
        )
        # Das Verdikt der Kampagne lautet „bestanden / nicht bestanden" —
        # Zielfunktion UND Deckel. Es dreht, wenn eine Seite von null
        # bestandenen Parametrisierungen auf mindestens eine wechselt.
        eintrag["verdikt_dreht"] = (eintrag["original"]["n_bestanden"] == 0) != (
            eintrag["bereinigt"]["n_bestanden"] == 0
        )
        eintrag["schlaegt_dreht"] = (eintrag["original"]["n_schlagen_bench"] == 0) != (
            eintrag["bereinigt"]["n_schlagen_bench"] == 0
        )
        ergebnis["welten"][label] = eintrag

    (OUT / "p12f_neulauf_bereinigt.json").write_text(
        json.dumps(ergebnis, indent=2), encoding="utf-8"
    )
    print(f"\n-> {OUT / 'p12f_neulauf_bereinigt.json'}")

    dreht = [w for w, e in ergebnis["welten"].items() if e["verdikt_dreht"]]
    wandert = [w for w, e in ergebnis["welten"].items() if e["optimum_wandert"]]
    print("\n" + "=" * 72)
    if dreht:
        print(f"BEFUND: Das Verdikt DREHT in {dreht} — ob eine Parametrisierung")
        print("        die Zielfunktion UND den DD-Deckel besteht, haengt an den")
        print("        Preisfehlern. Alle betroffenen Phasen sind neu zu bewerten.")
    else:
        print("BEFUND: Das Verdikt DREHT IN KEINER Steuerwelt. Ob eine")
        print("        Parametrisierung Zielfunktion UND DD-Deckel besteht, ist")
        print("        gegen die Preisfehler robust.")
    print(
        f"        Optimum wandert in: {wandert if wandert else 'keiner Welt'} "
        "(P2 schloss aus dem\n        unbewegten Optimum, dass die Steuer nicht "
        "bindet)."
    )
    print("=" * 72, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
