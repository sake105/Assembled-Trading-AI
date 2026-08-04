"""P12i — Dreht ein Verdikt, wenn recycelte Ticker getrennt sind?

DIE FRAGE
---------
P12h hat gezeigt, dass 29 Panel-Spalten (Lücke >= 500 Handelstage) ihr Symbol
an ein zweites Unternehmen weitergeben und dass die
Delisting-Hygiene bei ihnen ausfällt: CGP lief 3.264 Handelstage im Bestand
weiter, obwohl die Firma seit 2001 nicht mehr existierte. Ob das ein
Kampagnen-Verdikt bewegt, beantwortet nur ein Neulauf.

Aufbau identisch zu P12f: **dasselbe Parametergitter wie P2**, einmal auf dem
Originalpanel und einmal auf dem getrennten, gemessen an der **Zielfunktion**
(Median über rollierende 10-Jahres-Fenster + MaxDD ≥ −35 % in jedem Fenster) —
nicht am Endwert.

TRIAL-BUCHHALTUNG
-----------------
Wiederholung desselben, bereits gezählten Gitters auf korrigierten Daten: 24
Zellen × 3 Steuerwelten × 2 Panels, kein neuer Parameter, kein neuer Suchraum.
Der Zähler bleibt deshalb unverändert (E-090). Die Entscheidung steht hier,
weil E-090 den Präzedenzfall gesetzt hat, dass sogar die 44 Trials eines reinen
Regenerationslaufs gezählt und offengelegt wurden — Schweigen wäre an einer
Größe, die über den DSR die Signifikanzschwelle steuert, keine Option.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import replace
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from research.mandat2.campaign_data import load_campaign  # noqa: E402
from research.mandat2.metrics import DD_DECKEL  # noqa: E402
from research.mandat2.p12f_neulauf_bereinigt import (  # noqa: E402
    WELTEN,
    bester_kandidat,
    gitter,
)
from research.mandat2.p12h_ticker_recycling import unterbrechungen  # noqa: E402
from research.mandat2.panel_getrennt import MIN_LUECKE, trenne  # noqa: E402
from research.mandat2.render_befund_datenqualitaet import (  # noqa: E402
    kipp_abstand,
    zeilen_mit_wechsel,
)

OUT = Path(__file__).resolve().parent / "results"


def main() -> int:
    OUT.mkdir(exist_ok=True)
    d = load_campaign()
    print(d, flush=True)
    print("Trial-Zaehler NICHT erhoeht: Wiederholung derselben Hypothesen (E-090).\n")

    # Kandidaten mit DERSELBEN Schwelle wie die Trennung — sonst haette die
    # Liste Namen, die gar nicht getrennt werden (und umgekehrt).
    treffer = unterbrechungen(d.close, min_luecke=MIN_LUECKE)
    close_neu, m_neu, div_neu, protokoll = trenne(
        d.close, d.membership, treffer, d.div_panel
    )
    print(f"Recycelte Ticker getrennt: {len(protokoll)} Symbole")
    print(f"Spalten {d.close.shape[1]} -> {close_neu.shape[1]}")
    mehrfach = [s for s, i in protokoll.items() if i["n_segmente"] > 2]
    if mehrfach:
        print(f"   davon mehr als zweimal vergeben: {sorted(mehrfach)}")
    for sym, info in sorted(protokoll.items())[:5]:
        print(f"   {sym:<7}{info['n_segmente']} Segmente, Schnitte {info['schnitte']}")
    # Wahrscheinliche FEHLTREFFER offen ausweisen: die Schwelle entscheidet nach
    # der Lueckenlaenge, ihre Herleitung nutzte zusaetzlich den Kursfaktor. Wo
    # der Kurs fortsetzt, ist die Trennung vermutlich falsch — und erzeugt dann
    # ein fabriziertes Delisting (F-senior-3).
    verdaechtig = {
        s: [
            f
            for f, nah in zip(i["faktoren"], i["faktor_nahe_eins"], strict=True)
            if nah
        ]
        for s, i in protokoll.items()
        if any(i["faktor_nahe_eins"])
    }
    ergebnis_fehltreffer = {
        "n_schnitte": sum(len(i["schnitte"]) for i in protokoll.values()),
        "n_faktor_nahe_eins": sum(len(v) for v in verdaechtig.values()),
        "symbole": {
            s: [round(f, 2) for f in v] for s, v in sorted(verdaechtig.items())
        },
    }
    print(
        f"   davon mit Kurs-Fortsetzung (Faktor 0,5..2,0) — wahrscheinliche "
        f"Fehltreffer: {ergebnis_fehltreffer['n_faktor_nahe_eins']} von "
        f"{ergebnis_fehltreffer['n_schnitte']} Schnitten"
    )
    if verdaechtig:
        print(f"      {ergebnis_fehltreffer['symbole']}")
    d_neu = replace(d, close=close_neu, membership=m_neu, div_panel=div_neu)
    print("", flush=True)

    ergebnis = {
        "dd_deckel": DD_DECKEL,
        "n_getrennt": len(protokoll),
        "fehltreffer_kandidaten": ergebnis_fehltreffer,
        "protokoll": protokoll,
        "welten": {},
    }
    kopf = f"{'Welt':<11}{'Panel':<12}{'Median K':>10}{'Median B':>10}"
    print(kopf + f"{'schl. DD':>10}{'schlaegt':>10}{'BESTANDEN':>11}")
    print("-" * 74)
    for label, name, kwargs in WELTEN:
        eintrag = {}
        for panel_name, daten in (("original", d), ("getrennt", d_neu)):
            zeilen, b_end = gitter(daten, label, name, kwargs)
            bester = bester_kandidat(zeilen)
            eintrag[panel_name] = {
                "benchmark": b_end,
                "bester": bester,
                "n_schlagen_bench": sum(1 for z in zeilen if z["schlaegt_bench"]),
                "n_bestanden": sum(1 for z in zeilen if z["bestanden"]),
                "zeilen": zeilen,
            }
            print(
                f"{label:<11}{panel_name:<12}{bester['median_kandidat']:>10.3f}"
                f"{bester['median_benchmark']:>10.3f}"
                f"{bester['schlimmster_maxdd']:>9.1%}"
                f"{eintrag[panel_name]['n_schlagen_bench']:>7}/{len(zeilen)}"
                f"{eintrag[panel_name]['n_bestanden']:>8}/{len(zeilen)}",
                flush=True,
            )
        o, b = eintrag["original"]["bester"], eintrag["getrennt"]["bester"]
        eintrag["optimum_wandert"] = any(
            o[k] != b[k] for k in ("haltetage", "rank_out", "hebel")
        )
        eintrag["verdikt_dreht"] = (eintrag["original"]["n_bestanden"] == 0) != (
            eintrag["getrennt"]["n_bestanden"] == 0
        )
        ergebnis["welten"][label] = eintrag

    # Wie weit war der Ausgang vom Kippen entfernt? Ohne diese Zahl ist
    # „dreht nicht" keine Robustheitsaussage, sondern eine gesaettigte Messung
    # (Lehre aus dem P12f-Audit, F-auditor-1). `kipp_abstand` erwartet die
    # Panel-Namen „original"/„bereinigt" — hier heisst das zweite „getrennt",
    # deshalb eine flache Umbenennung fuer den Aufruf.
    fuer_abstand = {
        "welten": {
            w: {"original": e["original"], "bereinigt": e["getrennt"], **e}
            for w, e in ergebnis["welten"].items()
        },
        "dd_deckel": DD_DECKEL,
    }
    abstand, wirkung_dd = kipp_abstand(fuer_abstand)
    ergebnis["kipp_abstand"] = {**abstand, "max_delta_maxdd": wirkung_dd}
    print("")
    print(
        f"Aufloesung: bester schlimmster MaxDD {abstand['bester_dd']:.1%} gegen "
        f"Deckel {abstand['deckel']:.0%}"
    )
    print(
        f"            = {abstand['abstand_pp']:.1f} pp Abstand; die Trennung "
        f"verschiebt den DD um {wirkung_dd * 100:.2f} pp."
    )

    # Das schwaechere Kriterium reagiert sehr wohl — das gehoert in den Befund,
    # sonst liest sich „Median identisch" als „ohne Wirkung" (F-senior-10).
    wechsel = zeilen_mit_wechsel(fuer_abstand, "schlaegt_bench")
    ergebnis["zeilen_mit_wechsel"] = wechsel
    if wechsel:
        print(
            f"            Zeilen mit gewechseltem `schlaegt_bench`: "
            f"{sum(wechsel.values())} ({wechsel})"
        )

    # ARTEFAKT ZULETZT (Stage-2-Finding F-senior-2): der Schreibaufruf stand
    # vor der kipp_abstand-Berechnung. Der Schluessel landete nie in der Datei,
    # die Zahl existierte nur auf stdout — und genau deshalb konnte im Dokument
    # unbemerkt die Zahl des VORHERIGEN Laufs stehen bleiben.
    (OUT / "p12i_neulauf_getrennt.json").write_text(
        json.dumps(ergebnis, indent=2), encoding="utf-8"
    )
    print(f"\n-> {OUT / 'p12i_neulauf_getrennt.json'}")

    dreht = [w for w, e in ergebnis["welten"].items() if e["verdikt_dreht"]]
    wandert = [w for w, e in ergebnis["welten"].items() if e["optimum_wandert"]]
    print("\n" + "=" * 74)
    if dreht:
        print(f"BEFUND: Das Verdikt DREHT in {dreht} — ob eine Parametrisierung")
        print("        Zielfunktion UND DD-Deckel besteht, haengt am Ticker-")
        print("        Recycling. Alle betroffenen Phasen sind neu zu bewerten.")
    else:
        print("BEFUND: Das Verdikt DREHT IN KEINER Steuerwelt. Die Trennung")
        print("        recycelter Ticker bewegt nicht, ob Zielfunktion und")
        print("        DD-Deckel gemeinsam bestanden werden.")
    print(f"        Optimum wandert in: {wandert or 'keiner Welt'}")
    if abstand["faktor"] != float("inf") and abstand["faktor"] > 3:
        print(
            f"        ACHTUNG: die Trennung haette rund "
            f"{abstand['faktor']:.0f}-mal staerker wirken muessen, um eine"
        )
        print("        einzige Zeile ueber den Deckel zu heben — der Ausgang")
        print("        konnte an dieser Stelle nicht kippen.")
    print("=" * 74, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
