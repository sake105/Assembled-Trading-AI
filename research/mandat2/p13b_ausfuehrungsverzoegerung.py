"""P13b — Überlebt der SPY-Trendfilter, wenn er einen Tag später handeln muss?

DIE ANNAHME, DIE P13 NOCH GESCHENKT BEKAM
-----------------------------------------
Alle drei Trend-Definitionen aus P5 werten `close[t]` aus, und die Engine
kauft und verkauft am Monatsende zu genau diesem `close[t]`. Das ist **kein
Blick in die Zukunft** — es fließt kein Kurs nach t ein — aber es ist die
optimistischste noch zulässige Annahme: Der Ausstieg gelingt zu exakt dem
Kurs, der ihn ausgelöst hat.

Für einen Trendfilter ist das nicht neutral. Sein ganzer Nutzen entsteht in
den Fenstern, in denen er vor einem Absturz aussteigt, und dort ist der
Unterschied zwischen „zum auslösenden Schlusskurs" und „einen Tag später"
systematisch zu seinen Gunsten. Wer real handelt, sieht den Schlusskurs erst,
wenn nicht mehr zu ihm gehandelt werden kann.

WAS HIER GEMESSEN WIRD
----------------------
Dasselbe Raster wie P13 — 12 Fenster x 3 Definitionen x 3 Steuerwelten — mit
einer einzigen Änderung: `gate.shift(1)`. Das Signal von t wirkt erst an t+1.
Sonst ist nichts anders, damit die Differenz genau eine Ursache hat.

WARUM DER TRIAL-ZÄHLER TROTZDEM STEIGT
--------------------------------------
Das hier ist keine Wiederholung im Sinne von E-090 (P12f/P12i maßen dieselbe
Frage auf reparierten Daten und zählten deshalb nicht). Hier steht eine andere
Ausführungsannahme zur Wahl, und wenn der Kandidat nur ohne Verzögerung
besteht, wäre das Behalten der Variante ohne Verzögerung eine **Auswahl**.
Damit ist es eine Suche und wird gezählt — die konservative Richtung: der
DSR-Abschlag wird härter, nicht milder.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd  # noqa: F401

warnings.filterwarnings(
    "ignore", message=".*Converting to PeriodArray.*", category=UserWarning
)

from research.mandat2.campaign_data import load_campaign  # noqa: E402
from research.mandat2.data_gate import TrialCounter  # noqa: E402
from research.mandat2.engine import run_buy_and_hold  # noqa: E402
from research.mandat2.metrics import auswerten  # noqa: E402
from research.mandat2.p5_gate_robustheit import DEFINITIONEN, FENSTER  # noqa: E402
from research.mandat2.p13_spy_trend_robustheit import WELTEN, zaehle_buchungen  # noqa: E402
from research.mandat2.tax_regimes import make_regime  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"

#: Handelstage zwischen Signal und Ausführung. 1 = der Schlusskurs von t
#: entscheidet, gehandelt wird zum Schlusskurs von t+1. Mehr als die
#: Tagesauflösung gibt die Datenbasis nicht her (kein Open, kein Intraday für
#: den Suchzeitraum — siehe Befund 6).
VERZOEGERUNG = 1


def verzoegertes_gate(
    fn, close, fenster: int, verzoegerung: int = VERZOEGERUNG, symbol: str = "SPY"
):
    """Signal von t wirkt erst an t+`verzoegerung` HANDELSTAGE des Instruments.

    Als Funktion herausgezogen, weil der einzige inhaltliche Unterschied zu
    P13 sonst in einem `main()`-Rumpf staende und kein Test bemerken wuerde,
    wenn die Verzoegerung wegfaellt (Stage-1-Befund N10). Die ganze Spalte
    "mit Verz." im Befund haengt an dieser einen Zeile.

    Verschoben wird entlang der Handelstage von `symbol`, nicht entlang des
    Panel-Index. Ein `shift` auf dem Panel wuerde an den Tagen, an denen andere
    Namen handeln und SPY nicht, um zwei SPY-Tage verzoegern — bei 99,8 %
    Abdeckung eine Handvoll Faelle, aber die Zusicherung "ein Handelstag"
    galte dann nicht (F-senior-11).

    Der erste Handelstag wird NaN, und die Engine liest das als risk_on. Das
    ist hier folgenlos, aber NICHT weil der Kandidat investiert startete:
    alle drei Definitionen bilden `(a > b).astype(float)`, und NaN-Vergleiche
    ergeben False — die Warmlaufphase ist also **0.0 = risk-off**, nicht NaN.
    Jeder gegatete Lauf startet damit in Cash, und zwar umso laenger, je
    groesser das Fenster ist (F-senior-8).
    """
    roh = fn(close, fenster)
    handelstage = close[symbol].dropna().index
    return roh.reindex(handelstage).shift(verzoegerung).reindex(roh.index)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--regen",
        action="store_true",
        help=(
            "Artefakte neu erzeugen, OHNE den Trial-Zaehler zu erhoehen. "
            "Fuer Wiederholungen nach einem Bugfix: dieselbe Suche zaehlt "
            "nicht zweimal (E-090). Eine NEUE Suche laeuft ohne diesen Schalter."
        ),
    )
    args = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    d = load_campaign()
    print(d, flush=True)
    n = len(FENSTER) * len(DEFINITIONEN) * len(WELTEN)
    if args.regen:
        print(
            f"[REGEN] Trial-Zaehler UNVERAENDERT bei {TrialCounter().total()} — "
            f"dieselbe Suche zaehlt nicht zweimal (E-090).",
            flush=True,
        )
    else:
        print(
            f"Trials kumuliert: "
            f"{TrialCounter().increment(n, label='P13b Ausfuehrungsverzoegerung')}",
            flush=True,
        )
    print(
        f"({n} Laeufe, Signal wirkt {VERZOEGERUNG} Handelstag(e) spaeter)\n", flush=True
    )

    zeilen = []
    for welt, name, kwargs in WELTEN:
        bench = run_buy_and_hold(d, make_regime(name, **kwargs))
        for defname, fn in DEFINITIONEN.items():
            print(f"=== {welt} | {defname} ===", flush=True)
            for f in FENSTER:
                gate = verzoegertes_gate(
                    fn, d.close, f
                )  # der einzige Unterschied zu P13
                with zaehle_buchungen() as z:
                    r = run_buy_and_hold(
                        d, make_regime(name, **kwargs), risk_off_gate=gate
                    )
                a = auswerten(
                    r.equity_netto, bench.equity_netto, label=f"{welt}/{defname}/{f}"
                )
                zeilen.append(
                    {
                        "welt": welt,
                        "definition": defname,
                        "fenster": f,
                        "verzoegerung": VERZOEGERUNG,
                        "endwert": float(r.equity_netto.iloc[-1]),
                        "median_kandidat": a.median_kandidat,
                        "median_benchmark": a.median_benchmark,
                        "schlimmster_maxdd": a.schlimmster_maxdd,
                        "gerissene_fenster": len(a.gerissene_fenster),
                        "kaeufe": z["kauf"],
                        "verkaeufe": z["verkauf"],
                        "kosten_gezahlt": float(r.portfolio.costs_paid),
                        "steuer_gezahlt": float(r.portfolio.tax_paid),
                        "bestanden": a.bestanden,
                    }
                )
                print(
                    f"  {f:>4}: DD {a.schlimmster_maxdd:>7.1%} | gerissen "
                    f"{len(a.gerissene_fenster):>3}/{a.n_fenster} | Median "
                    f"{a.median_kandidat:.3f} | Buchungen "
                    f"{z['kauf'] + z['verkauf']:>3} | "
                    f"{'BESTANDEN' if a.bestanden else '-'}",
                    flush=True,
                )

    # Artefakt als LETZTE Anweisung (E-116): keine Kennzahl darf nach dem
    # Schreiben noch entstehen, sonst existiert sie nur auf der Konsole.
    (OUT / "p13b_ausfuehrungsverzoegerung.json").write_text(
        json.dumps(
            {"n_trials": n, "verzoegerung": VERZOEGERUNG, "zeilen": zeilen}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"\n-> {OUT / 'p13b_ausfuehrungsverzoegerung.json'}")
    print("Auswertung: render_befund_p13.py (liest beide Artefakte)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
