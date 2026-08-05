"""P13 — Trägt der SPY-Trendfilter, wenn man ihn genauso hart prüft wie P5?

WARUM DIESER STRANG ÜBERHAUPT NOCH LEBT
---------------------------------------
Die Befunde 6 und 7 haben die Datenbasis für Vergleiche gegen einen passiven
Index unbrauchbar gemacht: Survivorship 2,36–2,90 pp p. a. bei 1,5 pp
Entscheidungsmarge, dazu Ticker-Recycling in 29 Spalten. Das trifft jede
Strategie, die **Namen auswählt**.

Es trifft diesen Strang nicht. Hier steht auf beiden Seiten derselbe Basiswert:
SPY mit Trendfilter gegen SPY ohne. Keine Auswahl, kein Survivorship, kein
Recycling, keine Gewichtungsfrage (E-079). Die SPY-Serie ist nachgeprüft
sauber — 99,8 % Abdeckung im Suchfenster, kein Skalenbruch, größte Lücke zwei
Handelstage, nicht unter den 25 korrumpierten und nicht unter den 37
unterbrochenen Namen.

Und P4 hat ihn nebenbei mitgemessen, ohne ihn je als Kandidaten zu behandeln:

======  ========  ==========  ========
Gate     MaxDD    gerissen    Median
======  ========  ==========  ========
kein     −55,2 %  144 / 144      1,948
SMA100   −31,9 %    0 / 144      1,668
SMA200   −19,2 %    0 / 144      2,525
SMA300   −24,2 %    0 / 144      1,964
======  ========  ==========  ========

Zwei Konfigurationen bestehen die Zielfunktion. Aber das ist ein
**Drei-Punkte-Raster** — genau das Muster, das P5 beim Aktien-Kandidaten als
„gefundener Parameter" verdächtigt hat.

DIE FRAGE
---------
Dieselben zwei Tests wie in P5, wortgleich übernommen:

1. **Fenster-Band statt Punkt.** Über ein feines Raster (100…320) darf das
   Ergebnis schwanken, aber es muss ein zusammenhängendes Band geben.
2. **Andere Trend-Definitionen.** Wenn nur „Preis > SMA" funktioniert, aber
   weder „SMA steigt" noch „Rendite > 0", trägt nicht die Trendfolge, sondern
   die eine Formel.

Beide sind billig und beide können den Kandidaten töten. Sie kommen deshalb
vor DSR/PBO und weit vor dem Holdout. **Der Aktien-Kandidat ist an Test 2
gescheitert** (5/12 und 3/12, lückig) — das ist der wahrscheinlichste Ausgang
auch hier.

WAS HIER NICHT PASSIERT
-----------------------
Kein Holdout. Keine Parametersuche über das Raster hinaus. Der Trial-Zähler
steigt um die Zahl der Läufe, weil dies eine **echte Suche** ist und keine
Wiederholung — anders als P12f/P12i.
"""

from __future__ import annotations

import argparse
import json
import warnings
from contextlib import contextmanager
from collections.abc import Iterator
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
from research.mandat2.portfolio import Portfolio  # noqa: E402
from research.mandat2.tax_regimes import make_regime  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"

#: Dieselben drei Steuerwelten wie im Rest der Kampagne. Ein Trendfilter
#: handelt selten (ein paar Wechsel pro Jahrzehnt), die Steuer sollte ihn also
#: kaum treffen — genau das ist prüfbar und wird hier mitgemessen.
WELTEN = [
    ("ZERO", "ZERO", {}),
    ("PRIVAT_DE", "PRIVAT_DE", {}),
    ("GMBH+FK", "GMBH_THESAURIEREND", {"fixkosten_pa": 3500.0}),
]


# KEINE eigene Buchungslogik hier (Lektion aus MAJOR-2, Rule 50): die Engine
# kann das bereits. `run_buy_and_hold` haelt genau ein Instrument — Default SPY,
# `asset=FONDS` wegen §20 InvStG — und nimmt denselben `risk_off_gate` entgegen
# wie `run_strategy`. Eine Handimplementierung waere eine zweite Wahrheit mit
# eigener Steuer- und Kostenbehandlung gewesen.


@contextmanager
def zaehle_buchungen() -> Iterator[dict[str, int]]:
    """Zaehlt, wie oft die Engine wirklich kauft und verkauft.

    NOETIG, weil `run_buy_and_hold` `n_trades=1` als **Konstante** zurueckgibt
    (engine.py:125) — anders als `run_strategy`, das echt mitzaehlt. Das Feld
    heisst wie ein Messwert und ist keiner; im Trockenlauf stand „Trades 1"
    unter einem Filter, der ueber 22 Jahre mehrfach umschalten muss. Wer die
    Zahl uebernimmt, berichtet eine Konstante als Ergebnis (E-109).

    Die Alternative — die Umschaltungen aus der Gate-Serie nachrechnen — waere
    eine zweite Buchungswahrheit gewesen, die an der `px > 0`-Bedingung der
    Engine vorbeilaeuft. Stattdessen wird die Engine instrumentiert wie in
    E-102: gezaehlt wird der Bestandswechsel NACH dem Aufruf, damit die
    internen Frueh-Returns (`notional <= 0`, `EPS_QTY`) nicht als Trade zaehlen.

    Enthalten ist auch der `liquidate_all`-Verkauf am Ende jedes Laufs — die
    Zahl ist also „Buchungen", nicht „Signalwechsel".
    """
    z = {"kauf": 0, "verkauf": 0}
    o_buy, o_sell = Portfolio.buy, Portfolio.sell

    def buy(self: Portfolio, sym: str, notional: float, px: float) -> None:
        vorher = self.qty(sym)
        o_buy(self, sym, notional, px)
        if self.qty(sym) > vorher:
            z["kauf"] += 1

    def sell(self: Portfolio, sym: str, qty: float, px: float) -> None:
        vorher = self.qty(sym)
        o_sell(self, sym, qty, px)
        if self.qty(sym) < vorher:
            z["verkauf"] += 1

    Portfolio.buy = buy  # type: ignore[method-assign]
    Portfolio.sell = sell  # type: ignore[method-assign]
    try:
        yield z
    finally:
        Portfolio.buy = o_buy  # type: ignore[method-assign]
        Portfolio.sell = o_sell  # type: ignore[method-assign]


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
            f"{TrialCounter().increment(n, label='P13 SPY-Trend-Robustheit')}",
            flush=True,
        )
    print(
        f"({n} Laeufe: {len(FENSTER)} Fenster x {len(DEFINITIONEN)} Definitionen "
        f"x {len(WELTEN)} Steuerwelten)\n",
        flush=True,
    )

    zeilen = []
    for welt, name, kwargs in WELTEN:
        bench = run_buy_and_hold(d, make_regime(name, **kwargs))
        for defname, fn in DEFINITIONEN.items():
            print(f"=== {welt} | {defname} ===", flush=True)
            for f in FENSTER:
                gate = fn(d.close, f)
                with zaehle_buchungen() as z:
                    r = run_buy_and_hold(
                        d, make_regime(name, **kwargs), risk_off_gate=gate
                    )
                buchungen = z["kauf"] + z["verkauf"]
                a = auswerten(
                    r.equity_netto, bench.equity_netto, label=f"{welt}/{defname}/{f}"
                )
                zeilen.append(
                    {
                        "welt": welt,
                        "definition": defname,
                        "fenster": f,
                        "endwert": float(r.equity_netto.iloc[-1]),
                        "median_kandidat": a.median_kandidat,
                        "median_benchmark": a.median_benchmark,
                        "schlimmster_maxdd": a.schlimmster_maxdd,
                        "gerissene_fenster": len(a.gerissene_fenster),
                        # NICHT r.n_trades — das ist in run_buy_and_hold eine
                        # hart verdrahtete 1 (siehe zaehle_buchungen).
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
                    f"{a.median_kandidat:.3f} | Buchungen {buchungen:>3} | "
                    f"{'BESTANDEN' if a.bestanden else '-'}",
                    flush=True,
                )

    # Artefakt als LETZTE Anweisung (E-116). Die Auswertung steht bewusst
    # NICHT hier: der erste Entwurf klassifizierte streng nach Lueckenlosigkeit
    # und stellte damit 10 von 12 bestandenen Fenstern mit einem einzigen Loch
    # gleich mit 2 von 12 — dieselbe Verkuerzung wie E-117 (zweidimensionale
    # Evidenz, eindimensionale Regel). Breite und Zusammenhang sind zwei
    # Merkmale und gehoeren beide in den Befund. Der wird aus diesem Artefakt
    # generiert (E-085): render_befund_p13.py.
    (OUT / "p13_spy_trend_robustheit.json").write_text(
        json.dumps({"n_trials": n, "zeilen": zeilen}, indent=2), encoding="utf-8"
    )
    print(f"\n-> {OUT / 'p13_spy_trend_robustheit.json'}")
    print("Auswertung: render_befund_p13.py (liest dieses Artefakt)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
