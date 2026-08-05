"""P13e — DSR und PBO für den SPY-Trendfilter.

WOZU
----
P13/P13b zeigen, dass der Filter nicht an einem Parameter hängt. Das ist
Robustheit gegen die Wahl innerhalb der Familie — es ist **keine** Korrektur
dafür, dass überhaupt gesucht wurde. Genau diese Korrektur hat den
Aktien-Kandidaten der Kampagne erledigt (DSR p = 0,9512 gegen eine
0,95-Schwelle, IID-Gegenprobe 0,8783). Sie muss hier genauso greifen, sonst
ist der Maßstab je nach Kandidat ein anderer.

DER ERSTE ENTWURF WAR EINE KOPIE DES FALSCHEN MODULS
----------------------------------------------------
Ursprünglich war dieses Modul bewusst als Kopie von `p7_dsr_pbo.py` gebaut,
mit der Begründung „ein eigener Weg wäre eine dritte Wahrheit". Das war falsch:
**P7 ist im eigenen Fehlerlog als E-077 verworfen** und durch
`p8_dsr_heterogen.py` ersetzt. P7 schätzt `variance_across_trials` aus 37
Fast-Klonen derselben Strategie; die Sharpes liegen dann eng beieinander, die
Schwelle sinkt künstlich und die Korrektur wird wirkungslos. Genau deshalb
wurde dem Aktien-Kandidaten der Holdout-Schuss verweigert.

Die Kopie hätte denselben Fehler in einen neuen Befund verlängert — getarnt
als Disziplin. Rule 50 schützt vor divergierenden **Implementierungen**, nicht
vor der Wiederverwendung einer verworfenen **Methode**.

Deshalb rechnet dieses Modul die DSR über **drei** Varianzschätzer und
entscheidet nach der Regel, die `p8_dsr_heterogen.py` vor dem Lauf
festgeschrieben hat: heterogen geschätztes V mit kumuliertem N.

* **klonfamilie** — V aus den 37 Fenstervarianten. Nur zur Dokumentation des
  Effekts, **nicht** entscheidungsfähig (E-077).
* **heterogen** — V aus P8s heterogener 30-Strategien-Familie. Das ist der
  Maßstab, an dem der Aktien-Kandidat gemessen wurde, und damit der einzige,
  der einen Vergleich zwischen beiden Kandidaten trägt.
* **IID-Näherung** — die konservative Gegenprobe, ebenfalls wie in P8.

Implementierungen bleiben unverändert übernommen
(`src/assembled_core/qa/deflated_sharpe.py`, `research/mandat/h011_kandidat_a.
cscv_pbo`) — dort war die Wiederverwendung richtig.

WAS PBO HIER NICHT LEISTET
--------------------------
Die CSCV-Matrix besteht aus Fast-Klonen desselben Instruments; E-077 hält fest,
dass CSCV heterogene Spalten voraussetzt und bei Klonen **zu niedrige** Werte
liefert (dort 20 % als Strukturartefakt). Ein hoher Wert ist unter dieser
Verzerrung also erst recht ein Fehlschlag — ein niedriger wäre nichts wert
gewesen. Zusätzlich rangiert `cscv_pbo` nach **Sharpe**, nicht nach dem
Zielmaß der Kampagne (Median unter DD-Deckel); PBO misst damit die Stabilität
einer anderen Auswahl als der tatsächlich getroffenen. Beides steht so im
Befund.

ZWEI GEWINNER-BEGRIFFE, BEIDE AUSGEWIESEN
-----------------------------------------
* **In-sample-Gewinner** — die Zelle mit dem höchsten Median unter den
  bestandenen. Das ist, was DSR korrigieren soll, und die Zahl, die zählt.
* **A-priori-Parameter** `preis>SMA200` — nicht aus dem Raster gewählt,
  sondern der Lehrbuchwert (siehe P13c). Er wird mit **demselben** N deflatiert,
  obwohl er nicht selektiert wurde; das ist die konservative Richtung und
  beantwortet die Frage „und wenn man gar nicht gesucht hätte?" nicht schön,
  sondern hart.

TRIAL-ZÄHLER
------------
Steigt **nicht**. Hier wird nichts gesucht, sondern das bereits Gesuchte
korrigiert (E-090). Die Familienmatrix enthält genau die Konfigurationen, über
die gesucht wurde — eine kleinere Matrix würde PBO künstlich drücken.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore", message=".*Converting to PeriodArray.*", category=UserWarning
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.mandat.h011_kandidat_a import cscv_pbo  # noqa: E402
from research.mandat2.campaign_data import load_campaign  # noqa: E402
from research.mandat2.data_gate import TrialCounter  # noqa: E402
from research.mandat2.engine import run_buy_and_hold  # noqa: E402
from research.mandat2.metrics import auswerten  # noqa: E402
from research.mandat2.p5_gate_robustheit import DEFINITIONEN, FENSTER  # noqa: E402
from research.mandat2.p13c_ereignisabhaengigkeit import FENSTER_APRIORI  # noqa: E402
from research.mandat2.tax_regimes import make_regime  # noqa: E402
from src.assembled_core.qa.deflated_sharpe import deflated_sharpe  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"

#: Wie in P7: die steuerfreie Welt ist die Referenz fuer die Korrektur, weil
#: die Steuer die Kurve verzerrt, ohne etwas ueber Ueberanpassung zu sagen.
WELT = "ZERO"

#: Der a-priori-Parameter in der Notation der Familienmatrix.
APRIORI = f"preis>sma/{FENSTER_APRIORI}"

#: Artefakt, aus dem die heterogene Varianz stammt. Sie wird NICHT neu
#: geschaetzt: derselbe Wert wie beim Aktien-Kandidaten ist die Voraussetzung
#: dafuer, dass die beiden Verdicts ueberhaupt vergleichbar sind (F-senior-2).
P8_ARTEFAKT = "p8_dsr_heterogen.json"


def heterogene_varianz() -> tuple[float, int]:
    """V und Strategienzahl aus dem P8-Artefakt.

    Fail-loud statt Rueckfall auf die Klonvarianz: fehlt das Artefakt, ist die
    entscheidungsfaehige Zahl nicht berechenbar, und ein stiller Rueckfall auf
    den von E-077 verworfenen Schaetzer waere genau der Fehler, gegen den
    dieses Modul umgebaut wurde.
    """
    p = OUT / P8_ARTEFAKT
    if not p.exists():
        raise SystemExit(
            f"[ERROR] {P8_ARTEFAKT} fehlt — ohne die heterogene Varianz ist die "
            f"DSR hier nicht entscheidungsfaehig (E-077). Erst p8 laufen lassen."
        )
    d = json.loads(p.read_text(encoding="utf-8"))
    return float(d["varianz_heterogen"]), int(d["n_strategien"])


def main() -> int:
    OUT.mkdir(exist_ok=True)
    d = load_campaign()
    print(d, flush=True)
    n_familie = len(DEFINITIONEN) * len(FENSTER) + 1
    n_gesamt = TrialCounter().total()
    print(f"Familie: {n_familie} Varianten | Trial-Zaehler kumuliert: {n_gesamt}\n")

    bench = run_buy_and_hold(d, make_regime(WELT))
    kurven: dict[str, pd.Series] = {}
    kennzahlen: dict[str, dict] = {}

    varianten: list[tuple[str, object | None, int | None]] = [("ohne Gate", None, None)]
    for defname, fn in DEFINITIONEN.items():
        for f in FENSTER:
            varianten.append((f"{defname}/{f}", fn, f))

    for name, fn, f in varianten:
        gate = None if fn is None else fn(d.close, f)
        r = run_buy_and_hold(d, make_regime(WELT), risk_off_gate=gate)
        a = auswerten(r.equity_netto, bench.equity_netto, label=name)
        kurven[name] = r.equity_netto.pct_change().dropna()
        kennzahlen[name] = {
            "median": a.median_kandidat,
            "maxdd": a.schlimmster_maxdd,
            "bestanden": a.bestanden,
            "endwert": float(r.equity.iloc[-1]),
        }
        print(
            f"  {name:<16} Median {a.median_kandidat:>6.3f} | DD "
            f"{a.schlimmster_maxdd:>7.1%} | {'BESTANDEN' if a.bestanden else '-'}",
            flush=True,
        )

    rm = pd.DataFrame(kurven).dropna()
    print(f"\nRenditematrix: {rm.shape[0]} Tage x {rm.shape[1]} Varianten")

    pbo = cscv_pbo(rm, n_blocks=8)
    print(f"PBO (CSCV, 8 Bloecke, C(8,4)=70 Splits): {pbo:.1%}")

    bestanden = {k: v for k, v in kennzahlen.items() if v["bestanden"]}
    gewinner = max(bestanden or kennzahlen, key=lambda k: kennzahlen[k]["median"])
    sharpes = rm.apply(lambda x: x.mean() / x.std() if x.std() > 0 else np.nan)
    var_emp = float(sharpes.var(ddof=1))
    print(f"\nIn-sample-Gewinner: {gewinner} | a priori: {APRIORI}")
    print(f"Empirische Varianz der Sharpes ueber die Familie: {var_emp:.3e}")
    if APRIORI not in rm.columns:
        raise SystemExit(
            f"[ERROR] {APRIORI} fehlt in der Familienmatrix — die a-priori-Zeile "
            f"waere still entfallen und der Befund haette nur den Gewinner gezeigt."
        )

    var_het, n_strategien = heterogene_varianz()
    print(
        f"Heterogene Varianz aus P8 ({n_strategien} Strategien): {var_het:.3e} "
        f"= {var_het / var_emp:.1f}x der Klonvarianz"
    )

    ergebnis: dict = {
        "welt": WELT,
        "pbo": pbo,
        "pbo_rangmass": "sharpe (nicht das Zielmass der Kampagne)",
        "gewinner": gewinner,
        "apriori": APRIORI,
        "varianz_klonfamilie": var_emp,
        "varianz_heterogen": var_het,
        "n_strategien_p8": n_strategien,
        "n_familie": n_familie,
        "n_kumuliert": n_gesamt,
        "kennzahlen": kennzahlen,
        "dsr": {},
    }
    # Reihenfolge = Rangfolge der Belastbarkeit. `heterogen` ist die
    # Entscheidungsgrundlage (Regel aus p8), `klonfamilie` steht nur da, um
    # den Effekt von E-077 sichtbar zu machen.
    schaetzer: list[tuple[str, float | None]] = [
        ("heterogen", var_het),
        ("IID-Naeherung", None),
        ("klonfamilie", var_emp),
    ]
    for wer, spalte in (("gewinner", gewinner), ("apriori", APRIORI)):
        ergebnis["dsr"][wer] = {}
        for label, V in schaetzer:
            res = deflated_sharpe(
                rm[spalte], n_trials=n_gesamt, variance_across_trials=V
            )
            ergebnis["dsr"][wer][label] = {
                "spalte": spalte,
                "n_trials": n_gesamt,
                "varianz": V,
                "entscheidungsfaehig": label == "heterogen",
                "sharpe_observed": float(res.sharpe_observed),
                "sharpe_threshold": float(res.sharpe_threshold),
                "dsr_probability": float(res.deflated_sharpe_probability),
                "passes_5pct": bool(res.passes_5pct),
            }
            marke = "  <- Entscheidung" if label == "heterogen" else ""
            if label == "klonfamilie":
                marke = "  <- E-077, NICHT entscheidungsfaehig"
            print(
                f"  DSR {wer:<9} {label:<14} (N={n_gesamt}): Sharpe "
                f"{res.sharpe_observed:.4f} gegen Schwelle "
                f"{res.sharpe_threshold:.4f} -> p = "
                f"{res.deflated_sharpe_probability:.4f} "
                f"{'BESTANDEN' if res.passes_5pct else 'DURCHGEFALLEN'}{marke}"
            )

    dsr_ok = ergebnis["dsr"]["gewinner"]["heterogen"]["passes_5pct"]
    pbo_ok = pbo < 0.5
    ergebnis["verdikt"] = {
        "dsr_kumuliert_bestanden": dsr_ok,
        "pbo_bestanden": pbo_ok,
        "reif_fuer_holdout": bool(dsr_ok and pbo_ok),
    }

    # Artefakt als LETZTE Anweisung (E-116).
    (OUT / "p13e_dsr_pbo_spy.json").write_text(
        json.dumps(ergebnis, indent=2), encoding="utf-8"
    )
    print(f"\n-> {OUT / 'p13e_dsr_pbo_spy.json'}")
    print("=" * 68)
    print(
        f"DSR (heterogenes V, kumuliertes N — Regel aus p8): "
        f"{'BESTANDEN' if dsr_ok else 'DURCHGEFALLEN'}"
    )
    print(f"PBO < 50 %: {'BESTANDEN' if pbo_ok else 'DURCHGEFALLEN'} ({pbo:.1%})")
    print(
        "\n-> reif fuer den EINEN Holdout-Schuss."
        if dsr_ok and pbo_ok
        else "\n-> NICHT reif fuer den Holdout. Kein Schuss."
    )
    print("=" * 68, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
