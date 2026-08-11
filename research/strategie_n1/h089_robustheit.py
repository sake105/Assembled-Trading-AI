"""H-089-Robustheitsbatterie — ist der Fund eine Parameter-Insel oder eine Flaeche?

REGISTRIERUNG (+14 Trials, vorab; ALLE Ergebnisse berichtet; KEIN
Nachselektieren — die Batterie prueft die NACHBARSCHAFT der bestaetigten
Configs, nicht bessere Alternativen):
  Kandidatenfamilie: mom12-Dial (I9 solo) und 3er-Chor
  (trend200+cross+mom12), beide zweifach fenster-bestaetigt (K9/K9b).
  V1-V9:  mom12-solo, Lookback {210, 252, 294} x Dial-Tiefe {0.5, 0.6, 0.7}
  V10-V12: 3er-Chor, Dial-Tiefe {0.5, 0.6, 0.7} (Lookbacks Standard)
  V13:    mom12-solo Standard, KOSTEN VERDOPPELT (10 bps/Seite)
  V14:    3er-Chor Standard, KOSTEN VERDOPPELT
KRITERIUM je Variante (identisch K1/K9): Doppel-Kriterium (MDD >= 25 %
besser, CAGR-Abgabe <= 1 pp) in gesamt+h1+h2 auf BEIDEN Epochen
(SPY 1996-2016 UND CRSP 1926-1995). ERWARTUNG (vorab): eine robuste
Flaeche besteht mehrheitlich; bestehen nur die exakten Originalparameter,
ist der Fund verdaechtig und wird herabgestuft.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HIER = Path(__file__).resolve().parent
ROOT = HIER.parents[1]
sys.path.insert(0, str(ROOT))

from research.mandat2.data_gate import TrialCounter  # noqa: E402
from research.strategie_n1.k2_k3_dial_raster import kennzahlen, pruefe  # noqa: E402

ZIEL = HIER / "h089_robustheit.json"


def dial_kurve(px, rendite, mitglieder, tiefe, kosten_seite):
    risiko = sum(mitglieder) / len(mitglieder)
    expo = (1 - tiefe * risiko).shift(1)
    kosten = expo.diff().abs().fillna(0) * kosten_seite
    return (1 + (expo * rendite - kosten).fillna(0)).cumprod()


def signale(px, lb_mom):
    sma50, sma200 = px.rolling(50).mean(), px.rolling(200).mean()
    return {
        "trend200": (px < sma200).astype(float),
        "cross": (sma50 < sma200).astype(float),
        "mom": (px.pct_change(lb_mom) < 0).astype(float),
    }


def epoche_spy():
    pv = pd.read_parquet(
        ROOT / "research" / "mandat" / "data" / "prices_verdict.parquet"
    )
    s = (
        pv[pv.symbol == "SPY"]
        .set_index("timestamp")["close"]
        .sort_index()
        .astype(float)
    )
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    px = s.loc["1994-06-01":"2016-12-31"]
    fenster = {
        "gesamt": ("1996-01-01", "2016-12-31"),
        "h1": ("1996-01-01", "2006-12-31"),
        "h2": ("2007-01-01", "2016-12-31"),
    }
    return px, px.pct_change(), px, fenster


def epoche_1926():
    df = pd.read_parquet(
        ROOT / "research" / "mandat2" / "data_gratis" / "fama_french_daily.parquet"
    )
    df = df.loc["1926-07-01":"1995-12-31"]
    px = (1 + df["mkt"]).cumprod()
    bh = (1 + df["mkt_rf"].fillna(0)).cumprod()
    fenster = {
        "gesamt": ("1926-07-01", "1995-12-31"),
        "h1": ("1926-07-01", "1960-12-31"),
        "h2": ("1961-01-01", "1995-12-31"),
    }
    return px, df["mkt_rf"], bh, fenster


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true")
    args = ap.parse_args()
    if args.regen:
        print(f"[REGEN] Trials unveraendert: {TrialCounter().total()}", flush=True)
    else:
        print(
            "Trials kumuliert: "
            + str(
                TrialCounter().increment(14, label="H-089 Robustheitsbatterie V1-V14")
            ),
            flush=True,
        )
    varianten = []
    for lb in (210, 252, 294):
        for tiefe in (0.5, 0.6, 0.7):
            varianten.append(
                (f"V_mom{lb}_d{int(tiefe * 100)}", ["mom"], lb, tiefe, 0.0005)
            )
    for tiefe in (0.5, 0.6, 0.7):
        varianten.append(
            (
                f"V_chor_d{int(tiefe * 100)}",
                ["trend200", "cross", "mom"],
                252,
                tiefe,
                0.0005,
            )
        )
    varianten.append(("V_mom252_d60_2xkosten", ["mom"], 252, 0.6, 0.001))
    varianten.append(
        ("V_chor_d60_2xkosten", ["trend200", "cross", "mom"], 252, 0.6, 0.001)
    )

    ergebnis = {"registriert": __doc__, "varianten": {}}
    bestanden_beide = []
    for name, mitglieder, lb, tiefe, kosten in varianten:
        eintrag = {}
        beide_ok = True
        for ep_name, (px, rendite, bh, fenster) in (
            ("spy_1996_2016", epoche_spy()),
            ("crsp_1926_1995", epoche_1926()),
        ):
            sig = signale(px, lb)
            kurve = dial_kurve(px, rendite, [sig[m] for m in mitglieder], tiefe, kosten)
            checks, ok = pruefe(kurve, bh, fenster)
            eintrag[ep_name] = {
                "kompakt": {
                    f: (c["mdd_verbesserung_pct"], c["cagr_abgabe_pp"])
                    for f, c in checks.items()
                },
                "besteht": ok,
            }
            beide_ok &= ok
        eintrag["besteht_beide_epochen"] = beide_ok
        if beide_ok:
            bestanden_beide.append(name)
        ergebnis["varianten"][name] = eintrag
        print(
            f"{name}: {'BESTEHT beide' if beide_ok else 'fail'} | spy {eintrag['spy_1996_2016']['besteht']} crsp {eintrag['crsp_1926_1995']['besteht']}",
            flush=True,
        )
    ergebnis["verdikt"] = {
        "bestanden_beide_epochen": bestanden_beide,
        "quote": f"{len(bestanden_beide)}/14",
        "lesart_vorab": "mehrheitlich = robuste Flaeche; nur Originale = Insel -> Herabstufung",
    }
    ZIEL.write_text(
        json.dumps(ergebnis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("VERDIKT:", ergebnis["verdikt"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
