"""K9 — Systematischer Kombinatorik-Sweep: 2er/3er/4er/5er-Indikator-Chöre.

AUFTRAG (Hans, 2026-08-10): Kombinationen der Einzelindikatoren SYSTEMATISCH
durchmustern — erst 2er, dann 3er, 4er, 5er. Keine Stichprobe.

REGISTRIERUNG (vor Datenkontakt fixiert; Trials = exakte Config-Zahl, im
Lauf gebucht; ALLE Ergebnisse im Artefakt):

INDIKATOR-INVENTAR (10 Stueck, je Risiko-Signal in [0,1], Definitionen fix,
KEIN Parameter-Tuning — Parameter sind die W33-Standards):
  I1  trend200   Close < SMA200
  I2  cross      SMA50 < SMA200
  I3  ema        EMA20 < EMA60
  I4  macd       MACD(12,26) < Signal(9)
  I5  donchian   Close < Mitte des 55T-Kanals
  I6  rsi        RSI14 < 50
  I7  boll       Close < unteres Bollinger-Band (20, 2 Sigma)
  I8  mom6       126T-Rendite < 0
  I9  mom12      252T-Rendite < 0
  I10 vol        clip(z(20T-Vol vs 180T), 0, 3)/3   (stetig)

KOMBINATION: fuer jede k-Teilmenge (k=2..5) ist RISIKO = Mittelwert der
Mitglieder (gleichgewichteter Chor; die Gewichtung ueber den Raum entsteht
durch die Zusammensetzung). EXPOSURE = 1 - 0.6*RISIKO, 1 Tag Lag, 5 bps je
Umschichtungseinheit — identische Mechanik wie K1/E1.
RAUM: C(10,2)+C(10,3)+C(10,4)+C(10,5) = 45+120+210+252 = 627 Kombis,
  je auf SPY (1996-2016, Haelften 1996-2006/2007-2016) und GLD
  (2005-07..2016-12, Haelften wie K3) = 1.254 Configs.
KRITERIUM je Config (identisch K1, vorab): vs Buy-and-Hold des Assets:
  MDD >= 25 % besser UND CAGR-Abgabe <= 1.0 pp p.a., gesamt + beide
  Haelften. ERWARTUNG OFFENGELEGT: bei 1.254 Versuchen bestehen etliche
  durch Zufall — das In-Fenster-Bestehen ist NUR ein Filter.
BESTAETIGUNG (der eigentliche Test): Ueberlebende werden nach vorab
  fixiertem Rang (kleinste CAGR-Abgabe gesamt) sortiert; MAXIMAL die Top
  10 je Asset bekommen je EINEN Versuch auf disjunkten Daten:
  SPY-Kombis -> CRSP-VW 1926-1995 (Ueberschussrechnung);
  GLD-Kombis -> LBMA-Fixing 1975-2004 (Total Return, Indikatoren auf der
  Goldreihe selbst). Je +1 Trial, gebucht bei Ausfuehrung.
  BESTAETIGT heisst: gleiches Doppel-Kriterium in gesamt + beiden
  Haelften des Bestaetigungsfensters.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HIER = Path(__file__).resolve().parent
ROOT = HIER.parents[1]
sys.path.insert(0, str(ROOT))

from research.mandat2.data_gate import TrialCounter  # noqa: E402

ZIEL = HIER / "k9_kombinatorik_sweep.json"
KOSTEN = 0.0005
DIAL = 0.6
MAX_BESTAETIGUNGEN_JE_ASSET = 10


def indikatoren(px: pd.Series) -> pd.DataFrame:
    rt = px.pct_change()
    sma50, sma200 = px.rolling(50).mean(), px.rolling(200).mean()
    ema20 = px.ewm(span=20, adjust=False).mean()
    ema60 = px.ewm(span=60, adjust=False).mean()
    ema12 = px.ewm(span=12, adjust=False).mean()
    ema26 = px.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    kanal_mitte = (px.rolling(55).max() + px.rolling(55).min()) / 2
    delta = px.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    mb = px.rolling(20).mean()
    sd = px.rolling(20).std()
    vol = rt.rolling(20).std()
    vz = (vol - vol.rolling(180).mean()) / vol.rolling(180).std()
    return pd.DataFrame(
        {
            "I1_trend200": (px < sma200).astype(float),
            "I2_cross": (sma50 < sma200).astype(float),
            "I3_ema": (ema20 < ema60).astype(float),
            "I4_macd": (macd < sig).astype(float),
            "I5_donchian": (px < kanal_mitte).astype(float),
            "I6_rsi": (rsi < 50).astype(float),
            "I7_boll": (px < mb - 2 * sd).astype(float),
            "I8_mom6": (px.pct_change(126) < 0).astype(float),
            "I9_mom12": (px.pct_change(252) < 0).astype(float),
            "I10_vol": (vz.clip(0, 3) / 3).fillna(0),
        }
    )


def kennzahlen(kurve: pd.Series) -> dict:
    jahre = (kurve.index[-1] - kurve.index[0]).days / 365.25
    cagr = float((kurve.iloc[-1] / kurve.iloc[0]) ** (1 / jahre) - 1)
    dd = float((kurve / kurve.cummax() - 1).min())
    return {"cagr_pct": round(cagr * 100, 2), "mdd_pct": round(dd * 100, 2)}


def pruefe(kurve: pd.Series, bh: pd.Series, fenster: dict) -> tuple[dict, bool]:
    checks = {}
    for f, (a, b) in fenster.items():
        ks = kennzahlen(kurve.loc[a:b] / kurve.loc[a:b].iloc[0])
        kb = kennzahlen(bh.loc[a:b] / bh.loc[a:b].iloc[0])
        mdd_impr = 1 - ks["mdd_pct"] / kb["mdd_pct"]
        abgabe = kb["cagr_pct"] - ks["cagr_pct"]
        checks[f] = {
            "mdd_verbesserung_pct": round(mdd_impr * 100, 1),
            "cagr_abgabe_pp": round(abgabe, 2),
            "besteht": bool(mdd_impr >= 0.25 and abgabe <= 1.0),
        }
    return checks, all(c["besteht"] for c in checks.values())


def dial_kurve(rendite: pd.Series, risiko: pd.Series) -> pd.Series:
    expo = (1 - DIAL * risiko).shift(1)
    kosten = expo.diff().abs().fillna(0) * KOSTEN
    return (1 + (expo * rendite - kosten).fillna(0)).cumprod()


def sweep(px: pd.Series, fenster: dict, label: str) -> dict:
    ind = indikatoren(px)
    rt = px.pct_change()
    aus = {}
    kombis = [k for groesse in (2, 3, 4, 5) for k in combinations(ind.columns, groesse)]
    for i, kombi in enumerate(kombis):
        risiko = ind[list(kombi)].mean(axis=1)
        kurve = dial_kurve(rt, risiko)
        checks, ok = pruefe(kurve, px, fenster)
        aus["+".join(kombi)] = {"k": len(kombi), "checks": checks, "besteht": ok}
        if (i + 1) % 100 == 0:
            print(f"[{label}] {i + 1}/{len(kombis)}", flush=True)
    return aus


def lade(symbol: str) -> pd.Series:
    pv = pd.read_parquet(
        ROOT / "research" / "mandat" / "data" / "prices_verdict.parquet"
    )
    s = (
        pv[pv["symbol"] == symbol]
        .set_index("timestamp")["close"]
        .sort_index()
        .astype(float)
    )
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s


def bestaetige_spy(kombi: tuple[str, ...]) -> tuple[dict, bool]:
    df = pd.read_parquet(
        ROOT / "research" / "mandat2" / "data_gratis" / "fama_french_daily.parquet"
    )
    df = df.loc["1926-07-01":"1995-12-31"]
    px = (1 + df["mkt"]).cumprod()
    ind = indikatoren(px)
    risiko = ind[list(kombi)].mean(axis=1)
    expo = (1 - DIAL * risiko).shift(1)
    kosten = expo.diff().abs().fillna(0) * KOSTEN
    strat = (1 + (expo * df["mkt_rf"] - kosten).fillna(0)).cumprod()
    bh = (1 + df["mkt_rf"].fillna(0)).cumprod()
    fenster = {
        "gesamt": ("1926-07-01", "1995-12-31"),
        "h1": ("1926-07-01", "1960-12-31"),
        "h2": ("1961-01-01", "1995-12-31"),
    }
    return pruefe(strat, bh, fenster)


def bestaetige_gld(kombi: tuple[str, ...]) -> tuple[dict, bool]:
    roh = json.load(open(HIER / "data_geo" / "lbma_gold_pm.json", encoding="utf-8"))
    gold = (
        pd.Series(
            {pd.Timestamp(r["d"]): r["v"][0] for r in roh if r["v"] and r["v"][0]}
        )
        .sort_index()
        .loc["1973-06-01":"2004-10-31"]
    )
    ind = indikatoren(gold)
    risiko = ind[list(kombi)].mean(axis=1)
    kurve = dial_kurve(gold.pct_change(), risiko)
    fenster = {
        "gesamt": ("1975-01-01", "2004-10-31"),
        "h1": ("1975-01-01", "1989-12-31"),
        "h2": ("1990-01-01", "2004-10-31"),
    }
    return pruefe(kurve, gold, fenster)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regen", action="store_true")
    args = ap.parse_args(argv)
    n_configs = 627 * 2
    if args.regen:
        print(f"[REGEN] Trials unveraendert: {TrialCounter().total()}", flush=True)
    else:
        print(
            "Trials kumuliert: "
            + str(
                TrialCounter().increment(n_configs, label="K9 Kombinatorik-Sweep 627x2")
            ),
            flush=True,
        )

    spy = lade("SPY").loc["1994-06-01":"2016-12-31"]
    fenster_spy = {
        "gesamt": ("1996-01-01", "2016-12-31"),
        "h1": ("1996-01-01", "2006-12-31"),
        "h2": ("2007-01-01", "2016-12-31"),
    }
    gld = lade("GLD").loc["2004-11-18":"2016-12-31"]
    fenster_gld = {
        "gesamt": ("2005-07-01", "2016-12-31"),
        "h1": ("2005-07-01", "2011-03-31"),
        "h2": ("2011-04-01", "2016-12-31"),
    }
    ergebnis: dict = {"registriert": __doc__, "assets": {}}
    ueberlebende: dict = {}
    for label, px, fenster in (("SPY", spy, fenster_spy), ("GLD", gld, fenster_gld)):
        res = sweep(px, fenster, label)
        n_ok = sum(1 for v in res.values() if v["besteht"])
        je_k = {
            k: [
                sum(1 for v in res.values() if v["k"] == k),
                sum(1 for v in res.values() if v["k"] == k and v["besteht"]),
            ]
            for k in (2, 3, 4, 5)
        }
        ergebnis["assets"][label] = {
            "n_configs": len(res),
            "n_bestanden_in_fenster": n_ok,
            "je_groesse_total_bestanden": je_k,
            "configs": res,
        }
        rang = sorted(
            (name for name, v in res.items() if v["besteht"]),
            key=lambda n: res[n]["checks"]["gesamt"]["cagr_abgabe_pp"],
        )
        ueberlebende[label] = rang[:MAX_BESTAETIGUNGEN_JE_ASSET]
        print(
            f"[{label}] {n_ok}/{len(res)} bestehen in-Fenster; je k: {je_k}; Top-10 zur Bestaetigung",
            flush=True,
        )

    n_conf = sum(len(v) for v in ueberlebende.values())
    if n_conf and not args.regen:
        print(
            "Trials kumuliert: "
            + str(
                TrialCounter().increment(n_conf, label=f"K9-Bestaetigungen ({n_conf}x)")
            ),
            flush=True,
        )
    ergebnis["bestaetigungen"] = {}
    for label, namen in ueberlebende.items():
        fn = bestaetige_spy if label == "SPY" else bestaetige_gld
        for name in namen:
            checks, ok = fn(tuple(name.split("+")))
            ergebnis["bestaetigungen"][f"{label}:{name}"] = {
                "checks": checks,
                "bestaetigt": ok,
            }
            print(
                f"[BESTAETIGUNG {label}] {name}: {'BESTANDEN' if ok else 'FAIL'}",
                flush=True,
            )

    bestaetigt = [k for k, v in ergebnis["bestaetigungen"].items() if v["bestaetigt"]]
    ergebnis["verdikt"] = {
        "in_fenster_bestanden": {
            k: v["n_bestanden_in_fenster"] for k, v in ergebnis["assets"].items()
        },
        "bestaetigt": bestaetigt,
        "hinweis": (
            "In-Fenster-Bestehen ist bei 1254 Versuchen NUR ein Filter; "
            "zaehlbar ist ausschliesslich die Bestaetigungs-Spalte."
        ),
    }
    ZIEL.write_text(
        json.dumps(ergebnis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("VERDIKT:", ergebnis["verdikt"])
    print(f"-> {ZIEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
