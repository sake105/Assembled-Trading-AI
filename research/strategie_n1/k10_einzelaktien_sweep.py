"""K10 — Kombinatorik-Sweep auf EINZELAKTIEN (Auftrag Hans, 2026-08-10).

REGISTRIERUNG (vor Datenkontakt fixiert; +627 Trials; alle Ergebnisse im
Artefakt):

UNIVERSUM: alle Symbole des survivorship-FREIEN Verdict-Panels (inkl.
Delistete) mit >= 2520 Beobachtungen (~10 Jahre) im Fenster
1994-06-01..2016-12-31. Delistete enden, wann sie enden — das Portfolio
mittelt taeglich ueber die lebenden Mitglieder (identisch fuer Strategie
und Benchmark -> fair, kein Survivorship-Trick).

MECHANIK je Kombination (dieselben 627 Choere und 10 Indikatoren wie K9,
W33-Standardparameter, KEIN Tuning): JEDE Aktie bekommt ihren EIGENEN
Chor-Dial (Risiko = Mitglieder-Mittel auf DIESER Aktie; Exposure =
1 - 0.6*Risiko, 1 Tag Lag, 5 bps je Umschichtungseinheit).
PORTFOLIO = gleichgewichtetes Tagesmittel der Aktien-Strategierenditen.
BENCHMARK = gleichgewichtetes Tagesmittel der rohen Aktienrenditen
(EW-Buy-and-Hold desselben Universums, identische Mittelung).

KRITERIUM (identisch K1/K9, vorab): MDD >= 25 % besser als Benchmark UND
CAGR-Abgabe <= 1.0 pp p.a., in GESAMT (1996-2016) und BEIDEN Haelften
(1996-2006 / 2007-2016).

DATENHYGIENE (Nachtrag 2026-08-10, VOR erster Ergebnis-Sicht des
Neulaufs; Erstlauf war durch Overflow UNGUELTIG — E-052-Klasse:
pct_change ueberbrueckte Delisting-Luecken, Penny-Datenfehler erzeugten
absurde Renditen): (1) pct_change(fill_method=None) — keine Pads ueber
Luecken; (2) Tagesrenditen mit |r| > 100 % gelten als Datenfehler und
werden auf NaN gesetzt — IDENTISCH fuer Strategie und Benchmark;
(3) Kurven in float64.

BESTAETIGUNG — OFFENGELEGTE GRENZE: Einzelaktien-Daten vor 1996 sind im
Haus nicht vorhanden (Verdict-Panel beginnt 1996). Ueberlebende sind
daher KANDIDATEN mit Index-Querverweis (K9-Bestaetigung 1926-1995), kein
per-Aktie-bestaetigtes Ergebnis. Der ehrliche zweite Beweis ist der
Forward-Shadow.
"""

from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HIER = Path(__file__).resolve().parent
ROOT = HIER.parents[1]
sys.path.insert(0, str(ROOT))

from research.mandat2.data_gate import TrialCounter  # noqa: E402

ZIEL = HIER / "k10_einzelaktien_sweep.json"
KOSTEN = 0.0005
DIAL = 0.6
MIN_OBS = 2520


def lade_universum() -> pd.DataFrame:
    pv = pd.read_parquet(
        ROOT / "research" / "mandat" / "data" / "prices_verdict.parquet"
    )
    pv = pv[pv["symbol"] != "SPY"]
    breit = pv.pivot_table(index="timestamp", columns="symbol", values="close")
    breit.index = pd.DatetimeIndex(breit.index).tz_localize(None).normalize()
    breit = breit.loc["1994-06-01":"2016-12-31"]
    genug = breit.notna().sum() >= MIN_OBS
    return breit.loc[:, genug[genug].index].astype(np.float32)


def indikator_matrizen(px: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rt = px.pct_change(fill_method=None)
    sma50 = px.rolling(50).mean()
    sma200 = px.rolling(200).mean()
    ema20 = px.ewm(span=20, adjust=False).mean()
    ema60 = px.ewm(span=60, adjust=False).mean()
    macd = px.ewm(span=12, adjust=False).mean() - px.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    kanal = (px.rolling(55).max() + px.rolling(55).min()) / 2
    delta = px.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    mb = px.rolling(20).mean()
    sd = px.rolling(20).std()
    vol = rt.rolling(20).std()
    vz = (vol - vol.rolling(180).mean()) / vol.rolling(180).std()
    lebt = px.notna()

    def m(df):  # nur wo die Aktie lebt; NaN-Signale = 0 Risiko-Beitrag
        return df.where(lebt, np.nan).astype(np.float32)

    return {
        "I1_trend200": m((px < sma200).astype(np.float32)),
        "I2_cross": m((sma50 < sma200).astype(np.float32)),
        "I3_ema": m((ema20 < ema60).astype(np.float32)),
        "I4_macd": m((macd < sig).astype(np.float32)),
        "I5_donchian": m((px < kanal).astype(np.float32)),
        "I6_rsi": m((rsi < 50).astype(np.float32)),
        "I7_boll": m((px < mb - 2 * sd).astype(np.float32)),
        "I8_mom6": m((px.pct_change(126) < 0).astype(np.float32)),
        "I9_mom12": m((px.pct_change(252) < 0).astype(np.float32)),
        "I10_vol": m((vz.clip(0, 3) / 3).fillna(0).astype(np.float32)),
    }


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
            "strategie": ks,
            "benchmark": kb,
            "mdd_verbesserung_pct": round(mdd_impr * 100, 1),
            "cagr_abgabe_pp": round(abgabe, 2),
            "besteht": bool(mdd_impr >= 0.25 and abgabe <= 1.0),
        }
    return checks, all(c["besteht"] for c in checks.values())


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regen", action="store_true")
    args = ap.parse_args(argv)
    if args.regen:
        print(f"[REGEN] Trials unveraendert: {TrialCounter().total()}", flush=True)
    else:
        print(
            "Trials kumuliert: "
            + str(
                TrialCounter().increment(627, label="K10 Einzelaktien-Sweep 627 Choere")
            ),
            flush=True,
        )
    t0 = time.time()
    px = lade_universum()
    print(
        f"Universum: {px.shape[1]} Aktien x {px.shape[0]} Tage (inkl. Delistete)",
        flush=True,
    )
    ind = indikator_matrizen(px)
    rt = px.pct_change(fill_method=None)
    rt = rt.where(rt.abs() <= 1.0)  # Hygiene: |r|>100 %/Tag = Datenfehler
    rt = rt.astype(np.float64)
    bh_r = rt.mean(axis=1)
    bh = (1 + bh_r.fillna(0)).cumprod()
    fenster = {
        "gesamt": ("1996-01-01", "2016-12-31"),
        "h1": ("1996-01-01", "2006-12-31"),
        "h2": ("2007-01-01", "2016-12-31"),
    }
    namen = list(ind.keys())
    kombis = [k for g in (2, 3, 4, 5) for k in combinations(namen, g)]
    ergebnis: dict = {
        "registriert": __doc__,
        "universum": int(px.shape[1]),
        "configs": {},
    }
    for i, kombi in enumerate(kombis):
        risiko = sum(ind[n] for n in kombi) / len(kombi)
        expo = (1 - DIAL * risiko).shift(1)
        kosten = expo.diff().abs() * KOSTEN
        strat = (expo.astype(np.float64) * rt - kosten).mean(axis=1)
        kurve = (1 + strat.fillna(0)).cumprod()
        checks, ok = pruefe(kurve, bh, fenster)
        ergebnis["configs"]["+".join(kombi)] = {
            "k": len(kombi),
            "checks": checks,
            "besteht": ok,
        }
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(kombis)} ({time.time() - t0:.0f}s)", flush=True)
    bestanden = [n for n, v in ergebnis["configs"].items() if v["besteht"]]
    je_k = {
        k: [
            sum(1 for v in ergebnis["configs"].values() if v["k"] == k),
            sum(
                1 for v in ergebnis["configs"].values() if v["k"] == k and v["besteht"]
            ),
        ]
        for k in (2, 3, 4, 5)
    }
    rang = sorted(
        bestanden,
        key=lambda n: ergebnis["configs"][n]["checks"]["gesamt"]["cagr_abgabe_pp"],
    )
    ergebnis["verdikt"] = {
        "n_bestanden": len(bestanden),
        "je_groesse_total_bestanden": je_k,
        "top10_nach_cagr_abgabe": rang[:10],
        "laufzeit_s": round(time.time() - t0, 1),
        "hinweis": (
            "Einzelaktien-Bestaetigungsdaten vor 1996 nicht im Haus — "
            "Bestehende sind Kandidaten mit Index-Querverweis (K9 1926-1995), "
            "zweiter Beweis nur via Forward-Shadow."
        ),
    }
    ZIEL.write_text(
        json.dumps(ergebnis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        "VERDIKT:",
        json.dumps(ergebnis["verdikt"], ensure_ascii=False)[:400],
        flush=True,
    )
    print(f"-> {ZIEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
