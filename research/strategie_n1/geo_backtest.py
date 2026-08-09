"""Strategie N1 — Backtest der GEO-Komponente auf GDELT 2015-2016.

REGISTRIERUNG (vor jedem Datenkontakt fixiert, +2 Trials; Aenderung = neue
Registrierung). Getestet wird NUR die GEO-Komponente der N1-Spez auf
Tagesbasis — nicht die Fusion, nicht Intraday, nicht die Latenz. Fenster
2015-01-01..2016-12-31 gemaess Spez §6 (Holdout 2017..2026-07 versiegelt).

SCORE (taeglich, PIT: GDELT-1.0-Tagesdatei ist am Folgetag verfuegbar ->
Signal am Tag T nutzt Events bis T-1):
  X_me     = Summe NumArticles ueber Events mit QuadClass=4 (material
             conflict) und ActionGeo-Land in NAHOST = {IR,IZ,SA,SY,YM,IS,
             LE,EG,TU,QA,KU,BA,AE,MU,JO} (FIPS-Codes)
  X_global = dito ohne Laenderfilter
  z_*      = (X - Mittel_180T) / Std_180T (rollierend, nur Vergangenheit)
  geo_score = max(z_me, z_global)
ENTRY: geo_score(T-1) >= 3.0 (fix, kein Tuning) -> Kauf zum OPEN von T,
  gleichgewichtet in die 3 Instrumente; eine offene Position je Instrument.
INSTRUMENTE (Eskalation -> long) — ABWEICHUNG VON DER REGISTRIERUNG,
  verfuegbarkeitsgetrieben und VOR jedem Ergebnis-Blick fixiert (Stooq
  liefert JS-Challenge, yfinance gedrosselt, lokal nur Close-Panel):
  GLD (wie registriert); Defense-Basket LMT/NOC/GD/RTN gleichgewichtet
  (Ersatz fuer ITA); Oel-Majors-Basket XOM/CVX/COP (Ersatz fuer USO —
  Aktien-Beta statt Crude, SCHWAECHERER Proxy). USO/ITA selbst bleiben
  UNGETESTET. Kurse: prices_verdict.parquet (nur Close).
EXITS (erster gewinnt; Spez §4) — CLOSE-BASIERT (kein OHLC verfuegbar;
  konservativ: Entry erst am naechsten CLOSE nach dem Signal, Exits zum
  Folge-Close): (1) Score-Zerfall geo_score < 1.5; (2) Zeit 15 Handels-
  tage; (3) Risiko: Close <= Entry -5 % -> Exit Folge-Close; Trailing ab
  +3 % Peak-Close mit 2 % Abstand.
KOSTEN: 2 bps Fee + 3 bps Slippage je Seite (liquide ETFs) = 10 bps
  Roundtrip.
KONTROLLE: je Instrument 100 Zufalls-Entry-Tage (Seed 51), gleiche Exits
  mit dem realen geo_score fuer den Zerfalls-Exit.
VERDIKT (vorab): interessant nur bei Netto-Mittel > 0 UND t > 2 UND
  PF > 1 gepoolt UND klarer Abstand zur Kontrolle. Sonst FAIL.
OFFENGELEGT: die z-Schwelle 3.0 ist eine Vorab-Setzung ohne Shadow-
  Verteilung (die existiert erst nach Wochen Sammelbetrieb); das Fenster
  2015-16 enthaelt u. a. Syrien/Jemen-Eskalationen und den Oel-Crash —
  Regime-Abhaengigkeit ist NICHT pruefbar mit 2 Jahren.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HIER = Path(__file__).resolve().parent
ROOT = HIER.parents[1]
sys.path.insert(0, str(ROOT))

from research.mandat2.data_gate import TrialCounter  # noqa: E402

DATEN = HIER / "data_geo"
ZIEL = HIER / "geo_backtest.json"
UA = "Mozilla/5.0 Forschung-N1 (hans.oertel2@gmail.com)"

START, ENDE = "2015-01-01", "2016-12-31"
NAHOST = {
    "IR",
    "IZ",
    "SA",
    "SY",
    "YM",
    "IS",
    "LE",
    "EG",
    "TU",
    "QA",
    "KU",
    "BA",
    "AE",
    "MU",
    "JO",
}
INSTRUMENTE = {
    "GLD": ["GLD"],
    "DEFENSE": ["LMT", "NOC", "GD", "RTN"],
    "OEL_MAJORS": ["XOM", "CVX", "COP"],
}
Z_ENTRY, Z_EXIT = 3.0, 1.5
MAX_TAGE = 15
STOP, TRAIL_AKTIV, TRAIL_ABSTAND = -0.05, 0.03, 0.02
FEE_SEITE = 0.0005  # 2 bps Fee + 3 bps Slippage
SEED = 51
KONTROLLEN = 100


_PANEL = None


def lade_basket(symbole: list[str]) -> pd.Series:
    """Gleichgewichteter Close-Basket (normierte Preisindizes) 2014-06..2017-02."""
    global _PANEL
    if _PANEL is None:
        _PANEL = pd.read_parquet(
            ROOT / "research" / "mandat" / "data" / "prices_verdict.parquet"
        )
    teile = []
    for sym in symbole:
        s = (
            _PANEL[_PANEL["symbol"] == sym]
            .set_index("timestamp")["close"]
            .sort_index()
            .loc["2014-06-01":"2017-03-01"]
        )
        teile.append(s / s.iloc[0])
    df = pd.concat(teile, axis=1).dropna()
    basket = df.mean(axis=1)
    basket.index = pd.DatetimeIndex(basket.index).tz_localize(None).normalize()
    return basket


def lade_gdelt_tag(datum: str) -> tuple[float, float]:
    """(artikel_nahost, artikel_global) fuer QuadClass=4 am Tag datum."""
    cache = DATEN / f"g{datum}.json"
    if cache.exists():
        d = json.loads(cache.read_text())
        return d["me"], d["gl"]
    url = f"http://data.gdeltproject.org/events/{datum}.export.CSV.zip"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for versuch in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as h:  # noqa: S310
                roh = h.read()
            break
        except Exception:
            if versuch == 2:
                raise
            time.sleep(5)
    me = gl = 0.0
    with zipfile.ZipFile(io.BytesIO(roh)) as z:
        with z.open(z.namelist()[0]) as fh:
            for zeile in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
                f = zeile.rstrip("\n").split("\t")
                if len(f) < 55 or f[29] != "4":
                    continue
                try:
                    artikel = float(f[33])
                except ValueError:
                    continue
                gl += artikel
                if f[51] in NAHOST:
                    me += artikel
    cache.write_text(json.dumps({"me": me, "gl": gl}))
    return me, gl


def baue_score() -> pd.Series:
    tage = pd.date_range("2014-06-01", ENDE, freq="D")
    zeilen = []
    for i, t in enumerate(tage):
        me, gl = lade_gdelt_tag(t.strftime("%Y%m%d"))
        zeilen.append({"tag": t, "me": me, "gl": gl})
        if i % 50 == 0:
            print(f"[GDELT] {t.date()} ({i}/{len(tage)})", flush=True)
        time.sleep(0.3)
    df = pd.DataFrame(zeilen).set_index("tag")
    z = {}
    for spalte in ("me", "gl"):
        mu = df[spalte].rolling(180, min_periods=120).mean().shift(1)
        sd = df[spalte].rolling(180, min_periods=120).std().shift(1)
        z[spalte] = (df[spalte] - mu) / sd
    score = pd.concat([z["me"], z["gl"]], axis=1).max(axis=1)
    score.name = "geo_score"
    return score


def simuliere(
    px: pd.Series, score: pd.Series, i_entry: int
) -> tuple[float, int] | None:
    c = px.to_numpy()
    if i_entry + 1 >= len(px):
        return None
    entry = c[i_entry] * (1 + FEE_SEITE)  # Entry am Signal-Folge-Close
    basis = c[i_entry]
    peak_close = -np.inf
    for n, j in enumerate(range(i_entry + 1, min(i_entry + 1 + MAX_TAGE, len(px) - 1))):
        peak_close = max(peak_close, c[j])
        s = score.reindex([px.index[j]]).iloc[0]
        stop_riss = c[j] <= basis * (1 + STOP)
        trail_aktiv = peak_close >= basis * (1 + TRAIL_AKTIV)
        trail_riss = trail_aktiv and c[j] <= peak_close * (1 - TRAIL_ABSTAND)
        zerfall = bool(np.isfinite(s) and s < Z_EXIT)
        if stop_riss or trail_riss or zerfall or n == MAX_TAGE - 1:
            return c[j + 1] * (1 - FEE_SEITE) / entry - 1, j + 1
    j = min(i_entry + MAX_TAGE, len(px) - 1)
    return c[j] * (1 - FEE_SEITE) / entry - 1, j


def lauf(px: pd.DataFrame, score: pd.Series, entries: list[int]) -> list[float]:
    aus, frei_ab = [], -1
    for i in entries:
        if i <= frei_ab:
            continue
        r = simuliere(px, score, i)
        if r is not None:
            aus.append(r[0])
            frei_ab = r[1]
    return aus


def statistik(x: list[float]) -> dict:
    s = pd.Series(x, dtype=float)
    gew, verl = s[s > 0].sum(), -s[s <= 0].sum()
    return {
        "n_trades": int(len(s)),
        "mittel_pp": round(float(s.mean()) * 100, 4) if len(s) else None,
        "t": round(float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))), 2)
        if len(s) > 2 and s.std(ddof=1) > 0
        else None,
        "trefferquote": round(float((s > 0).mean()), 3) if len(s) else None,
        "profit_faktor": round(float(gew / verl), 3) if verl > 0 else None,
        "summe_pct": round(float(s.sum()) * 100, 2) if len(s) else None,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regen", action="store_true")
    args = ap.parse_args(argv)
    DATEN.mkdir(parents=True, exist_ok=True)
    if args.regen:
        print(f"[REGEN] Trials unveraendert: {TrialCounter().total()}", flush=True)
    else:
        print(
            "Trials kumuliert: "
            + str(TrialCounter().increment(2, label="N1 GEO-Komponente GDELT 2015-16")),
            flush=True,
        )
    score = baue_score()
    rng = np.random.default_rng(SEED)
    ergebnis: dict = {"registriert": __doc__, "instrumente": {}}
    pool: dict[str, list[float]] = {"signal": [], "kontrolle": []}
    for sym, symbole in INSTRUMENTE.items():
        px = lade_basket(symbole).loc[START:ENDE]
        sc = score.reindex(px.index)
        # Signal am Tag T-1 -> Entry Open T
        sig_tage = [
            i
            for i in range(1, len(px))
            if np.isfinite(sc.iloc[i - 1]) and sc.iloc[i - 1] >= Z_ENTRY
        ]
        zufall = sorted(
            rng.choice(
                np.arange(1, len(px) - MAX_TAGE - 1), size=KONTROLLEN, replace=False
            )
        )
        s_tr = lauf(px, score, sig_tage)
        k_tr = lauf(px, score, [int(i) for i in zufall])
        ergebnis["instrumente"][sym] = {
            "basket": symbole,
            "handelstage": len(px),
            "n_signaltage": len(sig_tage),
            "signal": statistik(s_tr),
            "kontrolle": statistik(k_tr),
        }
        pool["signal"] += s_tr
        pool["kontrolle"] += k_tr
        print(f"{sym}: {len(sig_tage)} Signaltage, {len(s_tr)} Trades", flush=True)
    ergebnis["gepoolt"] = {k: statistik(v) for k, v in pool.items()}
    g = ergebnis["gepoolt"]["signal"]
    ergebnis["verdikt"] = {
        "besteht": bool(
            g.get("n_trades", 0) > 2
            and (g.get("mittel_pp") or 0) > 0
            and (g.get("t") or 0) > 2
            and (g.get("profit_faktor") or 0) > 1
        ),
        "kriterium": "netto>0 & t>2 & PF>1 gepoolt + Abstand zur Kontrolle (vorab)",
    }
    ZIEL.write_text(
        json.dumps(ergebnis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for k, v in ergebnis["gepoolt"].items():
        print(k, "->", v)
    print("VERDIKT:", ergebnis["verdikt"])
    print(f"-> {ZIEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
