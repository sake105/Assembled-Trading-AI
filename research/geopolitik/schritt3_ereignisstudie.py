"""Schritt 2/3 — Ereignisstudie auf dem erinnerungsfreien Universum (Welle 48).

+2 Trials, gebucht VOR dem Lauf. Abbruchkriterium aus der Registrierung:
signierte Ueberrendite nicht > 0 mit t > 2 in mindestens einem der zwei
Horizonte, in BEIDEN Regel-Klassen -> These tot, Feld zu.

HANDELBARKEIT MIT TAGESDATEN (Schritt 2)
----------------------------------------
Das Panel fuehrt nur Schlusskurse. Der erste Kurs, zu dem nach einem Post
real gehandelt werden kann, ist damit der ERSTE SCHLUSS NACH dem Zeitstempel:

* Post vor 16:00 ET an einem Handelstag -> Schluss desselben Tages.
* Post nach 16:00 ET oder an einem handelsfreien Tag -> Schluss des naechsten
  Handelstags.

Die Bewegung vom letzten Schluss VOR dem Post bis zum Einstiegsschluss
(Eroeffnungsluecke plus Intraday-Drift) wird separat ausgewiesen als
`nicht_handelbar` — sie ist Information, kein Ertrag. Gemessen wird die
signierte Ueberrendite NACH dem Einstieg: +1 und +5 Handelstage.

ANALYSE-KONVENTIONEN (mechanisch, vor Sicht auf irgendein Ergebnis fixiert)
---------------------------------------------------------------------------
* Mehrere Ereignisse desselben Tickers am selben Handelstag werden zu EINEM
  zusammengefasst (erster Post zaehlt); widerspricht sich die Richtung an
  einem Tag, faellt der Tag. Sonst zaehlte ein Tweetsturm als zehn unabhaengige
  Beobachtungen (E-078).
* Regel A: Ueberrendite = richtung * (r_ticker - r_spy). Regel B:
  richtung * r_spy.
* Kontrollgruppe: je Ereignis 20 Zufalls-Handelstage DESSELBEN Tickers im
  selben Fenster, Seed fest. Die Kontrolle traegt das dokumentierte
  Namens-Rauschen (SO/"Southern Border" etc.): ein Nicht-Signal muss auf
  Zufallstagen dieselbe Statistik liefern wie auf Posttagen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HIER = Path(__file__).resolve().parent
ROOT = HIER.parents[1]
sys.path.insert(0, str(ROOT))

from research.mandat2.data_gate import TrialCounter  # noqa: E402

DATA = ROOT / "research" / "mandat" / "data"
EREIGNISSE = HIER / "data" / "ereignisse.parquet"
ZIEL = HIER / "schritt3_ereignisstudie.json"

MARKTSCHLUSS = 16  # ET, Archivstempel sind ET (verifiziert)
KONTROLLEN_JE_EVENT = 20
SEED = 48  # Welle 48
HORIZONTE = (1, 5)


def lade_kurse() -> pd.DataFrame:
    sc = pd.read_parquet(DATA / "_sc_close.parquet")
    # Panel-Index ist tz-aware UTC-Mitternacht, die Ereigniszeiten sind naive
    # ET-Stempel. Fuer den TAGES-Abgleich zaehlt nur das Kalenderdatum — der
    # Index wird naiv gemacht. (Der Stunden-Anteil der Ereignisse geht NUR in
    # die vor/nach-16-Uhr-Entscheidung ein, nie in einen Index-Vergleich.)
    sc.index = pd.DatetimeIndex(sc.index).tz_localize(None).normalize()
    pv = pd.read_parquet(DATA / "prices_verdict.parquet")
    spy = pv[pv["symbol"] == "SPY"].set_index("timestamp")["close"]
    spy.index = spy.index.tz_localize(None).normalize()
    sc = sc.copy()
    sc["SPY"] = spy.reindex(sc.index)
    return sc


def einstiegstag(
    zeit_et: pd.Timestamp, handelstage: pd.DatetimeIndex
) -> pd.Timestamp | None:
    """Erster Schluss, zu dem nach dem Post gehandelt werden kann."""
    tag = zeit_et.normalize()
    if tag in handelstage and zeit_et.hour < MARKTSCHLUSS:
        return tag
    nach = handelstage[handelstage > tag]
    return nach[0] if len(nach) else None


def studie(ev: pd.DataFrame, kurse: pd.DataFrame, regel: str) -> dict:
    handelstage = pd.DatetimeIndex(kurse.index)
    rng = np.random.default_rng(SEED)

    # Ein Ereignis je Ticker-Handelstag; Richtungskonflikt -> Tag faellt.
    ev = ev.sort_values("zeit_et").copy()
    ev["etag"] = [einstiegstag(z, handelstage) for z in ev["zeit_et"]]
    ev = ev.dropna(subset=["etag"])
    gr = ev.groupby(["ticker", "etag"])["richtung"]
    eindeutig = gr.nunique() == 1
    ev = ev.drop_duplicates(["ticker", "etag"], keep="first")
    ev = ev.set_index(["ticker", "etag"]).loc[eindeutig[eindeutig].index].reset_index()

    def renditen(ticker: str, etag: pd.Timestamp) -> dict | None:
        s = kurse[ticker].dropna() if ticker in kurse.columns else None
        spy = kurse["SPY"].dropna()
        if s is None or etag not in s.index or etag not in spy.index:
            return None
        pos = s.index.get_loc(etag)
        pos_spy = spy.index.get_loc(etag)
        aus = {}
        # nicht handelbarer Teil: letzter Schluss vor Einstieg -> Einstieg
        if pos >= 1:
            aus["gap"] = float(s.iloc[pos] / s.iloc[pos - 1] - 1)
        for h in HORIZONTE:
            if pos + h < len(s) and pos_spy + h < len(spy):
                r_t = float(s.iloc[pos + h] / s.iloc[pos] - 1)
                r_m = float(spy.iloc[pos_spy + h] / spy.iloc[pos_spy] - 1)
                aus[h] = r_t - r_m if regel == "A" else r_t
        return aus

    zeilen, gaps = [], []
    for _, z in ev.iterrows():
        r = renditen(z["ticker"], z["etag"])
        if r is None:
            continue
        if "gap" in r:
            gaps.append(z["richtung"] * r["gap"])
        zeile = {"ticker": z["ticker"], "richtung": z["richtung"]}
        for h in HORIZONTE:
            if h in r:
                zeile[f"h{h}"] = z["richtung"] * r[h]
        zeilen.append(zeile)
    df = pd.DataFrame(zeilen)

    # Kontrollgruppe: Zufallstage desselben Tickers, gleiche Richtung.
    kontrollen = {h: [] for h in HORIZONTE}
    for _, z in df.iterrows():
        s = kurse[z["ticker"]].dropna()
        gueltig = s.index[252 : -max(HORIZONTE) - 1]
        if len(gueltig) < KONTROLLEN_JE_EVENT:
            continue
        for tag in rng.choice(gueltig, size=KONTROLLEN_JE_EVENT, replace=False):
            r = renditen(z["ticker"], pd.Timestamp(tag))
            if r:
                for h in HORIZONTE:
                    if h in r:
                        kontrollen[h].append(z["richtung"] * r[h])

    def tstat(x: pd.Series) -> float:
        x = x.dropna()
        return (
            float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))
            if len(x) > 2
            else float("nan")
        )

    aus = {
        "regel": regel,
        "n_ereignisse": len(df),
        "gap_nicht_handelbar_median": round(float(np.median(gaps)), 5)
        if gaps
        else None,
        "gap_nicht_handelbar_mittel": round(float(np.mean(gaps)), 5) if gaps else None,
        "horizonte": {},
    }
    for h in HORIZONTE:
        e = df[f"h{h}"] if f"h{h}" in df else pd.Series(dtype=float)
        k = pd.Series(kontrollen[h])
        aus["horizonte"][f"{h}T"] = {
            "n": int(e.notna().sum()),
            "mittel_pp": round(float(e.mean()) * 100, 3) if len(e) else None,
            "t": round(tstat(e), 2) if len(e) else None,
            "kontrolle_mittel_pp": round(float(k.mean()) * 100, 3) if len(k) else None,
            "kontrolle_n": len(k),
            "pass_t2": bool(len(e) and e.mean() > 0 and tstat(e) > 2),
        }
    return aus


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--regen", action="store_true", help="Neulauf ohne Trial-Increment (E-090)."
    )
    args = ap.parse_args(argv)
    ev = pd.read_parquet(EREIGNISSE)
    ev["zeit_et"] = pd.to_datetime(ev["zeit_et"])
    kurse = lade_kurse()
    # BUCHUNGS-OFFENLEGUNG: Der Erstlauf buchte +2 und brach am tz-Fehler ab.
    # Der Reparatur-Patch meldete "gepatcht", aber der main()-Ersatz griff
    # nicht (stiller str.replace-No-Match) — der Neulauf buchte NOCHMAL +2.
    # Stand damit 3537 statt 3535; append-only, nicht zurueckgeschrieben.
    # Dieselbe Klasse wie E-129 (Schutz behauptet, nicht wirksam), diesmal im
    # Patch-Werkzeug statt im Flag.
    if args.regen:
        print(
            f"[REGEN] Trial-Zaehler UNVERAENDERT bei {TrialCounter().total()}\n",
            flush=True,
        )
    else:
        print(
            f"Trials kumuliert: "
            f"{TrialCounter().increment(2, label='Welle 48 Ereignisstudie')}\n",
            flush=True,
        )

    ergebnis = {
        "registriert": "Welle 48, Abbruchkriterium vorab",
        "A": studie(ev[ev.regel == "A"], kurse, "A"),
        "B": studie(ev[ev.regel == "B"], kurse, "B"),
    }

    a_pass = any(v["pass_t2"] for v in ergebnis["A"]["horizonte"].values())
    b_pass = any(v["pass_t2"] for v in ergebnis["B"]["horizonte"].values())
    ergebnis["verdikt"] = {
        "a_besteht": a_pass,
        "b_besteht": b_pass,
        "these_lebt": bool(a_pass or b_pass),
        "regel_text": (
            "Abbruch, wenn KEINE Klasse in mindestens einem Horizont "
            "signierte Ueberrendite > 0 mit t > 2 zeigt."
        ),
    }
    ZIEL.write_text(
        json.dumps(ergebnis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for r in ("A", "B"):
        e = ergebnis[r]
        print(
            f"Regel {r}: {e['n_ereignisse']} Ereignisse | Gap (nicht handelbar) "
            f"Median {e['gap_nicht_handelbar_median']}"
        )
        for hz, v in e["horizonte"].items():
            print(
                f"  {hz}: mittel {v['mittel_pp']} pp (t={v['t']}) | Kontrolle "
                f"{v['kontrolle_mittel_pp']} pp (n={v['kontrolle_n']}) | "
                f"{'PASS' if v['pass_t2'] else 'fail'}"
            )
    print(
        "\nVERDIKT:",
        "These lebt"
        if ergebnis["verdikt"]["these_lebt"]
        else "FAIL in beiden Klassen — Feld zu",
    )
    print(f"-> {ZIEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
