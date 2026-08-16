# -*- coding: utf-8 -*-
"""H-090 Phase 2 — Sekundaerkriterien (Deployability) fuer die Exit-Familie.

Registriert in research/registry.md Welle 50, VOR dem ersten Lauf:
  *Sekundaer (nur fuer primaer bestandene Varianten):*
  1. Netto-PRIVAT_DE (26,375 %, FIFO, Verlusttopf, End-Liquidation) >= EW-Pfad
     desselben PIT-Universums (Hauptvergleich, gleiche Datenwelt);
     SPY als FONDS-Klasse zusaetzlich ausgewiesen (E-069).
  2. DSR passes_5pct beim kumulierten Trial-Zaehler — heterogenes V aus dem
     P8-Artefakt (Regel aus p8_dsr_heterogen, E-077: Klonvarianz ist NICHT
     entscheidungsfaehig, wird nur zur Dokumentation mitgerechnet).
  3. PBO (CSCV, 8 Bloecke) <= 0,5 ueber die 8er-Familie. Caveat wie P13e:
     die Spalten sind Fast-Klone -> CSCV ist nach UNTEN verzerrt (E-077),
     und cscv_pbo rangiert nach Sharpe. Ein Bestehen ist unter dieser
     Verzerrung schwaecher, ein Scheitern staerker als die Zahl nahelegt.

TRIAL-ZAEHLER: steigt NICHT (E-090) — hier wird nichts gesucht, sondern das
bereits Gesuchte korrigiert und auf Deployability geprueft. Die 8 Trials der
Familie wurden im Phase-1-Erstlauf gebucht (Zaehler 6.277).

EINE WAHRHEIT (Rule 50): Trades kommen aus h090_momentum_exits.run_variant
ueber prepare_inputs — kein zweiter Entscheidungscode. Ein Guard vergleicht
n_trades je Variante mit dem Phase-1-Ergebnis-JSON und bricht bei Abweichung
ab (der Refactor prepare_inputs darf die Trades nicht veraendert haben).

PFAD-SIMULATION: Die Trade-Listen (BRUTTO-Kurse) werden auf ein
FIFO-Portfolio (research/mandat2/portfolio.py, autoritativ seit F-senior-9)
abgespielt: Positionsgroesse = Portfoliowert/5 beim Entry, cost_bps=15 wie
Phase 1, Dividenden aus div_panel (Steuer + Basis-Anhebung, E-068; das
Kurspanel ist TR-adjustiert). Welten: ZERO (fuer DSR/PBO — Steuer verzerrt
die Kurve, ohne etwas ueber Ueberanpassung zu sagen, wie P7/P13e) und
PRIVAT_DE (fuer den Steuer-Vergleich). EW-Benchmark = P11-Konstruktion
(gleichgewicht_score, top_in=500, rank_out=200, min_haltetage=730), aber in
PRIVAT_DE und mit cost_bps=15 (gleiche Reibung wie der Kandidat; P11 lief
mit dem 10-bp-Default — dokumentierte Abweichung zugunsten der
Vergleichbarkeit INNERHALB dieses Laufs).

Ergebnis-JSON als LETZTE Anweisung (E-116); Befund wird GENERIERT (E-085).

Aufruf:
    python research/mandat2/h090_phase2_sekundaer.py
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from research.mandat.h011_kandidat_a import cscv_pbo  # noqa: E402
from research.mandat2.campaign_data import load_campaign  # noqa: E402
from research.mandat2.data_gate import TrialCounter  # noqa: E402
from research.mandat2.engine import run_buy_and_hold, run_strategy  # noqa: E402
from research.mandat2.h090_momentum_exits import (  # noqa: E402
    TOP_K,
    VARIANTS,
    evaluate,
    prepare_inputs,
    run_variant_from_inputs,
)
from research.mandat2.portfolio import Portfolio  # noqa: E402
from research.mandat2.tax_regimes import AssetClass, make_regime  # noqa: E402

OUT = HERE / "results"
PHASE1_JSON = OUT / "h090_momentum_exits.json"
OUT_PATH = OUT / "h090_phase2_sekundaer.json"
P8_ARTEFAKT = OUT / "p8_dsr_heterogen.json"

COST_BPS = 15.0
START_KAPITAL = 100_000.0
EW_PARAMS = dict(top_in=500, rank_out=200, min_haltetage=730, hebel=1.0)


def gleichgewicht_score(close: pd.DataFrame) -> pd.DataFrame:
    """Konstanter Score — exakt die P11-Konstruktion des EW-Benchmarks."""
    return pd.DataFrame(1.0, index=close.index, columns=close.columns)


def div_auf_kalender(div_panel: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Dividenden vom rohen Panel-Kalender auf den SPY-Kalender bringen.

    Phantomtage wurden aus dem Simulationskalender entfernt; eine Dividende
    mit Ex-Tag auf einem Phantomtag wird dem NAECHSTEN Handelstag zugeordnet
    statt still zu entfallen.
    """
    out = pd.DataFrame(0.0, index=days, columns=div_panel.columns)
    pos = np.searchsorted(days.values, div_panel.index.values, side="left")
    for src_i, dst_i in enumerate(pos):
        if dst_i >= len(days):
            continue
        row = div_panel.iloc[src_i]
        vals = row[row.notna() & (row > 0)]
        if len(vals):
            out.loc[days[dst_i], vals.index] += vals
    return out


def replay(
    trades: list[dict],
    days: pd.DatetimeIndex,
    close: pd.DataFrame,
    div_days: pd.DataFrame,
    regime_name: str,
) -> dict:
    """Trade-Liste als taeglichen Portfolio-Pfad abspielen."""
    pf = Portfolio(
        START_KAPITAL,
        make_regime(regime_name),
        cost_bps=COST_BPS,
        asset=AssetClass.AKTIE,
    )
    buys: dict[pd.Timestamp, list[tuple[str, float]]] = {}
    sells: dict[pd.Timestamp, list[tuple[str, float]]] = {}
    for t in trades:
        buys.setdefault(pd.Timestamp(t["entry_date"], tz="UTC"), []).append(
            (t["symbol"], t["entry_px"])
        )
        sells.setdefault(pd.Timestamp(t["exit_date"], tz="UTC"), []).append(
            (t["symbol"], t["exit_px"])
        )

    close_ff = close.ffill()
    equity: list[float] = []
    equity_netto: list[float] = []
    for d in days:
        pf.set_date(d)
        # Erst Exits (macht Slots + Cash frei), dann Entries — gleiche
        # Reihenfolge wie run_variant (a vor b). Zero-Day-Trades (Entry-Fill
        # und Zwangsverkauf am selben Tag, z. B. Delisting am Entry-Tag):
        # der Sell trifft auf qty==0, wird zurueckgestellt und NACH den Buys
        # desselben Tages erneut ausgefuehrt — sonst bleibt die Position bis
        # zum Fensterende eingefroren (B-1, empirisch: MEE 2011-06-01 in
        # allen 8 Varianten, ~1/5 des Portfoliowerts tot).
        deferred: list[tuple[str, float]] = []
        for sym, px in sells.get(d, ()):
            q = pf.qty(sym)
            if q > 0:
                pf.sell(sym, q, px)
            else:
                deferred.append((sym, px))
        for sym, px in buys.get(d, ()):
            wert = pf.value(close_ff.loc[d])
            pf.buy(sym, wert / TOP_K, px)
        for sym, px in deferred:
            q = pf.qty(sym)
            if q > 0:
                pf.sell(sym, q, px)
            else:
                # Genau das Muster von B-1: ein still verworfener Sell
                # verfaelscht jede Folgezahl. Hart abbrechen (F-senior-15).
                raise SystemExit(
                    f"[ERROR] Replay: deferred-Sell {sym} am "
                    f"{d.date()} traf erneut qty==0 (Buy nicht ausgefuehrt, "
                    f"z. B. cash<=0) — Trade verloren. STOPP."
                )
        # div_days ist auf `days` indiziert — kein Existenz-Guard noetig
        # (ein immer-wahrer Guard suggeriert Absicherung, F-senior-16).
        zeile = div_days.loc[d]
        for sym in list(pf.lots):
            dv = float(zeile.get(sym, 0.0))
            if dv > 0:
                pf.book_dividend(sym, dv)
        preise = close_ff.loc[d]
        equity.append(pf.value(preise))
        equity_netto.append(pf.wert_nach_latenter_steuer(preise))

    # Sicherheitsnetz: run_variant bucht eow-Exits selbst; hier darf nur
    # Rundungsstaub uebrig sein.
    rest = list(pf.lots)
    pf.liquidate_all(close_ff.iloc[-1])
    e = pd.Series(equity, index=days)
    en = pd.Series(equity_netto, index=days)
    e.iloc[-1] = pf.value(close_ff.iloc[-1])
    en.iloc[-1] = e.iloc[-1]
    return {
        "equity": e,
        "equity_netto": en,
        "end": float(e.iloc[-1]),
        "tax_paid": float(pf.tax_paid),
        "costs_paid": float(pf.costs_paid),
        "rest_positionen_vor_liquidation": rest,
    }


def main() -> int:
    if not PHASE1_JSON.exists():
        raise SystemExit(f"[ERROR] {PHASE1_JSON} fehlt — erst Phase 1 laufen lassen.")
    phase1 = json.loads(PHASE1_JSON.read_text(encoding="utf-8"))
    if phase1["window"].get("smoke"):
        raise SystemExit("[ERROR] Phase-1-JSON ist ein Smoke-Artefakt — kein Massstab.")
    primaer_pass = [
        v
        for v, verdict in phase1["primary_verdicts"].items()
        if verdict == "PASS_PRIMAER"
    ]
    print(f"[H-090/P2] primaer bestanden: {primaer_pass}")

    if not P8_ARTEFAKT.exists():
        raise SystemExit(
            f"[ERROR] {P8_ARTEFAKT} fehlt — ohne heterogene Varianz keine "
            f"entscheidungsfaehige DSR (E-077)."
        )
    p8 = json.loads(P8_ARTEFAKT.read_text(encoding="utf-8"))
    var_het = float(p8["varianz_heterogen"])
    n_gesamt = TrialCounter().total()
    print(f"[H-090/P2] N kumuliert = {n_gesamt} | V_het = {var_het:.3e}")

    from src.assembled_core.qa.deflated_sharpe import deflated_sharpe

    data = load_campaign(trials=0)
    inp = prepare_inputs(data)
    days = inp["days"]
    close = inp["close"]
    div_days = div_auf_kalender(data.div_panel, days)

    # --- Trades regenerieren + Guard gegen Phase-1-JSON ---
    all_trades: dict[str, list[dict]] = {}
    for v in VARIANTS:
        all_trades[v] = run_variant_from_inputs(v, inp)
        n_soll = phase1["results"][v]["cost_15bp"]["n_trades"]
        if len(all_trades[v]) != n_soll:
            raise SystemExit(
                f"[ERROR] {v}: {len(all_trades[v])} Trades regeneriert, Phase-1-JSON "
                f"hat {n_soll} — prepare_inputs-Refactor hat die Simulation "
                f"veraendert. STOPP."
            )
        # Inhaltlicher Guard (MIN-1): gleiche Anzahl reicht nicht — der PF
        # bindet Preise, Daten und Hälften-Zuordnung mit.
        pf_ist = evaluate(all_trades[v], 15.0)["pf_total"]
        pf_soll = phase1["results"][v]["cost_15bp"]["pf_total"]
        # _sanitize schreibt inf/nan als STRING ins Phase-1-JSON — ein
        # Nicht-float muss hier abbrechen statt TypeError werfen (F-senior-17).
        if (
            not isinstance(pf_ist, float)
            or not isinstance(pf_soll, float)
            or abs(pf_ist - pf_soll) > 1e-9
        ):
            raise SystemExit(
                f"[ERROR] {v}: PF regeneriert {pf_ist} != Phase-1 {pf_soll} — "
                f"inhaltliche Trade-Drift. STOPP."
            )
    print("[H-090/P2] Trade-Regeneration deckungsgleich mit Phase 1 (8/8 Varianten)")

    # --- Pfad-Simulation: ZERO (DSR/PBO) + PRIVAT_DE (Steuer-Vergleich) ---
    pfade: dict[str, dict[str, dict]] = {"ZERO": {}, "PRIVAT_DE": {}}
    for welt in ("ZERO", "PRIVAT_DE"):
        for v in VARIANTS:
            pfade[welt][v] = replay(all_trades[v], days, close, div_days, welt)
            rest = pfade[welt][v]["rest_positionen_vor_liquidation"]
            if rest:
                # Hartes Netz statt totem Feld (M-1): run_variant bucht eow-
                # Exits selbst — bleibt hier etwas uebrig, hat das Replay
                # Trades verloren und alle Folgezahlen sind Artefakte.
                raise SystemExit(
                    f"[ERROR] Replay {welt}/{v}: Positionen am Fensterende "
                    f"nicht geraeumt: {rest} — Trade-Abspielung defekt. STOPP."
                )
            print(
                f"[H-090/P2] {welt:9} {v:6} End {pfade[welt][v]['end']:>12,.0f} "
                f"| Steuer {pfade[welt][v]['tax_paid']:>10,.0f}"
            )

    # --- Benchmarks in PRIVAT_DE: EW-Index (Hauptvergleich) + SPY (FONDS) ---
    regime_p = make_regime("PRIVAT_DE")
    ew = run_strategy(
        data,
        regime_p,
        score=gleichgewicht_score(data.close),
        cost_bps=COST_BPS,
        label="EW-Index PRIVAT_DE",
        **EW_PARAMS,
    )
    spy = run_buy_and_hold(data, make_regime("PRIVAT_DE"))
    ew_end = float(ew.equity.iloc[-1])
    spy_end = float(spy.equity.iloc[-1])
    print(f"[H-090/P2] EW-Benchmark PRIVAT_DE End {ew_end:>12,.0f}")
    print(f"[H-090/P2] SPY (FONDS)  PRIVAT_DE End {spy_end:>12,.0f}")

    # --- DSR + PBO auf der ZERO-Welt ---
    rm = pd.DataFrame(
        {v: pfade["ZERO"][v]["equity_netto"].pct_change() for v in VARIANTS}
    ).dropna()
    print(f"[H-090/P2] Renditematrix {rm.shape[0]} Tage x {rm.shape[1]} Varianten")
    pbo = float(cscv_pbo(rm, n_blocks=8))
    print(f"[H-090/P2] PBO (CSCV, 8 Bloecke): {pbo:.1%}  [E-077: nach unten verzerrt]")

    sharpes = rm.apply(lambda x: x.mean() / x.std() if x.std() > 0 else np.nan)
    var_klon = float(sharpes.var(ddof=1))

    dsr: dict[str, dict] = {}
    for v in primaer_pass:
        dsr[v] = {}
        for label, V in (
            ("heterogen", var_het),
            ("IID-Naeherung", None),
            ("klonfamilie", var_klon),
        ):
            res = deflated_sharpe(rm[v], n_trials=n_gesamt, variance_across_trials=V)
            dsr[v][label] = {
                "entscheidungsfaehig": label == "heterogen",
                "sharpe_observed": float(res.sharpe_observed),
                "sharpe_threshold": float(res.sharpe_threshold),
                "dsr_probability": float(res.deflated_sharpe_probability),
                "passes_5pct": bool(res.passes_5pct),
            }
            marke = {
                "heterogen": " <- Entscheidung",
                "klonfamilie": " <- E-077, nicht entscheidungsfaehig",
            }.get(label, "")
            print(
                f"[H-090/P2] DSR {v:6} {label:<14} p={res.deflated_sharpe_probability:.4f} "
                f"{'BESTANDEN' if res.passes_5pct else 'DURCHGEFALLEN'}{marke}"
            )

    # --- Sekundaer-Verdicts (vorab fixierte Kriterien) ---
    sekundaer: dict[str, dict] = {}
    for v in primaer_pass:
        end_p = pfade["PRIVAT_DE"][v]["end"]
        steuer_ok = end_p >= ew_end
        dsr_ok = dsr[v]["heterogen"]["passes_5pct"]
        pbo_ok = pbo <= 0.5
        sekundaer[v] = {
            "netto_privat_end": end_p,
            "ew_privat_end": ew_end,
            "spy_privat_end": spy_end,
            "steuer_vs_ew_bestanden": steuer_ok,
            "dsr_heterogen_bestanden": dsr_ok,
            "pbo_bestanden": pbo_ok,
            "sekundaer_bestanden": bool(steuer_ok and dsr_ok and pbo_ok),
        }
        print(
            f"[H-090/P2] {v:6} Netto-PRIVAT {end_p:>12,.0f} vs EW {ew_end:>12,.0f} "
            f"{'OK' if steuer_ok else 'FAIL'} | DSR {'OK' if dsr_ok else 'FAIL'} | "
            f"PBO {'OK' if pbo_ok else 'FAIL'} -> "
            f"{'SEKUNDAER-PASS' if sekundaer[v]['sekundaer_bestanden'] else 'SEKUNDAER-FAIL'}"
        )

    feld_pass = any(s["sekundaer_bestanden"] for s in sekundaer.values())
    print(f"\n[H-090/P2] FELD-VERDICT: {'PASS' if feld_pass else 'FAIL'}")

    payload = {
        "h_id": "H-090-Phase2",
        "registered": "research/registry.md Welle 50 (Sekundaerkriterien, vorab fixiert)",
        "trials_kumuliert": n_gesamt,
        "trial_zaehler_unveraendert": True,
        "cost_bps": COST_BPS,
        "primaer_pass": primaer_pass,
        "benchmarks_privat_de": {
            "ew_index": {
                "end": ew_end,
                "konstruktion": f"P11: {EW_PARAMS}, cost_bps=15",
            },
            # run_buy_and_hold reicht cost_bps nicht durch -> Portfolio-Default
            # 10 bp statt 15. Bei ~2 SPY-Trades immateriell; ausgewiesen statt
            # verschwiegen (F-senior-20; engine bleibt unangetastet).
            "spy_fonds": {"end": spy_end, "cost_bps": 10.0},
        },
        # Aus Phase 1 durchgereicht (F-senior-19): das primaere PASS-Label
        # laeuft gegen die V7-Nullmessung; beats_basis traegt die eigentliche
        # Hypothese ("verbessert die BASIS") bis in den Endbefund.
        "beats_basis": phase1.get("beats_basis"),
        "pfade_end": {
            welt: {v: pfade[welt][v]["end"] for v in VARIANTS} for welt in pfade
        },
        "steuern_privat_de": {v: pfade["PRIVAT_DE"][v]["tax_paid"] for v in VARIANTS},
        "pbo": pbo,
        "pbo_caveat": "Fast-Klone -> CSCV nach unten verzerrt (E-077); Rangmass Sharpe",
        "varianz_klonfamilie": var_klon,
        "varianz_heterogen": var_het,
        "dsr": dsr,
        "sekundaer_verdicts": sekundaer,
        "feld_verdict": "PASS" if feld_pass else "FAIL",
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[H-090/P2] Ergebnis -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
