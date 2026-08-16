# -*- coding: utf-8 -*-
"""H-090 — Rettet eine Kurs-Exit-Familie das Top-K-Momentum-Portfolio?

Registriert VOR dem ersten Lauf: research/registry.md, Welle 50 (2026-08-16).
Quelle: extern zugelieferte Strategiespezifikation (User-Dokument). Getestet
wird AUSSCHLIESSLICH die dort beschriebene Exit-Familie auf der bereits
mehrfach gefallenen Basisstrategie (H-011/H-012/H-049) — auf dem PIT-Panel,
das die Quelle selbst als fehlenden Schritt 7 ihres Pruefprotokolls benennt.

Basisstrategie (alle Varianten identisch):
  - Auswahl am Monatsultimo: Top-5 nach mom_12_1 = close[t-21]/close[t-252]-1,
    nur mom > 0, nur PIT-Mitglieder des Monats, Mindesthistorie 260 Handelstage
  - Marktfilter: keine NEUEN Positionen, wenn SPY-Close <= SMA200(SPY) am Ultimo
  - Stop-Loss fest -15 % (Schlusskursbasis), max. Haltedauer 120 Handelstage
  - Momentum-Exit: am Ultimo nicht mehr in Top-5 -> Exit-Signal

Exit-Varianten (vorregistrierte 8er-Familie, je +Basisregeln):
  BASIS  nur Basisregeln
  V1     festes Ziel +20 % (close-approximiert)
  V2     Close < min(Close[t-5..t-1]) ab +8 % Gewinn
  V3     RSI14(Wilder) > 70 ab +5 % Gewinn
  V4     RSI14(Wilder) > 75
  V5     festes Ziel +12 % (close-approximiert)
  V6     Trailing-Stop -12 % vom hoechsten Schluss seit Einstieg
  V7     Zeit-Exit nach genau 20 Handelstagen (Referenz/Nullmessung;
         ersetzt die 120-Tage-Grenze)

Ausfuehrungskonvention (Praezisierung in der Registry, VOR dem Lauf):
  Signale am Close von Tag t, Ausfuehrung einheitlich am Close von t+1
  (Ein- UND Ausstiege; die Quelle verlangt t+1 open — Panel ist close-only).
  Delisting: Zwangsverkauf zum letzten verfuegbaren Kurs (Kampagnenkonvention).
  Exit-Prioritaet je Tag: Stop -> Ziel -> Zeit -> Indikator -> Momentum.

Bugfix nach Stage-1-Review (2026-08-16, dokumentiert im Registry-Nachtrag):
  Der erste Volllauf war kontaminiert — der Delisting-Fallback nutzte
  last_valid ueber das GESAMTE Fenster und verkaufte bei temporaeren
  Kursluecken zu Zukunftspreisen (Median +501 Handelstage); zusaetzlich
  enthielt der Panel-Kalender Phantomtage (US-Feiertage mit Kursen weniger
  Symbole), an denen ALLE offenen Positionen faelschlich zwangsverkauft
  wurden. Fixes: (1) Kalender = SPY-Handelstage (Phantomtage raus),
  (2) Delisting nur wenn i > last_valid[col] (Kurs garantiert Vergangenheit),
  (3) temporaere Luecken: Fills werden auf den naechsten Tag mit Kurs
  verschoben, Positionen gehalten, Signal-Pruefung uebersprungen,
  (4) am Fensterende offene Positionen werden als reason="eow" zum letzten
  Kurs gebucht (vorher stillschweigend verworfen).

Kosten: 15 bp je Seite (Haertetest 30/50 bp auf denselben Trades).
Fenster: NUR Suchfenster 1995-2016 via load_campaign(); Holdout bleibt zu.
Haelften (disjunkt, E-078): Entry <= 2005-12-31 vs. Entry >= 2006-01-01.

Pass/Fail primaer (je Variante, vorab): PF > 1,2 in BEIDEN Haelften UND
PF > V7-Referenz in BEIDEN Haelften. Ergebnis-JSON wird als LETZTE Anweisung
geschrieben (E-116); der Befund wird daraus GENERIERT (E-085).

Aufruf:
    python research/mandat2/h090_momentum_exits.py            # voller Lauf, bucht 8 Trials
    python research/mandat2/h090_momentum_exits.py --smoke    # 2 Jahre, keine Trial-Buchung
    python research/mandat2/h090_momentum_exits.py --no-book  # Re-Run nach Bugfix,
                                                              # KEINE erneute Buchung
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from campaign_data import load_campaign  # noqa: E402

OUT_PATH = HERE / "results" / "h090_momentum_exits.json"

TOP_K = 5
MIN_HISTORY = 260
STOP_PCT = -0.15
MAX_HOLD = 120
COST_BPS_MAIN = 15.0
COST_BPS_STRESS = (30.0, 50.0)
HALF_SPLIT = pd.Timestamp("2005-12-31", tz="UTC")

VARIANTS = ["BASIS", "V1", "V2", "V3", "V4", "V5", "V6", "V7"]


def wilder_rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """RSI nach Wilder (EWM alpha=1/n), panelweit. Muster aus K10."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    ma_up = up.ewm(alpha=1.0 / n, adjust=False).mean()
    ma_dn = dn.ewm(alpha=1.0 / n, adjust=False).mean()
    rs = ma_up / ma_dn
    return 100.0 - 100.0 / (1.0 + rs)


def run_variant(
    variant: str,
    days: pd.DatetimeIndex,
    close_np: np.ndarray,
    mom_np: np.ndarray,
    rsi_np: np.ndarray,
    llp5_np: np.ndarray,
    gate_np: np.ndarray,
    ultimo_np: np.ndarray,
    members_at: dict,
    first_valid: np.ndarray,
    last_valid: np.ndarray,
    symbols: list,
) -> list[dict]:
    """Simuliert eine Exit-Variante. Gibt die Trade-Liste (BRUTTO-Kurse) zurueck.

    Kosten werden bewusst NICHT hier verrechnet: dieselben Trades werden
    anschliessend mit 15/30/50 bp bewertet (Haertetest auf identischer Basis).

    Restannahme (F-senior-5, Konvention — kein Preis-Look-ahead): die
    UNTERSCHEIDUNG temporaere Luecke vs. Delisting nutzt last_valid und damit
    Fensterwissen. In Echtzeit ist am ersten fehlenden Tag nicht bekannt, ob
    der Name tot ist oder pausiert. Die verwendeten PREISE liegen in beiden
    Zweigen garantiert in der Vergangenheit.
    """
    n_days = len(days)
    positions: dict[int, dict] = {}  # col -> {entry_px, entry_i, peak, reason}
    pending_exit: dict[int, str] = {}  # col -> reason (Ausfuehrung am naechsten Tag)
    pending_entry: list[int] = []
    trades: list[dict] = []

    max_hold = 20 if variant == "V7" else MAX_HOLD

    def book(col: int, pos: dict, exit_i: int, px: float, reason: str) -> None:
        trades.append(
            {
                "symbol": symbols[col],
                "entry_date": str(days[pos["entry_i"]].date()),
                "exit_date": str(days[exit_i].date()),
                "entry_px": float(pos["entry_px"]),
                "exit_px": float(px),
                "days_held": int(exit_i - pos["entry_i"]),
                "reason": reason,
            }
        )

    for i in range(n_days):
        row = close_np[i]

        # --- a) signalisierte EXITS zum heutigen Close ausfuehren ---
        # Bei temporaerer Kursluecke bleibt der pending-Exit stehen und wird
        # am naechsten Tag MIT Kurs gefuellt (kein Zukunftspreis-Fallback).
        for col, reason in list(pending_exit.items()):
            px = row[col]
            if np.isfinite(px):
                pos = positions.pop(col, None)
                del pending_exit[col]
                if pos is not None:
                    book(col, pos, i, px, reason)

        # --- b) signalisierte ENTRIES zum heutigen Close ---
        # Bei temporaerer Kursluecke wird der Fill auf den naechsten Tag mit
        # Kurs verschoben; ist das Symbol delisted (i > last_valid), entfaellt er.
        still_pending: list[int] = []
        for col in pending_entry:
            if col in positions or len(positions) >= TOP_K:
                continue
            px = row[col]
            if not np.isfinite(px):
                if i <= last_valid[col]:
                    still_pending.append(col)
                continue
            positions[col] = {"entry_px": float(px), "entry_i": i, "peak": float(px)}
        pending_entry = still_pending

        # --- c) Delisting-Check: NUR wenn der letzte Kurs des Symbols
        # VOR heute liegt (i > last_valid) -> Zwangsverkauf zum letzten Kurs
        # (garantiert Vergangenheit). Temporaere Luecken: Position halten.
        for col, pos in list(positions.items()):
            if i > last_valid[col]:
                positions.pop(col)
                pending_exit.pop(col, None)
                # exit_i = last_valid: Datum, Preis und days_held konsistent
                # zum tatsaechlich letzten Handelstag (F-senior-6).
                lv = int(last_valid[col])
                book(col, pos, lv, close_np[lv, col], "delisting")

        # --- d) Exit-SIGNALE fuer heute (Ausfuehrung morgen), Prioritaet 1.4 ---
        for col, pos in positions.items():
            if col in pending_exit:
                continue  # Exit bereits signalisiert, wartet auf Fill
            px = row[col]
            if not np.isfinite(px):
                continue  # temporaere Luecke: heute kein Signal moeglich
            pos["peak"] = max(pos["peak"], px)
            gain = px / pos["entry_px"] - 1.0
            held = i - pos["entry_i"]
            reason = None
            if gain <= STOP_PCT:
                reason = "stop"
            elif variant == "V1" and gain >= 0.20:
                reason = "target"
            elif variant == "V5" and gain >= 0.12:
                reason = "target"
            elif held >= max_hold:
                reason = "time"
            elif variant == "V2" and gain >= 0.08 and px < llp5_np[i, col]:
                reason = "llp5"
            elif variant == "V3" and gain >= 0.05 and rsi_np[i, col] > 70.0:
                reason = "rsi"
            elif variant == "V4" and rsi_np[i, col] > 75.0:
                reason = "rsi"
            elif variant == "V6" and px <= pos["peak"] * (1.0 - 0.12):
                reason = "trail"
            if reason is not None:
                pending_exit[col] = reason

        # --- e) Monatsultimo: Momentum-Exit + neue Auswahl ---
        if ultimo_np[i]:
            # Stale Entries verwerfen: ein am VORIGEN Ultimo signalisierter,
            # wegen Kursluecke nie gefuellter Entry darf nicht Monate spaeter
            # ausgefuehrt werden — die Auswahl wird je Ultimo frisch gebaut
            # (F-senior-3/4; verhindert zugleich Queue-Ueberbuchung > TOP_K).
            pending_entry = []
            mem = members_at.get(i)
            top: list[int] = []
            if mem is not None:
                mrow = mom_np[i]
                cands = [
                    c
                    for c in mem
                    if np.isfinite(mrow[c])
                    and mrow[c] > 0.0
                    and first_valid[c] <= i - MIN_HISTORY
                    and np.isfinite(row[c])
                ]
                cands.sort(key=lambda c: mrow[c], reverse=True)
                top = cands[:TOP_K]

            top_set = set(top)
            for col in list(positions.keys()):
                if col not in top_set and col not in pending_exit:
                    pending_exit[col] = "momentum"

            if gate_np[i]:
                held_after = {c for c in positions if c not in pending_exit}
                free = TOP_K - len(held_after)
                for col in top:
                    if free <= 0:
                        break
                    if col in positions or col in pending_exit:
                        continue
                    pending_entry.append(col)
                    free -= 1

    # --- Fensterende: verbleibende Positionen (inkl. offener pending-Exits)
    # zum letzten verfuegbaren Kurs als "eow" buchen statt still verwerfen.
    last_i = n_days - 1
    for col, pos in positions.items():
        exit_i = min(last_i, int(last_valid[col]))
        book(col, pos, exit_i, close_np[exit_i, col], "eow")

    return trades


def evaluate(trades: list[dict], cost_bps: float) -> dict:
    """Trade-Statistik netto Kosten; Haelften disjunkt nach Entry-Datum."""
    c = cost_bps / 10000.0

    def net_ret(t: dict) -> float:
        return (t["exit_px"] * (1 - c)) / (t["entry_px"] * (1 + c)) - 1.0

    def pf(rets: list[float]) -> float | None:
        gains = sum(r for r in rets if r > 0)
        losses = -sum(r for r in rets if r < 0)
        if losses == 0:
            return None if gains == 0 else float("inf")
        return gains / losses

    rets = [net_ret(t) for t in trades]
    h1 = [
        net_ret(t)
        for t in trades
        if pd.Timestamp(t["entry_date"], tz="UTC") <= HALF_SPLIT
    ]
    h2 = [
        net_ret(t)
        for t in trades
        if pd.Timestamp(t["entry_date"], tz="UTC") > HALF_SPLIT
    ]

    return {
        "n_trades": len(trades),
        "n_h1": len(h1),
        "n_h2": len(h2),
        "hit_rate": round(float(np.mean([r > 0 for r in rets])), 4) if rets else None,
        "avg_ret": round(float(np.mean(rets)), 5) if rets else None,
        "avg_days": round(float(np.mean([t["days_held"] for t in trades])), 1)
        if trades
        else None,
        # PFs UNGERUNDET (Verdict-Vergleich laeuft auf vollen Werten;
        # Rundung nur in der Konsolenausgabe)
        "pf_total": pf(rets),
        "pf_h1": pf(h1) if h1 else None,
        "pf_h2": pf(h2) if h2 else None,
        "exit_reasons": {
            r: sum(1 for t in trades if t["reason"] == r)
            for r in sorted({t["reason"] for t in trades})
        },
    }


def prepare_inputs(data, *, smoke: bool = False) -> dict:
    """Simulation-Inputs aus den Kampagnendaten — EINE Wahrheit fuer Phase 1
    (dieses Skript) und Phase 2 (h090_phase2_sekundaer.py, Rule 50)."""
    close = data.close
    if "SPY" not in close.columns:
        raise SystemExit("SPY fehlt im Panel — Marktfilter nicht berechenbar")

    # Kalender = SPY-Handelstage. Der rohe Panel-Kalender enthaelt Phantomtage
    # (US-Feiertage, an denen nur wenige Symbole Kurse haben) — die erzeugten
    # im ersten Lauf faelschliche Massen-"Delistings" (Stage-1-BLOCKER).
    n_raw = len(close)
    close = close[close["SPY"].notna()]
    n_phantom = n_raw - len(close)

    # Panelweite Vorprodukte — VOR dem Smoke-Slice, damit der Burn-in
    # (252 T mom, 200 T SMA) das Smoke-Fenster nicht auffrisst.
    mom = close.shift(21) / close.shift(252) - 1.0  # exakt engine.momentum_score
    rsi = wilder_rsi(close)
    llp5 = close.shift(1).rolling(5).min()  # 5-Tage-SCHLUSS-Tief ohne Tag t
    spy = close["SPY"]
    gate_s = spy > spy.rolling(200).mean()

    if smoke:
        sl = slice("2010-01-01", "2011-12-31")
        close = close.loc[sl]
        mom, rsi, llp5, gate_s = mom.loc[sl], rsi.loc[sl], llp5.loc[sl], gate_s.loc[sl]

    days = close.index
    symbols = [s for s in close.columns]
    col_ix = {s: j for j, s in enumerate(symbols)}
    close_np = close.to_numpy(dtype="float64")

    # Monatsultimos = letzter Handelstag je Monat im Panel-Kalender
    is_ultimo = pd.Series(True, index=days).groupby(days.to_period("M")).tail(1)
    ultimo_np = days.isin(is_ultimo.index)

    # PIT-Mitgliedschaft je Ultimo-Tag -> Spaltenindizes (SORTIERT, E-051)
    members_at: dict[int, list[int]] = {}
    mem_series = data.membership
    mem_dates = list(mem_series.index)
    for i in np.flatnonzero(ultimo_np):
        d = days[i]
        eligible = [md for md in mem_dates if md <= d]
        if not eligible:
            continue
        mem = mem_series[eligible[-1]]
        cols = sorted(col_ix[s] for s in sorted(mem) if s in col_ix)
        members_at[i] = cols

    finite = np.isfinite(close_np)
    first_valid = np.where(finite.any(axis=0), finite.argmax(axis=0), len(days))
    last_valid = len(days) - 1 - finite[::-1].argmax(axis=0)

    return {
        "close": close,
        "days": days,
        "close_np": close_np,
        "mom_np": mom.to_numpy(dtype="float64"),
        "rsi_np": rsi.to_numpy(dtype="float64"),
        "llp5_np": llp5.to_numpy(dtype="float64"),
        "gate_np": gate_s.to_numpy(),
        "ultimo_np": ultimo_np,
        "members_at": members_at,
        "first_valid": first_valid,
        "last_valid": last_valid,
        "symbols": symbols,
        "n_phantom": n_phantom,
    }


def run_variant_from_inputs(variant: str, inp: dict) -> list[dict]:
    """run_variant mit prepare_inputs-Ausgabe — gemeinsamer Einstieg fuer
    Phase 1 und Phase 2."""
    return run_variant(
        variant,
        inp["days"],
        inp["close_np"],
        inp["mom_np"],
        inp["rsi_np"],
        inp["llp5_np"],
        inp["gate_np"],
        inp["ultimo_np"],
        inp["members_at"],
        inp["first_valid"],
        inp["last_valid"],
        inp["symbols"],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 Jahre, keine Trial-Buchung")
    ap.add_argument(
        "--no-book",
        action="store_true",
        help="voller Lauf OHNE Trial-Buchung (nur fuer Re-Runs nach Bugfix "
        "eines bereits gebuchten Designs; Begruendung landet im JSON)",
    )
    args = ap.parse_args()

    trials = 0 if (args.smoke or args.no_book) else 8
    label = (
        None
        if args.smoke or args.no_book
        else "H-090 Exit-Familie BASIS+V1-V7 (Welle 50)"
    )
    data = load_campaign(trials=trials, trial_label=label)

    inp = prepare_inputs(data, smoke=args.smoke)
    n_phantom = inp["n_phantom"]

    results: dict[str, dict] = {}
    all_trades: dict[str, list] = {}
    for v in VARIANTS:
        trades = run_variant_from_inputs(v, inp)
        all_trades[v] = trades
        results[v] = {
            f"cost_{int(b)}bp": evaluate(trades, b)
            for b in (COST_BPS_MAIN, *COST_BPS_STRESS)
        }
        r15 = results[v]["cost_15bp"]

        def _r(x: float | None) -> float | None:
            return round(x, 3) if isinstance(x, float) and np.isfinite(x) else x

        print(
            f"[H-090] {v:6} n={r15['n_trades']:5} hit={r15['hit_rate']} "
            f"avg={r15['avg_ret']} PF={_r(r15['pf_total'])} "
            f"(H1={_r(r15['pf_h1'])} H2={_r(r15['pf_h2'])}) days={r15['avg_days']}"
        )

    # Primaeres Kriterium (vorab fixiert): PF > 1,2 in BEIDEN Haelften UND
    # PF > V7-Referenz in BEIDEN Haelften — bei 15 bp.
    ref = results["V7"]["cost_15bp"]
    verdicts = {}
    for v in VARIANTS:
        if v == "V7":
            continue
        r = results[v]["cost_15bp"]
        ok = (
            r["pf_h1"] is not None
            and r["pf_h2"] is not None
            and ref["pf_h1"] is not None
            and ref["pf_h2"] is not None
            and r["pf_h1"] > 1.2
            and r["pf_h2"] > 1.2
            and r["pf_h1"] > ref["pf_h1"]
            and r["pf_h2"] > ref["pf_h2"]
        )
        verdicts[v] = "PASS_PRIMAER" if ok else "FAIL_PRIMAER"

    # ZWEITE Referenz (F-senior-1): das PASS-Label oben laeuft gegen die
    # vorregistrierte V7-NULLMESSUNG. Die eigentliche Hypothese („verbessert
    # die BASIS") braucht den Vergleich gegen die unveraenderte Basisstrategie
    # — sonst kann der Befund beim Zusammenschreiben das Vorzeichen wechseln,
    # ohne dass eine Zahl falsch ist. Label bleibt unveraendert (vorregistriert);
    # beats_basis wird als eigenes Flag ausgewiesen.
    basis = results["BASIS"]["cost_15bp"]
    beats_basis = {}
    for v in VARIANTS:
        if v == "BASIS":
            continue
        r = results[v]["cost_15bp"]
        beats_basis[v] = bool(
            r["pf_h1"] is not None
            and r["pf_h2"] is not None
            and basis["pf_h1"] is not None
            and basis["pf_h2"] is not None
            and r["pf_h1"] > basis["pf_h1"]
            and r["pf_h2"] > basis["pf_h2"]
        )

    payload = {
        "h_id": "H-090",
        "registered": "research/registry.md Welle 50, 2026-08-16, VOR dem Lauf",
        "run_note": (
            "Re-Run nach Stage-1-BLOCKER-Fix (Zukunftspreis-Delisting + "
            "Phantomtage); Trials bereits im Erstlauf gebucht, --no-book"
            if args.no_book
            else None
        ),
        "n_phantom_days_removed": int(n_phantom),
        "window": {
            "fenster": data.fenster,
            "von": str(data.von.date()),
            "bis": str(data.bis.date()),
            "half_split": str(HALF_SPLIT.date()),
            "smoke": bool(args.smoke),
        },
        "params": {
            "top_k": TOP_K,
            "min_history": MIN_HISTORY,
            "stop_pct": STOP_PCT,
            "max_hold": MAX_HOLD,
            "cost_bps_main": COST_BPS_MAIN,
            "cost_bps_stress": list(COST_BPS_STRESS),
            "execution": "signal at close t, fill at close t+1 (entries AND exits)",
        },
        "results": results,
        "primary_verdicts": verdicts,
        "v7_reference": {"pf_h1": ref["pf_h1"], "pf_h2": ref["pf_h2"]},
        "basis_reference": {"pf_h1": basis["pf_h1"], "pf_h2": basis["pf_h2"]},
        "beats_basis": beats_basis,
    }

    def _sanitize(obj):
        """inf/-inf/nan -> JSON-konforme Strings (json.dumps schriebe sonst
        non-standard Infinity-Literale)."""
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, float) and not np.isfinite(obj):
            return repr(obj)
        return obj

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(_sanitize(payload), indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"[H-090] Ergebnis -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
