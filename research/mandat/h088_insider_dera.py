"""H-088 — §4.6.1-Insider-Nachtest auf dem echten Small-Cap-Universum.

Ausfuehrung der in **Welle 46** vorab registrierten Bedingungen. Die
Pass/Fail-Kriterien stehen dort und werden hier nicht neu erfunden.

WARUM UEBERHAUPT NOCH EIN ANLAUF
--------------------------------
H-031 (S&P) FAIL. H-053 (breite Preise) FAIL — aber mit einem Vorbehalt im
eigenen Registry-Eintrag, der das Verdikt fuer die eigentliche These entwertet:

    "nur 723 Symbole Form-4 (S&P-Historie, NICHT echtes Small-Cap-Universum
     -> §4.6.1-These 'kleine Firmen' ungetestet)"

Der DERA-Bestand schliesst diese Luecke: 17.134 Emittenten, 2006-2026,
universumsunabhaengig und damit survivorship-frei fuer as_of >= 2006-01-01.
8.065 der abgedeckten Namen waren nie im S&P-Panel; ihr Median-ADV liegt bei
1,83 Mio USD gegen 63,1 Mio bei den S&P-Namen.

WAS HIER ANDERS IST ALS IN H-053
--------------------------------
Die Signalquelle, und drei Verschaerfungen, die Welle 46 verlangt:

1. **Kontrollkorb statt SPY-Sharpe.** H-053 verglich den Sharpe gegen SPY. Das
   misst den Small-Cap-Effekt mit. Hier laeuft zusaetzlich ein Korb OHNE Signal
   aus demselben gegateten Universum, mit derselben monatlichen Korbgroesse.
   Schlaegt das Signal ihn nicht, misst es die Universumsauswahl, nicht Insider.
2. **DSR mit heterogenem V.** H-053 rief `deflated_sharpe(v, n_trials=141)` ohne
   `variance_across_trials` — das ist die IID-Naeherung. Welle 46 verlangt das
   heterogen geschaetzte V aus dem P8-Artefakt und den kumulierten Zaehler
   (E-077).
3. **Sekundaergate.** ADV60 >= 5 Mio zusaetzlich. Es kann nur verschaerfen: ein
   Ergebnis, das nur unter dem Primaergate haelt, gilt als FAIL.

Dazu die **Mindestgroesse** aus Welle 46: unter 300 gegateten Namen oder 2.000
opportunistischen Kaufereignissen wird KEIN Verdikt gefaellt, sondern
"nicht aussagefaehig" berichtet.

TRIALS
------
+2 (genau zwei Varianten: alle opportunistischen Kaeufe, und Cluster >= 2
Insider). Das Sekundaergate zaehlt nicht — es waehlt nichts aus, es kann nur
verschaerfen.

STOPP
-----
Ein FAIL schliesst das Insider-Feld **endgueltig** (Welle 46). Eine
Wiederaufnahme verlangt dann eine neue DATENQUELLE, nicht eine neue Auslegung.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HIER = Path(__file__).resolve().parent
ROOT = HIER.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HIER))

from h031_insider import (  # noqa: E402
    START_CAPITAL,
    classify_opportunistic,
    run_insider,
)
from smallcap_data import FLOOR_ADV, FLOOR_PRICE, load_smallcap  # noqa: E402

from research.mandat2.data_gate import TrialCounter  # noqa: E402

DATA = HIER / "data"
DERA = DATA / "form4_dera"
ZIEL = HIER / "h088_insider_dera.json"
P8 = ROOT / "research" / "mandat2" / "results" / "p8_dsr_heterogen.json"

#: Der DERA-Bestand beginnt 2006Q1 — nicht 2005 wie in H-053.
SIGNAL_START = pd.Timestamp("2006-01-01", tz="UTC")

#: Sekundaergate (Welle 46): kann nur verschaerfen, nie lockern.
FLOOR_ADV_STRENG = 5_000_000.0

#: Mindestgroesse (Welle 46). Darunter kein Verdikt, sondern
#: "nicht aussagefaehig" — sonst laesst sich ein schwaches Ergebnis
#: nachtraeglich als Datenproblem statt als FAIL lesen.
MIN_NAMEN, MIN_EREIGNISSE = 300, 2000

#: Steuerabschlag auf dem ETF-Pfad, wie in H-053.
ETF_TAX = 0.185

#: Seeds fuer den No-Signal-Kontrollkorb.
KONTROLL_SEEDS = 20


def load_purchases_dera() -> pd.DataFrame:
    """Open-Market-Kaeufe aus dem DERA-Bestand, im Schema von `h031_insider`.

    Unterschiede zum Altbestand, alle aus dem Manifest:
    * `transaction_type` fuehrt bereits {P, S, unknown} (Core-Klassifikation),
    * NONDERIV_TRANS enthaelt per Konstruktion nur nicht-derivative Zeilen,
    * `datum_plausibel = False` (0,18 %) DARF nicht eingehen — dort liegt das
      Transaktionsdatum nach dem Meldedatum, das waere ein Lookahead,
    * `available_at` ist bereits UTC und Meldetag+1 (konservativ).
    """
    dateien = sorted(glob.glob(str(DERA / "*.parquet")))
    if not dateien:
        raise SystemExit("[ERROR] DERA-Bestand fehlt — erst pull_form4_dera.py laufen.")
    spalten = [
        "symbol",
        "transaction_type",
        "datum_plausibel",
        "available_at",
        "transaction_date",
        "RPTOWNERCIK",
        "NONDERIV_TRANS_SK",
    ]
    df = pd.concat(
        [pd.read_parquet(f, columns=spalten) for f in dateien], ignore_index=True
    )
    df = df[(df["transaction_type"] == "P") & df["datum_plausibel"]]
    df = df.dropna(subset=["symbol"])
    df = df.rename(columns={"RPTOWNERCIK": "reporting_owner_cik"})
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], utc=True)
    df["available_at"] = pd.to_datetime(df["available_at"], utc=True)
    return df


def heterogene_varianz() -> float:
    """V aus dem P8-Artefakt — NICHT aus der eigenen Familie (E-077)."""
    if not P8.exists():
        raise SystemExit(f"[ERROR] {P8.name} fehlt — DSR nicht entscheidungsfaehig.")
    return float(json.loads(P8.read_text(encoding="utf-8"))["varianz_heterogen"])


def main() -> int:
    from src.assembled_core.qa.deflated_sharpe import deflated_sharpe

    close, adv = load_smallcap()
    idx = close.index
    monatsenden = list(idx.to_series().groupby(idx.to_period("M")).max())
    leer_div = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    cols = set(close.columns)

    roh = load_purchases_dera()
    opp = classify_opportunistic(roh)
    print(
        f"[SIG] {len(roh):,} P-Kaeufe -> {len(opp):,} opportunistisch "
        f"({100 * len(opp) / max(len(roh), 1):.1f} %), "
        f"{opp['symbol'].nunique():,} Symbole",
        flush=True,
    )

    def handelbar(me: pd.Timestamp, floor_adv: float) -> set[str]:
        if me not in adv.index:
            return set()
        pr, av = close.loc[me], adv.loc[me]
        ok = av[av >= floor_adv].index
        return {
            s
            for s in ok
            if s in cols
            and np.isfinite(pr.get(s, np.nan))
            and pr.get(s, 0.0) >= FLOOR_PRICE
        }

    gates = {
        "primaer": {
            me: handelbar(me, FLOOR_ADV) for me in monatsenden if me >= SIGNAL_START
        },
        "sekundaer": {
            me: handelbar(me, FLOOR_ADV_STRENG)
            for me in monatsenden
            if me >= SIGNAL_START
        },
    }

    def signal(df: pd.DataFrame, min_insider: int, trad: dict) -> dict:
        sig = {}
        for me in monatsenden:
            if me < SIGNAL_START:
                continue
            frisch = df[
                (df["available_at"] <= me)
                & (df["available_at"] > me - pd.DateOffset(months=3))
            ]
            if not len(frisch):
                continue
            # Cluster ueber EIGENSTAENDIGE Insider, nicht ueber Zeilen: der
            # DERA-Bestand dupliziert Transaktionen je Meldepflichtigem (E-124).
            je_titel = frisch.groupby("symbol")["reporting_owner_cik"].nunique()
            kand = set(je_titel[je_titel >= min_insider].index)
            s = kand & trad.get(me, set())
            if s:
                sig[me] = s
        return sig

    # --- Mindestgroesse VOR dem Rechnen pruefen (Welle 46)
    namen_gegated = (
        len(set().union(*gates["primaer"].values())) if gates["primaer"] else 0
    )
    ereignisse = int(
        len(opp[opp["symbol"].isin(set().union(*gates["primaer"].values()))])
        if gates["primaer"]
        else 0
    )
    print(
        f"[GATE] {namen_gegated:,} handelbare Namen, {ereignisse:,} opportunistische "
        f"Kaufereignisse darauf (Mindestgroesse {MIN_NAMEN}/{MIN_EREIGNISSE})",
        flush=True,
    )
    aussagefaehig = namen_gegated >= MIN_NAMEN and ereignisse >= MIN_EREIGNISSE

    print(
        f"Trials kumuliert: {TrialCounter().increment(2, label='H-088 Insider DERA')}\n",
        flush=True,
    )

    ergebnis: dict = {
        "hypothese": "H-088",
        "registriert": "Welle 46, VOR dem Lauf",
        "quelle": "SEC DERA Form 4, 2006Q1-2026Q1",
        "signal_start": str(SIGNAL_START.date()),
        "gate_primaer": {"preis": FLOOR_PRICE, "adv60": FLOOR_ADV},
        "gate_sekundaer": {"preis": FLOOR_PRICE, "adv60": FLOOR_ADV_STRENG},
        "namen_gegated": namen_gegated,
        "opportunistische_ereignisse": ereignisse,
        "aussagefaehig": aussagefaehig,
        "laeufe": {},
    }
    if not aussagefaehig:
        ergebnis["verdikt"] = {
            "PASS": None,
            "begruendung": (
                "NICHT AUSSAGEFAEHIG — Mindestgroesse aus Welle 46 unterschritten. "
                "Kein Verdikt; das Insider-Feld bleibt offen, weil kein Test "
                "stattgefunden hat."
            ),
        }
        ZIEL.write_text(json.dumps(ergebnis, indent=2, default=str), encoding="utf-8")
        print("[VERDIKT] NICHT AUSSAGEFAEHIG", flush=True)
        return 0

    varianten = {"all_opp": 1, "cluster2": 2}
    rets: dict[str, pd.Series] = {}
    for gname, trad in gates.items():
        for vname, mini in varianten.items():
            sig = signal(opp, mini, trad)
            med = int(np.median([len(v) for v in sig.values()])) if sig else 0
            label = f"{gname}/{vname}"
            res, _e, ret = run_insider(close, leer_div, sig, monatsenden, label=label)
            ergebnis["laeufe"][label] = {
                "signalmonate": len(sig),
                "median_namen_je_monat": med,
                "endwert": round(res["final_value"]),
                "cagr_netto": round(res["cagr_net"], 4),
                "sharpe_netto": round(res["sharpe_net"], 3),
                "maxdd_netto": round(res["maxdd_net"], 4),
            }
            if gname == "primaer":
                rets[vname] = ret
            print(
                f"[RUN] {label:<20} Endwert {res['final_value']:>12,.0f} | "
                f"Sharpe {res['sharpe_net']:>6.3f} | MaxDD {res['maxdd_net'] * 100:>6.1f}% "
                f"| {len(sig)} Signalmonate, Median {med} Namen",
                flush=True,
            )

    # --- No-Signal-Kontrollkorb aus DEMSELBEN gegateten Universum
    groessen = {me: len(s) for me, s in signal(opp, 1, gates["primaer"]).items()}
    kontroll_sharpes = []
    for seed in range(KONTROLL_SEEDS):
        rng = np.random.default_rng(seed)
        sig_k = {}
        for me, k in groessen.items():
            pool = sorted(gates["primaer"].get(me, set()))
            if pool and k:
                pick = rng.choice(len(pool), size=min(k, len(pool)), replace=False)
                sig_k[me] = {pool[i] for i in pick}
        _r, _e, ret_k = run_insider(
            close, leer_div, sig_k, monatsenden, label=f"kontrolle{seed}"
        )
        kontroll_sharpes.append(float(ret_k.mean() / ret_k.std() * np.sqrt(252)))
    kontroll_median = float(np.median(kontroll_sharpes))
    print(
        f"\n[KONTROLLE] {KONTROLL_SEEDS} Koerbe ohne Signal aus demselben Gate: "
        f"Sharpe-Median {kontroll_median:.3f} "
        f"({min(kontroll_sharpes):.3f} .. {max(kontroll_sharpes):.3f})",
        flush=True,
    )

    # --- Benchmark
    spy = close["SPY"].dropna()
    spy = spy[spy.index >= SIGNAL_START]
    spy_r = spy.pct_change().dropna()
    spy_sharpe = float(spy_r.mean() / spy_r.std() * np.sqrt(252))
    spy_maxdd = float((spy / spy.cummax() - 1).min())
    etf_netto = START_CAPITAL + START_CAPITAL * (spy.iloc[-1] / spy.iloc[0] - 1) * (
        1 - ETF_TAX
    )

    bester = max(rets, key=lambda k: ergebnis["laeufe"][f"primaer/{k}"]["endwert"])
    b = ergebnis["laeufe"][f"primaer/{bester}"]
    n_kum = TrialCounter().total()
    dsr = deflated_sharpe(
        rets[bester], n_trials=n_kum, variance_across_trials=heterogene_varianz()
    )

    # Kriterium 5: kippt das Ergebnis zwischen Primaer- und Sekundaergate?
    kippt = any(
        (ergebnis["laeufe"][f"primaer/{v}"]["endwert"] > etf_netto)
        != (ergebnis["laeufe"][f"sekundaer/{v}"]["endwert"] > etf_netto)
        for v in varianten
    )

    k1 = b["endwert"] > etf_netto
    k2 = b["sharpe_netto"] > kontroll_median
    k3 = bool(dsr.passes_5pct)
    k4 = b["maxdd_netto"] >= spy_maxdd
    k5 = not kippt
    ergebnis["benchmark"] = {
        "etf_netto_pfad": round(float(etf_netto)),
        "spy_sharpe": round(spy_sharpe, 3),
        "spy_maxdd": round(spy_maxdd, 4),
        "kontrollkorb_sharpe_median": round(kontroll_median, 3),
    }
    ergebnis["verdikt"] = {
        "bester_lauf": bester,
        "k1_ueber_etf_pfad": bool(k1),
        "k2_sharpe_ueber_kontrollkorb": bool(k2),
        "k3_dsr_heterogen": {
            "p": round(float(dsr.deflated_sharpe_probability), 4),
            "n_trials": n_kum,
            "pass": k3,
        },
        "k4_maxdd_nicht_schlechter_als_spy": bool(k4),
        "k5_kein_kippen_zwischen_gates": bool(k5),
        "PASS": bool(k1 and k2 and k3 and k4 and k5),
    }

    # Artefakt als LETZTE Anweisung (E-116).
    ZIEL.write_text(json.dumps(ergebnis, indent=2, default=str), encoding="utf-8")
    v = ergebnis["verdikt"]
    print("\n" + "=" * 72)
    print(
        f"  k1 Endwert > ETF-Pfad          {b['endwert']:>12,} vs {etf_netto:>12,.0f}  {'OK' if k1 else 'FAIL'}"
    )
    print(
        f"  k2 Sharpe > Kontrollkorb       {b['sharpe_netto']:>12.3f} vs {kontroll_median:>12.3f}  {'OK' if k2 else 'FAIL'}"
    )
    print(
        f"  k3 DSR (heterogen, N={n_kum})  {v['k3_dsr_heterogen']['p']:>12.4f}                 {'OK' if k3 else 'FAIL'}"
    )
    print(
        f"  k4 MaxDD >= SPY                {b['maxdd_netto']:>12.3f} vs {spy_maxdd:>12.3f}  {'OK' if k4 else 'FAIL'}"
    )
    print(
        f"  k5 kein Kippen zwischen Gates                                {'OK' if k5 else 'FAIL'}"
    )
    print("=" * 72)
    print(
        "VERDIKT: PASS" if v["PASS"] else "VERDIKT: FAIL — Insider-Feld endgueltig zu"
    )
    print("=" * 72, flush=True)
    print(f"\n-> {ZIEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
