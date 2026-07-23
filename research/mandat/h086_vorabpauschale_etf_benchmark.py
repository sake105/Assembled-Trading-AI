"""H-086 — Vorabpauschale (§18 InvStG) im ETF-Benchmark-Pfad (GESAMTBEWERTUNG P6/W2).

Schliesst die dokumentierte Benchmark-Luecke (registry.md „Benchmark-Notiz: Vorab-
pauschale im ETF-Netto-Pfad"): der bisherige ETF-Pfad rechnet
    final = START + Gewinn * (1 - 0.185)
d. h. NUR Endbesteuerung 18,5 % (= 26,375 % * 0,7 Teilfreistellung, gerundet),
KEINE Vorabpauschale. Dieses Script rechnet denselben SPY-Buy&Hold-Pfad ueber
dasselbe Fenster wie H-032b (window-matched an H-032 low-div) NEU mit:

  1) Vorabpauschale ab Steuerjahr 2018 (§18 InvStG):
     - Basisertrag = Ruecknahmepreis Jahresanfang * Basiszins * 0,7
     - Vorabpauschale = min(Basisertrag, Wertzuwachs des Jahres); negativ/0 -> 0
     - im Verkaufsjahr (= letztes Fensterjahr) KEINE Vorabpauschale fuer das
       laufende Jahr (vereinfachend; gesetzlich korrekt: VP fliesst erst am
       ersten Werktag des Folgejahres zu — beim Verkauf im Jahr N gibt es keine
       VP fuer N).
     - Aktienfonds-Teilfreistellung 30 % gilt AUCH auf die Vorabpauschale
       (steuerpflichtig = VP * 0,7), Steuersatz 26,375 % wie im bestehenden Pfad.
     - Angesetzte Vorabpauschalen mindern den steuerpflichtigen
       Veraeusserungsgewinn bei End-Liquidation (§19 InvStG).
  2) Teilfreistellungs-Check (zweite potenzielle Luecke): Teilfreistellung IST
     im bestehenden Pfad bereits modelliert — als gerundete 18,5 % statt exakt
     26,375 % * (1 - 0,30) = 18,4625 %. Der Alt-Pfad besteuert den ETF also um
     0,0375 Pp ZU HOCH (mini-Fehler GEGEN den ETF). Wird hier exakt gerechnet
     und der Effekt separat ausgewiesen (keine Rosinenpickerei: beide
     Korrekturen — VP senkt ETF, exakte TFS hebt ETF — gehen ins Netto ein).

Basiszins-Historie (BMF-Schreiben, jaehrlich; negativ -> 0 keine VP):
  2018: 0,87 %   (BMF v. 04.01.2018)
  2019: 0,52 %   (BMF v. 09.01.2019)
  2020: 0,07 %   (BMF v. 29.01.2020)
  2021: 0,00 %   (negativ -0,45 % -> keine VP)
  2022: 0,00 %   (negativ -0,05 % -> keine VP)
  2023: 2,55 %   (BMF v. 04.01.2023)
  2024: 2,29 %   (BMF v. 05.01.2024)
  2025: 2,53 %   (BMF v. 10.01.2025)
  2026: 2,53 %   ANNAHME: Carry-Forward des 2025er Satzes (2026er BMF-Satz hier
                 nicht hinterlegt); im H-032-Fenster irrelevant, da 2026 =
                 Verkaufsjahr -> ohnehin keine VP.

Modellgrenzen (ehrlich, mit Richtung):
  - VOR 2018: alte Rechtslage (ausschuettungsgleiche Ertraege thesaurierender
    auslaendischer Fonds) wird NICHT modelliert -> ETF-Pfad bleibt fuer
    1996-2017 weiterhin ZU STARK (Benchmark-freundlich, gegen die Kandidaten).
  - VP-Steuer wird durch Anteilsverkauf zum Jahresultimo bezahlt (Units-
    Reduktion); der Gewinnanteil dieser Mikro-Verkaeufe wird NICHT zusaetzlich
    besteuert -> minimal PRO ETF. Basis und angesetzte VP werden pro-rata
    mitreduziert.
  - Sparerpauschbetrag: der bestehende ETF-Pfad kennt KEINEN Sparerpauschbetrag
    (die Strategie-Engine TaxedPortfolio nutzt 1000 EUR/J). Beide Varianten
    werden ausgewiesen: ohne SPB (konsistent zum Alt-ETF-Pfad, Headline) und
    mit 1000 EUR/J auf die VP (konsistent zur Strategie-Engine; real waeren es
    801 EUR bis 2022 — vereinfacht 1000 flat wie in TaxedPortfolio).
  - Zufluss der VP erfolgt gesetzlich am ersten Werktag des Folgejahres;
    modelliert wird die Zahlung am letzten Handelstag des Jahres (wenige Tage
    frueher, immateriell).
  - Kandidaten-Endwerte (H-032 low-div, H-024 = EW-Band-full) werden UNVERAENDERT
    aus results/h032b_terminal_liq.json uebernommen (deterministische Laeufe
    nach E-051-Fix); hier wird nur der ETF-Benchmark neu gerechnet.

Deterministisch: reine Preisreihen-Arithmetik, keine Sets/Dicts mit
hash-abhaengiger Ordnung im Ergebnispfad.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h011_kandidat_a import OUTD, START_CAPITAL  # noqa: E402
from verdict_engine import load_verdict_prices  # noqa: E402

TAX = 0.26375  # Abgeltungsteuer + Soli (wie bestehender Pfad)
TFS = 0.30  # Aktienfonds-Teilfreistellung §20 InvStG
ETF_TAX_OLD = 0.185  # Alt-Pfad (gerundet); exakt waere 0.184625
ETF_TAX_EXACT = TAX * (1.0 - TFS)  # 0.1846250

# BMF-Basiszins je Steuerjahr (siehe Docstring; negativ -> 0.0 = keine VP)
BASISZINS = {
    2018: 0.0087,
    2019: 0.0052,
    2020: 0.0007,
    2021: 0.0,
    2022: 0.0,
    2023: 0.0255,
    2024: 0.0229,
    2025: 0.0253,
    2026: 0.0253,  # Carry-Forward-Annahme (im Verkaufsjahr ohnehin keine VP)
}


def etf_path_old(spy: pd.Series) -> float:
    """Bestehender Pfad: Endsteuer 18,5 % flat auf den Gewinn, keine VP."""
    gain = START_CAPITAL * (spy.iloc[-1] / spy.iloc[0] - 1.0)
    return float(START_CAPITAL + gain * (1.0 - ETF_TAX_OLD))


def etf_path_vp(
    spy: pd.Series, *, spb_per_year: float = 0.0, terminal_rate: float = ETF_TAX_EXACT
) -> dict:
    """SPY-Buy&Hold mit Vorabpauschale ab 2018 + End-Liquidation.

    spb_per_year: Sparerpauschbetrag, der jaehrlich gegen die steuerpflichtige
    VP gerechnet wird (0.0 = keiner, wie im Alt-ETF-Pfad).
    """
    units = START_CAPITAL / float(spy.iloc[0])
    basis = float(START_CAPITAL)  # Anschaffungskosten der verbleibenden Units
    acc_vp = 0.0  # angesetzte Vorabpauschalen (verbleibende Units)
    vp_tax_paid = 0.0
    per_year: dict[str, dict] = {}

    years = sorted(set(spy.index.year))
    sale_year = years[-1]
    for y in years:
        if y < 2018 or y == sale_year:
            continue  # vor 2018: alte Rechtslage (nicht modelliert); Verkaufsjahr: keine VP
        zins = BASISZINS.get(y)
        if zins is None:
            raise KeyError(f"Basiszins fuer {y} fehlt in BASISZINS-Tabelle")
        if zins <= 0.0:
            per_year[str(y)] = {"basiszins": zins, "vp": 0.0, "vp_tax": 0.0}
            continue
        yr = spy[spy.index.year == y]
        p_first, p_last = float(yr.iloc[0]), float(yr.iloc[-1])
        v_start = units * p_first  # Ruecknahmepreis Jahresanfang * Bestand
        v_end = units * p_last
        basisertrag = v_start * zins * 0.7
        wertzuwachs = v_end - v_start
        vp = max(0.0, min(basisertrag, wertzuwachs))
        taxable = max(0.0, vp * (1.0 - TFS) - spb_per_year)
        tax = taxable * TAX
        acc_vp += vp
        if tax > 0.0:
            # Steuer durch Anteilsverkauf zum Jahresultimo; Basis + angesetzte
            # VP pro-rata mitreduzieren (Gewinnanteil des Mikro-Verkaufs
            # unbesteuert -> minimal PRO ETF, dokumentiert).
            frac_sold = (tax / p_last) / units
            units -= tax / p_last
            basis *= 1.0 - frac_sold
            acc_vp *= 1.0 - frac_sold
            vp_tax_paid += tax
        per_year[str(y)] = {
            "basiszins": zins,
            "basisertrag": round(basisertrag, 2),
            "wertzuwachs": round(wertzuwachs, 2),
            "vp": round(vp, 2),
            "vp_tax": round(tax, 2),
        }

    proceeds = units * float(spy.iloc[-1])
    gain = proceeds - basis - acc_vp  # §19 InvStG: angesetzte VP mindern den Gewinn
    terminal_tax = max(0.0, gain) * terminal_rate
    final = proceeds - terminal_tax
    return {
        "final": float(final),
        "vp_tax_paid_total": round(vp_tax_paid, 2),
        "acc_vp_at_sale": round(acc_vp, 2),
        "terminal_tax": round(terminal_tax, 2),
        "per_year": per_year,
    }


def main() -> int:
    close = load_verdict_prices()
    spy = close["SPY"].dropna()

    # Fenster + Kandidaten-Endwerte unveraendert aus H-032b (post-E-051-Fix)
    h032b = json.loads((OUTD / "h032b_terminal_liq.json").read_text(encoding="utf-8"))
    w0, w1 = (
        pd.Timestamp(h032b["window"][0], tz="UTC"),
        pd.Timestamp(h032b["window"][1], tz="UTC"),
    )
    spy_w = spy[(spy.index >= w0) & (spy.index <= w1)]
    low_div = float(h032b["H032_low_div"]["final_net_postliq"])
    h024_ew = float(h032b["EW_full"]["final_net_postliq"])

    etf_old = etf_path_old(spy_w)
    # Reproduktions-Check gegen den dokumentierten Alt-Wert 1.610.149
    repro_ok = abs(etf_old - float(h032b["ETF_net_window_matched"])) < 1.0

    # Zerlegung: (a) nur exakte TFS (18,4625 % statt 18,5 %), keine VP
    gain_w = START_CAPITAL * (spy_w.iloc[-1] / spy_w.iloc[0] - 1.0)
    etf_tfs_exact_only = float(START_CAPITAL + gain_w * (1.0 - ETF_TAX_EXACT))
    # (b) VP + exakte TFS, ohne Sparerpauschbetrag (Headline; konsistent zum
    #     Alt-ETF-Pfad, der ebenfalls keinen SPB kennt)
    vp_no_spb = etf_path_vp(spy_w, spb_per_year=0.0)
    # (c) VP + exakte TFS, mit 1000 EUR/J SPB auf die VP (Sensitivitaet,
    #     konsistent zur Strategie-Engine)
    vp_spb = etf_path_vp(spy_w, spb_per_year=1000.0)
    # (d) Isolierter VP-Effekt bei ALTEM Terminal-Satz 18,5 % (nur Attribution)
    vp_oldrate = etf_path_vp(spy_w, spb_per_year=0.0, terminal_rate=ETF_TAX_OLD)

    etf_new = vp_no_spb["final"]

    def cmp_block(cand: float, etf: float) -> dict:
        return {
            "candidate": round(cand),
            "etf": round(etf),
            "candidate_beats_etf": bool(cand > etf),
            "delta_pct": round((cand / etf - 1.0) * 100.0, 2),
        }

    out = {
        "window": [str(w0.date()), str(w1.date())],
        "reproduced_old_etf_matches_h032b": bool(repro_ok),
        "teilfreistellung_finding": (
            "Teilfreistellung 30% war im Alt-Pfad BEREITS modelliert (18,5% "
            "~ 26,375%*0,7), nur gerundet statt exakt 18,4625% -> Alt-Pfad "
            "besteuerte den ETF um 0,0375 Pp zu hoch. KEINE zweite Luecke."
        ),
        "ETF_old_185_no_vp": round(etf_old),
        "ETF_tfs_exact_only_no_vp": round(etf_tfs_exact_only),
        "ETF_new_vp_tfs_exact_no_spb": round(etf_new),
        "ETF_new_vp_tfs_exact_spb1000": round(vp_spb["final"]),
        "ETF_vp_only_at_old_rate_185": round(vp_oldrate["final"]),
        "vp_detail_no_spb": vp_no_spb,
        "vp_detail_spb1000": {k: v for k, v in vp_spb.items() if k != "per_year"},
        "effects": {
            "vp_effect_pct_at_exact_rate": round(
                (etf_new / etf_tfs_exact_only - 1.0) * 100.0, 3
            ),
            "tfs_rounding_effect_pct": round(
                (etf_tfs_exact_only / etf_old - 1.0) * 100.0, 3
            ),
            "net_effect_pct_old_to_new": round((etf_new / etf_old - 1.0) * 100.0, 3),
        },
        "H032_low_div_vs_ETF": {
            "old": cmp_block(low_div, etf_old),
            "new": cmp_block(low_div, etf_new),
        },
        "H024_ew_band_full_vs_ETF": {
            "old": cmp_block(h024_ew, etf_old),
            "new": cmp_block(h024_ew, etf_new),
        },
        "model_notes": [
            "VP erst ab 2018; 1996-2017 alte Rechtslage (ausschuettungsgleiche "
            "Ertraege) NICHT modelliert -> ETF-Pfad dort weiterhin zu stark "
            "(Richtung: pro Benchmark / gegen Kandidaten).",
            "Verkaufsjahr (letztes Fensterjahr) ohne VP (Zufluss erst Folgejahr).",
            "VP-Steuer via Anteilsverkauf am Jahresultimo, Mikro-Verkaufsgewinn "
            "unbesteuert (minimal pro ETF); Basis+acc_vp pro-rata reduziert.",
            "2026er Basiszins = Carry-Forward 2,53% (Annahme, hier ohne Wirkung).",
            "Kandidaten-Endwerte unveraendert aus h032b_terminal_liq.json.",
        ],
    }
    (OUTD / "h086_vorabpauschale.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    print(
        "[VERDICT]",
        json.dumps(
            {
                k: out[k]
                for k in (
                    "reproduced_old_etf_matches_h032b",
                    "ETF_old_185_no_vp",
                    "ETF_new_vp_tfs_exact_no_spb",
                    "ETF_new_vp_tfs_exact_spb1000",
                    "effects",
                    "H032_low_div_vs_ETF",
                    "H024_ew_band_full_vs_ETF",
                )
            },
            indent=2,
            default=str,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
