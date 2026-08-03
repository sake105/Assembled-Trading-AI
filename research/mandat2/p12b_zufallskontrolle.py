"""P12b — Ist Momentum am langen Ende WIRKLICH schlechter als Zufall?

DIE OFFENE STELLE AUS P12
-------------------------
P12 zeigte Momentum bei den langen Haltedauern im unteren Bereich von fuenf
Zufallsziehungen. Fuenf Seeds koennen das nicht tragen: die Aufloesung betraegt
20 Prozentpunkte, und „unter allen fuenf" hat unter der Nullhypothese schon
eine Wahrscheinlichkeit von 1/6. Die Richtung war ueber vier Haltedauern
konsistent — aber konsistent ist nicht signifikant.

WICHTIG — was der erste Lauf dieses Skripts NICHT messen konnte
---------------------------------------------------------------
Der Zufalls-Score hatte keine NaN und war deshalb ab dem ersten Bar investiert,
waehrend das Momentum-Signal noch im Rueckblick-Vorlauf steckte (bei „2 Jahre"
vier Jahre lang). Verglichen wurden damit Startzeitpunkte, nicht Signale
(Stage-1-Finding F-test-3). Jetzt erbt der Zufalls-Score die NaN-Maske des
echten Signals, und beide starten am selben Bar.

Die langen Haltedauern sind billig (5 bis 99 Umschichtungen). Es gibt also
keinen Grund, die Frage offen zu lassen, statt sie zu messen: 60 Seeds je
Haltedauer.

DIES IST KEINE ALPHA-SUCHE
--------------------------
Es wird kein Signal optimiert und kein Kandidat gewaehlt. Gemessen wird die
Nullverteilung, gegen die ein bereits gerechnetes Ergebnis einzuordnen ist.
Der Trial-Zaehler steigt trotzdem, weil Backtests laufen.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from research.mandat2.data_gate import TrialCounter  # noqa: E402
from research.mandat2.intraday_data import load_intraday  # noqa: E402
from research.mandat2.p12_intraday_haltedauer import (  # noqa: E402
    RUECKBLICK_FEST,
    WARMUP,
    buy_and_hold,
    kennzahlen,
    momentum,
    simuliere,
    zufall,
)

OUT = Path(__file__).resolve().parent / "results"
N_SEEDS = 60
LANG = [(147, "1 Monat"), (441, "1 Quartal"), (1764, "1 Jahr"), (3528, "2 Jahre")]


def main() -> int:
    OUT.mkdir(exist_ok=True)
    d = load_intraday()
    print(d, flush=True)
    print(
        f"Trials kumuliert: "
        f"{TrialCounter().increment(len(LANG) * N_SEEDS, label='P12b Zufallskontrolle')}\n",
        flush=True,
    )

    close = d.close.ffill()
    fenster = close.index[WARMUP:]
    jahre = (fenster[-1] - fenster[0]).days / 365.25
    b_end = kennzahlen(buy_and_hold(close, start=WARMUP), jahre)["endwert"]
    print(
        f"[Benchmark] Buy-and-Hold {close.shape[1]} Namen ab {fenster[0]:%Y-%m-%d}: "
        f"{b_end:.3f}x ({jahre:.2f} Jahre)\n"
    )

    print(
        f"{'Haltedauer':<12}{'Momentum':>10}{'Zufall Md':>12}{'Zufall 5-95%':>20}"
        f"{'Perzentil':>11}{'p (einseitig)':>15}"
    )
    print("-" * 80)

    zeilen = []
    for bars, name in LANG:
        sig = momentum(close, RUECKBLICK_FEST)
        n_m, _, _, _ = simuliere(close, halte_bars=bars, score=sig, start=WARMUP)
        m_end = kennzahlen(n_m, jahre)["endwert"]

        z = []
        for seed in range(N_SEEDS):
            n_z, _, _, _ = simuliere(
                close, halte_bars=bars, score=zufall(close, seed, wie=sig), start=WARMUP
            )
            z.append(kennzahlen(n_z, jahre)["endwert"])
        z_arr = np.array(z)

        # Anteil der Zufallsziehungen, die Momentum SCHLAGEN -> einseitiger
        # p-Wert fuer „Momentum ist nicht besser als Zufall".
        besser = int((z_arr > m_end).sum())
        p = (besser + 1) / (N_SEEDS + 1)
        perz = 100.0 * (z_arr < m_end).mean()
        zeilen.append(
            {
                "halte_bars": bars,
                "name": name,
                "rueckblick_bars": RUECKBLICK_FEST,
                "momentum_end": m_end,
                "zufall_median": float(np.median(z_arr)),
                "zufall_p05": float(np.percentile(z_arr, 5)),
                "zufall_p95": float(np.percentile(z_arr, 95)),
                "zufall_schlagen_momentum": besser,
                "n_seeds": N_SEEDS,
                "momentum_perzentil": float(perz),
                "p_einseitig": float(p),
                "bench_end": b_end,
                "zufall_schlagen_bench": int((z_arr > b_end).sum()),
            }
        )
        print(
            f"{name:<12}{m_end:>9.3f}x{np.median(z_arr):>11.3f}x"
            f"{np.percentile(z_arr, 5):>10.3f}x..{np.percentile(z_arr, 95):>7.3f}x"
            f"{perz:>10.0f}%{p:>14.3f}",
            flush=True,
        )

    (OUT / "p12b_zufallskontrolle.json").write_text(
        json.dumps(
            {"n_seeds": N_SEEDS, "benchmark_end": b_end, "zeilen": zeilen}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"\n-> {OUT / 'p12b_zufallskontrolle.json'}")

    schlechter = [z for z in zeilen if z["momentum_perzentil"] < 50]
    print("\n" + "=" * 80)
    print(
        f"Momentum liegt in {len(schlechter)}/{len(zeilen)} Haltedauern UNTER dem "
        f"Zufallsmedian."
    )
    for z in zeilen:
        print(
            f"  {z['name']:<12} Perzentil {z['momentum_perzentil']:>3.0f}% | "
            f"{z['zufall_schlagen_momentum']}/{N_SEEDS} Zufallsziehungen besser | "
            f"{z['zufall_schlagen_bench']}/{N_SEEDS} schlagen den Halte-Benchmark"
        )
    print("=" * 80, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
