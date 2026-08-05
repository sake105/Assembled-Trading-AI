"""Handelbarkeit des neu erschlossenen Small-Cap-Universums — Grundlage fuer das Gate.

WOZU
----
Der DERA-Bestand macht die §4.6.1-These erstmals auf echten Small Caps
pruefbar. Genau dort lauert aber der Fehlschluss, an dem H-035/H-036 schon
einmal gescheitert sind: **"Size" war ein Illiquiditaets-Artefakt.** Ein
Backtest, der Namen anfasst, die man real nicht handeln kann, misst
Spread-Fantasie statt Rendite.

Dieses Skript beziffert, wie gross das Problem ist, damit die Schwelle des
Liquiditaets-Gates aus der Verteilung folgt und nicht aus dem Bauch (E-122).

WAS GEMESSEN WIRD
-----------------
Fuer die Namen mit mindestens einem plausiblen Open-Market-Kauf im
DERA-Bestand: Median-ADV und Median-Kurs ueber die Tage, an denen der Titel
ueberhaupt notiert. Getrennt nach "jemals im S&P-Panel" und "nie" — die
zweite Gruppe ist das, was neu dazugekommen ist.

PIT-HINWEIS
-----------
Die hier gerechneten Mediane sind **Beschreibung, kein Filter**. Ein Gate im
Backtest muss die Handelbarkeit zu jedem `as_of` neu pruefen (rollierendes
ADV-Fenster), sonst waehlt es Titel danach aus, ob sie SPAETER liquide wurden
— ein Survivorship-artiger Lookahead. Das steht hier, weil die Verwechslung
naheliegt.

KEIN TRIAL
----------
Reine Beschreibung der Datenlage (E-090).
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATA = Path(__file__).resolve().parent / "data"
DERA = DATA / "form4_dera"
SC_CLOSE = DATA / "_sc_close.parquet"
SC_ADV = DATA / "_sc_adv.parquet"
FORM4_BROAD = DATA / "form4_broad"
ZIEL = Path(__file__).resolve().parent / "liquiditaet_smallcap.json"

#: Kandidatenschwellen, ueber die berichtet wird. Die Wahl fuer das Gate
#: faellt in der Registrierung, nicht hier — dieses Skript liefert nur die
#: Verteilung, an der man sie begruenden kann.
ADV_STUFEN = (100_000, 200_000, 500_000, 1_000_000, 5_000_000)
KURS_STUFEN = (1.0, 3.0, 5.0)


def main() -> int:
    for p in (SC_CLOSE, SC_ADV):
        if not p.exists():
            raise SystemExit(f"[ERROR] {p.name} fehlt — Preispanel nicht vorhanden.")
    dateien = sorted(glob.glob(str(DERA / "*.parquet")))
    if not dateien:
        raise SystemExit("[ERROR] DERA-Bestand fehlt — erst pull_form4_dera.py laufen.")

    kauf = pd.concat(
        [
            pd.read_parquet(
                p, columns=["symbol", "transaction_type", "datum_plausibel"]
            )
            for p in dateien
        ],
        ignore_index=True,
    )
    kauf = kauf[(kauf["transaction_type"] == "P") & kauf["datum_plausibel"]]
    mit_signal = set(kauf["symbol"].dropna().unique())

    close = pd.read_parquet(SC_CLOSE)
    adv = pd.read_parquet(SC_ADV)
    gemeinsam = [c for c in close.columns if c in adv.columns and c in mit_signal]

    # Jemals im S&P-Panel? Der Altbestand form4_broad ist genau die
    # S&P-Historie — die Differenz ist das neu Erschlossene.
    sp = set()
    for p in glob.glob(str(FORM4_BROAD / "*.parquet")):
        sp.update(Path(p).stem.split("__")[1:])

    med_adv = adv[gemeinsam].median()
    med_kurs = close[gemeinsam].median()
    neu = [c for c in gemeinsam if c not in sp]
    alt = [c for c in gemeinsam if c in sp]

    def gruppe(namen: list[str], label: str) -> dict:
        if not namen:
            return {"label": label, "n": 0}
        a, k = med_adv[namen], med_kurs[namen]
        return {
            "label": label,
            "n": len(namen),
            "median_adv": float(a.median()),
            "median_kurs": float(k.median()),
            "unter_adv": {str(s): int((a < s).sum()) for s in ADV_STUFEN},
            "unter_adv_anteil": {
                str(s): round(float((a < s).mean()), 4) for s in ADV_STUFEN
            },
            "unter_kurs": {str(s): int((k < s).sum()) for s in KURS_STUFEN},
            "unter_kurs_anteil": {
                str(s): round(float((k < s).mean()), 4) for s in KURS_STUFEN
            },
        }

    ergebnis = {
        "namen_mit_kaufsignal": len(mit_signal),
        "davon_im_preispanel": len(gemeinsam),
        "gesamt": gruppe(gemeinsam, "alle mit Signal + Kurs"),
        "neu_erschlossen": gruppe(neu, "nie im S&P-Panel (neu)"),
        "bereits_abgedeckt": gruppe(alt, "jemals im S&P-Panel (alt)"),
        "pit_hinweis": (
            "Mediane sind Beschreibung, kein Filter. Ein Gate im Backtest muss "
            "die Handelbarkeit je as_of rollierend pruefen — sonst waehlt es "
            "Titel danach aus, ob sie SPAETER liquide wurden (Lookahead)."
        ),
    }

    for s in ("gesamt", "neu_erschlossen", "bereits_abgedeckt"):
        g = ergebnis[s]
        if not g.get("n"):
            continue
        print(
            f"  {g['label']:<28} n={g['n']:>5} | Median-ADV "
            f"{g['median_adv']:>12,.0f} | Median-Kurs {g['median_kurs']:>7.2f}"
        )
        print(
            "      unter ADV: "
            + "  ".join(
                f"{int(k) // 1000}k: {v:.0%}" for k, v in g["unter_adv_anteil"].items()
            )
        )
    # Artefakt als LETZTE Anweisung (E-116).
    ZIEL.write_text(
        json.dumps(ergebnis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n-> {ZIEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
