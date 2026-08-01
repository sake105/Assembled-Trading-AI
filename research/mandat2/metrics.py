"""Zielfunktion und Fenster-Auswertung (Mandat II, Phase 0).

Die Zielfunktion ist am 2026-08-01 gesperrt worden (Entscheidung Hans):

    maximize   Median-Endvermoegen ueber ALLE rollierenden 10-Jahres-Fenster
    s.t.       MaxDD >= -35 %   in JEDEM Fenster

Zwei Entwurfsentscheidungen, die den Unterschied zwischen einer ehrlichen und
einer geschoenten Auswertung ausmachen:

1. **Alle rollierenden Fenster, nicht ein Fenster.** „Schlaegt SPY ueber 10
   Jahre" ist ohne Fensterverteilung eine Einladung zum Rosinenpicken: fast
   jede Strategie hat IRGENDEIN 10-Jahres-Fenster, in dem sie gewinnt. Der
   Median ueber alle Startmonate beantwortet die Frage, die tatsaechlich
   gestellt war.

2. **Der DD-Deckel gilt pro Fenster, nicht im Mittel.** Ein Kandidat, der in
   einem einzigen Fenster -60 % faehrt und in allen anderen -20 %, ist
   durchgefallen — nicht „im Schnitt akzeptabel". Sonst waere der Deckel
   wirkungslos und gehebeltes SPY gewaenne trivial.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Harte Nebenbedingung (Entscheidung Hans, 2026-08-01). SPY selbst lag
#: 2007-2009 bei ca. -55 % — der Kandidat darf also WENIGER Risiko als der
#: Benchmark, nicht mehr.
DD_DECKEL = -0.35
#: Zielhorizont in Jahren.
FENSTER_JAHRE = 10


def max_drawdown(equity: pd.Series) -> float:
    """Groesster Peak-to-Trough-Ruecksetzer als negative Zahl (-0.35 = -35 %)."""
    if len(equity) < 2:
        return 0.0
    e = equity.astype(float)
    peak = e.cummax()
    dd = (e / peak) - 1.0
    return float(dd.min())


def rolling_windows(
    index: pd.DatetimeIndex, jahre: int = FENSTER_JAHRE, schritt_monate: int = 1
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Alle (Start, Ende) mit ``jahre`` Abstand, im Monatsraster.

    Monatsraster statt Tagesraster: 144 Fenster ueber 22 Jahre sind genug
    Verteilung fuer einen Median, und taegliche Fenster waeren zu ~99 %
    ueberlappend, also fast keine zusaetzliche Information bei 250x Rechenzeit.
    """
    if len(index) == 0:
        return []
    idx = pd.DatetimeIndex(index)
    monatsanfaenge = (
        idx.to_series().groupby(idx.to_period("M")).min().sort_values().tolist()
    )
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for i in range(0, len(monatsanfaenge), schritt_monate):
        start = monatsanfaenge[i]
        ende = start + pd.DateOffset(years=jahre)
        if ende > idx[-1]:
            break
        pos = idx.searchsorted(ende, side="right") - 1
        if pos > 0 and idx[pos] > start:
            out.append((start, idx[pos]))
    return out


@dataclass
class FensterErgebnis:
    start: pd.Timestamp
    ende: pd.Timestamp
    kandidat_faktor: float  # Endvermoegen / Startvermoegen
    benchmark_faktor: float
    kandidat_maxdd: float
    benchmark_maxdd: float

    @property
    def schlaegt_benchmark(self) -> bool:
        return self.kandidat_faktor > self.benchmark_faktor

    @property
    def deckel_gerissen(self) -> bool:
        return self.kandidat_maxdd < DD_DECKEL


@dataclass
class Auswertung:
    """Das Urteil ueber einen Kandidaten — mit allem, was dagegen spricht."""

    label: str
    fenster: list[FensterErgebnis] = field(default_factory=list)

    # ------------------------------------------------------------- Kennzahlen
    @property
    def n_fenster(self) -> int:
        return len(self.fenster)

    @property
    def median_kandidat(self) -> float:
        return float(np.median([f.kandidat_faktor for f in self.fenster]))

    @property
    def median_benchmark(self) -> float:
        return float(np.median([f.benchmark_faktor for f in self.fenster]))

    @property
    def anteil_fenster_geschlagen(self) -> float:
        if not self.fenster:
            return 0.0
        return sum(f.schlaegt_benchmark for f in self.fenster) / len(self.fenster)

    @property
    def schlimmster_maxdd(self) -> float:
        return min((f.kandidat_maxdd for f in self.fenster), default=0.0)

    @property
    def gerissene_fenster(self) -> list[FensterErgebnis]:
        return [f for f in self.fenster if f.deckel_gerissen]

    # ---------------------------------------------------------------- Verdikt
    @property
    def deckel_eingehalten(self) -> bool:
        """In JEDEM Fenster, nicht im Median."""
        return not self.gerissene_fenster

    @property
    def schlaegt_benchmark(self) -> bool:
        return self.median_kandidat > self.median_benchmark

    @property
    def bestanden(self) -> bool:
        """Beide Bedingungen. Ein Kandidat ohne Fenster besteht NICHT.

        Fail-closed: eine leere Auswertung ist keine bestandene Pruefung,
        sondern eine ausgefallene.
        """
        return (
            bool(self.fenster) and self.schlaegt_benchmark and self.deckel_eingehalten
        )

    def bericht(self) -> str:
        if not self.fenster:
            return f"{self.label}: KEINE Fenster auswertbar — kein Urteil moeglich."
        verdikt = "BESTANDEN" if self.bestanden else "DURCHGEFALLEN"
        gruende = []
        if not self.schlaegt_benchmark:
            gruende.append(
                f"Median {self.median_kandidat:.3f}x <= Benchmark "
                f"{self.median_benchmark:.3f}x"
            )
        if not self.deckel_eingehalten:
            gruende.append(
                f"DD-Deckel in {len(self.gerissene_fenster)}/{self.n_fenster} "
                f"Fenstern gerissen (schlimmster {self.schlimmster_maxdd:.1%})"
            )
        zusatz = ("  |  " + "; ".join(gruende)) if gruende else ""
        return (
            f"{self.label}: {verdikt}  |  {self.n_fenster} Fenster  |  "
            f"Median {self.median_kandidat:.3f}x vs {self.median_benchmark:.3f}x  |  "
            f"{self.anteil_fenster_geschlagen:.0%} der Fenster geschlagen  |  "
            f"schlimmster MaxDD {self.schlimmster_maxdd:.1%}{zusatz}"
        )


def auswerten(
    kandidat: pd.Series,
    benchmark: pd.Series,
    *,
    label: str,
    jahre: int = FENSTER_JAHRE,
    schritt_monate: int = 1,
) -> Auswertung:
    """Kandidat gegen Benchmark ueber alle rollierenden Fenster.

    Beide Kurven werden auf den gemeinsamen Index beschraenkt — sonst
    vergliche man unterschiedliche Zeitraeume und der Median waere bedeutungslos.
    """
    gemeinsam = kandidat.index.intersection(benchmark.index)
    k = kandidat.reindex(gemeinsam).astype(float)
    b = benchmark.reindex(gemeinsam).astype(float)
    a = Auswertung(label=label)
    for start, ende in rolling_windows(
        pd.DatetimeIndex(gemeinsam), jahre=jahre, schritt_monate=schritt_monate
    ):
        ks, ke = k.loc[start], k.loc[ende]
        bs, be = b.loc[start], b.loc[ende]
        if not (np.isfinite(ks) and np.isfinite(bs)) or ks <= 0 or bs <= 0:
            continue
        a.fenster.append(
            FensterErgebnis(
                start=start,
                ende=ende,
                kandidat_faktor=float(ke / ks),
                benchmark_faktor=float(be / bs),
                kandidat_maxdd=max_drawdown(k.loc[start:ende]),
                benchmark_maxdd=max_drawdown(b.loc[start:ende]),
            )
        )
    return a
