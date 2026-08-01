# P1 Erstlauf — Befund (2026-08-01)

Erster echter Lauf von Mandat II. **Ausschließlich auf dem SUCH-Fenster**
(1995-01-03 … 2016-12-30, 1.037 Symbole, 5.549 Handelstage). Der Holdout wurde
nicht angefasst. Rohdaten: `results/p1_baseline.json`.

Kandidat: 12-1-Momentum, monatlich, Top-20 rein / Rang-60 raus, kein Hebel,
keine Mindesthaltedauer. Benchmark: SPY Buy-and-Hold, als **Fonds** besteuert.

---

## Befund 1 — Der DD-Deckel ist schärfer als der Benchmark. In jedem Fenster.

| | schlimmster MaxDD | bester MaxDD | Fenster über dem −35 %-Deckel |
|---|---|---|---|
| **SPY (Benchmark)** | −55,2 % | −47,5 % | **144 / 144** |
| Momentum | −61,2 % … −64,8 % | — | 144 / 144 |

Jedes 10-Jahres-Fenster zwischen 1995 und 2016 enthält Dotcom **oder** die
Finanzkrise — meistens eine von beiden vollständig. SPY kommt in **keinem
einzigen** Fenster mit weniger als −47,5 % durch.

**Konsequenz für die Zielfunktion (unverändert gültig, aber sie bedeutet etwas
anderes als „SPY schlagen"):** Ein reiner Long-only-Aktienkandidat, der immer
voll investiert ist, kann die Nebenbedingung auf diesem Suchfenster **nicht
erfüllen** — unabhängig von der Aktienauswahl. Wer den Deckel halten will,
braucht zwingend einen **risikoreduzierenden Bestandteil**: Cash-Quote,
Timing/Gate, Absicherung oder eine unkorrelierte Sleeve.

Das ist keine Aufweichung der Vorgabe. Hans hat den Deckel bewusst schärfer als
den Benchmark gesetzt („der Kandidat darf weniger Risiko als der Benchmark,
nicht mehr"). Der Lauf zeigt nur, **wie viel** schärfer das ist: es verlangt
eine Halbierung des Krisen-Drawdowns gegenüber dem Index, nicht eine
Verbesserung am Rand.

Damit verschiebt sich der Suchraum ab P2: gesucht ist nicht „die bessere
Aktienauswahl", sondern „die bessere Aktienauswahl **plus** ein Mechanismus,
der durch 2000–2002 und 2008–2009 trägt".

## Befund 2 — Momentum verliert in ALLEN vier Steuerwelten, auch ohne Steuer

| Regime | Momentum Endwert | SPY Endwert | Median-Faktor Kandidat vs Benchmark | Fenster geschlagen |
|---|---|---|---|---|
| `ZERO` | 701.760 | 726.197 | 1,444× vs 1,948× | 52 % |
| `PRIVAT_DE` | 385.798 | 610.752 | 1,265× vs 1,909× | 45 % |
| `GMBH_THESAURIEREND` | 416.823 | 653.766 | 1,235× vs 1,925× | 43 % |
| `GMBH_AUSSCHUETTUNG` | 333.261 | 507.710 | 1,235× vs 1,925× | 43 % |

**Das bestätigt Kernbefund 1 aus Mandat I an einem unabhängig gebauten Aufbau:
auch bei null Steuer verliert Momentum gegen den Index.** Die Steuer war nicht
die Ursache. Bei `ZERO` beträgt der Rückstand über 22 Jahre nur noch 3,4 %
(702k vs 726k) — aber der Median über die Fenster ist mit 1,44× gegen 1,95×
deutlich schlechter, das heißt: Momentum gewinnt gelegentlich groß und verliert
regelmäßig, während der Index gleichmäßig liefert.

## Befund 3 — Die GmbH hilft dem Turnover-Kandidaten, aber nicht genug

Der Turnover-Steuervorteil ist real und messbar. Momentum, ~8.600 Trades:

* Steuer auf **Kursgewinne**: 68.328 (privat) → **20.907** (GmbH) = −69 %
* Steuer auf **Dividenden**: 104.435 (privat) → **128.033** (GmbH) = +23 %

Genau die vorhergesagte Asymmetrie: Umschichten wird billig, Dividenden werden
teurer. Netto bleibt die GmbH für diesen Kandidaten vorn (417k gegen 386k), aber
der Vorsprung von 31k wird von den nicht modellierten Rechtsformkosten
aufgefressen — bei 3.500 €/Jahr über 22 Jahre sind das 77k. **Für diesen
Kandidaten trägt die GmbH-Struktur nicht.**

Zu beachten: der Benchmark profitiert von der GmbH *ebenfalls* (611k → 654k),
weil Fondsgewinne dort mit 11,57 % statt 18,46 % belastet werden. Der
Steuervorteil hebt also beide Seiten, nicht nur den Kandidaten.

## Befund 4 — Transaktionskosten sind bei Momentum eine eigene Größe

33.081 € Kosten bei `ZERO` (8.000 Trades) gegen 827 € beim Benchmark. Das ist
ein Drittel des gesamten Rückstands im steuerfreien Fall.

---

## Was daraus für die nächsten Phasen folgt

1. **P2 muss den Risikoteil zuerst lösen, nicht den Renditeteil.** Ohne einen
   Mechanismus gegen die zwei großen Krisen ist jede Rendite-Optimierung
   vergeblich, weil die Nebenbedingung vorher greift.
2. **Hebel ist auf diesem Suchfenster fast sicher tot** — er vergrößert genau
   den Drawdown, der schon ohne ihn dreifach über dem Deckel liegt. Er wird
   trotzdem gemessen (Auftrag), aber die Erwartung steht hier vorab.
3. **Die Turnover-Hypothese der GmbH ist bestätigt, aber sie reicht allein
   nicht.** Sie kann einen ohnehin knappen Kandidaten über die Linie tragen —
   einen, der um 35 % zurückliegt, nicht.
4. Die Fixkosten der GmbH gehören in jeden Vergleich, sobald es um „GmbH oder
   privat?" geht. 22 Jahre × 3.500 € = 77.000 € sind bei 100.000 € Startkapital
   keine Fußnote.

**Verbrauchte Trials:** dieser Lauf ist als Diagnostik-Lauf geführt und noch
nicht im Trial-Zähler registriert (Kandidat wurde nicht gegen den Holdout
gestellt). Ab der ersten registrierten Hypothese in P2 zählt `TrialCounter`.
