# P1 Erstlauf — Befund (2026-08-01, **korrigierte Fassung**)

> **Korrekturhinweis.** Die erste Fassung dieses Dokuments (Commit `b0071665`)
> stand auf fehlerhaften Zahlen. Der Stage-2-Review fand zwei BLOCKER und vier
> MAJORs, die alle behoben wurden. **Befund 3 hat sich dadurch ins Gegenteil
> gedreht** — die GmbH trägt für diesen Kandidaten sehr wohl. Was sich *nicht*
> geändert hat: Befund 1 und Befund 2. Die alten Zahlen stehen in der
> Git-Historie; hier stehen nur die korrigierten.
>
> Behoben: Holdout-Leck im Dividendenpanel (22.507 Zeilen aus 2017–2027 wurden
> auf den letzten Suchtag gestapelt — 728 Symbole an einem Tag statt Median 5),
> nominale Dividenden auf einem adjustierten Panel (Überzeichnung 1,16–1,48×),
> Zielfunktion maß auf der Mark-to-market-Kurve (Buy-and-Hold bekam die
> Steuerstundung geschenkt), fehlender Delisting-Zwangsverkauf, wirkungsloser
> Hebel, Rang-Gleichstände. Details: E-070/E-071 in `docs/CLAUDE_CODING_ERRORS.md`.

Lauf ausschließlich auf dem SUCH-Fenster (1995-01-03 … 2016-12-30, 1.037
Symbole, 5.549 Handelstage). Der Holdout wurde nicht angefasst. Rohdaten:
`results/p1_baseline.json`.

Kandidat: 12-1-Momentum, monatlich, Top-20 rein / Rang-60 raus, kein Hebel,
keine Mindesthaltedauer. Benchmark: SPY Buy-and-Hold, als **Fonds** besteuert
(§20 InvStG), nicht als Aktie.

Ausgewertet wird auf der **Netto-Kurve** (Marktwert abzüglich latenter Steuer
auf offene Buchgewinne). Ohne das trüge der umschichtende Kandidat seine Steuer
laufend und der Buy-and-Hold-Benchmark nie.

---

## Befund 1 — Der DD-Deckel ist schärfer als der Benchmark. In jedem Fenster. *(unverändert)*

| | schlimmster MaxDD | bester MaxDD | Fenster über dem −35 %-Deckel |
|---|---|---|---|
| **SPY (Benchmark, ZERO)** | −55,2 % | −47,5 % | **144 / 144** |
| Momentum | −65,9 % … −67,2 % | — | 144 / 144 |

Jedes 10-Jahres-Fenster zwischen 1995 und 2016 enthält Dotcom **oder** die
Finanzkrise. SPY kommt in **keinem einzigen** Fenster mit weniger als −47,5 %
durch.

**Konsequenz:** Ein reiner Long-only-Aktienkandidat, der immer voll investiert
ist, kann die Nebenbedingung auf diesem Suchfenster **nicht erfüllen** —
unabhängig von der Aktienauswahl. Wer den Deckel halten will, braucht zwingend
einen **risikoreduzierenden Bestandteil**: Cash-Quote, Timing-Gate, Absicherung
oder eine unkorrelierte Sleeve.

Das ist keine Aufweichung der Vorgabe. Hans hat den Deckel bewusst schärfer als
den Benchmark gesetzt. Der Lauf zeigt nur, **wie viel** schärfer: verlangt ist
eine Halbierung des Krisen-Drawdowns gegenüber dem Index, nicht eine
Verbesserung am Rand. Ab P2 ist gesucht: „bessere Auswahl **plus** ein
Mechanismus, der durch 2000–2002 und 2008–2009 trägt".

## Befund 2 — Momentum verliert in ALLEN vier Steuerwelten, auch ohne Steuer *(unverändert, Zahlen aktualisiert)*

| Regime | Momentum | SPY | Median-Faktor Kandidat vs Benchmark | Fenster geschlagen |
|---|---|---|---|---|
| `ZERO` | 679.935 | 726.197 | 1,250× vs 1,948× | 41 % |
| `PRIVAT_DE` | 427.194 | 610.752 | 1,071× vs 1,870× | 38 % |
| `GMBH_THESAURIEREND` | 565.472 | 653.766 | 1,145× vs 1,900× | 40 % |
| `GMBH_AUSSCHUETTUNG` | 442.704 | 507.710 | 1,145× vs 1,900× | 40 % |

**Auch bei null Steuer verliert Momentum gegen den Index.** Die Steuer war nicht
die Ursache. Der Fenster-Median ist mit 1,25× gegen 1,95× deutlich schlechter:
Momentum gewinnt gelegentlich groß und verliert regelmäßig, der Index liefert
gleichmäßig.

**Einordnung, ehrlich abgegrenzt:** Das *reproduziert* Kernbefund 1 aus
Mandat I — es *bestätigt* ihn nicht unabhängig. Geteilt mit Mandat I sind
dieselben drei Rohdateien, die byte-gleich übernommene Truncation-Hygiene, die
bewusst 1:1 übernommene FIFO-/Kosten-/Reihenfolge-Mechanik (mit einem Test, der
die Gleichheit sogar *pinnt*) und dasselbe 12-1-Signal inklusive der
Turnover-Bremse aus H-012. Neu gebaut ist allein die Auswertungsschicht. Ein
solcher Aufbau kann Fehler auf Daten-, Universums- und Portfoliomechanik-Ebene
strukturell **nicht** entdecken — und dieser Review hat gezeigt, dass genau dort
welche saßen. Eine unabhängige Bestätigung bräuchte eine zweite Datenquelle.

## Befund 3 — Die GmbH trägt für diesen Kandidaten **(ins Gegenteil korrigiert)**

Die erste Fassung schrieb „der Vorsprung wird von den Rechtsformkosten
aufgefressen". Das war eine Folge des Dividenden-Fehlers. Korrigiert:

| Momentum, ~10.000 Trades | privat | GmbH |
|---|---|---|
| Steuer auf **Kursgewinne** | 123.827 | **29.528** (−76 %) |
| Steuer auf **Dividenden** | 34.653 | **50.546** (+46 %) |
| **Endvermögen** | 427.194 | **565.472** |

Der Vorsprung beträgt **+138.278 €** (+32 %). Selbst nach 22 Jahren
Rechtsformkosten (3.500 €/J = 77.000 €) bleiben **+61.000 €**. Die
Turnover-Asymmetrie wirkt genau wie vorhergesagt — Umschichten wird billig,
Dividenden werden teurer — und sie ist bei dieser Handelsfrequenz groß genug,
um die Struktur zu rechtfertigen.

**Aber:** Der Kandidat verliert trotzdem gegen SPY. Die GmbH macht einen
Verlierer weniger schlecht; sie macht ihn nicht zum Gewinner. Und der Benchmark
profitiert ebenfalls (611k → 654k), weil Fondsgewinne dort mit 11,57 % statt
18,46 % belastet werden.

## Befund 4 — Transaktionskosten sind bei Momentum eine eigene Größe

38.899 € Kosten bei `ZERO` (16.293 Trades) gegen 827 € beim Benchmark — fast
der gesamte Rückstand im steuerfreien Fall (46.262 €).

---

## Was daraus für die nächsten Phasen folgt

1. **P2 muss den Risikoteil zuerst lösen, nicht den Renditeteil.** Ohne
   Mechanismus gegen die zwei großen Krisen greift die Nebenbedingung vor jeder
   Rendite-Optimierung.
2. **Die GmbH-Hypothese ist bestätigt und trägt** — für Strategien mit hohem
   Turnover. Sie gehört ab jetzt in jeden Vergleich, mit Fixkosten.
3. **Hebel ist auf diesem Suchfenster wahrscheinlich tot**, weil er den
   Drawdown vergrößert, der schon ohne ihn doppelt über dem Deckel liegt. Er
   wird trotzdem gemessen — die Erwartung steht hier vorab, und der Mechanismus
   ist inzwischen per Test belegt (vorher war er wirkungslos, was den Sweep zu
   einer selbsterfüllenden Bestätigung gemacht hätte).
4. Fixkosten der GmbH gehören in jeden Struktur-Vergleich: 22 × 3.500 € =
   77.000 € sind bei 100.000 € Startkapital keine Fußnote.

**Offen und nicht behauptet:** 244 nicht ausführbare Aufträge im ZERO-Lauf
(Delisting-Randfälle) sind gezählt, aber nicht einzeln untersucht.

**Verbrauchte Trials:** Diagnostik-Lauf, noch nicht im Zähler registriert
(kein Kandidat gegen den Holdout gestellt). Ab der ersten registrierten
Hypothese in P2 zählt `TrialCounter`.
