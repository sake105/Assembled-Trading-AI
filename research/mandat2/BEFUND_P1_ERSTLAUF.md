# P1 Erstlauf — Befund (2026-08-01, **vierte Fassung**)

> **Regel ab dieser Fassung: keine Zahl in diesem Dokument, die nicht aus
> `results/p1_baseline.json` stammt.** Alle drei überlebenden Fehler — die
> „plausible" SPY-Rendite (E-073), die 77.000-€-Handrechnung (E-076) und die
> 22-statt-21-Jahre — waren im Fließtext gerechnet statt im Lauf gemessen.
>
> **Korrekturhinweis.** Dieses Dokument stand dreimal auf fehlerhaften Zahlen.
> Zwei Review-Runden fanden insgesamt **drei BLOCKER und zehn MAJORs**. Alle
> behoben. Was sich über alle drei Fassungen **nicht** geändert hat: Befund 1
> und Befund 2. Befund 3 hat sich in Fassung 2 ins Gegenteil gedreht und ist
> seitdem stabil. Die alten Zahlen stehen in der Git-Historie.
>
> Behoben in Runde 1: Holdout-Leck im Dividendenpanel (22.507 Zeilen aus
> 2017–2027 auf den letzten Suchtag gestapelt), nominale Dividenden auf
> adjustiertem Panel, Zielfunktion maß mark-to-market, fehlender
> Delisting-Zwangsverkauf, wirkungsloser Hebel, Rang-Gleichstände.
> Behoben in Runde 2: **Anker der Rohpfad-Rekonstruktion lag außerhalb des
> Suchfensters** (Dividenden dadurch noch immer +20 % überzeichnet, und Suche
> und Holdout liefen auf *unterschiedlichen* Skalen), **unsterbliche Staublots**
> (7.940 Phantom-Trades von 16.293), **latente Steuer regime-asymmetrisch**
> (~0,6 pp zugunsten der GmbH), **Ausschüttungsebene erreichte die Zielfunktion
> nicht** (vier von fünf Kennzahlen für `GMBH_AUSSCHUETTUNG` waren bitgleich die
> von `GMBH_THESAURIEREND`), `geliehen`-Reset auf zwei Pfaden übersprungen.
> Behoben in Runde 3 (Stage-3-Audit): **Befund 3 war durch die eigene Engine
> widerlegt** — die GmbH-Fixkosten waren im Text nominal subtrahiert statt
> gerechnet, der Zinseszins fehlte. Details: E-070 bis E-076 in
> `docs/CLAUDE_CODING_ERRORS.md`.
>
> **Abdeckung:** gelaufen ist **1 von 80+ P1-Familien** (12-1-Momentum), und
> zwar in **einer** von 12 Parametrisierungen — siehe `BEFUND_P2_SWEEP.md`.
> PLAN.md §5 definiert P1 als Re-Run aller Mandat-I-Familien — das steht aus.

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

## Befund 1 — Der DD-Deckel ist schärfer als der Benchmark. In jedem Fenster. *(Richtung unverändert, Zahlen aktualisiert)*

| | schlimmster MaxDD | bester MaxDD | Fenster über dem −35 %-Deckel |
|---|---|---|---|
| **SPY (Benchmark, ZERO)** | −55,2 % | −47,5 % | **144 / 144** |
| Momentum | −65,9 % … −67,2 % | — | 144 / 144 |

Jedes 10-Jahres-Fenster zwischen 1995 und 2016 enthält Dotcom **oder** die
Finanzkrise. SPY kommt in **keinem einzigen** Fenster mit weniger als −47,5 %
durch.

**Gemessen** ist: SPY reißt den Deckel in 144/144 Fenstern, Momentum ebenso.
**Gefolgert** — nicht gemessen — ist der Schluss, dass das für jeden immer voll
investierten Long-only-Kandidaten gilt: belegt sind zwei Kurven, nicht 80. Die
Folgerung ist plausibel, weil der Deckel am Index selbst scheitert und nicht an
der Auswahl, aber sie bleibt eine Folgerung.

**Konsequenz:** Wer den Deckel halten will, braucht zwingend
einen **risikoreduzierenden Bestandteil**: Cash-Quote, Timing-Gate, Absicherung
oder eine unkorrelierte Sleeve.

Das ist keine Aufweichung der Vorgabe. Hans hat den Deckel bewusst schärfer als
den Benchmark gesetzt. Der Lauf zeigt nur, **wie viel** schärfer: verlangt ist
eine Halbierung des Krisen-Drawdowns gegenüber dem Index, nicht eine
Verbesserung am Rand. Ab P2 ist gesucht: „bessere Auswahl **plus** ein
Mechanismus, der durch 2000–2002 und 2008–2009 trägt".

## Befund 2 — ~~Momentum verliert in ALLEN vier Steuerwelten~~ **ÜBERHOLT durch P2**

> **Diese Aussage war parametrisierungsabhängig und ist so falsch.** Sie gilt
> für die hier gefahrene Kombination (`hold0 / out60`), nicht für Momentum als
> Familie. Der P2-Sweep zeigt: mit `hold730 / out200` schlägt dieselbe Strategie
> den Index in **allen drei** getesteten Steuerwelten (ZERO 2,737 gegen 1,948;
> PRIVAT_DE 2,168 gegen 1,870; GMBH+FK 2,095 gegen 1,862). Die Spannweite über
> das Parametergitter beträgt Faktor 6,4 bei identischem Signal.
> Siehe `BEFUND_P2_SWEEP.md`. Die Zahlen unten bleiben als Referenzpunkt für
> genau eine Zelle des Gitters stehen.

| Regime | Momentum | SPY | Median-Faktor Kandidat vs Benchmark | Fenster geschlagen |
|---|---|---|---|---|
| `ZERO` | 679.935 | 726.197 | 1,250× vs 1,948× | 41 % |
| `PRIVAT_DE` | 430.623 | 610.752 | 1,091× vs 1,870× | 38 % |
| `GMBH_THESAURIEREND` | 572.365 | 653.766 | 1,151× vs 1,900× | 40 % |
| `GMBH_AUSSCHUETTUNG` | 447.779 | 507.710 | 1,137× vs 1,786× | 40 % |

**Für diese eine Parametrisierung** verliert Momentum auch bei null Steuer. Der
Schluss „die Steuer war nicht die Ursache" bleibt richtig — die Ursache war
aber, wie P2 zeigt, der **Turnover**, und der ist eine Stellschraube, keine
Eigenschaft der Familie. Was P2 zusätzlich zeigt: das Optimum liegt in allen
Steuerwelten bei denselben Parametern, die Steuer verschiebt es nicht.

**Einordnung, ehrlich abgegrenzt:** Das *reproduziert* Kernbefund 1 aus
Mandat I — es *bestätigt* ihn nicht unabhängig. Geteilt mit Mandat I sind
dieselben drei Rohdateien, die byte-gleich übernommene Truncation-Hygiene, die
bewusst 1:1 übernommene FIFO-/Kosten-/Reihenfolge-Mechanik (mit einem Test, der
die Gleichheit sogar *pinnt*) und dasselbe 12-1-Signal inklusive der
Turnover-Bremse aus H-012. Neu gebaut ist allein die Auswertungsschicht. Ein
solcher Aufbau kann Fehler auf Daten-, Universums- und Portfoliomechanik-Ebene
strukturell **nicht** entdecken — und dieser Review hat gezeigt, dass genau dort
welche saßen. Eine unabhängige Bestätigung bräuchte eine zweite Datenquelle.

## Befund 3 — Turnover-Asymmetrie real, GmbH-Struktur bei 100.000 € trotzdem nicht tragfähig

**Die Steuerasymmetrie wirkt exakt wie vorhergesagt** (Momentum, ~5.930 Trades):

| | privat | GmbH |
|---|---|---|
| Steuer auf **Kursgewinne** | 124.919 | **29.799** (−76 %) |
| Steuer auf **Dividenden** | 31.254 | **45.659** (+46 %) |
| Endvermögen *ohne* Rechtsformkosten | 430.623 | **572.365** (+141.742) |

**Aber die Rechtsformkosten fressen den gesamten Vorsprung** — gemessen, nicht
gerechnet (`fixkosten_pa = 3.500`, Lauf `GMBH_THESAURIEREND+fixkosten`):

| | privat | GmbH + Fixkosten |
|---|---|---|
| **Endvermögen** | 430.623 | **434.702** |
| **Median-Faktor (die gesperrte Zielfunktion)** | **1,0910** | **1,0524** |
| Benchmark-Endvermögen | 610.752 | 580.266 |

Der Vorsprung schrumpft von +141.742 € auf **+4.079 €** (+0,9 %). Und auf der
Zielfunktion, die über das Verdikt entscheidet, ist die GmbH **schlechter** als
privat.

**Warum die vorige Fassung hier falsch lag:** Sie subtrahierte 77.000 € nominal
im Fließtext. Real gemessen kosten 73.500 € eingezahlte Fixkosten (21
Jahreswechsel, nicht 22) über die Laufzeit **137.663 €** Endvermögen — die
entgangene Verzinsung fehlte, und sie ist fast so groß wie der Nominalbetrag
noch einmal. Der Parameter existierte, er wurde nur nicht gesetzt. PLAN.md §2
verlangt genau das wörtlich: *„für die Frage ‚GmbH oder privat?' muss er
gesetzt werden, sonst ist das Ergebnis geschönt."*

**Was das heißt:** 3.500 €/Jahr sind bei 100.000 € Startkapital 3,5 % p. a. —
mehr als der gesamte Steuervorteil auf realisierte Gewinne. Die GmbH-Frage ist
eine **Skalenfrage**, keine Strategiefrage. Ab welchem Kapital sie kippt, ist
offen und als eigener Sweep notiert.

**Unabhängig davon:** Der Kandidat verliert in jeder Variante gegen SPY. Der
Benchmark profitiert in der GmbH ohne Fixkosten ebenfalls (611k → 654k), weil
Fondsgewinne dort mit 11,57 % statt 18,46 % belastet werden — mit Fixkosten
fällt auch er zurück (580k).

## Befund 4 — Transaktionskosten sind bei Momentum eine eigene Größe

38.899 € Kosten bei `ZERO` (**5.927** Trades) gegen 827 € beim Benchmark — fast
der gesamte Rückstand im steuerfreien Fall (46.262 €).

> Die Fassung 2 nannte hier 16.293 Trades. Davon waren ~7.940 Phantom-Trades
> aus einer nicht terminierenden Zwangsverkaufs-Schleife auf vier Staublots
> (E-075). Die Kostenzahl war davon unberührt (Staublots erzeugen keine Kosten),
> die Trade-Zahl nicht.

**Ursachen-Zuordnung, korrigiert:** Der Rückgang von 701.760 (Fassung 1) auf
679.935 stammt **nicht** aus dem Delisting-Zwangsverkauf — der bewegt 7 €. Er
stammt fast vollständig aus dem neuen Verkaufs-Fallback auf den letzten gültigen
Kurs: vorher wurden Verkäufe toter Namen dauerhaft verschluckt und die Position
zum eingefrorenen Kurs weitergeführt. Fachlich richtig, aber in Fassung 2 falsch
etikettiert — wer die Zahl zurückverfolgt, hätte an der falschen Stelle gesucht.

---

## Was daraus für die nächsten Phasen folgt

1. **P2 muss den Risikoteil zuerst lösen, nicht den Renditeteil.** Ohne
   Mechanismus gegen die zwei großen Krisen greift die Nebenbedingung vor jeder
   Rendite-Optimierung.
2. **Die GmbH-Steuerasymmetrie ist bestätigt, die GmbH-STRUKTUR bei 100.000 €
   nicht.** Sie gehört ab jetzt nur noch **mit** Fixkosten in Vergleiche. Offen
   und als eigener Sweep notiert: ab welchem Startkapital sie kippt.
3. **Hebel ist auf diesem Suchfenster wahrscheinlich tot**, weil er den
   Drawdown vergrößert, der schon ohne ihn doppelt über dem Deckel liegt. Er
   wird trotzdem gemessen — die Erwartung steht hier vorab, und der Mechanismus
   ist inzwischen per Test belegt (vorher war er wirkungslos, was den Sweep zu
   einer selbsterfüllenden Bestätigung gemacht hätte).
4. Fixkosten der GmbH gehören in jeden Struktur-Vergleich: 22 × 3.500 € =
   77.000 € sind bei 100.000 € Startkapital keine Fußnote.

**Offen und nicht behauptet:**
- 244 nicht ausführbare Aufträge im ZERO-Lauf (Delisting-Randfälle) sind
  gezählt, aber nicht einzeln untersucht.
- Die Trade-Zahl ist regimeabhängig (5.922–5.933).
- Bei `GMBH_AUSSCHUETTUNG` rechnet die latente Ausschüttungsebene gegen die
  **ursprüngliche Einlage**, nicht gegen den Fensterstart. Vertretbare
  Modellierung, aber sie erklärt einen Teil des Median-Unterschieds.

**Verbrauchte Trials:** Diagnostik-Lauf, noch nicht im Zähler registriert
(kein Kandidat gegen den Holdout gestellt). Ab der ersten registrierten
Hypothese in P2 zählt `TrialCounter`.
