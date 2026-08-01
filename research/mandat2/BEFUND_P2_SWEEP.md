# P2 Sweep — Haltedauer × Gewinnmitnahme × Hebel × Steuerwelt (2026-08-01)

72 Läufe: 12 Parameterkombinationen × 3 Steuerwelten, ausschließlich auf dem
SUCH-Fenster (1995–2016). Holdout unberührt. Trial-Zähler jetzt **1.988**.
Rohdaten: `results/p2_sweep.json`.

Achsen: `min_haltetage` ∈ {0, 90, 365, 730} · `rank_out` ∈ {30, 60, 200}
(200 ≈ „nie auf Rang verkaufen") · `hebel` ∈ {1,0; 1,5} mit Finanzierungskosten.

---

> ## ⚠ Befund A ist durch P3 WIDERLEGT
>
> Der Kontrolltest (`BEFUND_P3_KONTROLLE.md`) zeigt: eine **Zufallsauswahl** mit
> derselben Haltedisziplin erreicht Median **2,692** im Mittel über 20 Seeds —
> Momentum liegt mit 2,737 im **50. Perzentil** dieser Verteilung. Das Signal
> war austauschbar. Gemessen wurde nicht Momentum-Alpha, sondern der Effekt von
> 20 gleichgewichteten Namen, lange gehalten, gegen einen kapitalgewichteten
> Index. Zusätzlich hatten 17 von 20 Zufallsläufen einen **besseren** Drawdown.
>
> Befund B (das Optimum wandert nicht zwischen Steuerwelten) und Befund C
> (0 von 72 halten den DD-Deckel) sind davon **nicht** betroffen — sie hängen
> nicht an der Signalgüte.

## Befund A — Mein P1-Befund 2 war parametrisierungsabhängig. Korrektur. *(inzwischen selbst widerlegt, s. o.)*

P1 schrieb: *„Momentum verliert in ALLEN vier Steuerwelten, auch ohne Steuer."*
Das galt für **eine** Kombination (`hold0 / out60`) — die ich nie variiert habe.
Mit langer Haltedauer und laufengelassenen Gewinnern kippt es:

| Steuerwelt | beste Kombination | Median-Faktor | Benchmark | Endwert | Benchmark |
|---|---|---|---|---|---|
| `ZERO` | hold730 out200 ×1,0 | **2,737** | 1,948 | 1.090.312 | 726.197 |
| `PRIVAT_DE` | hold730 out200 ×1,0 | **2,168** | 1,870 | 620.869 | 610.752 |
| `GMBH+FK` | hold730 out200 ×1,0 | **2,095** | 1,862 | 635.135 | 580.266 |

**In allen drei Steuerwelten schlägt diese Parametrisierung den Index.** Dass
es an *Momentum* liegt, war der Fehlschluss — P3 zeigt, dass Rauschen dasselbe
leistet. Richtig bleibt: **wenn man aufhört zu churnen**, schlägt eine
konzentrierte, gleichgewichtete Auswahl den kapitalgewichteten Index. Ungehebelt, im steuerfreien Fall um 50 % Endvermögen (1,09 Mio gegen
726 k). Die Spannweite über das Gitter ist gewaltig: im `ZERO`-Fall von 278.203 €
(hold365/out30/×1,5) bis 1.773.450 € (hold730/out200/×1,5) — **Faktor 6,4 allein
durch die Parametrisierung**, bei identischem Signal und identischen Daten.

Das ist die direkte Antwort auf deinen Einwand, und du hattest recht: ich habe
die Stellschrauben gebaut und nicht gedreht.

## Befund B — Aber die Steuerwelt verschiebt das Optimum NICHT

Das war deine Hypothese: ohne Steuerdruck könne man ganz anders handeln. Die
Daten sagen nein.

Das Optimum liegt in **allen drei Welten bei derselben Kombination**:
`hold730 / out200`. Nur die Hebelwahl unterscheidet sich (ZERO und GmbH sähen
×1,5 besser, was aber am Drawdown scheitert).

| | optimale Haltedauer | optimales `rank_out` |
|---|---|---|
| `ZERO` (0 % Steuer) | 730 Tage | 200 |
| `PRIVAT_DE` (26,375 %) | 730 Tage | 200 |
| `GMBH+FK` (1,49 % + Fixkosten) | 730 Tage | 200 |

**Interpretation:** Die bindende Restriktion war nie die Steuer, sondern der
**Turnover selbst**. Häufiges Umschichten kostet in jeder Welt — durch
Transaktionskosten, durch das Verkaufen von Gewinnern vor ihrem Lauf, und in
Steuerwelten zusätzlich durch die Steuer. Wer die Steuer wegnimmt, ändert die
Höhe des Ergebnisses, nicht die Richtung der optimalen Strategie.

Konkret: `out30` (schnelles Rauswerfen) ist in **jeder** Welt die schlechteste
Wahl, `out200` in jeder die beste. `hold0` ist in jeder Welt schlechter als
`hold730`. Die Rangfolge der Parameter ist über die Steuerwelten stabil.

## Befund C — 0 von 72 Kombinationen halten den DD-Deckel

Bester Drawdown im gesamten Gitter: **−64,6 %** (PRIVAT_DE, hold730/out200/×1,0).
Der Deckel liegt bei −35 %.

Das verstärkt P1-Befund 1 erheblich: dort war es eine Kombination, jetzt sind es
72. **Keine reine Long-only-Aktienstrategie in diesem Gitter erfüllt die
Nebenbedingung** — auch die nicht, die den Index bei der Rendite klar schlägt.

Hebel verschlimmert es systematisch: ×1,5 verschiebt den Drawdown von −65 % auf
−82 bis −95 %, ohne den Median verlässlich zu heben. Die vorab notierte
Erwartung („Hebel ist auf diesem Fenster tot") ist damit gemessen — und diesmal
mit einem Mechanismus, der nachweislich funktioniert.

---

## Was das NICHT ist

Diese Zahlen sind **in-sample, best-of-12, auf dem Suchfenster**. Kein
Kandidat hat den Holdout gesehen. Vor einer Aussage „das schlägt SPY" fehlt:

1. **DSR/PBO** mit dem kumulierten Trial-Zähler (1.988). Bei 12 Kombinationen
   ist die beste per Konstruktion nach oben verzerrt.
2. **Der Holdout-Schuss** (2017–2026, enthält COVID und 2022).
3. ~~Eine Kontrollgruppe mit Zufallsauswahl.~~ **Gelaufen — siehe P3. Sie
   schlägt den Index genauso. Damit ist Befund A erledigt.**

## Nächster Schritt

Vor jeder weiteren Rendite-Optimierung: der Risikoteil. 72 von 72
Kombinationen scheitern am Drawdown, nicht an der Rendite. Ein Cash-/Timing-Gate
oder eine unkorrelierte Sleeve ist die einzige Achse, die die Nebenbedingung
überhaupt erreichen kann — und sie ist im Gitter noch gar nicht enthalten.
