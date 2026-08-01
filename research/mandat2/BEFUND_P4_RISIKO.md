# P4 Risiko-Sleeve — der erste Kandidat, der die Nebenbedingung erfüllt (2026-08-01)

Rohdaten: `results/p4_risiko.json`, `results/p3_kontrolle_gate.json`.
Trial-Zähler **2.052**. Suchfenster, Holdout unberührt.

Getestet wurde der **einfachste** Mechanismus, der den Drawdown adressieren
kann: ein SMA-Trendfilter auf den Index. Risk-off = alles in Cash, ausgewertet
am Monatsende, PIT-sauber (die SMA am Tag t nutzt nur Kurse bis t).

---

## Befund A — Der Deckel ist erreichbar. Aber es ist das Gate, nicht die Auswahl.

**Kontrolle zuerst: der reine Index mit Gate.**

| SPY | schlimmster MaxDD | gerissene Fenster | Median | Endwert |
|---|---|---|---|---|
| ohne Gate | −55,2 % | 144/144 | 1,948 | 726.197 |
| **+ SMA200** | **−19,2 %** | **0/144** | **2,525** | 810.867 |
| + SMA100 | −31,9 % | 0/144 | 1,668 | 463.576 |
| + SMA300 | −24,2 % | 0/144 | 1,964 | 543.414 |

**Der gefilterte Index allein hält den Deckel — und schlägt den ungefilterten
Index zugleich.** Damit ist P1-Befund 1 („kein Long-only-Kandidat kann die
Nebenbedingung erfüllen") in seiner allgemeinen Form widerlegt: er galt für
*immer voll investierte* Strategien. Mit einer Cash-Option ist der Deckel
erreichbar.

**Und die Strategie darüber:**

| ZERO | Endwert | Median | Benchmark | MaxDD | Verdikt |
|---|---|---|---|---|---|
| 20 Namen, ohne Gate | 1.090.312 | 2,737 | 1,948 | −65,5 % | durchgefallen |
| **20 Namen + SMA200** | **2.656.374** | **4,124** | 1,948 | **−30,5 %** | **BESTANDEN** |

6 von 12 Kombinationen bestehen (alle drei Steuerwelten mit SMA200; PRIVAT_DE
zusätzlich mit SMA100 und SMA300).

## Befund B — Mit Gate überlebt Momentum die Zufallskontrolle. Ohne Gate nicht.

Das ist das eigentliche Ergebnis dieser Runde. Dieselbe Kontrolle wie in P3,
20 Seeds, nur diesmal mit dem SMA200-Gate:

| | Median | Perzentil in der Zufallsverteilung |
|---|---|---|
| Momentum **ohne** Gate | 2,737 | **50.** → Artefakt |
| Momentum **mit** Gate | **4,124** | **100.** → überlebt |
| Zufall mit Gate (20 Seeds) | Mittel 2,866 · P95 3,437 · Max 3,533 | — |

Ohne Gate war das Signal austauschbar. Mit Gate liegt Momentum **über allen
zwanzig** Zufallsläufen. Das ist eine Interaktion, keine Addition: der
Trendfilter entfernt die Crash-Phasen, und in den verbleibenden Risk-on-Phasen
diskriminiert das Momentum-Signal offenbar tatsächlich.

Nebenbefund: die Zufallsläufe mit Gate haben Drawdowns von −19,8 bis −28,2 % —
also **besser** als Momentums −30,5 %. Das Signal kauft Rendite mit Risiko, es
verschenkt sie nicht.

---

## Was dagegen spricht — und es ist einiges

**1. Die Fensterwahl ist nicht robust.** SMA200 besteht in allen drei
Steuerwelten, SMA100 reißt in ZERO und GmbH 64 bzw. 64 Fenster, SMA300 reißt in
GmbH 61 Fenster. **Drei Werte getestet, der mittlere gewinnt.** Das ist genau
das Muster, das man bei einem gefundenen Parameter erwartet — nicht bei einem
Mechanismus. Ein Mechanismus sollte über 100/200/300 monoton oder wenigstens
stabil sein; das ist er nicht.

**2. Das Suchfenster ist für Trendfilter maximal günstig.** 1995–2016 enthält
zwei lange, angekündigte Bärenmärkte (2000–2002, 2007–2009), in denen ein
SMA-Filter genau einmal aussteigt und lange draußen bleibt. Der Ruf des
SMA200-Gates ruht historisch fast ausschließlich auf diesen beiden Episoden.

**3. Der Holdout enthält den Gegentest.** 2017–2026 enthält den COVID-Crash —
einen V-förmigen Einbruch, bei dem Trendfilter typischerweise verspätet
aussteigen und verspätet zurückkommen (Whipsaw). Das ist der Fall, in dem der
Mechanismus scheitern *sollte*, wenn er nur ein Artefakt der zwei langsamen
Bärenmärkte ist.

**4. Trial-Zähler 2.052.** Bei dieser Zahl an Versuchen ist ein
in-sample-Bestehen wenig wert, bis DSR/PBO gerechnet sind.

**5. Der Fehler in der ersten Fassung.** Meine SPY-Kontrollzeile war zunächst
kaputt (−90 % Drawdown für SPY allein — unmöglich). Ursache: SPY ist ein ETF und
steht nicht in der S&P-Mitgliederliste; mein Konstrukt hat die Auswahl
stillschweigend verworfen und eine beliebige Aktie gehalten. Repariert und neu
gerechnet. Genannt, weil es zeigt, wie leicht eine Kontrollzeile falsch sein
kann, ohne dass es auffällt.

---

## Status und nächster Schritt

**Das ist der erste Kandidat der Kampagne, der die gesperrte Zielfunktion
in-sample erfüllt.** Er ist damit ein Kandidat für den Holdout-Schuss — aber
noch nicht reif dafür.

Vorher zwingend:
1. **DSR/PBO** mit N = 2.052.
2. **Robustheitsprüfung der Fensterwahl** — feineres Raster (SMA 120…260) und
   die Frage, ob ein *Band* funktioniert oder nur ein Punkt.
3. **Kostenrealismus des Gates**: Ein- und Ausstiege sind Komplettumschichtungen
   des Depots. Bei den vorhandenen 10 bps ist das enthalten, aber Slippage in
   Krisenphasen ist es nicht.

Erst danach der eine Schuss auf 2017–2026. Der Holdout ist genau deshalb
gesperrt worden — und er ist mit dem COVID-Crash der härtestmögliche Test für
genau diesen Mechanismus.
