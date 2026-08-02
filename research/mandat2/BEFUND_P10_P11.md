# P10 + P11 — GmbH-Break-even und der richtige Maßstab (2026-08-02)

Zwei Fragen, die ich mir selbst offengelassen hatte. Beide beantwortet, beide
mit Konsequenzen. Trial-Zähler **2.200**. Rohdaten in `results/`.

---

# P10 — Ab welchem Kapital trägt die vermögensverwaltende GmbH?

Die Antwort ist eine Skalen-Antwort, und sie fällt günstiger aus als der
100.000-€-Einzelfall vermuten ließ.

## Der Kandidat (aktiv gehandelt, ~5.900 Trades)

| Startkapital | privat | GmbH (FK 3.500/J) | Differenz |
|---|---|---|---|
| 100.000 | 620.869 | 635.135 | **+14.266 (+2,3 %)** |
| 250.000 | 1.460.543 | 1.982.800 | +522.257 (+35,8 %) |
| 500.000 | 3.014.556 | 4.412.672 | +1.398.116 (+46,4 %) |
| 1.000.000 | 6.156.563 | 8.864.995 | +2.708.432 (+44,0 %) |
| 5.000.000 | 28.533.832 | 46.664.985 | +18.131.153 (+63,5 %) |

## Der reine ETF-Sparer (1 Trade)

| Startkapital | privat | GmbH (FK 3.500/J) | Differenz |
|---|---|---|---|
| 100.000 | 610.752 | 580.266 | **−30.486 (−5,0 %)** |
| 250.000 | 1.526.602 | 1.560.914 | +34.312 (+2,2 %) |
| 500.000 | 3.053.019 | 3.195.329 | +142.310 (+4,7 %) |
| 1.000.000 | 6.105.854 | 6.464.158 | +358.304 (+5,9 %) |
| 5.000.000 | 30.528.532 | 32.614.789 | +2.086.257 (+6,8 %) |

## Break-even

| Rechtsformkosten | aktiver Kandidat | reiner ETF |
|---|---|---|
| 2.000 €/J | ab 100.000 € | ab 100.000 € |
| **3.500 €/J** | **ab 100.000 €** (knapp) | **ab 250.000 €** |
| 5.000 €/J | ab 250.000 € | ab 250.000 € |

**Zwei klare Muster:**

1. **Je mehr gehandelt wird, desto früher lohnt die GmbH.** Beim aktiven
   Kandidaten liegt der Vorteil ab 250.000 € bei 36–64 %, beim reinen
   ETF-Sparer nie über 7 %. Das ist konsistent: der Steuervorteil greift auf
   *realisierte* Gewinne, und wer nie realisiert, hat nichts davon.
2. **Bei 100.000 € und realistischen Kosten ist es ein Nullsummenspiel** —
   +2,3 % beim Kandidaten, −5,0 % beim ETF. Für einen Buy-and-Hold-Sparer mit
   sechsstelligem Vermögen ist die GmbH bei 3.500 €/J **schlechter** als das
   Privatdepot.

**Wichtige Korrektur an meiner früheren Zahl:** Ich hatte „+4.079 €" berichtet.
Das galt für die *P1-Parametrisierung* (hold0/out60), nicht für den
P2-Gewinner. Mit hold730/out200 sind es +14.266 €. Beide Zahlen sind für ihren
Parametersatz korrekt; ich hätte sie nicht ohne Angabe des Parametersatzes
nebeneinanderstellen dürfen.

---

# P11 — Der Maßstab war falsch. Von Anfang an.

Ich hatte in P3 notiert, der Zufallsbefund müsse „gegen einen gleichgewichteten
Index-Benchmark statt gegen SPY" geprüft werden. Das Ergebnis:

| | Endwert | Median | MaxDD |
|---|---|---|---|
| **SPY** (kapitalgewichtet) | 726.197 | 1,948 | −55,2 % |
| **EW-Index** (gleichgewichtet, breit) | **1.263.916** | **2,594** | −58,6 % |

**Der gleichgewichtete Index allein schlägt SPY um Faktor 1,33 im Median.**
Ohne jede Auswahl, ohne Signal, ohne Timing.

Und damit verschwindet der Zufallsbefund aus P3:

| 20 Zufallsnamen | Median-Verhältnis |
|---|---|
| gegen SPY | **1,40×** |
| gegen EW-Index | **1,05×** |

Zwei von acht Seeds liegen sogar *unter* dem EW-Index. **Der P3-Effekt war die
Gleichgewichtung** — eine bekannte Faktorexposition, kein Befund.

## Was das für den Hauptkandidaten heißt

| Momentum + SMA140-Gate | Median-Verhältnis |
|---|---|
| gegen SPY | 3,43× |
| **gegen EW-Index** | **2,57×** |

Der Vorsprung schrumpft um rund ein Viertel, bleibt aber deutlich. Der
Kandidat lebt also **nicht** allein von der Gewichtung — anders als die
Zufallsauswahl.

**Aber:** das ändert nichts am Verdikt. Der Kandidat ist bereits an DSR
gescheitert (P8) und an der Dünnheit seines Mechanismus (P9, 12–18
Entscheidungen). Ein kleinerer Vorsprung gegen den richtigen Maßstab macht ihn
nicht besser.

## Konsequenz für die Kampagne

**Der gesamte P1–P8-Komplex hat gegen einen ungeeigneten Benchmark gemessen.**
SPY ist kapitalgewichtet, alle Kandidaten sind gleichgewichtet — ein Drittel des
gemessenen „Vorsprungs" war reine Gewichtung. Das entwertet die Verdicts nicht
(sie waren negativ, und gegen einen *härteren* Maßstab bleiben sie es erst
recht), aber es hätte in die andere Richtung fatal sein können.

**Regel für künftige Kampagnen:** Der Benchmark muss dieselbe
Gewichtungsmethode haben wie der Kandidat. Alles andere misst
Faktorexposition und nennt sie Alpha.

---

## Zwei eigene Fehler in diesem Lauf

1. **`sma_gate(d.close, 140)`** — `140` landete im Parameter `symbol` statt in
   `fenster`, weil die Signatur `(close, symbol="SPY", fenster=200)` lautet.
   Der Lauf brach mit `KeyError: 140` ab. Positionsargumente bei Funktionen mit
   optionalen Strings davor sind eine Falle; im ganzen Modul auf
   Schlüsselwort-Aufruf umgestellt.
2. **Die +4.079-€-Zahl** aus dem ABSCHLUSS stammte aus einem anderen
   Parametersatz als der P2-Gewinner (s. o.). Korrigiert.
