# Befund P13 — SPY mit Trendfilter

> Erzeugt von `render_befund_p13.py` aus den Artefakten in `results/`. Nicht von Hand bearbeiten.
>
> **Reichweite dieser Zusicherung:** Alle Ergebniszahlen dieses Strangs — Tabellen, Mediane, Drawdowns, Bandbreiten, p-Werte, DSR/PBO — stammen aus `results/p13*.json` und werden von Tests gegen die Artefakte nachgerechnet. **Zitate aus anderen Läufen** (P5, P8, Befund 6 und 7: Survivorship-Spanne, Entscheidungsmarge, Zahl recycelter Spalten, SPY-Abdeckung, die 0,9512 des Aktien-Kandidaten) sind übernommen, nicht hier neu gerechnet — sie stehen in den dort genannten Dokumenten. Der erste Entwurf behauptete pauschal „jede Zahl hier hat einen Schlüssel dort“ und deckte damit genau die Stelle zu, an der real gedriftet wurde (E-122).

## Warum dieser Strang die Datenkritik überlebt

Die Befunde 6 und 7 des Mandats haben die Datenbasis für Vergleiche gegen einen Index unbrauchbar gemacht: Survivorship 2,36–2,90 pp p. a. bei 1,5 pp Entscheidungsmarge, dazu Ticker-Recycling in 29 Spalten. Das trifft jede Strategie, die **Namen auswählt**.

Hier steht auf beiden Seiten derselbe Basiswert: SPY mit Filter gegen SPY ohne. Keine Auswahl, kein Survivorship, kein Recycling, keine Gewichtungsfrage. Die SPY-Serie ist geprüft sauber (99,8 % Abdeckung im Suchfenster, kein Skalenbruch, größte Lücke zwei Handelstage).

## Was geprüft wurde

Dieselbe Mühle, an der der Aktien-Kandidat in P5 gescheitert ist: 12 Fensterwerte × 3 Trend-Definitionen × 3 Steuerwelten, einmal ohne und einmal mit einem Handelstag Ausführungsverzögerung. Bestehen heißt: Median über alle 144 rollierenden 10-Jahres-Fenster **über** dem von Buy-and-Hold **und** der Drawdown-Deckel von −35 % in **keinem** Fenster gerissen.

| Steuerwelt | Definition | besteht ohne Verz. | mit Verz. | in **beiden** | längste lückenlose Kette (beide) |
|---|---|---:|---:|---:|---|
| ZERO | `preis>sma` | 10/12 | 10/12 | **9/12** | 160–320 (9) |
| ZERO | `sma steigt` | 11/12 | 11/12 | **11/12** | 120–320 (11) |
| ZERO | `rendite>0` | 10/12 | 11/12 | **9/12** | 140–200 (4) |
| PRIVAT_DE | `preis>sma` | 6/12 | 8/12 | **5/12** | 160–240 (5) |
| PRIVAT_DE | `sma steigt` | 8/12 | 9/12 | **7/12** | 240–280 (3) |
| PRIVAT_DE | `rendite>0` | 8/12 | 7/12 | **7/12** | 260–320 (4) |
| GMBH+FK | `preis>sma` | 2/12 | 0/12 | **0/12** | — |
| GMBH+FK | `sma steigt` | 2/12 | 3/12 | **2/12** | 140 |
| GMBH+FK | `rendite>0` | 0/12 | 0/12 | **0/12** | — |

Der Aktien-Kandidat kam in P5 auf 5 und 3 von 12 Fenstern, lückig. In der steuerfreien Welt bestehen hier `preis>sma` 9/12 (längste Kette 9), `sma steigt` 11/12 (längste Kette 11), `rendite>0` 9/12 (längste Kette 4). 2 der 3 Definitionen tragen ein lückenloses Band über mindestens 8 der 12 Fensterwerte. Ein gefundener Parameter sieht anders aus — aber `rendite>0` zeigt, dass die Breite nicht in jeder Definition auch zusammenhängend ist.

## Die Ausführungsannahme trägt das Ergebnis nicht

Alle drei Gates entscheiden auf `close[t]`, und die Engine handelt am selben `close[t]` — kein Blick in die Zukunft, aber die optimistischste zulässige Annahme: der Ausstieg gelingt zu genau dem Kurs, der ihn ausgelöst hat. Mit einem Handelstag Verzögerung bleibt das Bild bestehen (Spalte „mit Verz.“ oben); in `PRIVAT_DE`/`preis>sma` wird das Band sogar breiter. Der Vorsprung stammt also nicht aus der Ausführung.

Eine Eigenheit gehört dazu genannt: Alle drei Definitionen bilden `(a > b).astype(float)`, und NaN-Vergleiche ergeben False — die Warmlaufphase ist **risk-off**, nicht neutral. Jede gegatete Variante startet also in Cash, und zwar umso länger, je größer das Fenster ist: Fenster 100 liefert ab 1995-05-24 ein Signal, Fenster 320 erst ab 1996-04-08 — **10.5 Monate** später. Das Fensterraster konfundiert damit Signallänge mit anfänglicher Marktabwesenheit. Die Richtung ist konservativ (1995 ff. war stark, wer später einsteigt, verliert), die großen Fenster sind also eher benachteiligt.

## Der Einwand, der bleibt: zwei Ereignisse, nicht 144 Fenster

Der Suchzeitraum 1995–2016 enthält zwei Bärenmärkte. Ein rollierendes 10-Jahres-Fenster darin startet zwischen 1995 und 2006 und trifft damit zwangsläufig mindestens einen. Gezählt statt vermutet:

* Benchmark-MaxDD je Fenster: schlimmster **-55.2%**, Median -55.2%, **mildester -47.5%**
* Fenster ohne Rückgang von mindestens 30%: **0**
* Der Kandidat gewinnt 140 von 144 Fenstern, Median-Vorsprung +57.7 pp

**Kein einziges krisenfreies Fenster.** Die Stichprobe kann „Trendfolge wirkt“ nicht von „Trendfolge hat diese beiden Abstürze umgangen“ unterscheiden. Die 144 Fenster überlappen zudem massiv — jeder Handelstag steckt in bis zu 120 von ihnen. Die effektive Stichprobe für den Mechanismus sind zwei Ereignisse.

## Was das Timing wert ist (Kontrollgruppe)

Ein Gate nimmt Zeit aus dem Markt **und** wählt wann. Die Kontrolle trennt beides: die Folge an/aus bleibt unverändert, gemischt werden nur die **Blocklängen innerhalb ihrer Wertklasse**. Auf **Signalebene** exakt erhalten bleiben damit der An-Anteil (65.2%), die Zahl der Blöcke (141, inklusive Warmlauf) und deren Längenverteilung; verändert wird nur, **wann** die langen und kurzen Episoden liegen. 60 Ziehungen, Parameter a priori `preis>SMA200` statt der besten Rasterzelle.

| | Median über 144 Fenster |
|---|---:|
| echter Filter | **2.525x** |
| Zufalls-Timing, Median | 1.353x |
| Zufalls-Timing, p95 | 1.886x |
| Zufalls-Timing, bestes von 60 | 2.137x |
| Buy-and-Hold | 1.948x |

Auf **Portfolioebene** ist die Erhaltung nur näherungsweise, und das ist gemessen statt angenommen: die Engine liest das Gate nur an Monatsenden, zwischen Signal und Wirkung liegt also ein Sampling-Schritt. Realisiert investiert war der echte Filter an 66.3% der Tage, die Zufallsläufe an 62.4%–67.3% (Median 65.1%). Der echte Filter ist damit **mehr** im Markt als der typische Zufallslauf — sein Vorsprung stammt nicht aus zusätzlicher Abwesenheit.

Auf der **buchenden** Ebene ist die Kontrolle sogar im Nachteil, und auch das ist gemessen: an den Monatsenden schaltet das echte Gate 18-mal, die gemischten 19- bis 37-mal (Median 27). Jede Schaltung kostet `cost_bps`, auch in der steuerfreien Welt — die Kontrollgruppe trägt also mehr Kostendrag als der Kandidat. Der Abstand unten ist zu groß, als dass das ihn erklären könnte, aber das ist eine Abschätzung und keine Bereinigung: es ist der einzige bekannte Effekt, der **zugunsten** des Kandidaten wirkt.

0 von 60 Zufallsläufen erreichen den echten Filter (**p = 0.016**), 1/60 bestehen die Zielfunktion. Der Zufallsmedian liegt **unter** Buy-and-Hold: zu zufälligen Zeiten auszusetzen kostet. Das Timing trägt also Information — innerhalb dieser Stichprobe.

## Die Mehrfachtest-Korrektur — und daran scheitert er, zweimal

Familienmatrix: 37 Varianten (3 Definitionen × 12 Fenster + ungegatet), N = 3529 (kumulierter Trial-Zähler beider Mandate). Die Entscheidungsregel stand vor dem Lauf in `p8_dsr_heterogen.py` fest: heterogen geschätztes V, kumuliertes N, und PBO unter 50 %.

| Varianzschätzer | Schwelle | p | | |
|---|---:|---:|---|---|
| heterogen | 0.0415 | 0.7838 | ❌ | **Entscheidungsgrundlage** |
| IID-Naeherung | 0.0483 | 0.6105 | ❌ | konservative Gegenprobe |
| klonfamilie | 0.0140 | 0.9974 | ✅ | nicht entscheidungsfähig (E-077) |
| PBO (CSCV, 8 Blöcke, 70 Splits) | — | 68.6% | ❌ | rangiert nach Sharpe, nicht nach dem Zielmaß |

Beobachteter Sharpe 0.0522. Die Klonfamilien-Varianz (1.503e-05) ist 8.8-mal kleiner als die heterogene aus P8 (1.328e-04, 30 Strategien) — sie senkt die Schwelle von 0.0415 auf 0.0140 und macht aus einem Fehlschlag ein Bestehen. Genau diese Konstruktion ist im Repo als **E-077** protokolliert; der erste Entwurf dieses Moduls war eine Kopie des dort verworfenen `p7_dsr_pbo.py` und hätte den Fehler in einen neuen Befund verlängert.

**Beide Korrekturen sind gerissen.** DSR p = 0.7838 gegen die 0,95-Schwelle, und PBO 68.6%: in mehr als der Hälfte der 70 Aufteilungen landet die in-sample beste Konfiguration out-of-sample unter dem Median der Familie. Welches Fenster das beste ist, ist über die Zeit nicht stabil.

Drei Einordnungen, keine davon entlastend:

* **Der Vergleich mit dem Aktien-Kandidaten fällt zu Ungunsten dieses Kandidaten aus.** Auf demselben Varianzschätzer kam jener auf 0,9512 (bestanden, aber Münzwurf am Rand) und scheiterte erst an der IID-Gegenprobe; dieser liegt mit 0.7838 deutlich darunter. Das N ist dabei nicht dasselbe — jener wurde gegen N = 2.144 gemessen, dieser gegen N = 3529; ein Teil der Differenz ist also Zählerwachstum und nicht Kandidatenqualität — die 0,95-Schwelle verfehlt dieser Kandidat aber deutlich, nicht knapp. Eine frühere Fassung dieses Befunds stellte 0,9974 aus der Klonfamilie neben jene 0,9512 aus der heterogenen Familie und leitete daraus eine „Symmetrie“ ab — zwei verschiedene Schätzer, verglichen zugunsten des eigenen Kandidaten.
* **PBO ist hier zusätzlich nach unten verzerrt.** E-077 hält fest, dass CSCV heterogene Spalten voraussetzt und bei Fast-Klonen zu niedrige Werte liefert. Ein Wert von 68.6% unter dieser Verzerrung ist ein deutlicherer Fehlschlag, als die Zahl nahelegt.
* **In-sample-Gewinner und a-priori-Parameter fallen zusammen (`preis>sma/200`).** Das stützt, dass hier kein Parameter gefunden wurde — ändert aber nichts, denn beide Korrekturen bewerten die Suche, und gesucht wurde nachweislich (3529 Trials).

Die Matrix zu verkleinern oder die Klonvarianz zu behalten, bis die Zahlen passen, wäre genau die Manipulation, vor der E-077 warnt. Die Regel stand vor dem Lauf: **alle Kriterien oder keines.**

## Was hier nicht behauptet wird

* **Kein Holdout.** Alles oben liegt im Suchfenster bis 2016-12-30. Der Zeitraum 2017-01 bis 2026-07 ist unangetastet und bleibt es, bis darüber entschieden wird.
* **Trials: 216 für P13/P13b, kampagnenweit 3529** (inklusive der 1964 aus Mandat I, die der Zähler bewusst nicht zurücksetzt). Der Wert stammt aus dem Korrektur-Artefakt, nicht aus dem Live-Zähler — ein Dokument, das aus Artefakten reproduzierbar sein soll, darf keinen laufenden Zustand zitieren. P13c, P13d und P13e zählen nicht mit: Zerlegung, Kontrollgruppe und Korrektur sind keine Suche (E-090).
* **Ein Markt, ein Instrument, ein Vierteljahrhundert.** Aus einem US-Index-ETF von 1995 bis 2016 folgt nichts über andere Märkte oder andere Regime.
* **Die 144 Fenster sind keine 144 Beobachtungen.** Der p-Wert der Kontrollgruppe bezieht sich auf die Timing-Frage, nicht auf die Frage, ob der Mechanismus außerhalb dieser Stichprobe existiert.
* **Bestehen heißt nicht „schlägt SPY im Endwert“.** Es heißt höherer Median über rollierende Fenster bei eingehaltenem Drawdown-Deckel — Buy-and-Hold reißt diesen Deckel in 144 von 144 Fenstern.

## Stand: kein Holdout-Schuss

Der Filter übersteht die P5-Mühle als erster Kandidat der Kampagne, auf der einzigen Datenbasis, die die Befunde 6 und 7 nicht entwerten, und keine der billigen Widerlegungen greift. Er scheitert trotzdem an beiden Hälften der Mehrfachtest-Korrektur — DSR 0.7838 und PBO 68.6%. Damit hat **kein** Kandidat dieser Kampagne die Korrektur vollständig bestanden.

Zwei Dinge bleiben verwertbar:

* **Die GmbH-Frage ist beantwortet** (bestätigt Befund 3 aus anderer Richtung): Ein Filter, der über das Raster 6 bis 62 Buchungen erzeugt, trägt bei diesem Kapitaleinsatz die Fixkosten der Rechtsform nicht — 0/12 bis 2/12 bestandene Fenster.
* **Die Stichprobe selbst ist die Grenze.** Selbst wenn beide Korrekturen bestanden hätten, wäre der Mechanismus nicht von zwei Ereignissen zu trennen gewesen. Ein besserer Test bräuchte Marktdaten vor 1995 oder andere Märkte — eine Beschaffungsfrage, keine Forschungsfrage.
