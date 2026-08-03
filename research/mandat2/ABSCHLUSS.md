# FORSCHUNGSMANDAT II — Abschluss (2026-08-02)

**Auftrag (Hans, 2026-08-01):** Mandat neu öffnen. Steueraspekt weglassen und
zusätzlich eine vermögensverwaltende GmbH rechnen. Über 10 Jahre besser werden
als SPY. Alle Einzelaktien-Strategien neu testen, Haltedauern von Stunden bis
Jahren, Hebel, neue Strategien, externe Quellen.

**Zielfunktion (gesperrt 2026-08-01):** Median-Endvermögen über alle
rollierenden 10-Jahres-Fenster, unter der bindenden Nebenbedingung
MaxDD ≥ −35 % in *jedem* Fenster.

**Suchfenster** 1995-01-03 … 2016-12-30 · **Holdout** 2017-01-01 … 2026-07-06,
**bis heute unangetastet** · **Trials** 2.144 (kumuliert ab Mandat I)

---

## Das Ergebnis: kein deployable Kandidat

Der beste Kandidat der Kampagne — 20 gleichgewichtete S&P-Namen, Momentum-
Auswahl, Mindesthaltedauer 2 Jahre, Verkauf erst bei Rang > 200, Trendfilter
`Preis > SMA140` — besteht in-sample die Zielfunktion (Median 6,68 gegen
Benchmark 1,95, MaxDD −32,1 %) und drei von vier Robustheitstests.

Er scheitert an der Mehrfachtest-Korrektur:

| Test | Ergebnis | |
|---|---|---|
| Zielfunktion (in-sample) | Median 6,68 vs 1,95 · MaxDD −32,1 % | ✅ |
| Zufallskontrolle mit Gate | 100. Perzentil von 20 Seeds | ✅ |
| Fenster-Band | 11/12 Fenster, zusammenhängend 120–320 | ✅ |
| andere Trend-Definitionen | 5/12 und 3/12, lückig | ❌ |
| PBO (heterogene Matrix) | 24,3 % | ✅ |
| **DSR, heterogene Varianz** | **p = 0,9512** (Schwelle 0,95) | ⚠️ |
| **DSR, IID-Gegenprobe** | **p = 0,8783** | ❌ |

**Kein Holdout-Schuss.** Die Regel stand vor dem Lauf: alle drei Kriterien oder
keiner. Und p = 0,9512 gegen eine 0,95-Schwelle ist kein Bestehen, sondern ein
Münzwurf am Rand — bei einem Kandidaten, dessen konservativere Gegenprobe klar
durchfällt.

Damit ist die Kampagne auf diesem Suchraum beendet. Jedes weitere Nachjustieren
erhöht N und damit die Schwelle; man kann sich hier nicht mehr freisuchen.

---

## Was trotzdem gilt — vier belastbare Befunde

### 1. Die Steuer war nie die bindende Restriktion. Der Turnover war es.

Das Optimum der Handelsweise liegt in **allen drei Steuerwelten bei denselben
Parametern**: Mindesthaltedauer 730 Tage, `rank_out` 200. Nur die Hebelwahl
unterscheidet sich. `out30` (schnelles Umschichten) ist überall die
schlechteste Wahl, `out200` überall die beste.

Damit ist Hans' Ausgangshypothese beantwortet: **ohne Steuerdruck handelt man
nicht anders.** Die Steuer ändert, wie viel man behält — nicht, wie man handeln
sollte. Die Spannweite über das Parametergitter beträgt Faktor 6,4 bei
identischem Signal; die Steuerwelt verschiebt das Optimum darin nicht.

### 2. Der DD-Deckel ist erreichbar — aber nur mit Cash-Option, nicht mit Auswahl.

| SPY allein | MaxDD | gerissene Fenster | Median |
|---|---|---|---|
| ohne Gate | −55,2 % | 144/144 | 1,948 |
| **+ SMA200** | **−19,2 %** | **0/144** | **2,525** |

Der gefilterte *Index* hält den Deckel und schlägt zugleich den ungefilterten
Index. Kein immer voll investierter Long-only-Kandidat schafft das (0 von 72
Kombinationen im P2-Gitter, bester Drawdown −64,6 %).

### 3. Die GmbH-Frage ist eine Skalenfrage, keine Strategiefrage.

Die Steuerasymmetrie ist real und groß: Kursgewinnsteuer −76 % (123.827 →
29.799), Dividendensteuer +46 %. Ohne Rechtsformkosten +141.742 € Vorsprung.

**Mit** Rechtsformkosten (3.500 €/J, gemessen statt gerechnet) bleiben +4.079 €
— und auf der Zielfunktion ist die GmbH *schlechter* als privat (1,0524 gegen
1,0910). 73.500 € eingezahlte Fixkosten kosten über 22 Jahre 137.663 €
Endvermögen; der Zinseszins ist fast so groß wie der Nominalbetrag noch einmal.

**Nachtrag 2026-08-02 (`BEFUND_P10_P11.md`):** Der Break-even ist gerechnet.
Bei 3.500 €/J Rechtsformkosten lohnt die GmbH für den **aktiv gehandelten**
Kandidaten bereits ab 100.000 € (knapp, +2,3 %) und ab 250.000 € deutlich
(+36 bis +64 %). Für den **reinen ETF-Sparer** erst ab 250.000 € — bei
100.000 € ist sie mit −5,0 % sogar schlechter als das Privatdepot. Muster: je
mehr realisiert wird, desto früher trägt die Struktur.

Korrektur: die zuvor genannten „+4.079 €" galten für die P1-Parametrisierung,
nicht für den P2-Gewinner (dort +14.266 €). Beide korrekt, aber nicht
vergleichbar ohne Angabe des Parametersatzes.

### 4. Signale sind austauschbar, solange man nicht filtert.

Ohne Trendfilter liegt Momentum im **50. Perzentil** von 20 Zufallsläufen mit
identischer Haltedisziplin — das Signal ist wertlos. Mit Filter liegt es im
**100. Perzentil**. Das ist eine Interaktion, kein additiver Effekt, und der
interessanteste unbestätigte Befund der Kampagne.

### 5. Der Benchmark war ein Drittel des „Vorsprungs" (Nachtrag 2026-08-02)

Die Kampagne maß gleichgewichtete Kandidaten gegen den **kapitalgewichteten**
SPY. Ein gleichgewichteter Index desselben Universums schlägt SPY allein um
Faktor 1,33 (Median 2,594 gegen 1,948) — ohne Auswahl, ohne Signal. Der
Zufallsbefund aus Punkt 4 schrumpft damit von 1,40× auf **1,05×** gegen den
richtigen Maßstab: er war die Gewichtung. Der Hauptkandidat verliert ein
Viertel (3,43× → 2,57×), bleibt aber deutlich vorn — er lebt also nicht
allein davon. Am Verdikt ändert das nichts, weil ein härterer Maßstab ein
negatives Urteil nur bestätigt. Registriert als **E-079**.

---

## Was offen bleibt

**Datenblockiert:** Der Fundamentalfaktor. PIT-korrekte XBRL-Daten beginnen
2009-04-15, das Suchfenster endet 2016 — kein einziges vollständiges
10-Jahres-Fenster. Braucht eine andere Datenquelle (Compustat/CRSP-Klasse);
EODHD-Fundamentals sind nicht freigeschaltet. **Beschaffungs-, keine
Forschungsentscheidung.**

**~~Unerklärt~~ — beantwortet am 2026-08-02 (`BEFUND_P9_FORENSIK.md`):** Bei
Fenster 200 tragen **alle drei** Definitionen (MaxDD −30,5 / −31,5 / −31,5 %);
sie unterscheiden sich in der Reaktionsgeschwindigkeit — `Preis > SMA` steigt
bei −8 % SPY-Drawdown aus, `SMA steigt` erst bei −13 bis −16 %, bleibt dafür
länger draußen.

Der eigentliche Befund ist gravierender: **das Gate trifft über 22 Jahre nur
12–18 wirksame Entscheidungen**, von denen etwa vier zählen. Die täglichen
Flips (140 gegen 14) sind irrelevant, weil nur an 264 Monatsenden gelesen wird;
die drei Definitionen stimmen dort zu 86,4 % überein. Das erklärt die binäre
Verteilung der gerissenen Fenster ebenso wie den DSR-Wert genau an der
Schwelle — und es heißt: **der Kandidat ist nicht knapp gescheitert, er war nie
belastbar genug, um knapp zu sein** (E-078).

**Nicht getestet, aber NICHT blockiert — Korrektur 2026-08-03:** Intraday.
Meine Aussage „bräuchte das EODHD-Intraday-Paket (ab ca. Okt 2020)" war in
beiden Teilen falsch. Das Paket ist freigeschaltet und wird bereits genutzt
(452k 5-Minuten-Bars für 4 ETFs 2020–2026 und 246k 1-Minuten-Bars für 20 Titel
2024–2026 liegen im Repo). Und „ab Okt 2020" gilt nur für den **1h**-Endpunkt —
der **1m**-Endpunkt reicht bei Einzelaktien bis **2004** zurück (empirisch
geprüft: AAPL, MSFT, GE, XOM, KO).

2004–2026 sind 22 Jahre und damit **genug für rollierende 10-Jahres-Fenster**.
Der Strang ist offen, nicht blockiert. Randbedingungen gemessen: max. 120 Tage
pro API-Call, SPY erst ab ~2014 (Benchmark bleibt deshalb der tägliche SPY),
1m-Rohdaten zu groß zum Lagern → Verdichtung auf Stundenbars beim Ingest
(~80k Bars und 2,4 MB je Symbol über die volle Strecke).

---

## Externe Quellen

**wikifolio** (300 erhoben, robots.txt-konform, keine personenbezogenen Daten):
Median **4,82 % p. a.**, nur 26,4 % erreichen SPY-Niveau, 22,8 % sind negativ —
und das ist die optimistische Zahl, weil gescheiterte wikifolios aus dem
Sitemap verschwinden. Dazu 10 % Performance-Fee plus 0,95 % p. a. Ohne
Hebelprodukte Median 5,65 %, mit Hebel 1,62 %.

**eToro** nicht erhoben: dessen robots.txt sperrt `/portfolio` und `*/api/*`
für alle Agents.

**Anthropic-Finanz-Templates** (`anthropics/financial-services`, 10 Plugins):
geprüft und eingeordnet. Ihre Bewertungsmethodik (P/E, EV/EBITDA, DCF-Treiber)
ist übertragbar, ihre Connectoren (FactSet, CapIQ) haben wir nicht, und
validieren können sie nicht — das bleibt unsere Pipeline. Ohne längere
Fundamentalhistorie ist der Strang ohnehin blockiert.

---

## Methodische Ausbeute

Acht Befunde habe ich in dieser Kampagne selbst gekippt — drei BLOCKER und
mehrere MAJORs, alle vor dem Holdout. Daraus sind acht Anti-Patterns entstanden
(**E-070 bis E-077**), darunter:

* Ein Zugangs-Gate deckte nur den Hauptdatensatz; `clip()` tarnte den Leak als
  gültige Daten (E-070).
* Ein Fix wirkte am Endpunkt, die entscheidende Kennzahl las ihn nie (E-071).
* Ein Stellparameter ohne Wirkung, aber mit voller Kostenseite — der
  Hebel-Sweep hätte sich selbst bestätigt (E-072).
* Kosten im Fließtext subtrahiert statt gemessen; der Zinseszins fehlte und
  drehte den Befund (E-076).
* DSR mit großem N und kleinem V aus der eigenen Klonfamilie (E-077).

**Die wichtigste Regel, die daraus entstand:** kein Auswahl-Befund ohne
Zufallskontrolle. Sie hat den Hauptbefund der Kampagne gekippt, bevor er in den
Holdout getragen wurde.

---

## Empfehlung

Die Antwort auf die Ausgangsfrage lautet: **auf diesen Daten und in diesem
Suchraum nein** — auch ohne Steuer, auch mit GmbH, auch mit Hebel, auch mit
allen getesteten Haltedauern.

Wer weitersuchen will, hat drei ehrliche Optionen, und keine davon ist ein
weiterer Parameter-Sweep:

1. **Andere Daten** — längere Fundamentalhistorie oder Intraday. Beides
   Beschaffung, nicht Forschung.
2. **Andere Nebenbedingung** — der −35 %-Deckel ist schärfer als der Benchmark
   und schließt den gesamten Long-only-Raum aus. Das war eine bewusste
   Entscheidung; sie zu ändern wäre eine neue Frage, keine neue Suche.
3. **Den unerklärten Befund erklären** — warum trägt `Preis > SMA` und die
   Alternativen nicht? Eine ökonomische Antwort darauf wäre mehr wert als
   weitere zweitausend Trials.

Der Holdout bleibt versiegelt und ist damit weiter einmal verwendbar — für
einen Kandidaten, der ihn verdient.
