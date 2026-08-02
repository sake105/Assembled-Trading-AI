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

Bei 100.000 € Startkapital trägt die Struktur nicht. Ab welchem Kapital sie
kippt, ist offen und wäre ein eigener Sweep.

### 4. Signale sind austauschbar, solange man nicht filtert.

Ohne Trendfilter liegt Momentum im **50. Perzentil** von 20 Zufallsläufen mit
identischer Haltedisziplin — das Signal ist wertlos, gemessen wird der
Gleichgewichtungs-Effekt. Mit Filter liegt es im **100. Perzentil**. Das ist
eine Interaktion, kein additiver Effekt, und der interessanteste unbestätigte
Befund der Kampagne.

---

## Was offen bleibt

**Datenblockiert:** Der Fundamentalfaktor. PIT-korrekte XBRL-Daten beginnen
2009-04-15, das Suchfenster endet 2016 — kein einziges vollständiges
10-Jahres-Fenster. Braucht eine andere Datenquelle (Compustat/CRSP-Klasse);
EODHD-Fundamentals sind nicht freigeschaltet. **Beschaffungs-, keine
Forschungsentscheidung.**

**Unerklärt:** Warum trägt `Preis > SMA`, aber „SMA steigt" und „Rendite > 0"
nicht? Solange das offen ist, bleibt der Verdacht, dass die spezifische Ein-
und Ausstiegszeitpunktwahl auf zwei Ereignisse passt. Die Zahl gerissener
Fenster ist über alle 72 Läufe fast binär (0, 64 oder 69) — der ganze Befund
hängt an einem einzigen Ereignis.

**Nicht getestet:** Intraday. „Wenige Stunden Haltedauer" ist mit dem
EOD-Panel nicht auflösbar; das bräuchte das EODHD-Intraday-Paket (ab ca.
Okt 2020) und wäre ein eigener Strang.

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
