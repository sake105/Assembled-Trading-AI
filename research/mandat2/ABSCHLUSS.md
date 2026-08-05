# FORSCHUNGSMANDAT II — Abschluss (2026-08-02)

**Auftrag (Hans, 2026-08-01):** Mandat neu öffnen. Steueraspekt weglassen und
zusätzlich eine vermögensverwaltende GmbH rechnen. Über 10 Jahre besser werden
als SPY. Alle Einzelaktien-Strategien neu testen, Haltedauern von Stunden bis
Jahren, Hebel, neue Strategien, externe Quellen.

**Zielfunktion (gesperrt 2026-08-01):** Median-Endvermögen über alle
rollierenden 10-Jahres-Fenster, unter der bindenden Nebenbedingung
MaxDD ≥ −35 % in *jedem* Fenster.

**Suchfenster** 1995-01-03 … 2016-12-30 · **Holdout** 2017-01-01 … 2026-07-06,
**bis heute unangetastet** · **Trials** 3.305 (kumuliert ab Mandat I:
1.964 + 1.341 aus Mandat II, Stand `trials.json`)

> **Korrektur 2026-08-03:** hier stand 2.144. Die Zahl war seit P4 nicht mehr
> nachgeführt worden und driftete gegen `trials.json` — dieselbe Fehlerklasse
> wie E-085, nur kampagnenweit statt in einem Befund. In den 1.341 sind
> **44 Trials aus einem reinen Regenerationslauf** enthalten (E-090); sie
> werden nicht zurueckgeschrieben, sondern hier offengelegt. Wirkung: der
> DSR-Haircut ist um diesen Betrag zu streng, also konservativ.

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

> **Nachtrag 2026-08-05:** Ein *anderer* Suchraum wurde seither geprüft und
> ebenfalls geschlossen — SPY mit Trendfilter, ohne jede Namensauswahl. Er
> besteht alle Robustheitstests, an denen der Kandidat oben gescheitert ist —
> scheitert aber an beiden Hälften der Mehrfachtest-Korrektur (DSR 0,7838 auf
> dem gültigen Schätzer, PBO 68,6 %). Kein Holdout-Schuss. Siehe „Der letzte
> Strang" weiter unten. Damit hat **kein** Kandidat der Kampagne die
> Mehrfachtest-Korrektur bestanden.

---

## Was trotzdem gilt — sieben belastbare Befunde

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

### 6. Die Datenbasis trägt keinen Vergleich gegen SPY (Nachtrag 2026-08-04)

Punkt 5 war der erste Riss; hier ist der ganze Bruch. Vollständige Herleitung
und alle Zahlen: **[`BEFUND_DATENQUALITAET.md`](./BEFUND_DATENQUALITAET.md)**
(generiert aus `results/p12d_*.json`, `p12e_*.json`, `p12f_*.json`,
`p12g_*.json`).

**Survivorship (P12d).** Das Intraday-Universum besteht aus Namen, die heute
noch handelbar sind. Liegenlassen dieser Namen liefert über 10,5 Jahre
**2,36 bis 2,90 Prozentpunkte p. a. mehr als ein survivorship-freies
PIT-Universum — ohne Strategie, ohne Signal.** Die Kampagne entscheidet Fragen
im Bereich von rund 1,5 pp p. a.; schon der untere Rand der Spanne liegt
darüber. Die Verzerrung ist also größer als der Effekt, um den gestritten wird.

Der Bezug ist bewusst das PIT-Universum und nicht SPY: gegen SPY wären es
3,36 pp, aber dort würden Gewichtung und Indexkonstruktion mitgemessen
(vgl. Befund 5). Die Spanne kommt daher, dass P12d zwei Delisting-Behandlungen
rechnet — eine einzelne Zahl wäre hier Scheinpräzision.

Verstärkend (P12e): von den Indexmitgliedern **mit** Preisspalte überleben
46,2 % bis zum Fensterende, von denen **ohne** nur 9,0 % — Anreicherungsfaktor
**5,15×**. Das Fehlen einer Preisspalte ist kein technischer Zufall, es sagt
das Ausscheiden voraus. Das Panel verliert bevorzugt die Verlierer.

> **Konsequenz:** Jeder Vergleich gegen SPY oder einen passiven ETF ist auf
> dieser Datenbasis nicht belastbar — in keine Richtung. Vergleiche *innerhalb*
> des Universums (Haltedauer gegen Haltedauer, Parametrisierung gegen
> Parametrisierung) bleiben gültig, weil beide Seiten dieselbe Verzerrung
> tragen.

**Und die Quelle kann die Lücke nicht schließen (P12g).** Der naheliegende
Ausweg war, das Intraday-Universum zu verbreitern — von 21 auf 298 Symbole.
Ob das zum Ziel führt, beantwortet eine **API-Probe**, nicht das
Dateiverzeichnis: sechs Ausscheider des Suchfensters, jeweils unter dem
Symbol geprüft, unter dem sie **damals im Index standen** — LEH, WM, EK, GM,
CC, BSC — liefern **null Bars**, ebenso ihre Post-Insolvenz-Ticker. Eine
Kontrollgruppe aus acht Überlebenden liefert im selben Fenster 7.008–7.992
Bars; der Aufruf funktioniert also.

Mehr Anfragen erhöhen damit die **Abdeckung** (viele Überlebende sind schlicht
noch nicht gezogen), aber nicht die **Unverzerrtheit** — sechs geprüft, sechs
stumm; die Ausscheider sind bei dieser Quelle nicht zu haben. Gezählt wurden
Minutenbars. Für einen belastbaren Vergleich gegen einen
passiven Index braucht es Tagesdaten mit Delisting-Kursen.

> Eine erste Fassung dieses Absatzes las die Verfügbarkeit aus fehlenden
> Dateien und meldete einen „Anreicherungsfaktor 3,06". Das war falsch: acht
> von acht geprüften Namen der vermeintlichen Fehlgruppe (AMZN, GILD, VRSN,
> ADBE, NVDA, COST, ROST, PAYX) liefern Bars — sie waren nie angefragt worden.
> Registriert als **E-112**; der Härtetest lief zusätzlich unter den falschen
> Symbolen (**E-113**).

**Preisfehler (P12e/P12f).** 25 Namen tragen Skalenbrüche über 48.380
Handelstage, und sie sind nachweislich in die Ergebnisse eingegangen: GPS wurde
an 2 Übergangstagen **gehalten** — der eine trug +12,36 % Portfolio-Rendite und
war damit Rang 2 von 5.548 Handelstagen —, und vier Namen (ABC, CFC, GPS, TWX)
wurden über kontaminierte Momentum-Beine gewählt (22 von 5.040 Auswahlplätzen,
0,44 %). Ein Neulauf des P2-Gitters auf dem
gespleißten Panel zeigt: **das Verdikt dreht in keiner Steuerwelt** — 0 von 24
Parametrisierungen bestehen Zielfunktion und DD-Deckel, vor wie nach der
Bereinigung. Auch Befund 1 hält: Mindesthaltedauer 730 und `rank_out` 200
bleiben in allen Welten und beiden Panels das Optimum; nur die Hebelwahl
kippt, und die trennte Erst- und Zweitplatzierten ohnehin nur um 0,55 %.

Die Bereinigung selbst ist eine **Untergrenze**: 25 von 458 auffälligen Tagen
beseitigt (5 %), 13 Namen bewusst unangetastet — 12 mit verschränkten Skalen,
einer (WFT) mit einem Sättigungs-Sentinel von 999.999,9999, der kein Kurs ist.
Sie hat **null** neue Ausreißer erzeugt, zweiseitig geprüft und gegen die
Kursniveaus des *Originalpanels* gemessen.

Bis dahin waren vier Fehlerklassen zu beheben — jede hätte für sich ein
falsches Ergebnis getragen:

| | Fehler | Wirkung |
|---|---|---|
| **E-107** | Spanne am ersten Gegenschlag geschlossen, ohne den Betrag zu prüfen | aus −77 % Kurssturz wurden **+6.802 %** |
| **E-108** | Wächter filterte nach den Kursen des *bereinigten* Panels | Reparatur schaltete ihre eigene Überwachung an 379 Symbol-Tagen ab |
| **E-110** | Detektor repariert, Artefakt des zweiten Konsumenten nicht neu erzeugt | Befund mischte zwei Detektorgenerationen (246 gegen 79 Übergangstage) |
| **E-111** | „nicht reparierbar" als leere Messfelder kodiert | die 13 kaputtesten Namen meldeten **null**; die gemessene Kontamination verdoppelte sich nach dem Fix von 24.123 auf 48.380 Tage |

E-108 verdeckte im konkreten Lauf genau **einen** Tag — der Fehler ist
strukturell, nicht in seiner hier gemessenen Wirkung. E-111 dagegen hatte die
Hälfte des Schadens unsichtbar gemacht, und zwar die schlimmere Hälfte.

### 7. Der Ticker war als Schlüssel behandelt worden (Nachtrag 2026-08-04)

Beim Anschluss an Befund 6 — Tagesdaten statt Intraday — fiel eine eigene
Fehlerklasse auf: **der Datenlieferant gibt unter einem Symbol die *heutige*
Firma zurück.** ABI (Anheuser-Busch, 2008 übernommen) beginnt 2025-06-26;
ABS (Albertsons, 2006) beginnt 2018; ALTR (Altera, 2015) beginnt 2017. Von 133
im Panel fehlenden PIT-Mitgliedern liegen 99 in der Rohdatei — ausschließlich
mit Kursen *nach* dem Suchfenster.

Belastbar ist dabei nicht „keine Kurse im Fenster" — das wiederholt nur die
Filterregel des Panels, die Spalten ohne jeden Kurs im Fenster ohnehin
verwirft —, sondern dass die Serie **erst danach beginnt**. Das gilt für
**99 von 99** der betroffenen Namen, alle gemessen, nicht gestichprobt.

Wo die Neuvergabe **innerhalb** des Fensters liegt, führt eine Panel-Spalte
zwei Unternehmen. Gefunden werden sie über eine **Signatur**: eine Lücke von
mindestens 500 Handelstagen, nach der die Serie weiterläuft. **29 Spalten**
tragen sie, alle 29 waren Index-Mitglieder — CTXS an 205 Terminen, TOY an 114.
TOY war einer der Namen, die P12d als „Niveausprung 100–200 %, Ursache offen"
markiert hatte; die Ursache ist damit gefunden.

> **Signatur ist nicht Ursache.** Die Schwelle wurde aus zwei Merkmalen
> hergeleitet — Lückenlänge *und* Kursfaktor —, entscheidet aber nur nach dem
> ersten. **8 der 30 Schnitte** liegen im Faktorband 0,5–2,0, in dem der Kurs
> fortsetzt (HSH, MWI, MYG, NLC, RX, RYC, WLL zweimal): dort ist die Trennung
> wahrscheinlich falsch und erzeugt ein *fabriziertes* Delisting. Die Liste
> steht im Artefakt, nicht im Fließtext. Registriert als **E-117**.
>
> Die Schranke dazu liegt in denselben Artefakten: **alle 8 Schnitte liegen
> nach dem letzten Mitgliedschaftstermin** des jeweiligen Symbols (MWI 2014-07
> gegen 1999-04, RYC 2007-07 gegen 1999-07, WLL 2008-12 und 2014-12 gegen
> 2002-01; RX, MYG, HSH, NLC ebenso). Bei membership-getriebener Auswahl sind
> diese Namen zum Zeitpunkt des fabrizierten Delistings nur noch über die
> Mindesthaltedauer erreichbar — und die Richtung ist konservativ, weil ein
> Zwangsverkauf keinen PASS erzeugen kann.

Zwei Schäden, beide am echten Bestand gemessen (`p12h_ticker_recycling.py`,
Engine instrumentiert über `Portfolio.set_date`). **Gemessen für EINE
Konfiguration:** 12-1-Momentum, `top_in=20`, Steuerwelt ZERO, Engine-Defaults
für Haltedauer und `rank_out` — nicht für das Kampagnen-Optimum aus Befund 1
(730 Tage / `rank_out` 200). Die Zahlen sind konfigurationsabhängig:

| | |
|---|---|
| Kurssprung am Wiedereinstieg | CGP **−3,49 %** Portfolio-Tagesrendite (Rang 5.419/5.548), NGH −1,21 %, NVLS +0,24 % |
| **Ausfall der Delisting-Hygiene** | 3 Namen, zusammen **5.035 Handelstage** im Bestand ohne echten Kurs — CGP allein 3.264 (13 Jahre) |

Der zweite ist der strukturelle: der Zwangsverkauf prüft `last_valid < t`, und
bei einem recycelten Ticker liegt `last_valid` am Ende der Serie von Firma B.
Für diese Namen greift er **nie**; die Position läuft auf dem eingefrorenen
letzten Kurs weiter. Keiner der bisherigen Detektoren sah das —
`pct_change(fill_method=None)` liefert über eine NaN-Lücke NaN, der Sprung war
für die Preisfehler-Prüfung unsichtbar.

**Ein Neulauf des P2-Gitters auf getrenntem Panel dreht kein Verdikt**
(`p12i_neulauf_getrennt.py`): 0 von 24 Parametrisierungen bestehen in beiden
Panels, der Median des besten Kandidaten ist in allen drei Steuerwelten
**identisch**, das Optimum wandert nicht.

> Auch hier gilt die Einschränkung aus Befund 6, mit den Zahlen dieses Laufs:
> der beste schlimmste Drawdown liegt bei **−63,6 %** gegen einen Deckel von
> −35 %, also **28,6 pp** daneben; die Trennung verschiebt den Drawdown um
> höchstens **6,98 pp**. Sie hätte rund **4-mal stärker** wirken müssen, um
> eine einzige Zeile über den Deckel zu heben. Der Test konnte an dieser Stelle
> nicht kippen — die Aussage ist „ohne Wirkung auf das Verdikt", nicht „ohne
> Wirkung".

Das schwächere Kriterium reagiert nämlich sehr wohl: ohne Drawdown-Deckel
wechseln einzelne Parametrisierungen ihren Status (ZERO 6 → 7, PRIVAT_DE
2 → 3 von 24). Die Trennung ist messbar, sie bewegt nur nichts an der
bindenden Nebenbedingung.

Registriert als **E-114**. Der erste Reparaturversuch hätte bei einer Schwelle
von 120 Handelstagen Coca-Cola Enterprises — sechs jährliche Vendor-Datenlöcher,
Firma durchgehend existent — in sieben Stücke zerlegt (**E-115**). Die Schwelle
liegt jetzt bei 500 und ist aus der Verteilung hergeleitet; der Bereich
120–500 bleibt unangetastet und offen.

---

## Der letzte Strang: SPY mit Trendfilter — an DSR und PBO gescheitert (Nachtrag 2026-08-05)

**Das reaktiviert den Aktien-Kandidaten nicht.** Er bleibt an der DSR
gescheitert, und der Suchraum „20 Namen aus dem S&P" bleibt geschlossen. Was
hier geöffnet wird, ist ein **anderer** Suchraum, und zwar der einzige, den
die Befunde 6 und 7 nicht entwerten: auf beiden Seiten steht derselbe
Basiswert. SPY mit Filter gegen SPY ohne — keine Auswahl, also kein
Survivorship, kein Ticker-Recycling, keine Gewichtungsfrage.

Anlass war die Datenlage: Der EODHD-Zugang besteht nicht mehr, Tagesdaten mit
Delisting-Kursen sind auf absehbare Zeit nicht beschaffbar (siehe unten). Die
Frage war also, welcher Strang **ohne neue Daten** überhaupt noch entscheidbar
ist. P4 hatte die Antwort seit Wochen in einem Kontrollblock stehen, ohne sie
als Kandidaten zu behandeln.

Vollständige Zahlen und alle Einschränkungen: `BEFUND_SPY_TREND.md` (generiert
aus `results/p13*.json`). Kurzfassung:

| Test | Ergebnis | |
|---|---|---|
| Fenster-Band, steuerfrei (Schnitt beider Läufe) | 9/12, 11/12, 9/12 · längste lückenlose Kette **9 / 11 / 4** | ✅ |
| andere Trend-Definitionen | alle drei bestehen, aber `rendite>0` nur lückig (Kette 4) | ⚠️ |
| Ausführung einen Tag später | Band bleibt, teils breiter | ✅ |
| Zufalls-Timing, 60 Ziehungen | 0/60 erreichen den Filter, p = 0,016 | ✅ |
| Steuerwelt PRIVAT_DE | 5/12 bis 7/12 | ⚠️ |
| Steuerwelt GmbH + 3.500 €/J Fixkosten | 0/12 bis 2/12 | ❌ |
| **Ereignisunabhängigkeit** | **0 von 144 Fenstern ohne Bärenmarkt** | ⚠️ |
| **DSR, heterogenes V (Regel aus P8), N = 3.529** | Sharpe 0,0522 gegen Schwelle 0,0415 → **p = 0,7838** | ❌ |
| DSR, IID-Gegenprobe | p = 0,6105 | ❌ |
| ~~DSR, V aus der Klonfamilie~~ | ~~p = 0,9974~~ — **nicht entscheidungsfähig (E-077)** | — |
| **PBO (CSCV, 8 Blöcke, 70 Splits)** | **68,6 %** | ❌ |
| Holdout | **kein Schuss** — beide Korrekturen gerissen | — |

Der Filter besteht die Zielfunktion nicht knapp, sondern deutlich, und keine
der billigen Widerlegungen greift. Er scheitert trotzdem an **beiden** Hälften
der Mehrfachtest-Korrektur.

Der erste Entwurf von `p13e` war eine Kopie von `p7_dsr_pbo.py` — dem Modul,
das in diesem Repo als **E-077** verworfen und durch `p8_dsr_heterogen.py`
ersetzt wurde. Er schätzte die Varianz aus 37 Fast-Klonen derselben Strategie,
senkte damit die Schwelle von 0,0415 auf 0,0140 und meldete „DSR bestanden".
Auf dem gültigen Schätzer fällt p von 0,9974 auf **0,7838**. Rule 50 schützt
vor divergierenden Implementierungen, nicht vor der Wiederverwendung einer
verworfenen Methode.

Damit ist auch der Vergleich mit dem Aktien-Kandidaten geklärt, und er fällt
zu Ungunsten dieses Kandidaten aus: Auf demselben Schätzer kam jener auf
0,9512 (formal bestanden, Münzwurf am Rand) und starb erst an der
IID-Gegenprobe; dieser liegt mit 0,7838 deutlich darunter. Eine frühere
Fassung dieses Nachtrags stellte 0,9974 aus der Klonfamilie neben jene 0,9512
aus der heterogenen Familie und leitete daraus eine „Symmetrie" ab — zwei
verschiedene Schätzer, verglichen zugunsten des eigenen Kandidaten.

PBO ist zudem nach unten verzerrt: E-077 hält fest, dass CSCV heterogene
Spalten voraussetzt und bei Fast-Klonen zu niedrige Werte liefert. 68,6 %
unter dieser Verzerrung ist ein deutlicherer Fehlschlag, als die Zahl
nahelegt. Die Regel stand vor dem Lauf — alle Kriterien oder keines.

Selbst ein bestandener PBO hätte den Strang nicht entschieden. Das mildeste
der 144 rollierenden Fenster enthält einen Rückgang von 47,5 %; die Stichprobe
kann „Trendfolge wirkt" nicht von „Trendfolge hat 2000–2002 und 2008 umgangen"
trennen. Die effektive Stichprobe für den Mechanismus sind zwei Ereignisse,
nicht 144 Fenster. Ein besserer Test bräuchte Daten vor 1995 oder andere
Märkte — Beschaffungs-, keine Forschungsfrage.

Das Ergebnis zur GmbH ist unabhängig davon verwertbar und bestätigt Befund 3
aus anderer Richtung: Ein Filter, der über das Raster 6 bis 62 Buchungen
erzeugt, trägt die Fixkosten der Rechtsform bei diesem Kapitaleinsatz nicht.

Trial-Stand nach P13/P13b: **3.529** kampagnenweit (+216). P13c (Zerlegung),
P13d (Kontrollgruppe) und P13e (Korrektur) zählen nicht mit — keines davon ist
eine Suche (E-090). Für Wiederholungsläufe nach einem Bugfix haben P13 und
P13b jetzt einen `--regen`-Schalter, der das Increment überspringt; ohne ihn
hätte jede Neugenerierung der Artefakte erneut 216 Trials addiert.

---

## Was offen bleibt

**Der EODHD-Zugang besteht nicht mehr** (Stand 2026-08-05). Damit ist jede
Beschaffungsfrage unten bis auf Weiteres blockiert — nicht durch eine
Forschungsentscheidung, sondern durch fehlenden Datenzugang. Was ohne neue
Daten noch entscheidbar war, ist im P13-Strang oben abgearbeitet.

**Tagesdaten mit Delisting-Kursen sind weiterhin nicht beschafft.** Befund 6
nennt sie als den Weg, auf dem die SPY-Frage überhaupt entscheidbar wäre; P12g
hat ausschließlich den **Intraday**-Endpunkt sondiert und dort belegt, dass er
Ausscheider nicht führt. Für den **EOD**-Endpunkt ist Delisted-Coverage
grundsätzlich verifiziert (SIVB, BBBY), für das Suchfenster 1995–2016 aber
ungeprüft. Der Anlauf dazu förderte stattdessen Befund 7 zutage — der Fehler
musste vor dem Pull gefunden werden, sonst hätte ein Tagespull über 1.011
Symbole denselben Ticker-als-Schlüssel-Fehler in größerem Maßstab geerbt.
Beschaffungs-, keine Forschungsfrage.

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
