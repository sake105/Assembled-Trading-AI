# Befund — Trägt der Datensatz die Verdikte?

*Erzeugt von `render_befund_datenqualitaet.py` aus `results/p12d_*.json`, `p12e_*.json`, `p12f_*.json`, `p12g_*.json`. Nicht von Hand bearbeiten — Änderungen gehen beim nächsten Lauf verloren.*

---

## Kurzfassung

**Nein — nicht in der Größenordnung, um die gestritten wird.** Die
Auswahl des Universums allein liefert 2,36 bis 2,90
Prozentpunkte p. a. gegenüber einem survivorship-freien Universum,
ohne jede Strategie. Die Kampagne entscheidet Fragen im Bereich von
rund 1,5 pp p. a. Die Verzerrung ist
größer als der zu messende Effekt.

Die Preisfehler im Panel sind dagegen **nachweislich** in die
Ergebnisse eingegangen (Abschnitt 2)
und drehen dennoch kein Verdikt (Abschnitt 4). Der begrenzende
Faktor ist nicht die Sauberkeit der Kurse, sondern die Auswahl
der Namen — und die ist durch keine Bereinigung reparierbar,
nur durch andere Daten.

---

## 1. Wie viel Rendite kommt allein aus der Auswahl der Namen?

Das Intraday-Universum von P12 besteht aus Namen, die **heute** noch
handelbar sind. Wer 2006 in genau diese Namen investierte, wusste 2006
nicht, dass sie überleben würden. Die Frage ist, wie groß dieser
Vorteil ist — nicht, ob es ihn gibt.

Gemessen über 10,5 Jahre (2006-06-22..2016-12-30), gleichgewichtet, gleiches Verfahren für alle Zeilen:

| Universum | Namen | Endwert (halten) | CAGR (halten) | CAGR (umschichten) | MaxDD |
|---|---:|---:|---:|---:|---:|
| intraday_p12 | 20 | 3,02× | 11,06 % | 11,06 % | -54,4 % |
| durchgehend_2004_2016 | 259 | 2,69× | 9,86 % | 9,86 % | -52,8 % |
| pit_2004 | 382 | 2,28× | 8,16 % | 8,70 % | -54,3 % |
| SPY (Referenz) | 1 | 2,18× | 7,70 % | — | -55,2 % |

**Überhöhung: 2,36 bis 2,90 Prozentpunkte p. a.** gegenüber
dem survivorship-freien **pit_2004** — allein aus der Zusammensetzung
des Universums, ohne jede Strategie, ohne jedes Signal, bei reinem
Liegenlassen. Die Spanne kommt daher, dass P12d zwei Delisting-
Behandlungen rechnet; eine einzelne Zahl wäre hier Scheinpräzision.

Gegen **SPY** wäre die Zahl mit 3,36 pp noch größer.
Der Vergleich gegen das PIT-Universum ist aber der ehrlichere: er
isoliert die Auswahl, während gegen SPY zusätzlich Gewichtung und
Indexkonstruktion mitgemessen würden (vgl. E-079).

Die Kampagne entscheidet Fragen im Bereich von rund 1,5 Prozentpunkten p. a.
Schon der **untere** Rand der Spanne liegt mit 2,36 pp
**über der Marge, um die gestritten wird.**

> Damit ist die entscheidende Aussage dieses Dokuments erreicht:
> **Der Datensatz kann die Frage nicht entscheiden.** Nicht „die
> Strategie verliert“ und nicht „die Strategie gewinnt“ — die
> Datengrundlage trägt das Urteil in dieser Größenordnung nicht.

Das ist keine Formalie. Ein Ergebnis, dessen Vorzeichen von einer
Verzerrung abhängt, die größer ist als der gemessene Effekt, ist
kein Ergebnis.

Zur Einordnung: das PIT-Universum enthält Insolvenzticker (EKDKQ, MTLQQ, WNDXQ),
das Intraday-Universum keinen einzigen. Der Unterschied ist keine
Feinheit der Stichprobe, sondern ihre Konstruktion.

---

## 2. Sind die kaputten Kurse in die Ergebnisse eingegangen?

Im Panel liegen **25 Namen** mit
Skalenbrüchen: 79 Übergangstage,
48.380 Tage auf einer falschen
Skala. Die Frage ist nicht, ob es sie gibt, sondern ob die Strategie
sie **angefasst** hat. Dafür zwei getrennte Kanäle:

**Kanal A — gehalten über einen Übergangstag.** Gemessen am echten
Bestand der Engine, nicht an der Auswahl: die Turnover-Bremse hält
Namen über die Auswahl hinaus, ein Auswahl-Proxy hätte hier
entwarnt, wo keine Entwarnung war (E-102).

| Name | Tage | größte Wirkung auf die Tagesrendite | Rang unter allen Tagen |
|---|---:|---:|---:|
| GPS | 2 | 12,36 % (1996-12-20) | 2 von 5.548 |

GPS lag an seinem Übergangstag auf Rang 2 von 5.548 — das ist der **2-extremste Tag** der ganzen Kampagne, gemessen vom
näheren Ende der Rangliste. Der Tag mit 12,36 % Portfolio-Rendite ist also nicht
irgendein Tag. Ein Vendor-Fehler an dieser Stelle ist kein Rauschen.

**Kanal B — mit kontaminiertem Momentum-Score gewählt.** Der Score ist
`close.shift(21) / close.shift(252)` — ein Quotient
aus **zwei** Stützstellen, kein Fenster. Kontaminiert ist er genau
dann, wenn die beiden Beine auf **verschiedenen** Skalen liegen; liegen
beide auf derselben falschen Skala, kürzt sich der Faktor heraus
(E-104).

Betroffen: **4 Namen, 22 von 5.040 Auswahlplätzen (0,44 %).**

* ABC: 4 Termine (2001-08-31 … 2001-11-30)
* CFC: 4 Termine (2007-02-28 … 2007-06-29)
* GPS: 11 Termine (1997-01-31 … 1997-11-28)
* TWX: 3 Termine (1998-12-31 … 1999-02-26)

Beide Kanäle sind klein. „Klein“ ist aber keine Antwort auf
„dreht es ein Verdikt?“ — diese Frage beantwortet nur ein
Neulauf (Abschnitt 4).

---

## 3. Was fehlt im Panel — und fehlt es zufällig?

Von den Indexmitgliedern haben im Median nur **91,3 %**
überhaupt eine Preisspalte (Spanne 84,0 % bis 96,2 %). Fehlende Spalten wären harmlos, wenn sie
zufällig fehlten. Sie fehlen nicht zufällig:

Von den 409 Mitgliedern **mit** Preisspalte am
1996-01-31 sind am 2016-12-30 noch
**46,2 %** im Index. Von den
78 Mitgliedern **ohne** Preisspalte nur
**9,0 %**.

> **Anreicherungsfaktor 5,15×.** Das Fehlen einer Preisspalte
> ist kein technischer Zufall — es sagt voraus, dass der Name
> ausscheidet. Das Panel verliert also bevorzugt die Verlierer.

Diese Verzerrung wirkt in dieselbe Richtung wie die aus Abschnitt 1
und ist von ihr **nicht unabhängig**: beide entstehen daraus, dass
Ausscheider schlechter dokumentiert sind als Überlebende. Die
Größenordnungen dürfen deshalb nicht addiert werden — wohl aber
gilt: die Schranke aus Abschnitt 1 ist eher eine Unter- als eine
Obergrenze.

---

## 4. Dreht ein Verdikt, wenn man die Brüche repariert?

Repariert wird durch **Spleißen**: innerhalb einer Korruptionsspanne
liegen die Kurse um einen konstanten Faktor daneben, geteilt durch
diesen Faktor liegen sie wieder auf der Basisskala. Je Spanne werden
damit genau zwei Renditen ersetzt — die an den beiden Rändern —, alle
übrigen bleiben bis auf Maschinengenauigkeit erhalten, weil sich ein konstanter Faktor
im Quotienten herauskürzt.

Bereinigt: **12 Namen, 16 Spannen.**
Nicht bereinigt: **13 Namen** (CFC, CIN, CNG, COMS, HPC, KRI, MCIC, MEL, SLR, SMI, USBC, WFT, YRCW).
Bei 12 davon sind die Skalen **verschränkt** — nach
einem Sprung folgt ein weiterer, ohne dass die Rückkehr zum
ersten passt; dort ist nicht bestimmbar, welcher Kurs auf welcher
Skala liegt.
**4 Namen tragen den Sättigungswert des
Datenlieferanten** (999.999,9999): COMS, MCIC, WFT, YRCW — 3 davon zusätzlich verschränkt.
Das ist kein Kurs, und Konstante geteilt durch Konstante ist kein
Spleiß.
Alle bleiben unberührt und stehen im Protokoll. Eine Bereinigung,
die einen Teil repariert und das sagt, ist ehrlicher als eine, die
alles zu reparieren behauptet (E-107).

**Gegenprobe in beide Richtungen.** Eine Reparatur, die ihre eigenen
Nebenwirkungen nicht misst, ist eine zweite, unbeobachtete
Datenquelle — und zwar genau dort, wo die Frage entschieden wird:

| auffällige Tage im Original | 458 |
|---|---:|
| davon beseitigt | 25 |
| **neu entstanden** | **0** |
| bleiben auffällig | 433 |

Kein einziger Ausreißer ist durch die Bereinigung entstanden —
geprüft nach oben (>+100 %) **und** nach unten (<−50 %). Ein
einseitiger Wächter hätte die halbe Fehlerklasse durchgelassen.

Beseitigt werden damit 5 % der auffälligen Tage;
433 bleiben stehen. Die Bereinigung
ist eine **Untergrenze**, kein sauberes Panel — „bereinigt“
heißt hier: die eindeutig auflösbaren Skalenbrüche sind weg.

Die Dividenden wurden mitskaliert; die Invariante *Dividende je
Kurseinheit* ändert sich um maximal 1.4e-17 — Rundungsrauschen.
Ohne diese Mitskalierung stiege die implizite Dividendenrendite in
der Spanne um genau den Spleißfaktor (bei WIN von 26 % auf 274 %),
und der Vergleich zweier Panels wäre unfair auf genau der Achse, um
die es in der GmbH-Frage geht.

Gerechnet wird **dasselbe Parametergitter wie in P2** — kein neuer
Parameter, keine neue Suche, der Trial-Zähler bleibt unverändert
(E-090).

Gemessen wird die **Zielfunktion der Kampagne**, nicht der Endwert:
Median über alle rollierenden 10-Jahres-Fenster gegen den Benchmark,
unter der bindenden Nebenbedingung MaxDD ≥ -35 %
in *jedem* Fenster. Ein Endwertvergleich hätte hier eine andere Frage
beantwortet: P2 hielt ausdrücklich fest, dass der beste Kandidat den
Index **bei der Rendite schlägt** und an der Nebenbedingung scheitert.

| Steuerwelt | Panel | Median bester | Median Bench | schlimmster DD | schlägt | **besteht** |
|---|---|---:|---:|---:|---:|---:|
| ZERO | original | 3,460 | 1,948 | -81,6 % | 6/24 | **0/24** |
| ZERO | bereinigt | 3,410 | 1,948 | -82,1 % | 7/24 | **0/24** |
| PRIVAT_DE | original | 2,168 | 1,870 | -64,6 % | 2/24 | **0/24** |
| PRIVAT_DE | bereinigt | 2,269 | 1,870 | -84,0 % | 4/24 | **0/24** |
| GMBH+FK | original | 2,392 | 1,862 | -84,4 % | 2/24 | **0/24** |
| GMBH+FK | bereinigt | 2,439 | 1,862 | -85,1 % | 2/24 | **0/24** |

**Das Verdikt dreht in keiner Steuerwelt.** Das ist allerdings keine
Robustheitsaussage, solange nicht dabeisteht, wie weit der Ausgang vom
Kippen entfernt war:

* Der **beste** schlimmste Drawdown über alle Parametrisierungen und
  beide Panels liegt bei -61,2 % — der Deckel
  fordert -35 %. Der beste Kandidat
  verfehlt ihn also um **26,2 Prozentpunkte.**
* Die Bereinigung verschiebt den schlimmsten Drawdown um höchstens
  **5,21 Prozentpunkte.**

Sie hätte also rund **5,0-mal stärker** wirken
müssen, um auch nur eine einzige Zeile über den Deckel zu heben — und
das in die richtige Richtung. Der Ausgang
konnte an dieser Stelle nicht kippen — das ist die ehrliche Fassung
von „robust“, und sie ist stärker, weil sie die Auflösung
des Tests
mitliefert.

Das ist eine Aussage über die **Preisfehler**, nicht über die
Datenqualität insgesamt. Die Survivorship-Schranke aus Abschnitt 1
bleibt davon vollständig unberührt — sie ist der größere Posten
und durch keine Bereinigung erreichbar.

**Das schwächere Kriterium reagiert dagegen sehr wohl.** Lässt man den
Drawdown-Deckel weg und fragt nur, ob der Benchmark bei der Rendite
geschlagen wird, wechseln **3 einzelne Parametrisierungen**
ihren Status (PRIVAT_DE: 2, ZERO: 1) — ohne dass sich an der Gesamtaussage etwas
ändert, weil keine von ihnen den Deckel hält.

Das ist der eigentliche Beleg dafür, dass die Preisfehler wirken: sie
verschieben die Rangfolge messbar. Sie verschieben sie nur nicht weit
genug, um an der bindenden Nebenbedingung etwas zu ändern.

Das Optimum wandert in: PRIVAT_DE — und zwar
ausschließlich in der Dimension Hebel.
**Mindesthaltedauer (730 Tage) und `rank_out` (200) sind
in allen Steuerwelten und in beiden Panels identisch.** Genau
darauf stützte P2 den Schluss, dass nicht die Steuer, sondern
der Turnover die bindende Restriktion ist — dieser Schluss
überlebt die Bereinigung unverändert.

Die Hebelwahl war schon im Original kein belastbarer Befund:
Erst- und Zweitplatzierter trennten dort 0,55 %.
Eine Rangfolge an dieser Marge kippt bei jeder Störung — sie
sagt nichts über den Hebel, sondern über die Auflösung der
Messung.

---

## 5. Kann der Intraday-Endpunkt die Lücke überhaupt schließen?

Nach Abschnitt 1 lag der Schluss nahe, einfach mehr Symbole zu ziehen.
Genau das ist geschehen — 298 Dateien liegen
inzwischen vor. Ob das zum Ziel führt, beantwortet aber nicht das
Dateiverzeichnis, sondern nur eine Abfrage.

Geprüft wurden Ausscheider des Suchfensters — jeweils unter dem Symbol,
unter dem sie **damals im Index standen**, und zusätzlich unter dem
Post-Insolvenz-Ticker. Das ist nicht dasselbe: die Q-Ticker entstehen
erst mit dem Chapter-11-Handel, ein Negativbefund auf ihnen ist fast
garantiert und beweist nichts (E-113). Gezählt werden **Minutenbars**.

| Name | Symbol damals | Bars | Q-Ticker | Bars |
|---|---|---:|---|---:|
| Lehman Brothers | LEH | 0 | LEHMQ | 0 |
| Washington Mutual | WM | 0 | WAMUQ | 0 |
| Eastman Kodak | EK | 0 | EKDKQ | 0 |
| General Motors | GM | 0 | MTLQQ | 0 |
| Circuit City | CC | 0 | CCTYQ | 0 |
| Bear Stearns | BSC | 0 | — | — |

**Kontrollgruppe:** dieselbe Abfrage für Überlebende — 8 von 8 liefern Bars (7.008–7.992 im Probefenster).
Der Aufruf funktioniert also; das Schweigen bei den Ausscheidern ist
kein Fehler der Abfrage. Die Kontrolle liegt zudem im **früheren**
Fenster als alle Ausscheider — eine reine Datumsgrenze scheidet damit
als Erklärung aus.

> **Alle 6 geprüften Ausscheider liefern keine
> einzige Bar** — auch nicht unter ihrem Handelssymbol vor der
> Insolvenz. Für die Survivorship-Korrektur ist dieser Weg zu.

Mehr Anfragen erhöhen also die **Abdeckung** des Universums (viele
Überlebende sind schlicht noch nicht gezogen), aber nicht seine
**Unverzerrtheit**: die geprüften Ausscheider sind bei dieser Quelle
nicht zu haben. Dafür braucht es Tagesdaten mit Delisting-Kursen.

*Der Stand des bisherigen Pulls (39,3 % der
PIT-Mitglieder) beschreibt die Zusammensetzung der bisherigen
Anfrageliste, nicht den Endpunkt — er wird hier bewusst nicht als
Verzerrungsmaß ausgewiesen (Stage-2-Findings F-senior-1/7).*

---

## Was daraus folgt

1. Die Intraday-Ergebnisse aus P12 (keine Haltedauer von 1 Stunde
   bis 2 Jahre schlägt das Liegenlassen desselben Universums)
   bleiben **innerhalb** des Universums gültig — der Vergleich ist
   dort ceteris paribus, weil beide Seiten dieselbe Verzerrung
   tragen.
2. Jeder Vergleich **gegen SPY** oder gegen einen passiven ETF ist
   auf dieser Datenbasis nicht belastbar.
3. Der nächste Schritt ist kein weiterer Backtest, sondern ein
   Panel, das Ausscheider mitführt (PIT-Universum mit
   Delisting-Kursen).

