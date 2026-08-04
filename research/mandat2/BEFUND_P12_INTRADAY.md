# P12 — Das kurze Ende der Haltedauer (2026-08-03)

> **Dieses Dokument wird generiert** (`render_befund_p12.py`), nicht von Hand
> geschrieben. Grund: die erste Fassung war beim Schreiben korrekt und nach der
> Review-Remediation komplett veraltet — zwei von drei Schlussfolgerungen
> hatten sich umgekehrt (E-085). Jede Zahl in den Tabellen und Kernaussagen
> stammt aus `results/p12*.json`. **Nicht** von dort: Kopfdatum, die
> Spannenangabe der Haltedauern, der Rückblick-Faktor und die im
> Buchhaltungs-Hinweis genannte Trial-Differenz — diese stehen im Generator
> und sind damit nicht gegen Drift geschützt.

Der Strang, den ich fälschlich für datenblockiert erklärt hatte (E-080).
Hans' Frage im Wortlaut: *„Du kannst Aktie auch kürzer halten, nur
mehrere wenige Monate oder wenige Stunden.“* P2 hatte das lange Ende
beantwortet. Hier ist das kurze.

## Die Datengrundlage

**20 Symbole**, Stundenbars aus dem EODHD-1m-Endpunkt
verdichtet. Ausgewertet wird ausschließlich das Suchfenster; der Holdout ist
in dieser Schicht nicht vorhanden, nicht bloß ungenutzt.

- Gemeinsames Fenster: **2006-06-22..2016-12-30** (10,52 Jahre)
- Warm-up 4.355 Bars, für **alle** Varianten identisch —
  gemessener Cash-Anteil max. 0,0000 %
- Verworfen: AON (Abdeckung nur 54,5%)

**Die Rohdaten sind unbereinigt.** Vor der Bereinigung enthält das Panel
16 Stundensprünge über 35 %, danach 8.
Die größten Rohsprünge sind Kapitalmaßnahmen, keine Marktbewegungen:

| Symbol | Zeitpunkt | roher Sprung |
|---|---|---|
| AIG | 2009-07-01 13:00 | 1740,7 % |
| AAPL | 2014-06-09 13:00 | -85,7 % |
| BEN | 2013-07-26 13:00 | -66,9 % |
| ADBE | 2005-05-24 13:00 | -49,9 % |
| BAC | 2004-08-30 13:00 | -49,8 % |
| BAX | 2015-07-01 13:00 | -46,0 % |

Bereinigt wird über den Anker, der in dieser Kampagne ohnehin gilt — das
tagesgenaue, total-return-adjustierte Panel. Der Tagesfaktor ist **innerhalb**
eines Tages konstant: Intraday-Renditen bleiben unverändert (Splits wirken
über Nacht), Übernacht-Renditen werden um Split *und* Dividende korrigiert.

### Survivorship — hart benannt

Das Universum ist **nicht** survivorship-frei: Namen, die 2004–2016
durchgehend im Index waren. Deshalb wird **ausschließlich innerhalb des
Universums** verglichen, nie gegen SPY (E-079). Absolute Renditeaussagen sind
aus diesem Strang nicht ableitbar; die relative Frage nach der Wirkung
kürzeren Haltens ist es.

## Der Test

Variiert wird die Haltedauer von **1 Stunde bis 2 Jahre**. Top 5
gleichgewichtet, 10 bps je Seite, Steuerwelt Null.
Positionen **driften zwischen den Terminen** (Stücke, nicht Gewichte) — sonst
wäre Haltedauer nicht das, was das Wort sagt.

Zwei Rückblick-Familien, weil ein mit der Haltedauer skalierender Rückblick
das Signal mitvariieren lässt und damit kein Ein-Parameter-Sweep mehr ist:

- **A** hält den Rückblick fest → echter Ein-Parameter-Sweep.
- **B** skaliert ihn mit der Haltedauer. Nur Zeilen ohne Deckelung; gedeckelte
  wären untereinander identisch parametriert und hatten im Vorlauf genau
  deshalb die Bestwerte geliefert (E-084).

**Offengelegter Freiheitsgrad:** Der Warm-up ist ein **gesetzter** Wert,
nicht abgeleitet. Er wirkt zweifach — er bestimmt den gemeinsamen
Fensterstart *und* welche Haltedauern Familie B überhaupt enthält (nur
solche mit 20× Rückblick ≤ Warm-up). Ein größerer Wert würde weitere
B-Zeilen zulassen und zugleich den Fensterstart nach hinten schieben.

**Benchmark — gleiches Universum, gleiche Gewichtungsmethode:**

| | Endwert | CAGR | MaxDD |
|---|---|---|---|
| Buy-and-Hold | 3,138× | 11,5 % | -54,7 % |
| EW monatlich rebalanciert | 3,244× | 11,8 % | -61,6 % |

## Das Ergebnis

### Familie A_fester_rueckblick

| Haltedauer | Rückblick | Umschicht. | netto | brutto | Zufall netto | Zufall brutto | MaxDD | Kostenlast |
|---|---|---|---|---|---|---|---|---|
| 1 Stunde | 882 | 18.504 | 0,710× | 2,360× | 0,000× | 3,104× | -61,3 % | 232,4 % |
| 2 Stunden | 882 | 9.252 | 1,041× | 2,416× | 0,000× | 2,382× | -58,4 % | 132,1 % |
| 4 Stunden | 882 | 4.626 | 1,397× | 2,498× | 0,003× | 3,398× | -56,6 % | 78,8 % |
| 1 Tag | 882 | 2.644 | 1,541× | 2,385× | 0,049× | 2,590× | -56,5 % | 54,8 % |
| 1 Woche | 882 | 529 | 2,729× | 3,239× | 1,155× | 2,560× | -52,9 % | 18,7 % |
| 1 Monat | 882 | 126 | 2,477× | 2,685× | 2,147× | 2,591× | -57,6 % | 8,4 % |
| 1 Quartal | 882 | 42 | 1,978× | 2,072× | 2,373× | 2,527× | -53,9 % | 4,7 % |
| 1 Jahr | 882 | 11 | 2,380× | 2,415× | 2,661× | 2,702× | -53,9 % | 1,5 % |
| 2 Jahre | 882 | 6 | 2,338× | 2,355× | 2,452× | 2,470× | -61,4 % | 0,7 % |

### Familie B_skalierter_rueckblick

| Haltedauer | Rückblick | Umschicht. | netto | brutto | Zufall netto | Zufall brutto | MaxDD | Kostenlast |
|---|---|---|---|---|---|---|---|---|
| 1 Stunde | 20 | 18.504 | 0,004× | 2,599× | 0,000× | 3,104× | -99,6 % | 58555,4 % |
| 2 Stunden | 40 | 9.252 | 0,100× | 2,564× | 0,000× | 2,382× | -91,8 % | 2470,8 % |
| 4 Stunden | 80 | 4.626 | 0,450× | 2,291× | 0,003× | 3,398× | -75,7 % | 408,9 % |
| 1 Tag | 140 | 2.644 | 1,272× | 3,236× | 0,049× | 2,590× | -64,8 % | 154,4 % |
| 1 Woche | 700 | 529 | 1,801× | 2,207× | 1,155× | 2,560× | -56,7 % | 22,6 % |
| 1 Monat | 2.940 | 126 | 1,921× | 2,025× | 2,147× | 2,591× | -53,7 % | 5,4 % |

Fett = schlägt Buy-and-Hold (3,138×).

## Was daraus folgt

**1. Das kurze Ende trägt nicht.** Keine Haltedauer bis einschließlich
einem Tag: der beste kurze Wert ist 1,541× gegen 3,138×.
Bei einstündigem Halten fallen 18.504 Umschichtungen an.

**2. Vor Kosten gewinnen die kurzen Haltedauern — aber schlechter als
das Los.** Brutto liegen sie zwischen 2,291× und 3,236×,
also im Plus. In **5 von 8** kurzen Zeilen
liegt Momentum brutto jedoch **unter der Zufallsauswahl** — die Rangfolge nach
jüngster Rendite wählt dort aktiv schlechter als das Los, und zwar bevor eine
einzige Gebühr anfällt. Ein Brutto-Alpha ist am kurzen Ende also nicht
nachweisbar; die Kosten verschärfen das Bild zusätzlich.

*Belastbarkeit:* Diese Aussage beruht auf **5** Zufallsziehungen je
Zeile ohne ausgewiesenes Streuungsmaß. Die große Kontrolle (P12b) lief am
**langen** Ende. Eine 60-Seed-Kontrolle am kurzen Ende ist offener
Folgeschritt — bis dahin ist Aussage 2 ein Hinweis, kein Beleg.

Eine frühere Fassung dieses Dokuments behauptete hier ein umgekehrtes
Vorzeichen (Bruttowert 0,159×). Das war ein Artefakt eines fehlerhaften
Laufs — feste Gewichte statt driftender Positionen, unterschiedliche
Startzeitpunkte, außerbörsliche Bars. Zurückgenommen.

**3. Keine einzige Zeile schlägt Buy-and-Hold.** Bester Netto-Wert
2,729× (1 Woche, A_fester_rueckblick) gegen
3,138×. Das reproduziert am Intraday-Panel, was P2 am Tagespanel fand:
Umschlag ist der schädliche Parameter.

Bester Brutto-Wert über alle Zeilen: 3,239×
(1 Woche, A_fester_rueckblick).

## Artefaktschranke des Bereinigungsverfahrens

Der Tagesfaktor *soll* eine Treppe sein, ist es aber nicht: er absorbiert
auch die Differenz zwischen Vendor-Tagesschluss und letzter Stundenbar —
ein reversierendes Rauschen, also gleichgerichtet mit dem Effekt, den ein
Intraday-Test am kurzen Ende sucht (E-083). Gegenprobe mit erzwungen
stufigem Faktor:

| Haltedauer | Familie | netto normal | netto stufig | Δ |
|---|---|---|---|---|
| 1 Stunde | A | 0,710× | 0,688× | -3,1 % |
| 2 Stunden | A | 1,041× | 1,010× | -3,0 % |
| 4 Stunden | A | 1,397× | 1,450× | 3,8 % |
| 1 Tag | A | 1,541× | 1,562× | 1,3 % |
| 1 Woche | A | 2,729× | 2,607× | -4,5 % |
| 1 Monat | A | 2,477× | 2,601× | 5,0 % |
| 1 Quartal | A | 1,978× | 1,969× | -0,5 % |
| 1 Jahr | A | 2,380× | 2,381× | 0,0 % |
| 2 Jahre | A | 2,338× | 2,332× | -0,2 % |
| 1 Stunde | B | 0,004× | 0,004× | -3,8 % |
| 2 Stunden | B | 0,100× | 0,099× | -0,7 % |
| 4 Stunden | B | 0,450× | 0,454× | 0,9 % |
| 1 Tag | B | 1,272× | 1,368× | 7,6 % |
| 1 Woche | B | 1,801× | 1,811× | 0,6 % |
| 1 Monat | B | 1,921× | 1,876× | -2,3 % |

## P12b — Momentum gegen Zufall am langen Ende

60 Zufallsziehungen je Haltedauer, jeweils mit derselben
NaN-Maske wie das echte Signal (sonst vergleicht die Kontrolle einen
anderen Zeitraum, E-082).

| Haltedauer | Momentum | Zufall Median | Zufall 5–95 % | Perzentil | p (einseitig) |
|---|---|---|---|---|---|
| 1 Monat | 2,477× | 2,562× | 1,194× – 3,629× | 47 % | 0.541 |
| 1 Quartal | 1,978× | 2,668× | 1,589× – 4,134× | 22 % | 0.787 |
| 1 Jahr | 2,380× | 2,781× | 1,575× – 4,305× | 28 % | 0.721 |
| 2 Jahre | 2,338× | 2,636× | 1,619× – 4,258× | 32 % | 0.689 |

## P12c — Trägt das umgekehrte Vorzeichen die Reibung?

| Haltedauer | Umschicht. | brutto | brutto stufig | Δ | bei 10 bps | Break-even vs. Halten |
|---|---|---|---|---|---|---|
| 1 Stunde | 18.504 | 6,140× | 5,956× | -3,0 % | 0,009× | 1 bps |
| 2 Stunden | 9.252 | 5,011× | 5,280× | 5,4 % | 0,171× | 1 bps |
| 4 Stunden | 4.626 | 4,282× | 4,180× | -2,4 % | 0,772× | 1 bps |
| 1 Tag | 2.644 | 2,697× | 3,020× | 12,0 % | 1,037× | nie |

*brutto stufig* ist die Gegenprobe mit erzwungen stufigem Tagesfaktor
(E-083). Reversal ist der Fall, den das reversierende Rauschen des
Bereinigungsverfahrens aufblähen würde — deshalb ist diese Spalte hier
Pflicht und nicht Fußnote.

Die Abweichung ist **nicht systematisch gerichtet** (das
Vorzeichen wechselt), die Bruttokante ist also kein
Verfahrensartefakt.

Für die tragende Zeile (**1 Stunde**, der höchste Bruttowert)
beträgt die Artefaktschranke 3,0 %. Ihr Break-even gegen
Buy-and-Hold liegt bei 1 bps — dort bleiben 3,180×
gegenüber 6,140× brutto, also 1,4 %
des Bruttovorsprungs über das schlichte Halten.

Schon **ein einzelner Basispunkt** kostet damit den Großteil
der Kante.
Sie ist real, aber die Aussage über ihre exakte Höhe ist es
nicht.

## Wie stark ist das Universum survivorship-verzerrt? (P12d)

Der Versuch, die Verzerrung durch ein Point-in-Time-Universum zu heilen,
scheiterte an der Datenquelle: der EODHD-**Intraday**-Endpunkt führt keine
delisteten Ticker (gemessen an 22 ausgeschiedenen Namen: 18 % Trefferquote
gegen 92 % bei Überlebenden). Das **Tages**panel enthält die Toten dagegen
vollständig — dort ist die Verzerrung wenigstens **bezifferbar**.

| Universum | n | B&H (Erlös gehalten) | B&H (umgeschichtet) | CAGR |
|---|---|---|---|---|
| intraday_p12 | 20 | 3,017× | 3,017× | 11,1 % |
| durchgehend_2004_2016 | 259 | 2,689× | 2,689× | 9,9 % |
| pit_2004 | 382 | 2,283× | 2,407× | 8,2 % |
| SPY (Referenz) | 1 | 2,183× | — | — |

**Überhöhung des P12-Benchmarks:** 2,9 % p. a., wenn
der Delisting-Erlös als totes Geld liegen bleibt, und 2,4 % p. a.,
wenn er pro rata auf die überlebenden Positionen verteilt wird. Beide Varianten sind
Buy-and-Hold und unterscheiden sich nur in dieser einen Annahme.

**Konsequenz für das Verdikt — und sie ist unbequem:** der Abstand
zwischen bestem Kandidaten (2,729×, 10,0 % p. a.) und Buy-and-Hold
(11,5 % p. a.) beträgt **1,5 % p. a.**
Die Verzerrung liegt mit 2,4 % bis 2,9 % **darüber**.

Das heißt nicht, dass eine Strategie das Halten schlägt. Es heißt, dass
**dieser Datensatz die Frage nicht entscheiden kann**: der gemessene
Vorsprung des Benchmarks ist kleiner als die bekannte Verzerrung seines
Universums. Ein survivorship-freier Intraday-Test wäre nötig — und ist
mit dieser Datenquelle nicht baubar, weil der Endpunkt keine delisteten
Ticker führt.

*Belastbarkeit der Ungleichung:* Marge und Verzerrung sind auf
**verschiedenen Kursquellen** gemessen — die Marge am Stundenpanel, die
Verzerrung am Tagespanel. An derselben Größe gemessen (identisches
Buy-and-Hold, dieselben 20 Namen, dasselbe Fenster) unterscheiden sich
die beiden Panels um 0,4 % p. a. — das sind
28 % der Marge. Der Schluss überlebt
das, weil der Abstand zwischen Verzerrung und Marge größer ist als diese
Basisdifferenz; ausgewiesen gehört sie trotzdem.

*Zerlegung der Überhöhung* (Variante „Erlös gehalten“): **1,7 % p. a.**
entfallen auf das Auswahlkriterium Dauermitgliedschaft (PIT → durchgehend),
**1,2 % p. a.** auf die weitere Verengung auf die 20
intraday-verfügbaren Namen. Beide Kanäle schauen vorwärts, die Summe ist
für die gestellte Frage also die richtige Zahl.

Gegen die Marge einzeln gehalten: **Dauermitgliedschaft** 1,7 % **über** der Marge, **Intraday-Auswahl** 1,2 % unter der Marge (1,5 %).

*Eine frühere Fassung dieses Abschnitts nannte hier +0,1 % p. a. und
schloss daraus, das Verdikt kippe nicht. Diese Zahl stammte aus einer
fehlerhaften Vergleichsrechnung — täglich rebalanciertes Portfolio statt
Buy-and-Hold, dessen Rebalancing-Bonus mit der Namenszahl wächst — und
ist zurückgenommen (E-096).*

**Die Tages-Engine der Kampagne (P1–P11): PIT-korrekt in der Auswahl,
aber nicht lückenlos.** `engine.run_strategy` wählt je Termin aus
`membership(t)` und erzwingt über `last_valid` den Delisting-Verkauf; das
Panel trägt Delistings (208 von 1.037 Symbolen enden vor Panelende), und
das PIT-Universum enthält nachweislich Pleite-Ticker
(EKDKQ, MTLQQ, WNDXQ).

Der Restkanal, den ich vorher zu Unrecht wegformuliert hatte: die
Preisabdeckung der Index-Mitglieder ist unvollständig, und die
fehlenden Namen sind überproportional mit Index-Austritten
angereichert. „Survivorship-frei“ ist zu stark — richtig ist:
*die Auswahl ist PIT-korrekt, die Abdeckung nicht vollständig, und die
Lücke ist nicht neutral.* **Die Zahlen dazu stehen im P12e-Abschnitt,
aus dem Artefakt** — hier stünden sie sonst ein zweites Mal und würden
beim nächsten Datenstand auseinanderlaufen. Der Intraday-Strang P12 ist
davon unabhängig und deutlich stärker betroffen.

*Nebenbefund Datenqualität:* 9 Namen des PIT-Universums sind
korrumpiert und wurden ausgeschlossen. Es handelt sich **nicht** um
einzelne Ausreißertage, sondern um Serien mit zwei ineinander
verschränkten Preisskalen über Dutzende Tage — bei MEL etwa liegt das
Niveau 2014-11-10..17 abwechselnd bei ~141.000 und ~7,80, wobei der
**niedrige** Wert der plausible ist. Weitere Fälle: **CIN**, **HPC**.

Die Truncation-Regel in `campaign_data` greift nur bei Vortagskursen unter
1 USD und lässt diese Klasse durch. Der Detektor hier sieht wiederum nur
diese eine Morphologie: dauerhafte Niveausprünge im Band 100–200 %
(AYE +170 %, TOY +155 %, HIG +102 %) passieren ungeprüft durch und bleiben
im Universum — ob sie echt sind, ist **offen**. Ob P1–P11 von den korrupten
Namen berührt sind, ist ebenfalls offen und ein eigener Prüfschritt.

## Sind die bisherigen Verdicts kontaminiert? (P12e)

P12d ließ zwei Fragen offen, die **alle** Phasen betreffen. Beide sind
jetzt gemessen — die erste mit einem Ergebnis, das ich zunächst falsch
hatte.

**Frage 1 — ist ein Preisfehler in eine Rendite eingegangen?** Das Panel
trägt über das volle Suchfenster **25 korrumpierte Namen**.
Davon sind **246 Übergangstage** — nur dort ist die *Tagesrendite* verzerrt.
An weiteren **32.332 Tagen** steht der Kurs auf einer falschen *Skala*; das ist
für Renditen folgenlos, solange Vortag und Tag dieselbe Skala teilen, und
wird nur für die beiden Momentum-Stützstellen relevant.

Zwei Kanäle können den Fehler in ein Ergebnis tragen:

**Kanal A — über einen Übergangstag gehalten:** 2 Namen an 11 Handelstagen.

| Name | Tage im Bestand | größte Tageswirkung | Rang unter allen Tagen |
|---|---|---|---|
| CFC | 9 (2007-05-17 … 2007-06-20) | -6,7 % am 2007-06-20 | 5532 von 5.548 |
| GPS | 2 (1996-12-20 … 1997-12-22) | 12,4 % am 1996-12-20 | 2 von 5.548 |

**Kanal B — im kontaminierten Momentum-Fenster gewählt:** 4 Namen, **22 von 5.040 Auswahlplätzen (0,44 %)**.
Das Fenster wird in HANDELSTAGEN gerechnet (21…252 nach dem Fehlertag), nicht in
Kalendertagen.

> **Zurückgenommen:** eine frühere Fassung meldete hier „4 berührt,
> keiner über den Halte-Kanal, 0,38 %“. Der Halte-Kanal war aus der
> **Auswahl** abgeleitet statt aus dem **Bestand** — die Engine verkauft
> aber erst bei `rang > rank_out`, hält also weit über den letzten
> Top-20-Termin hinaus. An der instrumentierten Engine gemessen ist der
> Halte-Kanal nicht leer, sondern der wirksamere der beiden (E-102).

Ob das ein Verdikt dreht, ist damit **nicht** gesagt — dafür bräuchte es
einen Lauf aller Phasen mit bereinigtem Panel. Der Befund grenzt die
Frage ein, er beantwortet sie nicht.

**Geltungsbereich:** gemessen für **eine** Konfiguration (momentum_score (12-1), top20,
ungegatet). Phasen mit anderem Score, anderem `top_in` oder einem
Risk-off-Gate wählen andere Namen — dafür gilt der Befund nicht.

**Die korrumpierten Serien, aus dem Artefakt:**

| Name | Fehlertage | von | auf | Faktor |
|---|---|---|---|---|
| TWX | 4739 | 1 | 1,62 | 5,70 | 3,5× |
| SWKS | 4686 | 1 | 1,32 | 4,09 | 3,1× |
| RHT | 4280 | 1 | 33,06 | 141,25 | 4,3× |
| WIN | 2564 | 2 | 27,97 | 295,25 | 10,6× |
| MCIC | 1930 | 9 | 17219,70 | 1000000,00 | 58,1× |
| RX | 1848 | 5 | 2,89 | 14,63 | 5,1× |

**Frage 2 — wie groß ist die Abdeckungslücke?** Jetzt aus dem Artefakt
statt aus der Prosa (vorher war ausgerechnet die Zahl, die eine
Einschränkung trägt, die einzige ohne Beleg):

- Anteil der Index-Mitglieder mit Preisspalte: **84,0 % bis 96,2 %**, Median 91,3 %
- Überlebensquote 1996-01-31 → 2016-12-30: mit Preisspalte **46,2 %**, ohne **9,0 %**
- Die Lücke ist **5,1-fach** mit Index-Austritten angereichert — sie ist
  nicht neutral.

## Was dieser Strang nicht beantwortet

- **Absolute Renditen** — Universum survivorship-verzerrt (beziffert in P12d).
- **Andere Signale am kurzen Ende.** Getestet wurde Momentum und sein
  Gegenteil, nicht Orderbuch-, Nachrichten- oder Volatilitätssignale. Der
  Befund lautet: dieses Signal trägt dort nicht — nicht: dort ist nichts.
- **Ausführungsrealismus.** Bar-Kurs plus Pauschale; echte Marktwirkung bei
  stündlichem Umschlag wäre schlechter, nicht besser.
- **Gefüllte Bars.** Gerechnet wird auf `close.ffill()`. Eine gefüllte Bar
  erzeugt exakt 0 Rendite und geht in Signal und Umschichtung ein; bei
  stündlicher Haltedauer ist das nicht vernachlässigbar.
  Der Abdeckungsfilter lässt strukturell gefüllte Bars zu; dieses
  Lauf-Artefakt führt die Schwelle noch nicht, deshalb ist sie hier
  nicht beziffert.
  Die Richtung ist konservativ — es dämpft das kurze Ende, rettet das
  negative Verdikt also nicht.
- **Der Holdout bleibt versiegelt.** Kein Kandidat aus P12 hat ihn verdient.

## Buchhaltungs-Hinweis zum Trial-Zähler

Ein Wiederholungslauf von P12c zur Artefakt-Hygiene (bit-identische
Ergebnisse, nur ein fehlendes Metadatenfeld) hat **44 Trials** gezählt,
ohne eine einzige neue Hypothese zu prüfen. Der Zähler steuert den
DSR-Haircut und bedeutet „Zahl geprüfter Hypothesen“ — Regenerationen
gehören nicht hinein. Die Skripte haben dafür jetzt `--regen`; die bereits
gezählten 44 werden **nicht** stillschweigend zurückgeschrieben, sondern
hier offengelegt (E-090). Wirkung: der Haircut ist um diesen Betrag zu
streng, also konservativ.

## Offene Folgeschritte

1. **60-Seed-Zufallskontrolle am kurzen Ende** (netto *und* brutto). Erst
   damit wird Aussage 2 vom Hinweis zum Beleg.
2. **Abdeckung je behaltenem Symbol** ins Ergebnis-JSON, damit der
   ffill-Anteil nachprüfbar ist statt nur beschränkt.
3. **CI-Status.** Tests und Lint sind lokal grün, nicht CI-bestätigt.

