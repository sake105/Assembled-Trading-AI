# P12 — Das kurze Ende der Haltedauer (2026-08-03)

> **Dieses Dokument wird generiert** (`render_befund_p12.py`), nicht von Hand
> geschrieben. Grund: die erste Fassung war beim Schreiben korrekt und nach der
> Review-Remediation komplett veraltet — zwei von drei Schlussfolgerungen
> hatten sich umgekehrt (E-085). Jede Zahl unten stammt aus
> `results/p12*.json`.

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
- Verworfen: AON (Abdeckung nur 54.5%)

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

**1. Das kurze Ende trägt nicht.** Keine Haltedauer bis einschließlich einem
Tag kommt netto in die Nähe des schlichten Haltens: der beste kurze Wert ist
1,541× gegen 3,138×. Bei einstündigem
Halten fallen 18.504 Umschichtungen an.

**2. Vor Kosten gewinnen die kurzen Haltedauern — aber schlechter als das
Los.** Brutto liegen sie zwischen 2,291× und 3,236×,
also im Plus. In **5 von 8** kurzen Zeilen
liegt Momentum brutto jedoch **unter der Zufallsauswahl** — die Rangfolge nach
jüngster Rendite wählt dort aktiv schlechter als das Los, und zwar bevor eine
einzige Gebühr anfällt. Ein Brutto-Alpha ist am kurzen Ende also nicht
nachweisbar; die Kosten verschärfen das Bild zusätzlich.

*Belastbarkeit:* Diese Aussage beruht auf **fünf** Zufallsziehungen je Zeile
ohne ausgewiesenes Streuungsmaß. Die 60-Seed-Kontrolle (P12b) lief am
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

Die Abweichung ist **nicht systematisch gerichtet** (das Vorzeichen
wechselt), die Bruttokante ist also kein Verfahrensartefakt. Ihre Größe
reicht aber bis 12,0 %, während der Break-even bei 1 bps
liegt: die Artefaktschranke ist damit von derselben Größenordnung wie
der verbleibende Spielraum. Beides zusammen gelesen heißt: die Kante
ist real, aber die Aussage über ihre exakte Höhe ist es nicht.

## Was dieser Strang nicht beantwortet

- **Absolute Renditen** — Universum survivorship-verzerrt.
- **Andere Signale am kurzen Ende.** Getestet wurde Momentum und sein
  Gegenteil, nicht Orderbuch-, Nachrichten- oder Volatilitätssignale. Der
  Befund lautet: dieses Signal trägt dort nicht — nicht: dort ist nichts.
- **Ausführungsrealismus.** Bar-Kurs plus Pauschale; echte Marktwirkung bei
  stündlichem Umschlag wäre schlechter, nicht besser.
- **Gefüllte Bars.** Gerechnet wird auf `close.ffill()`. Eine gefüllte Bar
  erzeugt exakt 0 Rendite und geht in Signal und Umschichtung ein; bei
  stündlicher Haltedauer ist das nicht vernachlässigbar. Der
  Abdeckungsfilter lässt strukturell bis zu 10 % gefüllte Bars zu. Die
  Richtung ist konservativ — es dämpft das kurze Ende, rettet das negative
  Verdikt also nicht.
- **Der Holdout bleibt versiegelt.** Kein Kandidat aus P12 hat ihn verdient.

## Offene Folgeschritte

1. **60-Seed-Zufallskontrolle am kurzen Ende** (netto *und* brutto). Erst
   damit wird Aussage 2 vom Hinweis zum Beleg.
2. **Abdeckung je behaltenem Symbol** ins Ergebnis-JSON, damit der
   ffill-Anteil nachprüfbar ist statt nur beschränkt.
3. **CI-Status.** Tests und Lint sind lokal grün, nicht CI-bestätigt.

