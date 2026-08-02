# P9 Forensik — die offene Frage ist beantwortet (2026-08-02)

Rohdaten: `results/p9_gate_forensik.json`.
**Kein Trial verbraucht** — hier wird ein gemessenes Ergebnis zerlegt, kein
Kandidat ausgewählt. Trial-Zähler bleibt bei 2.144.

Die aus P5/P8 offene Frage: warum trägt `Preis > SMA`, aber „SMA steigt" und
„Rendite > 0" nicht?

---

## Antwort 1 — Sie tragen alle drei. Der P5-Unterschied war schmaler als er aussah.

Beim Fenster 200 bestehen **alle drei** Definitionen, mit fast identischem
Drawdown:

| Definition | MaxDD | investiert | Median |
|---|---|---|---|
| `Preis > SMA` | −30,5 % | 65,2 % | 4,124 |
| `SMA steigt` | −31,5 % | 67,3 % | 3,522 |
| `Rendite > 0` | −31,5 % | 73,8 % | 3,091 |

Die P5-Lückigkeit entstand bei *anderen* Fenstern, nicht hier. Meine
Formulierung „es trägt die Formel, nicht die Trendfolge" war zu scharf.

## Antwort 2 — Der eigentliche Befund: die Entscheidung fällt 12- bis 18-mal.

Das Gate wird nur an **264 Monatsenden** gelesen. Die täglichen Flips sind
weitgehend irrelevant:

| Definition | tägliche Flips | **wirksame Regimewechsel** |
|---|---|---|
| `Preis > SMA` | 140 | **18** |
| `SMA steigt` | 14 | **12** |
| `Rendite > 0` | 123 | **17** |

**Über 22 Jahre trifft der Mechanismus zwölf bis achtzehn Entscheidungen.**
Und davon zählen im Ergebnis etwa vier: Ausstieg und Wiedereinstieg in jedem
der beiden Bärenmärkte.

Dazu passt, dass die drei Definitionen an **86,4 %** der Rebalance-Termine
identisch sind (paarweise 89–93 %). Sie sind fast dasselbe Signal; sie
unterscheiden sich an rund 15 bis 35 von 264 Tagen — und diese Handvoll Tage
entscheidet über Bestehen oder Durchfallen.

## Antwort 3 — Die Definitionen unterscheiden sich in der Reaktionsgeschwindigkeit

Wie tief war SPY schon, als das Gate zum ersten Mal ausstieg?

| | Dotcom | Finanzkrise |
|---|---|---|
| `Preis > SMA` | −8,0 % (18.02.2000) | −7,3 % (03.08.2007) |
| `Rendite > 0` | −9,0 % (24.02.2000) | −9,5 % (21.11.2007) |
| `SMA steigt` | −12,9 % (16.10.2000) | −16,0 % (25.01.2008) |

`Preis > SMA` reagiert am schnellsten. `SMA steigt` reagiert deutlich später,
bleibt dafür aber sehr lange draußen (Okt 2000 bis Jan 2002; Jan 2008 bis
Aug 2009 — nur zwei Schaltungen in der ganzen Finanzkrise).

Das ist ein echter, erklärbarer Mechanismus-Unterschied: **Lage-Signal reagiert
schnell, Steigungs-Signal reagiert träge und hält länger durch.** Welches
besser ist, hängt von der Form der Krise ab — langsame Bärenmärkte belohnen
beides, ein V-förmiger Einbruch bestraft das träge Signal.

## Antwort 4 — Es gibt nicht *ein* Ereignis, sondern mindestens zwei

Ohne Gate reißen 144 von 144 Fenstern, und sie haben **keinen gemeinsamen
Teilzeitraum** (das früheste Fenster endet vor Beginn des spätesten). Meine
P5-Vermutung „ein einziges Ereignis" war falsch: es sind Dotcom *und*
Finanzkrise, und jedes 10-Jahres-Fenster in 1995–2016 enthält mindestens eines
davon.

Das macht die Sache nicht besser. Zwei Ereignisse sind für einen Mechanismus,
der über vier entscheidende Kalls gemessen wird, immer noch fast nichts.

---

## Was das für das Gesamturteil bedeutet

**Die DSR-Rechnung aus P7/P8 war zu optimistisch — und zwar aus einem Grund,
den keine Varianzkorrektur behebt.** Sie behandelt 5.548 Tagesrenditen als
Stichprobe. Die *Entscheidung*, die den Kandidaten trägt, fällt aber zwölf- bis
achtzehnmal, und ihr Ergebnis hängt an vier Kalls in zwei Ereignissen.

Effektive Freiheitsgrade in dieser Größenordnung erklären alles, was vorher
seltsam aussah:

* warum die Zahl gerissener Fenster fast binär ist (0 / 64 / 69) — ein
  einzelner Kall kippt eine ganze Fensterklasse,
* warum der DSR-Wert genau an der Schwelle liegt (0,9512),
* warum die Alternativen „lückig" bestehen — sie unterscheiden sich an einer
  Handvoll Tagen.

**Damit ist das negative Verdikt aus P8 nicht nur bestätigt, sondern
verstärkt.** Der Kandidat ist nicht knapp gescheitert; er war nie belastbar
genug, um überhaupt knapp zu sein.

## Korrektur an meiner eigenen P5-Aussage

In `BEFUND_P5_GATE_ROBUSTHEIT.md` steht: *„nicht ‚im Aufwärtstrend investiert
sein' trägt das Ergebnis, sondern die spezifische Ein- und
Ausstiegs-Zeitpunktwahl von `preis > SMA`."* Das ist so nicht haltbar — bei
Fenster 200 tragen alle drei. Richtig ist: die Definitionen unterscheiden sich
in der **Reaktionsgeschwindigkeit**, und weil die effektive Stichprobe winzig
ist, entscheidet diese Geschwindigkeit an wenigen Tagen über das Gesamturteil.
