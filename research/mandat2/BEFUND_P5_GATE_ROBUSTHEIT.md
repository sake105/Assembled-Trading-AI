# P5 Gate-Robustheit — gemischtes Urteil (2026-08-01)

72 Läufe: 12 Fenster (100…320) × 3 Trend-Definitionen × 2 Steuerwelten.
Rohdaten: `results/p5_gate_robustheit.json`. Trial-Zähler **2.124**.

Zwei Tests, beide vor dem Lauf festgelegt:
1. **Fenster-Band statt Punkt** — funktioniert nur ein Wert, ist es Rauschen.
2. **Andere Trend-Definitionen** — trägt die Trendfolge oder nur die eine Formel?

---

## Test 1 — BESTANDEN. Und meine P4-Sorge war falsch.

In P4 hatte ich geschrieben: *„Drei Werte getestet, der mittlere gewinnt — das
Muster eines gefundenen Parameters."* Das war eine Fehldiagnose aus zu grobem
Raster. Mit feinem Raster:

| Fenster (ZERO, `preis>sma`) | MaxDD | Marge zum Deckel | Median |
|---|---|---|---|
| 100 | −35,3 % | −0,3 % ✗ | 4,471 |
| 120 | −34,2 % | +0,8 % | 4,638 |
| 140 | −32,1 % | +2,9 % | **6,678** |
| 160 | −31,1 % | +3,9 % | 6,677 |
| 180 | −30,5 % | +4,5 % | 4,708 |
| 200 | −30,5 % | +4,5 % | 4,124 |
| 220 | −26,1 % | +8,9 % | 3,811 |
| 240 | −23,7 % | +11,3 % | 3,586 |
| 260 | −29,2 % | +5,8 % | 3,261 |
| 280 | −27,8 % | +7,2 % | 3,300 |
| 300 | −34,6 % | +0,4 % | 3,031 |
| 320 | −29,0 % | +6,0 % | 3,652 |

**11 von 12 Fenstern bestehen, zusammenhängend von 120 bis 320.** In
`PRIVAT_DE` bestehen alle 12. Das ist ein Band, kein Punkt. Der einzige
Ausreißer ist 100 — und genau den hatte ich in P4 als einen von drei Werten
getestet, was den falschen Eindruck erzeugte.

Der Median liegt über das gesamte Band zwischen 3,03 und 6,68 — überall
deutlich über dem Benchmark (1,948). Auch das spricht für ein Band statt für
eine gefundene Zelle.

## Test 2 — DURCHGEFALLEN. Es trägt die Formel, nicht die Trendfolge.

| Welt | Definition | bestanden | Fenster |
|---|---|---|---|
| ZERO | `preis > SMA` | **11/12** | zusammenhängend |
| ZERO | `SMA steigt` | 5/12 | **lückig** [120,160,180,200,220] |
| ZERO | `Rendite > 0` | 3/12 | **lückig** [100,140,200] |
| PRIVAT_DE | `preis > SMA` | **12/12** | zusammenhängend |
| PRIVAT_DE | `SMA steigt` | 7/12 | lückig |
| PRIVAT_DE | `Rendite > 0` | 6/12 | lückig |

Die beiden Alternativen sind normalerweise stark mit `preis > SMA` korreliert.
Dass sie hier so deutlich auseinanderlaufen — und lückig statt bandförmig
bestehen — heißt: **nicht „im Aufwärtstrend investiert sein" trägt das
Ergebnis, sondern die spezifische Ein- und Ausstiegs-Zeitpunktwahl von
`preis > SMA`.**

## Die eigentliche Fragilität: ein einziges Ereignis entscheidet

Die Zahl gerissener Fenster ist über alle 72 Läufe fast binär:

| gerissene Fenster | Anzahl Läufe |
|---|---|
| **0** | 44 |
| 64 | 7 |
| **69** | 21 |

Nichts dazwischen. Das heißt: das Gate steigt vor dem großen Einbruch aus —
oder es tut es nicht. Wenn nicht, reißen auf einen Schlag alle ~69 Fenster,
die das Ereignis enthalten.

**Der ganze Befund hängt damit an einem einzigen Ereignis** (mit hoher
Wahrscheinlichkeit 2008). Das ist keine Verteilung von Evidenz, das ist ein
Datenpunkt mit 144 Ausprägungen. Genau deshalb ist der Holdout mit dem
COVID-Crash der entscheidende Test: ein zweites, andersartiges Ereignis.

---

## Verdikt

**Der P4-Kandidat überlebt P5 nur zur Hälfte.** Er ist robust gegen die
Fensterwahl und nicht robust gegen die Formulierung. Das ist besser als
befürchtet und schlechter als nötig.

**Was ich daraus nicht ableite:** dass die Strategie funktioniert. Was ich
ableite: sie ist reif genug für DSR/PBO, und *danach* für den einen
Holdout-Schuss. Vorher nicht — bei Trial-Zähler 2.124 und einer
Ein-Ereignis-Abhängigkeit wäre alles andere leichtfertig.

**Korrektur an meiner eigenen P4-Formulierung:** Der Satz „Drei Werte getestet,
der mittlere gewinnt — das Muster eines gefundenen Parameters" war eine
voreilige Diagnose. Die richtige Reaktion auf drei Datenpunkte ist ein feineres
Raster, nicht ein Urteil.
