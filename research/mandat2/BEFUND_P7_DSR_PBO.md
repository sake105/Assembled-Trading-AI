# P7 — DSR/PBO: kein Holdout-Schuss (2026-08-01)

Rohdaten: `results/p7_dsr_pbo.json`. Renditematrix 5.548 Tage × 37 Varianten
(3 Trend-Definitionen × 12 Fenster + ungegatet). Trial-Zähler **2.144**.

Genutzt wurden die bestehenden Implementierungen —
`src/assembled_core/qa/deflated_sharpe.py` und `cscv_pbo` aus Mandat I — statt
einer dritten Wahrheit.

---

## Das Ergebnis hängt an einer Annahme, und die konservative fällt durch

In-sample-Gewinner nach dem Zielmaß: `preis>sma/140`
(Median 6,678 · MaxDD −32,1 % · Sharpe täglich 0,0623 = **annualisiert 0,99**).

| Varianz-Annahme | N | DSR-Schwelle | p | Urteil |
|---|---|---|---|---|
| empirisch (37 Varianten) | 37 | 0,0132 | 0,9999 | bestanden |
| empirisch (37 Varianten) | 2.144 | 0,0211 | 0,9988 | bestanden |
| **IID-Näherung** | 37 | 0,0290 | 0,9932 | bestanden |
| **IID-Näherung** | **2.144** | **0,0465** | **0,8783** | **DURCHGEFALLEN** |

**PBO (CSCV, 8 Blöcke, 70 Splits): 20,0 %** — formal bestanden.

## Warum die konservative Zahl die richtige ist

Die empirische Varianz über die Familie beträgt 3,7 · 10⁻⁵ und ist damit
**4,8-fach kleiner** als die IID-Näherung. Das ist kein Zeichen von Stabilität,
sondern ein Artefakt: die 37 „Varianten" sind Fast-Klone derselben Strategie
mit leicht verschobenem Gate. Ihre Sharpes liegen naturgemäß eng beieinander.

`variance_across_trials` soll aber die **Streuung der Suche** abbilden — also
wie weit die Ergebnisse über alles auseinanderliegen, was man ausprobiert hat.
Sie aus einer engen Klonfamilie zu schätzen und gleichzeitig N aus der ganzen
Kampagne zu nehmen, mischt zwei unvereinbare Bezugsgrößen: **großes N, kleines
V** — und das drückt die Schwelle künstlich nach unten.

Dasselbe gilt für PBO. CSCV setzt voraus, dass die Spalten der Matrix
unterschiedliche Strategien sind. Bei Fast-Klonen überlebt der In-sample-Sieger
fast immer OOS, weil er sich von den anderen kaum unterscheidet. **20 % PBO
sind hier ein Struktur-Artefakt, kein Robustheitsnachweis.**

## Verdikt

**Kein Holdout-Schuss.** Der Kandidat besteht die Mehrfachtest-Korrektur nicht,
sobald man die Varianz nicht aus der eigenen Klonfamilie schätzt.

Das ist die ganze Funktion dieser Disziplin: Der Kandidat sah nach vier
bestandenen Tests (Zielfunktion, Zufallskontrolle mit Gate, Fensterband,
PBO) reif aus. Die eine Annahme, bei der man sich selbst betrügen kann, kippt
ihn. Der Holdout bleibt versiegelt — er ist einmal verwendbar, und dafür ist
dieser Kandidat nicht gut genug.

## Was ihn reif machen würde

1. **Eine echte Streuungsschätzung.** Sharpes über *unterschiedliche*
   Strategiefamilien (Momentum, Value, Zufall, Buy-and-Hold, gegatet und
   ungegatet), nicht über Gate-Fenster. Das ist die richtige Bezugsgröße für
   V — und sie ist mit den vorhandenen Läufen weitgehend schon erhoben.
2. **PBO über heterogene Spalten** statt über Klone.
3. **Eine ökonomische Begründung**, warum `preis > SMA` trägt, aber „SMA
   steigt" und „Rendite > 0" nicht (P5). Solange das unerklärt ist, bleibt der
   Verdacht, dass die spezifische Ein-/Ausstiegszeitpunktwahl auf zwei
   Ereignisse passt.

---

# Nachtrag: der Fundamentalfaktor ist datenblockiert

Der vierte Strang (Fundamentalbewertung mit der Methodik der
Anthropic-Finanz-Skills) lässt sich **auf diesem Suchfenster nicht testen**.

Die Skills nutzen P/E, EV/EBITDA, EV/Revenue, Bruttomarge, EBITDA-Marge,
Umsatzwachstum, ROE, ROIC, PEG, FCF-Yield (Comps) bzw. WACC, Terminal Growth,
EBIT-Marge, CapEx, Beta (DCF). Alle davon wären aus unseren 1,45 Mio
XBRL-Zeilen (743 Ticker) grundsätzlich baubar.

**Aber:** `filed_date` und `disclosure_date` beginnen erst am **2009-04-15**
(SEC-XBRL-Pflicht lief 2009–2011 gestaffelt an). PIT-korrekte
Fundamentaldaten gibt es also frühestens ab 2009, das Suchfenster endet
2016-12-31 — **keine acht Jahre, also nicht ein einziges vollständiges
10-Jahres-Fenster.** Die gesperrte Zielfunktion ist auf diesen Daten nicht
auswertbar.

Ohne PIT-Datum zu rechnen wäre Look-ahead; ein kürzeres Fenster zu nehmen
wäre eine andere Zielfunktion und nicht vergleichbar. Und längere
Fundamentalhistorie ist bei EODHD nicht freigeschaltet (403, siehe Memory
„EODHD-Plan-Entitlements").

**Verdikt: datenblockiert, nicht widerlegt.** Der Faktor bleibt eine offene
Frage, die eine andere Datenquelle braucht (Compustat/CRSP-Klasse) — das ist
eine Beschaffungs-, keine Forschungsentscheidung.
