# Indikator-Audit über alle bisherigen Trials (Stand 2026-08-09, N = 3.559)

**Auftrag Hans:** Reine Recherche, keine neuen Tests. Prüfen: basierten die
bisherigen Tests auf EINZELNEN Indikatoren oder MEHREREN? Falls mehreren:
mit welchen Gewichtungen, und wo ist an der Gewichtung noch etwas zu
variieren? Ergebnis als Liste für spätere Sessions.

**Quellen:** `research/registry.md` (H-011–H-088, Wellen 1–48b),
`research/ledger.md`, `research/mandat/FINAL_REPORT.md`, Produktions-Code
(mfv2, Overlays), heutige N1-/K1-Läufe. Kein Datenkontakt, keine Trials.

---

## 1. Kernbefund in einem Satz

Von ~3.559 Trials basierten **fast alle auf Einzelindikatoren oder auf
BINÄREN Kombinationen (UND/ODER/2-von-3-Gates)** — die **gewichtete,
kontinuierliche Fusion mehrerer Indikatoren wurde erstmals HEUTE getestet**
(12 Configs), und genau dort zeigte sich als einziges die bestandene
Bestätigung (K1/E1). Dein Verdacht war berechtigt: Dieser Raum ist fast
unerschlossen.

## 2. Klassifikation des Bestands

### 2a. Einzelindikator / Einzelsignal (der große Block, grob ~85 % der Trials)

| Familie | Beispiele | Verdikt |
|---|---|---|
| Momentum-Varianten | H-012/021/023/035/044/049, EU (W40c) | alle FAIL |
| Trend/Timing einzeln | H-071-Einzelfamilien (SMA/EMA/MACD/Donchian/RSI/Bollinger/TSMom, 1.032 Configs über 41 Assets) | „Technische Analyse: TOT" (0/25 SPY, 0/25 GLD, BTC-Fußnote idiosynkratisch) |
| Events einzeln | Insider (H-031/053/088), Congress (H-033/034), 13F (H-029/030, W39-Nachtrag 60/0), News (H-038), PEAD | alle FAIL |
| Vol/Low-Vol/Quality einzeln | H-017/040/041 | FAIL |
| Kalender/Saisonalität | H-045/075 | FAIL |
| Intraday einzeln | H-072 (7 Signale brutto negativ), P12-Haltedauern | FAIL |
| Geopolitik einzeln | Welle 48/48b (Truth Social), N1-GEO (GDELT, heute) | FAIL — Post-/Eskalationstage = Zufallstage |

### 2b. BINÄRE Kombinationen (getestet, aber ohne Gewichte — an/aus)

| Test | Kombinationslogik | Ergebnis |
|---|---|---|
| H-071-Kombis (W33) | UND/ODER/2-von-3: SMA200×Momentum, MACD×RSI, Donchian×VolFilter | 0 Überlebende auf SPY/GLD |
| H-019/025/026 Regime-Gates | Trend-/Vol-Gate schaltet Strategie an/aus | FAIL (gate_both bestätigend negativ) |
| H-074 | Makro-Regime konditioniert Endportfolio (an/aus) | FAIL |
| H-080b Event×Technik | Insider-Kauf UND über SMA200; Congress UND Momentum (Konfirmation) | 12/0 |
| H-087 Trendfilter | binär alles-rein/alles-raus am CAGR-Kriterium | 0/338 Fenster |

**Muster:** Kombination hieß bisher immer *Schnittmenge* (beide müssen
feuern) oder *Schalter*. Nie: „Indikator A zählt 50 %, B 30 %, C 20 %".

### 2c. GEWICHTETE Systeme (die kurze, vollständige Liste)

| System | Gewichte | Was variiert wurde | Was NIE variiert wurde |
|---|---|---|---|
| **mfv2** (Produktions-Faktorstack) | 9 aktive Faktoren mit festen, evidenz-abgeleiteten Gewichten; 6 auf 0,00 | Faktor-AUFNAHME (evidence-gated) | **die Gewichte selbst** — nie ein Raster; Full-Stack-OOS ergab Sharpe-Delta 0,00 (Paket 3c.2) |
| H-078 Portfolio-Labor | 48 ASSET-Gewichtskonstruktionen (SPY/Gold/BTC/ETH), Monte-Carlo-bewertet | Asset-Allokation → Endspez 65/25/5/5 bestätigt | Asset- ≠ Indikator-Gewichte; Signale spielten keine Rolle |
| H-054/055 Risk-Parity/Vol-Target | formelbasiert (inverse Vol) | — | keine freien Gewichte, Formel statt Raster |
| Produktions-Overlay-Kette | multiplikativ: georisk × profit_lock × turnover × vol_target | — | Kette nie als gewichtete Summe getestet |
| **N1-Komposit (HEUTE)** | 6 Gewichts-Configs Geo/Fin/TA | erstes echtes Indikator-Gewichtsraster | FAIL n. Bonferroni; Gradient: TA+FIN tragen, Geo verwässert |
| **K1-Dial (HEUTE)** | 8 Configs Trend/Vol/DD → Exposure-Regler | dito | E1 (100/0/0) auf 1926–1995 **BESTANDEN**; jede Beimischung verwässerte |

## 3. Die ehrliche Lehre aus 2c (heute doppelt gemessen)

Gewichtete Fusion half bisher **nicht durch Mischen, sondern durch
Weglassen**: In beiden heutigen Rastern war die beste Config die, die den
toten Komponenten Gewicht **entzog** (Geo → 0 im Komposit; Vol/DD → 0 im
Dial). Das widerlegt Gewichtungs-Exploration nicht — es sagt, dass das
Raster die 0 enthalten muss und dass „viele schwache Signale addieren"
kein Selbstläufer ist.

## 4. Offene Gewichtungs-Variationsräume (Kandidatenliste für morgen)

Geordnet nach Prior-Stärke; jede Zeile wäre eine eigene Registrierung mit
vorab fixiertem Raster, gebuchten Trials, Bonferroni und Kontrolle.
**Nichts davon ist getestet.**

| # | Kandidat | Raster-Idee | Prior |
|---|---|---|---|
| **K2** | **E1-Dial-Feinstruktur**: Dial-Tiefe (0,4/0,6/0,8) × graduierter Trendabstand (Sigmoid statt binär unter/über MA200) | 6–9 Configs | **positiv** (E1 bestätigt; einzige lebende Spur) |
| **K3** | **Asset-übergreifender Dial**: je Asset (SPY/GLD/ggf. BTC) eigener E1-Dial, Portfolio-Gewichte × Risiko-Score verknüpft (verbindet K1 mit H-078-Allokation) | 6 Configs | leicht positiv (beide Bausteine einzeln belegt/bestätigt) |
| **K4** | **mfv2-Gewichtsraster**: die 9 aktiven Faktoren grob umgewichten (z. B. Top-3 hochgewichtet vs. flach), NUR OOS bewertet | 4–6 Configs | neutral (Stack-OOS war 0,00 — aber Gewichte nie variiert) |
| **K5** | **Overlay-Kette als gewichtete Summe** statt Multiplikation (georisk/profit_lock/vol_target mit Raster) | 4 Configs | neutral |
| **K6** | **N1-Forward-Shadow** W1/W2/W5 inkl. Social (bereits eingefroren; braucht Wochen Sammler-Daten) | 3 Configs, läuft passiv | offen (einzige Fusion mit positivem historischem Gradienten) |
| **K7** | Event-Signale als GEWICHTETE Evidenz statt UND-Gate (Insider/Congress-Score fließt graduell in den Dial) | 4 Configs | **negativ** (alle Event-Stränge einzeln 0/x; nur der Vollständigkeit halber gelistet) |
| K8 | Konfirmations-GRADE: „2-von-3" durch kontinuierliche Zustimmungs-Quote ersetzen (0…1 statt an/aus) über die W33-Kombifamilien | 5 Configs | negativ (Basis-Signale tot; billig, da Daten/Code vorhanden) |

**Budget-Hinweis für morgen:** K2+K3 zusammen ≈ 12–15 Trials — verträglich.
K4–K8 nur nach Ergebnis von K2/K3 entscheiden, sonst Raster-Inflation
(heutige Lektion: die Latte steigt mit jedem Versuch).

## 5. Nicht vergessen (Kontext für jede weitere Runde)

- Holdout 2017-01..2026-07 bleibt versiegelt; freie Fenster: ≤2016 und
  Serien außerhalb der Mandats-Panels (1926-Reihe, GDELT-Cache).
- K1/E1-Bestätigung ist ein DEFENSIV-Ergebnis (MDD-Schutz), kein
  Renditemotor — Erwartungsmanagement für alle Folge-Raster.
- Jede Config = 1 Trial, vorab; alle Ergebnisse berichten; Bonferroni.
