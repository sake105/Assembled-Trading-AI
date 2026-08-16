# Test-Budget-Ledger — FORSCHUNGSMANDAT (append-only)

Regel (Mandat §4.2): Jeder ausgeführte Backtest-Lauf zählt — auch Varianten, Fehlläufe,
"nur mal kurz geschaut". Das kumulative N fließt in jede DSR-Berechnung ein.
Reine Benchmark-Reihen (SPY TR, ETF-Netto-Pfad, EW-Baseline) sind KEINE Trials
(kein Selektionsdruck — sie sind die Messlatte, nicht der Kandidat), werden aber je
Experiment transparent mitgeführt.

## N₀-Rekonstruktion (festgeschrieben 2026-07-05)

| Quelle | Trials | Beleg |
|---|---|---|
| 10 Closure-Strategien (OOS negativ) | 10 | docs/PROJEKT_ABSCHLUSS_2026_05.md |
| Fable-Exploration 2026-06-13, Runden 1–5 (Mill-Trials H1–H6, Varianten, Polygon-Intraday) | 30 | research/fable_exploration/ERGEBNIS.md — DSR-Rechnungen liefen dort zuletzt bei n_trials=40 kumulativ |
| **N₀ gesamt** | **40** | konservativ = der höchste dort verwendete Stand |

Hinweis: Die 252 Experiment-Configs aus Fable Round 4 waren ein Overlay-Sweep mit
EINEM vorab spezifizierten Auswahlkriterium und wurden dort nicht als 252 Einzeltrials
gezählt; diese Konvention wird beibehalten (Familie = registrierte Trials laut Registry).

## Zufallsmaßstab (E[max Sharpe] unter H₀ bei N Trials, ~√(2·ln N))

| N | E[max SR] (annualisiert, ~10J daily) |
|---|---|
| 40 | ~0.43 |
| 44 | ~0.44 |
| 60 | ~0.45 |

(Interpretation: Ein "bestes" Ergebnis unterhalb dieses Sharpe ist mit Zufall aus N
Versuchen voll erklärbar. Formel: E[max Z] ≈ √(2·ln N), skaliert auf Jahres-Sharpe
bei T=2520 Handelstagen: E[max SR] ≈ √(2·ln N / T) · √252.)

---

## Läufe

| # | Datum | H-ID | Lauf | N kumulativ | Ergebnis (Kurz) |
|---|---|---|---|---|---|
| 1–40 | ≤2026-06-13 | H-001…H-010 + Fable | historisch | 40 | alle FAIL/NULL (Closure + ERGEBNIS.md) |
| 41 | 2026-07-05 | H-011 | V1_basis | 41 | net CAGR 8.23 %, Sharpe 0.62, MaxDD −23.8 % — FAIL |
| 42 | 2026-07-05 | H-011 | V2_mom_only | 42 | net CAGR 13.23 %, Sharpe 0.74 — beste Variante, trotzdem < ETF-Pfad |
| 43 | 2026-07-05 | H-011 | V3_no_gate | 43 | net CAGR 10.95 %, Sharpe 0.71 |
| 44 | 2026-07-05 | H-011 | V4_tight_buffer | 44 | net CAGR 8.57 %, Sharpe 0.63 |

### H-011 Ergebnisblock (2026-07-05) — VERDICT: FAIL (explorativ, nicht verdict-fähig)

Daten: Alpaca Daily adj=all, 2016-01→2026-07 (~10,5 J, effektiv ab 2017 nach Warmup),
503 Symbole (survivorship-biased, Bias SCHMEICHELT der Strategie), PIT-XBRL-ROE
(540 Symbole, available_at-gated). Benchmarks über identische Steuer-/Kosten-Engine.

| Reihe | Endwert (100k Start) | net CAGR | net Sharpe | MaxDD |
|---|---|---|---|---|
| **V1 Kandidat A (Verdict-Variante)** | 229.360 | 8,23 % | 0,621 | −23,8 % |
| V2 Momentum-only (Ablation) | 368.407 | 13,23 % | 0,737 | −24,6 % |
| V3 ohne Gate (Ablation) | 297.414 | 10,95 % | 0,706 | −25,9 % |
| V4 enger Puffer (Ablation) | 236.841 | 8,57 % | 0,630 | −23,7 % |
| **ETF-Netto-Pfad (Beat-Kriterium)** | **373.261** | **13,38 %** | — | — |
| EW-Survivorship-Baseline (Kontrolle) | 416.612 | 14,57 % | 0,851 | −37,6 % |
| SPY TR brutto (dekorativ) | — | 15,05 % | 0,887 | −33,8 % |

Vorab-Kriterien: (1) > ETF-Pfad? **NEIN** (−144k). (2) Sharpe > EW-Baseline? **NEIN**
(0,62 < 0,85). (3) DSR: prob 0,414, passes_5pct **NEIN** (n_trials=44). (4) PBO CSCV
0,514 > 0,50 → **Fail-Flag**. (5) Teilperioden: V1 > EW nur in 1/5 Fenstern (2024-25)
→ **NEIN**. → **FAIL auf allen fünf Linien.**

**Learnings (Daten, keine Meinung):**
1. **Steuer/Turnover ist der größte einzelne Feind:** V1 zahlte 39.852 € Steuern +
   13.477 € Kosten auf 100k Start — die EW-Baseline nur 1.704 € Kosten. Monatliches
   Rebalancing + ATR-Exits realisieren Gewinne permanent → FIFO-Steuerleck exakt wie
   Mandat §2.4 warnt.
2. **Das ROE-Quality-Leg SUBTRAHIERT** (V2 ohne Quality +5,0 %p CAGR vs V1): unser
   Quality ist nur ROE (kein COGS → Novy-Marx Gross Profitability nicht verfügbar) —
   ein Daten-Problem, kein Faktor-Urteil.
3. **Das 50 %-Gate SUBTRAHIERT über dieses Fenster** (V3 > V1): 2016–2026 ist fast
   durchgehend Bullenmarkt; das Gate konnte nur kosten. Kein Urteil über Krisenschutz
   (dafür ist das Fenster ungeeignet).
4. **Survivorship-Gift live repliziert:** EW-Baseline (416k) schlägt sogar den
   ETF-Pfad — auf diesem Universum ist „nichts tun" der beste Lauf. Jede künftige
   Strategie MUSS an dieser Baseline gemessen werden, nicht an SPY.
5. Selbst die beste Ablation (V2, 368k) bleibt UNTER dem ETF-Netto-Pfad (373k) —
   nach deutschen Steuern schlägt hier nichts den passiven Pfad, obwohl der
   Survivorship-Bias für die Strategie arbeitet. Diese NULL ist deshalb robust.

| 45 | 2026-07-05 | H-012 | out60 | 45 | net CAGR 18.90 %, Sharpe 0.76, MaxDD −41.5 % — Familie-Bestes |
| 46 | 2026-07-05 | H-012 | out80 | 46 | net CAGR 16.52 %, Sharpe 0.70 |
| 47 | 2026-07-05 | H-012 | out100 | 47 | net CAGR 16.67 %, Sharpe 0.71 |

### H-012 Ergebnisblock (2026-07-05) — VERDICT: FAIL (4 von 5 Kriterien; explorativ)

Auswahl (vorab definiert: bestes Netto-Endvermögen): out60 → 614.905 € (100k Start).

Vorab-Kriterien: (1) > ETF-Pfad 373.261? **JA** (+241.644 — einziges PASS-Kriterium).
(2) Sharpe 0,763 > EW-Baseline 0,851? **NEIN.** (3) DSR prob 0,585, passes **NEIN**
(N=47). (4) PBO(3er-Familie) **0,771 > 0,5** → stark overfit-verdächtige Auswahl.
(5) Fenster ≥ EW: nur 2/5 (2022-23, 2024-25) → **NEIN**. → **FAIL.**

**Learnings:**
1. **Der Steuer-Turnover-Hebel ist REAL und groß:** identisches Momentum-Signal,
   nur Umsetzung geändert (kein Retrim, kein ATR, breiter Puffer): 368k (H-011-V2,
   monatlich getrimmt) → 615k (out60). Delta +247k stammt NICHT aus dem Signal,
   sondern aus Steuerstundung + Let-winners-run (Bessembinder-Asymmetrie §2.2,
   Design-Priorität §2.4 bestätigt). Kosten 5,8k vs 13,5k; Steuer später gezahlt.
2. **Aber kein Alpha-Beweis:** Mehr-Endvermögen kommt mit mehr Risiko (MaxDD −41,5 %
   vs −37,6 % EW; Sharpe darunter), PBO 0,77 sagt: die out60-Auswahl innerhalb der
   Familie ist wahrscheinlich Rauschen. Auf Survivor-Daten nicht von der Baseline
   trennbar.
3. Die praktische Lehre fürs Mandat: **Steuer-/Umsetzungs-Design ist der validierbare
   Teil** (mechanisch, kein Signal-Claim) — Signal-Alpha bleibt auf diesen Daten
   unentscheidbar (wie Fable-Closure).

| 48 | 2026-07-05 | H-013 | B1_1x_cash | 48 | net CAGR 6.75 %, Sharpe 0.57 — klar unter ETF-Pfad |
| 49 | 2026-07-05 | H-013 | B2_2x_1x | 49 | net CAGR 15.09 %, Sharpe 0.64, MaxDD −50.7 % |
| 50 | 2026-07-05 | H-013 | B3_2x_cash | 50 | net CAGR 10.23 %, Sharpe 0.51, MaxDD −54.0 % |

### H-013 Ergebnisblock (2026-07-05) — VERDICT: FAIL (4 von 5; survivorship-immun, aber 10J-Bullenfenster)

Fenster nach SMA200-Warmup: 2016-10→2026-07 (~9,7 J); ETF-Netto-Pfad dieses Fensters: 350.791.
Auswahl (bestes Endvermögen): B2 (2x/1x) → 391.103.

Vorab-Kriterien: (1) B2 > ETF-Pfad? **JA** (einziges PASS; B1 und B3 klar darunter).
(2) DSR prob 0,385 **NEIN** (N=50). (3) MaxDD −50,7 % vs SPY −33,8 % → **NEIN**.
(4) PBO 0,714 → **NEIN**. (5) Fenster ≥ SPY: 2/5 → **NEIN**. → **FAIL.**

**Learnings:**
1. **B2s Endwert-Sieg ist Hebel-Beta, kein Timing-Skill:** Sharpe 0,64 < SPY 0,90 —
   mehr Endvermögen nur über mehr (gehebeltes) Risiko im Bullenmarkt. Genau das
   Muster, vor dem die Warnbox (RORO-Liquidation) warnt.
2. **Das Timing selbst war in diesem Fenster negativ:** 2018-19 (0,15 vs 0,82) und
   2022-23 (−0,47 vs 0,18) — die Gate-Switches kosteten Rendite UND Steuern
   (B2: 97k Steuern durch Switch-Realisationen).
3. **B1 (unlevered) ist die ehrlichste Version und die schwächste** (189k vs 351k
   ETF-Pfad): monatliche SMA200-Rotation nach deutschen Steuern zerstört in einem
   Bullenjahrzehnt ~46 % des passiven Endvermögens.
4. Kein Urteil über Krisenschutz-Wert des Gates — dafür braucht es ein Fenster mit
   echten Bärenmärkten (2000-03, 2008) = Norgate-Daten.

---

## Zwischenstand nach 3 Experimenten (2026-07-05, N=50)

Alle 3 Mandats-Kandidaten-Familien getestet: H-011 (Kandidat A) FAIL 5/5,
H-012 (Steuer-Momentum) FAIL 4/5, H-013 (Kandidat B) FAIL 4/5. Das einzige
reproduzierte positive Muster ist MECHANISCH, kein Signal: Turnover-/Steuerstundungs-
Minimierung (+247k bei identischem Signal, H-012-Learning 1). Kumulative
DSR-Hürde bei N=50 entspricht E[max SR|Zufall] ≈ 0,45 — jedes künftige Ergebnis
unter dieser Schwelle ist mit Zufall erklärbar.

---

## Welle 2 (2026-07-06) — Läufe 51–63, N=63. ALLE 6 FAMILIEN: FAIL.

Referenzen: ETF-Netto-Pfad 373.261 · EW-Baseline 416.612 / Sharpe 0,851 ·
H-012-out60 (backstop-/gate-/TLH-los) 614.905.

| Familie | Läufe (final €) | Selected | DSR | PBO | Verdict |
|---|---|---|---|---|---|
| H-014 TLH 15 % | out60 516.351 · out80 518.830 · out100 453.779 | out80 | 0,50 ✗ | **1,00** ✗ | **FAIL** — Delta>0 nur 1/3 Paare; bestes TLH −96k unter H-012-Best |
| H-015 ATR-Exits | 3x 291.022 · 4x 385.179 · 5x 480.415 | 5x | 0,54 ✗ | 0,37 ✓ | **FAIL** — monoton: je enger der Stop, desto schlechter; ALLE unter backstop-los 614.905 |
| H-016 GEM SPY/IEF | classic 249.184 · abs 247.815 | classic | 0,49 ✗ | 0,66 ✗ | **FAIL** — Sharpe 0,75 < SPY-B&H 0,89; survivorship-immun, deshalb besonders glaubwürdige NULL |
| H-017 Low-Vol | 248.693 (Sharpe 0,67, Steuern nur 23k) | — | 0,43 ✗ | n/a | **FAIL** — turnover-ärmste Strategie, aber Rendite fehlt |
| H-018 52wk-High | 217.439 (Sharpe 0,56) | — | 0,30 ✗ | n/a | **FAIL** — klar schlechter als 12-1-Momentum (Paarvergleich −397k) |
| H-019 Gate-Familie | sma 600.666 · vol **660.509** · both 613.133 | vol | 0,55 ✗ | **0,91** ✗ | **FAIL** — Vol-Gate +46k über gate-los, aber PBO 0,91 = Selektionsrauschen |

**Learnings Welle 2:**
1. **TLH ist im Momentum-Kontext kontraproduktiv** (PBO 1,0; 2/3 Paare negativ):
   realisierte Verluste + Rebound-Verpassung + Kosten > Topf-Nutzen — zumal in einem
   Bullenjahrzehnt die Töpfe klein bleiben. Der §2.4-Steuer-Edge liegt im NICHT-Handeln
   (Stundung), nicht im aktiven Ernten.
2. **Jeder getestete Exit-/Schutz-Mechanismus kostet monoton Endvermögen** (ATR 3x→5x:
   291k→480k, alle unter 615k). Konsistent mit H-011 (Gate) und H-013 (Rotation):
   auf 2016–2026 ist JEDE Absicherung reine Prämie ohne Auszahlung. Ehrlich: das
   Fenster enthält keinen mehrjährigen Bärenmarkt — der Wert von Schutz ist hier
   strukturell UNTERSCHÄTZT. Nicht als "Schutz ist wertlos" verallgemeinern.
3. **Signal-Ranking bestätigt:** 12-1-Momentum > 52wk-High > Low-Vol auf diesem
   Universum; nichts davon schlägt die No-Signal-EW-Baseline im Sharpe.
4. **H-016 (survivorship-IMMUN) ist die glaubwürdigste NULL der Welle:** GEM-Switches
   kosten deutsche Steuern, die die Bond-Phasen nie zurückverdienen.
5. Das Vol-Gate-Plus (H-019, +46k) hat PBO 0,91 → nicht belastbar; als EINZIGE
   Beobachtung der Welle knapp über der H-012-Referenz, Kandidat für einen
   sauberen Re-Test NUR auf survivorship-freien Daten mit Bärenmarkt-Fenstern.

---

## Welle 3 (2026-07-07) — Läufe 64–67, N=67. No-Signal-Steuer-Designs.

Paar-Design: Survivorship steckt in BEIDEN Seiten jedes Vergleichs → der gemessene
Delta ist der Steuer-/Turnover-Effekt, nicht der Survivor-Gift. Referenz: EW monatlich
416.612 / Sharpe 0,851 (Steuern 83.676, Kosten 1.704).

| Lauf | final € | Sharpe | Steuern | Kosten | vs. Referenz |
|---|---|---|---|---|---|
| H020_band25 (EW, 25 %-Band) | 477.113 | 0,925 | 56.164 | 720 | **+60.501 / +0,074 Sharpe** |
| H020_band50 (EW, 50 %-Band) | 476.523 | 0,956 | 42.177 | 445 | **+59.911 / +0,105 Sharpe** |
| H021_annual (Momentum jährl.) | 577.485 | 0,752 | 67.049 | 2.546 | −37.420 vs. H-012-out60 |
| H022_buyhold (B&H-Extrem) | 699.552 | 1,034 | **0** | 100 | +282.940 (Diagnose-Obergrenze) |

**H-020 VERDICT: PASS (erster des Mandats)** — beide Vorab-Kriterien erfüllt (beide
Varianten > 416.612 UND Sharpe ≥ 0,80), **PBO 0,00** (Band-Wahl split-konsistent,
kein Selektionsrisiko), DSR 0,73 (kein Signal-Alpha-Claim — mechanischer Paar-Effekt;
DSR war bewusst nicht Pass-Bedingung). Einordnung: UMSETZUNGS-Ergebnis, explorativ.
**H-021 VERDICT: FAIL** — jährliches Momentum-Rebalancing verliert mehr Signal
(12-Monats-Decay) als die Stundung einspart (577k < 615k monatlich-mit-Puffer).
**H-022 (Diagnose):** Die Stundungs-Obergrenze ist enorm (+68 % Endvermögen, 0 €
Steuern bis Horizont) — ABER B&H eines heutigen Survivor-Universums ist die reinste
Form des Hindsight-Bias; nur der RELATIVE Befund zählt: Handelsfrequenz ↓ ⇒
Netto-Endvermögen ↑, monoton über alle Wellen.

**Konvergente Gesamtaussage nach 10 Hypothesen (H-011…H-022, 27 Läufe):**
Auf die deutsche Steuer wirkt genau EIN Hebel robust und paar-sauber: **weniger
handeln** (Band- statt Kalender-Rebalancing, breite Puffer, keine Zwangs-Exits).
Jeder Signal-/Timing-/Schutz-Anspruch stirbt an DSR/PBO oder an der EW-Baseline.

---

## Welle 4, Lauf 1 (2026-07-07) — Läufe 68–75: KONTAMINIERT (Datenfehler, kein Verdict)

Erste verdict-fähige Läufe (H-023/024/025 + EW-PIT-Baseline) auf EODHD-PIT-Universum
lieferten unplausible Ergebnisse (EW-PIT 17,9 % CAGR = +7,7 %p über SPY; MaxDD −94 %).
Diagnose (Pflicht §2.5): **3.079 Tage mit >+100 %-Sprüngen**, Extremfall CBE
+34.000× an einem Tag von $0.005 — kaputte adjusted-Reihen bei Delisted-Titeln;
monatliches Rebalancing erntet die künstlichen Sprünge („rebalancing into noise").
**Alle 8 Trial-Läufe ungültig.** Sie zählen im Budget (Fehlläufe zählen, §4.2): N=75.

**Korrektur (dokumentiert, KEIN Parameter-Fishing):** (a) Kursreihen werden am ersten
unmöglichen Sprung gekappt (|r|>100 % bei prev<$1 → Reihe endet, Engine-Force-Sell
zum letzten sauberen Kurs); (b) Kauf nur bei Kurs ≥ $1 (der ohnehin von §3.1
geforderte Mindestliquiditätsfilter). Re-Runs ersetzen die kontaminierten Läufe mit
UNVERÄNDERTEN Parametern/Kriterien — kein neuer Selektionsdruck, zählen nicht doppelt
(Präzedenz: Fable daily.parquet-Bugfix vor Verdict). Verdicts erst nach Re-Run.

## Welle 4, Re-Run mit Datenhygiene (2026-07-07) — VERDICTS (N=75)

78 Reihen gekappt. Fenster ~1996-06→2026-07 (~30 J, Warmup-getrimmt), inkl. 2000-03,
2008, 2020, 2022. Benchmarks: SPY brutto 10,27 %/Sharpe 0,605/MaxDD −55,2 %;
**ETF-Netto-Pfad 1.610.149 (9,57 % net)**; EW-PIT-monatlich 1.305.849 / Sharpe 0,552 /
MaxDD −75,6 % (die ehrliche Baseline — 17,9 % aus Lauf 1 war Datenartefakt).
Rest-Caveat: 36 fehlende Bankruptcy-Ticker (LEHMQ, WCOEQ, …) schmeicheln allen
Aktien-Läufen leicht; in PAAR-Vergleichen neutral, in Absolut-Vergleichen nicht.

| Lauf | final € | CAGR | Sharpe | MaxDD | Steuern |
|---|---|---|---|---|---|
| H023_out60 | 1.214.720 | 8,56 % | 0,605 | **−93,0 %** | 296.550 |
| H023_out80 | 1.126.770 | 8,29 % | 0,542 | −92,4 % | 275.376 |
| H023_out100 | 1.475.985 | 9,26 % | 0,584 | −93,4 % | 353.617 |
| H024_band25 | 1.616.344 | 9,58 % | 0,559 | −75,5 % | 288.530 |
| H024_band50 | 1.690.822 | 9,75 % | 0,559 | −75,2 % | 235.319 |
| H025_gate_sma | 1.516.872 | 9,35 % | 0,595 | −92,0 % | 418.572 |
| H025_gate_vol | 1.432.422 | 9,15 % | 0,598 | −93,3 % | 353.838 |
| H025_gate_both | 1.937.493 | 10,24 % | 0,624 | −93,0 % | 488.766 |

**H-023 VERDICT: FAIL** (selected out100): (1) 1,48M < ETF 1,61M ✗; (2) Sharpe 0,584 >
EW 0,552 ✓; (3) DSR 0,94 < 0,95 ✗; (4) PBO 0,829 ✗; (5) 4/8 Fenster ✓ (Grenzfall).
→ Die zentrale Mandats-Frage ist erstmals VERDICT-FÄHIG beantwortet: steueroptimiertes
Momentum schlägt den ETF-Netto-Pfad über ~30 J NICHT — und trägt ein absurdes
Risiko-Profil (MaxDD −93 % im Dotcom-Tal; konzentriertes Top-20-Momentum = Tech pur).

**H-024 VERDICT: PASS (erster verdict-fähiger PASS des Mandats):** beide Bänder >
EW-PIT-monatlich (+24 %/+29 % Endvermögen), Sharpe nicht schlechter (0,559 vs 0,552),
PBO 0,429 ≤ 0,5 ✓. Welle-3-Befund REPLIZIERT auf survivorship-freien Daten über 30 J
inkl. zweier Bärenmärkte. Charakter: Steuer-/Kosten-MECHANIK (DSR 0,83 — kein
Alpha-Claim, war vorab kein Kriterium). Bonus-Beobachtung (NICHT pass-fähig wegen
Bankruptcy-Rest-Caveat): band50 1,69M > ETF-Pfad 1,61M — ein selbstgehaltenes
EW-PIT-Portfolio mit 50 %-Band schlägt den ETF-Pfad knapp; operativ aber ~500 Namen.

**H-025 VERDICT: FAIL** (selected gate_both): Sharpe 0,624 > 0,605 ✓, DSR 0,97 ✓ (!),
PBO 0,314 ✓ — aber **MaxDD −93,0 % vs gate-los −93,0 % = NICHT ≥10 %p besser ✗**
(hartes Vorab-Kriterium). Der faire Bärenmarkt-Test widerlegt die Schutz-These:
monatliche Gates mit 50 %-Exposure verhindern das Momentum-Tal NICHT (Vol-Perzentil
reagiert zu spät, SMA-Gate fängt −93 % nicht ab). Notiz: gate_boths Endvermögen
(1,94M, DSR 0,97) stammt aus der Rückschau-Beobachtung „Bärenmarkt-Cash + billiger
Reentry" — dürfte NUR als NEU registrierter Confirmatory-Test weiterverfolgt werden.

---

## Welle 5 (2026-07-07) — Läufe 76–81 (H-026 Confirmatory), N=81. FAIL — Thema Gates ZU.

Störfamilie {P70/P80/P90}×{SMA150/250} (Original P80/S200 bewusst ausgeschlossen):
finals 981.299–1.339.500 — **kein Nachbar** erreicht auch nur gate-los (1.214.720:
4 von 6 darunter), **alle 6 klar unter ETF-Pfad** (1.610.149); PBO fail. Die
1,94M-Beobachtung aus H-025 war ein **isolierter Parameter-Peak** (klassisches
Overfitting-Muster; DSR 0,97 des Originals bewertete den Peak, nicht die Region).
Per Vorab-Registrierung: Thema kombinierte Gates ENDGÜLTIG ZU. Die Registry-Disziplin
hat hier exakt geliefert, wofür sie da ist.

**Insider-Patrone (§4.6.1): bleibt aufgespart** — Form-4-Universum deckt nur 260
Watchlist-Survivors (+ transaction_date-Schmutz bis 2050); Voraussetzung = breiter
Form-4-Pull über ~1.200 CIKs (Multi-Tage-Job, notiert). Kein Trial verbraucht.

## Welle 5b (2026-07-07) — Läufe 82–84 (H-027, 50-Namen-Band), N=84. Formal FAIL.

| Lauf | final € | Sharpe | MaxDD | Steuern | Kosten |
|---|---|---|---|---|---|
| EW50 monatlich (Referenz) | 960.787 | 0,586 | −80,0 % | 223.855 | 8.126 |
| Band 25 % | 1.237.879 (+28,8 %) | 0,578 | −77,6 % | 210.532 | 5.710 |
| Band 50 % | 1.296.004 (+34,9 %) | 0,609 | −79,9 % | 189.709 | 4.817 |

Kriterien: final ✓✓, Sharpe ✓ — **PBO 0,543 > 0,5 ✗ → formal FAIL** (Registry bindend).
**Design-Learning (Registrierungsfehler, dokumentiert):** PBO über ein 2-Trial-Paar
nahezu identischer Reihen (Korr. ~0,99) ist ein Münzwurf-Maß — der IS-Winner wechselt
zufällig; ~0,5 ist mechanisch erwartbar UNABHÄNGIG von der Realität des Effekts.
Richtig wäre Fenster-Delta-Konsistenz gewesen. KEINE nachträgliche Aufweichung; der
50-Namen-Fall gilt als konsistente ZUSATZ-Evidenz (dritte Replikation der Richtung:
H-020 explorativ, H-024 verdict-PASS, H-027 formal FAIL bei ökonomisch identischem
Bild +29–35 %) — die Praxis-Empfehlung stützt sich allein auf H-024.

---

## Datenbasis-Ausbau (2026-07-07, keine Trials)

- **13F-Bulk komplett:** 53 Quartals-ZIPs (2013Q2–2026), 2,75 GB, alle Filer/Holdings
  (SEC structured data). Parser + PIT-faire Manager-Auswahl offen.
- **XBRL-Breitpull:** +205 → **743 Symbole** (1,45 M rows). **Strukturelle Grenze:**
  ~455 tote Ticker ohne CIK-Auflösung (SEC-Ticker-Map enthält nur Aktive) →
  **Fundamentals bleiben survivor-lastig**, auch wo Preise es nicht sind. Jeder
  künftige Value/Quality-Test trägt diese Einschränkung im Explorativ-/Verdict-Flag.
- **Form-4-Breitpull:** Tranchen laufen (1–3/≈24; T1: 25.782 rows, 27 Symbole, inkl. toter Ticker ✓).
- **13F Phase 1 KOMPLETT:** 23,8 M Holding-Zeilen, 52 Quartale, Top-100-Manager je
  Quartal (PIT-fair aus Portfolio-Summen, kein Rückschau-Star-Picking), FILING_DATE
  als PIT-Anker → data/13f_top100.parquet (313 MB). Phase 2: CUSIP→Ticker via
  SEC-FTD-Dateien (Route validiert: CUSIP|SYMBOL enthalten).
- **Dividenden-Pull (EODHD):** läuft. **Dividendensteuer-Engine:** implementiert
  (26,375 % am Ex-Tag, konservativ ohne Pauschbetrag/Topf-Verrechnung).

## Welle 6 (2026-07-07) — Läufe 85–86 (H-028 GEM International), N=86. VERDICT: FAIL.

Fenster ~2003-08→2026-07 (~23 J, EFA+Warmup; enthält 2008/2020/2022), Div-Steuer-Drag
aktiv, keine Teilfreistellung (konservativ). Benchmarks gleiches Fenster: SPY brutto
11,2 %/Sharpe 0,67/MaxDD −55,2 %; ETF-Netto-Pfad 956.886.

| Lauf | final € | CAGR | Sharpe | MaxDD | Steuern |
|---|---|---|---|---|---|
| GEM classic (SPY/EFA/IEF) | 438.031 | 6,65 % | 0,475 | −34,0 % | 175.356 |
| relative-only (SPY/EFA) | 506.115 | 7,32 % | 0,456 | −63,4 % | 191.809 |

Kriterien (selected: relative_only): (1) 506k < ETF 957k ✗ (−47 %!); (2) DSR 0,39 ✗;
(3) MaxDD −63,4 % vs SPY −55,2 % ✗; (5) beide Hälften unter SPY ✗. → **FAIL total.**
Learnings: (a) Auch die ORIGINAL-Formulierung stirbt am deutschen Switch-Steuerleck
(~19 Switches × FIFO-Realisation) + EFAs strukturelle US-Underperformance seit 2003;
GEM classic hat immerhin das beste MaxDD-Profil aller je getesteten Varianten
(−34 %) — aber bei 6,65 % CAGR ist das Drawdown-Schutz zum Preis der halben Rendite.
(b) Survivorship-immun ⇒ belastbare NULL. (c) Konsistent mit Antonaccis eigenem
Live-ETF-Track-Record. Rotations-/Timing-Familie ist damit über DREI Varianten
(H-013 US, H-016 US-verkürzt, H-028 Original) tot.

## Dividendensteuer-Sensitivität (2026-07-07, Modell-Korrektur-Reruns — keine neuen Trials)

26,375 % auf Dividenden am Ex-Tag (ohne Pauschbetrag, ohne Topf-Verrechnung — konservativ).
Effekt über ~30 J: **−24 bis −33 % Endvermögen** — die Dividendensteuer ist real massiv.

| Lauf | ohne Div-Steuer | mit Div-Steuer | Delta |
|---|---|---|---|
| EW_PIT_monthly | 1.305.849 | 992.719 | −24,0 % |
| H023_out100 | 1.475.985 | 986.947 | −33,1 % |
| H024_band25 | 1.616.344 | 1.225.744 | −24,2 % |
| H024_band50 | 1.690.822 | 1.291.112 | −23,6 % |

**Verdicts STABIL:** H-023 weiter (deutlicher) unter ETF-Pfad → FAIL bestätigt.
H-024-Paar: beide Bänder weiter +23/+30 % über EW-monatlich → PASS bestätigt
(der Band-Effekt ist steuer-modell-robust).

**NEUER Befund (wichtig):** Die Bonus-Beobachtung „band50 > ETF-Pfad" KIPPT unter
realistischer Dividendensteuer (1.291.112 < 1.610.149). **Mit vollem deutschem
Steuermodell schlägt KEIN getestetes Direkt-Aktien-Design den thesaurierenden ETF** —
dessen Teilfreistellung + Thesaurierung (Dividenden wachsen brutto weiter) ist ein
struktureller Vorteil, den Direktbesitz in unseren Tests nie aufholt. Damit ist die
ehrlichste Zusammenfassung des Mandats bisher: Der beste bekannte Weg, den
S&P-500-ETF-Pfad zu schlagen, ist, ihn zu halten.

## Welle 7, Lauf 1 (2026-07-07) — Läufe 87–88 (H-029): KONTAMINIERT, kein Verdict. N=88.

Diagnose (Pflicht §2.5, ausgelöst durch implausible 16,3 J Fenster + MaxDD −84 %):
(1) 2009–2012 nur 2–114 Konsens-Zeilen/Jahr (Late-Filing-Amendments) → Serie startet
3 Jahre zu früh mit 1–3-Namen-Rumpf-Portfolio; (2) SEC-ZIPs ab 2024 sind 3-Monats-
SPANNEN → per-ZIP-Top-100 splittert Quartals-Konsens (155–862 rows statt ~1.500).
Nebenbefund positiv: SPY-Korrelation 0,17 — Congress-Copy-Abgrenzung (0,95) gelungen;
Fenster-Konsistenz 6/7 über EW trotz Kontamination. Korrektur (kein Fishing):
Manager-Ranking je PERIODOFREPORT über alle ZIPs + Coverage-Gate ≥ 50 Manager/Periode.
Re-Run ersetzt (Konvention wie Welle 4).

## Engine-Bugfix (2026-07-07, betrifft ALLE Verdict-Läufe — Metriken-Korrektur)

**Bewertungs-Artefakt gefunden** (via H-029-Diagnose: die 5 schlimmsten „Tagesverluste"
−80 % lagen alle nach US-Feiertagen): An Panel-Tagen, wo nur einzelne Symbole Zeilen
haben, wurden gehaltene Positionen mit NaN→0 bewertet → fake ±80 %-Equity-Spikes →
MaxDD/Sharpe/DSR aller verdict_engine-Läufe (Wellen 4–7) verzerrt; Endvermögen kaum
(Spikes symmetrisch-temporär). **Fix:** Bewertung auf forward-gefüllte Kurse; Trades
weiterhin nur an echten Kurszeilen. Re-Runs der Kern-Metriken ersetzen die alten
Zahlen (Modell-Korrektur-Konvention). Die FAIL-Verdicts H-023/025/026/028 standen
NICHT allein auf MaxDD — Endvermögens-/DSR-Kriterien scheiterten unabhängig; die
MaxDD-Zahlen (−93 %) werden mit dem Re-Run neu ausgewiesen.

## Welle 7, Re-Run nach Fixes (2026-07-07) — H-029 VERDICT: FAIL (4/5 ✓, DSR ✗). N=88.

Fenster 2013-08→2026-07 (~12,8 J, OHNE 2008 — dokumentierte Einschränkung). Volle
deutsche Steuern inkl. Div-Drag. Benchmarks gleiches Fenster: SPY brutto 14,5 %/
Sharpe 0,882/MaxDD −33,7 %; ETF-Netto-Pfad 482.986; EW-PIT-Fenster-Sharpe 0,55.

| Lauf | final € | CAGR | Sharpe | MaxDD | Steuern |
|---|---|---|---|---|---|
| H029_k5 | 479.441 | 12,97 % | 0,741 | −33,1 % | 23.905 |
| **H029_k10 (selected)** | **752.752** | **17,01 %** | 0,861 | −33,6 % | **18.433** |

Kriterien: (1) > ETF-Pfad ✓ (**+56 %**); (2) Sharpe > EW ✓; (3) **DSR 0,729 ✗** (N=88);
(4) Fenster 4/7 ✓; (5) MaxDD ✓. PBO 0,0 (k5/k10 konsistent). → **FAIL** (Registry bindend).

**Ehrliche Einordnung — stärkster Kandidat des Mandats bisher:** Erster Kandidat, der
4 von 5 Kriterien gleichzeitig schafft, inkl. ETF-Pfad-Beat nach vollen Steuern bei
SPY-gleichem MaxDD und nur 18k Steuern über 13 J (Konsens-Mega-Caps werden praktisch
nie verkauft = die H-024-Mechanik emergent). ABER die Skepsis-Punkte sind gewichtig:
(a) SPY-Korrelation 0,936 (nach Fix; die 0,17 vorher waren Spike-Artefakt) — nahe der
Congress-Copy-Schwelle: im Kern konzentriertes Mega-Cap-Beta; (b) Fenster 2013–2026 =
exakt das Jahrzehnt, in dem Mega-Cap-Growth alles schlug — ohne 2000/2008-Test ist
„Konsens" von „Growth-Beta" nicht trennbar; (c) DSR sagt: bei 88 Trials nicht
signifikant. **Nächster legitimer Schritt (nur als NEU registrierter Confirmatory,
Lehre H-026): Störungsfamilie (Top-8/12-Positionen, K=7, Manager-Top-50/150) +
Mega-Cap-Beta-Kontrolle (cap-weighted-Top-50-Baseline).** Erst wenn der Effekt
Nachbarn UND Beta-Kontrolle überlebt, ist er mehr als das beste Jahrzehnt von Big Tech.

## Metriken-Re-Runs Wellen 4 mit Bewertungs-Fix + Div-Steuer (2026-07-08) — ERLEDIGT
EW_PIT 983.303 · H023_out100 953.648 · H024_band25 1.231.246 · H024_band50 1.315.515.
**Verdicts stabil:** H-023 < ETF-Pfad (FAIL bestätigt), H-024 beide > EW (+25/+34 %,
PASS bestätigt). Der NaN→0-Fix ändert keine Entscheidung; die alten −93 %-MaxDD-Werte
sind als Spike-kontaminiert markiert (H-025/026-Detailzahlen nicht neu gerechnet —
deren Verdicts hingen nicht an MaxDD allein).

## Welle 8 (2026-07-08) — Läufe 89–94 (H-030 Confirmatory 13F), N=94. Formal FAIL (PBO), substanziell stark.

| Nachbar | final € | Sharpe | MaxDD |
|---|---|---|---|
| top8 | 771.394 | 0,867 | −37,5 % |
| top12 | 670.663 | 0,854 | −31,6 % |
| k7 | 632.849 | 0,832 | −32,3 % |
| k13 | 745.291 | 0,873 | −34,6 % |
| m50 | 812.940 | 0,894 | −34,3 % |
| m150 | 689.862 | 0,850 | −31,1 % |
| **Beta-Kontrolle (13F-Markt-Portfolio)** | **351.112** | 0,575 | — |
| ETF-Netto-Pfad | 482.986 | — | — |

Kriterien: (1) **ALLE 6 > ETF-Pfad ✓** — kein isolierter Peak (der H-026-Test, den
der Gate-Kandidat krachend verlor, ist hier BESTANDEN); (2) **Median 717.577 > Beta-
Kontrolle 351.112 ✓** — die Top-10-Konzentration addiert +104 % über „was
Institutionen eh halten"; NICHT bloßes Beta (das 13F-Markt-Portfolio schlägt nicht
mal den ETF); (4) 6/6 Sharpe > EW ✓; **(3) PBO 0,743 ✗ → formales VERDICT: FAIL**
(Registry bindend). Design-Learning (2. Mal): CSCV-PBO über eng korrelierte
Störungsfamilien desselben Effekts misst primär Winner-Springen zwischen
~identischen Reihen — strukturell streng; die Confirmatory-Substanz tragen crit1/2.
KEINE nachträgliche Aufweichung; ein drittes Kriterien-Nachjustieren wäre
Kriterien-Shopping.

**Nachbefund (2026-07-08, aus Shadow-Snapshot):** Der aktuelle k10-Basket (14 Namen)
enthält **IVV/SPY/VOO — Index-ETFs** aus Manager-Top-10s. SPY (einziger mit
Preisdaten im Panel) war damit auch in den H-029/030-Baskets → verwässert Richtung
Benchmark (macht den ETF-Beat eher KONSERVATIVER, ist aber unsauber für den
„Best-Ideas"-Charakter). Regel für alle Folgeläufe + Shadow-Tracking: ETF-Ausschluss
(NAMEOFISSUER/Ticker-Filter). Aktueller Konsens ex-ETF = die Mega-Cap-11: AAPL 89,
MSFT 88, NVDA 87, AMZN 80, GOOGL 80, AVGO 73, META 58, GOOG 55, TSLA 31, JPM 22, LLY 20.

**Ehrlicher Stand des 13F-Konsens-Themas nach 2 formalen FAILs (H-029 DSR, H-030 PBO)
bei durchweg starken Begleitwerten:** Auf diesen Daten/diesem Fenster NICHT
verdict-fähig bestätigt — und mit Registry-Mitteln nicht weiter testbar, ohne
Kriterien zu shoppen. Die zwei redlichen Wege zu einem echten Urteil:
(a) strukturell anderes Fenster: 13F VOR 2013 (EDGAR-Alt-Filings, Text-Format —
großer Parser-Job) → 2008-Test; (b) **echtes Out-of-Time: den k10-Basket ab jetzt
6–12 Monate PAPER-tracken** (Guardrail-2-konform, kostet nichts, verbraucht kein
Test-Budget, liefert unbestechliche OOS-Evidenz). Empfehlung: (b) starten, (a) als
Wintervorhaben.

- **Kumulatives N: 84** (N₀=40 + 44 Läufe in 5 Wellen; inkl. 8 kontaminierte Welle-4-Läufe)
- **Beste DSR bisher:** H-025 gate_both 0,97 — durch H-026-Confirmatory als isolierter
  Parameter-Peak WIDERLEGT (Nachbarn 4/6 unter gate-los). Beste ÜBERLEBENDE DSR: keine ≥ 0,95.
- **PBO des besten Kandidaten (H-024):** 0,43 ✓
- **Investierte Rechenzeit:** ~5 Sessions, ~60 Backtest-Läufe (inkl. Benchmarks/Diagnosen)
- **Ehrliches Gesamt-Verdict (1 Satz):** Nach 17 registrierten Experimenten auf jetzt
  survivorship-freien 30-Jahres-Daten existiert KEIN belastbares Signal-Alpha
  (Momentum/Quality/LowVol/52wk/GEM/Gates/TLH — alle FAIL), und der einzige
  verdict-fähige PASS ist die Steuer-MECHANIK „seltener handeln" (H-024: Band- statt
  Kalender-Rebalancing, +24–29 % Endvermögen über 30 J, dreifach repliziert).
- **Nächste registrierte Hypothesen:** offen — Kandidaten: (a) Form-4-Breitpull als
  Voraussetzung der §4.6.1-Patrone (Multi-Tage-Vorbereitung, kein Trial); (b) Kandidat-C-
  Watchlist operationalisieren (kein Trial); (c) weitere Steuer-Design-Varianten nur
  mit sauber spezifizierten Kriterien (Lehre aus H-027).


---

## Welle 9a (2026-07-08) — Läufe 95–96 (H-032 Dividend-Tilt), N=96. **VERDICT: PASS (2. des Mandats).**

30,4 J, volle Steuer-Engine inkl. Div-Drag. low_div (Top 50 niedrigster trailing-Yield,
jährlich, no-retrim) 2.695.589 / 11,44 % CAGR / MaxDD −52,6 % vs. high_div 643.768 /
6,32 % / −55,9 %. **Ratio 4,19× — beide Vorab-Kriterien PASS** (>1,10×; > EW-PIT 988k).
low_div schlägt zudem als ERSTER Lauf den ETF-Pfad über das volle 30-J-Fenster
(1.593k, +69 %) — Bonus-Beobachtung, nicht Kriterium.

**Pflicht-Dekomposition (Diagnose-Läufe, brutto):** low 3.405k vs high 2.402k =
nur 1,42× brutto (Growth-Fenster-Tilt). **Der Steuer-KEIL liefert den Rest (×2,95):
high_div verliert −73 % seines Brutto-Endwerts an die deutsche Steuer, low_div nur
−21 %.** Die registrierte Steuer-These (Ausschüttung = Zwangsrealisation ohne
Stundung/Topf) ist mechanisch exakt bestätigt; der Brutto-Anteil ist als
fensterabhängig gekennzeichnet.

**Praktische Anti-Empfehlung (direkt verwertbar):** Klassische High-Yield-
„Dividendenstrategien" — in DE-Retail extrem populär — sind nach deutschem
Steuerrecht strukturell ruinös (−73 % Steuerfraß über 30 J); wer deutsch versteuert,
gehört in Thesaurierer/Nicht-Zahler. Konsistent mit ALLEN Mandats-Befunden:
Stundung ist der einzige robuste Edge.


---

## Welle 9b (2026-07-08) — Läufe 97–98 (H-031 Insider-Patrone §4.6.1), N=98. VERDICT: FAIL — FELD ENDGÜLTIG ZU.

Datenbasis: 23.211 Open-Market-P-Käufe, 723 Symbole, 4.258 Insider (volles jemals-
Universum inkl. WAMUQ/WCOEQ; PIT via available_at); 98,8 % opportunistisch nach
CMP-Routine-Filter. Fenster 2005–2026 (~21 J, INKL. 2008). Volle Steuern + Div-Drag.

| Lauf | final € | CAGR | Sharpe | MaxDD | Steuern |
|---|---|---|---|---|---|
| all_opp | 1.038.072 | 11,50 % | 0,617 | −57,2 % | 331.795 |
| **officer_10k (sel.)** | **1.295.845** | **12,65 %** | **0,660** | −57,5 % | 423.098 |

Benchmarks: ETF-Pfad 772.823 · SPY 10,9 %/0,642/−55,2 % · EW-PIT-Sharpe 0,439.
Kriterien: (1) > ETF **✓ +68 %**; (2) Sharpe > EW ✓ (und > SPY); (3) **DSR 0,705 ✗**;
(4) **6/6 Fenster > EW ✓** (inkl. 2005–08!); (5) MaxDD −57,5 % vs −55,2 % **✗ (2,3 %p)**.
PBO 0,143 ✓. → **FAIL (2 von 5 verfehlt). Per §4.6.1 ist das Insider-Feld damit
ENDGÜLTIG GESCHLOSSEN** — die eine Patrone ist unter den bestmöglichen Bedingungen
(survivorship-frei, 2008 im Fenster, PIT, Routine-Filter) verschossen worden.

**Ehrliche Einordnung:** Substanziell zweitstärkstes Ergebnis des Mandats (nach 13F):
konsistent über ALLE 6 Fenster, ETF-Beat über 21 J inkl. Finanzkrise, PBO sauber,
Sharpe über SPY — aber nicht DSR-fest bei N=98 und mit leicht tieferem Drawdown.
Gemeinsames Muster der zwei „Fast-Kandidaten" (13F-Konsens, Insider-Officer): beide
sind informierte-Akteure-Signale mit niedrigem Turnover — und beide sterben an der
kumulativen Trial-Zählung, nicht an den Daten. Das ist der ehrliche Preis von 98
Versuchen.


---

## Welle 10 (2026-07-09) — Läufe 99–101 (H-033 Congress-Re-Test, Sperrlisten-Override Hans), N=101. VERDICT: FAIL.

Fenster 2013–2026 (~13 J); 64 % der Käufe im Preisuniversum (Non-S&P-Picks fehlen —
dokumentierte Grenze). Benchmarks: ETF-Pfad 546.380 · SPY Sharpe 0,909/−33,7 % · EW 0,592.

| Variante | final € | CAGR | Sharpe | **SPY-Korr** |
|---|---|---|---|---|
| copy_all (sel.) | 512.002 | 7,89 % | 0,631 | **0,965** |
| big_buys ≥50k | 241.741 | 4,19 % | 0,407 | 0,888 |
| cluster ≥3 | 374.584 | 6,33 % | 0,531 | 0,915 |

Kriterien: (1) ETF ✗ (alle drunter); (2) ✓; (3) DSR 0,653 ✗; (4) ✓; (5) MaxDD ✗.
→ **FAIL — und die Sperrlisten-Begründung ist mit eigenen Daten REPRODUZIERT:
copy_all korreliert 0,965 mit SPY (NANC/KRUZ-Wert!) bei schlechterem Sharpe (0,63
vs 0,91) = teures Beta.** Schärfer noch: Differenzierung macht es SCHLECHTER —
große Käufe (−56 % vs copy_all) und Cluster sind ANTI-Signale. Fables „marginal"-H2
stirbt auf sauberen Daten vollständig. Rest-Zweifel: die fehlenden 36 % Non-S&P-
Picks; eine EODHD-erweiterte Variante wäre die LETZTE Congress-Patrone (nur auf
expliziten Wunsch — Prior nach dieser Evidenz: sehr niedrig).


---

## Welle 10b (2026-07-09) — Läufe 102–104 (H-034 Congress volles Universum), N=104. VERDICT: FAIL — CONGRESS-FELD ENDGÜLTIG ZU.

100 % Käufe-Coverage (2.839 Symbole inkl. 1.672 Non-S&P via EODHD; +41 Reihen
hygiene-gekappt). Ergebnis SCHLECHTER als H-033: copy_all 310.852 (vorher 512k!),
big_buys 219.559, cluster3 360.160 — alle WEIT unter ETF-Pfad 546.380; SPY-Korr
0,86–0,96; DSR 0,47. **Die fehlenden 36 % Non-S&P-Picks waren keine versteckten
Gewinner, sondern zusätzliche Verlierer** — die Politiker-Small-Cap-Picks
verschlechtern jedes Portfolio. Beide Congress-Patronen (H-033/H-034) verbraucht:
Feld endgültig zu, Sperrliste steht mit eigener Evidenz doppelt bestätigt.


---

## Welle 12 (2026-07-09) — Läufe 105–106 (H-037 Krypto §23-Steuer-Keil), N=106. VERDICT: PASS (3. des Mandats).

| Asset | HODL (steuerfrei) | aktiv-netto | aktiv-brutto | Steuer-Keil |
|---|---|---|---|---|
| BTC (2011–2026) | 11,9 Mrd | 5,17 Mrd | 11,74 Mrd | **−55,9 %** |
| ETH (2016–2026) | 27,1 Mio | 18,5 Mio | 25,8 Mio | **−28,4 %** |

Beide Assets: HODL > aktiv-netto × 1,20 ✓ UND aktiv-netto < aktiv-brutto × 0,85 ✓
→ **PASS.**

**HARTER CAVEAT (Pflicht):** Die absoluten Endwerte (11,9 Mrd aus 100k) sind REINER
HINDSIGHT-Moonshot und KEIN Strategie-Claim — BTC ex-post gekauft. Der Test misst
AUSSCHLIESSLICH den RELATIVEN §23-Steuer-Keil: aktives Krypto-Trading verliert in DE
bis zu 56 % des Brutto-Gewinns an die <1-Jahr-Steuer (44 % Spitzensatz), während
HODL ≥ 1 J **steuerfrei** ist. Kein „Krypto schlägt Aktien", keine Kaufempfehlung,
Spot only (Guardrail 4 unberührt).

**Einordnung:** Dritter verdict-fähiger PASS — und wieder DASSELBE Prinzip wie H-024
(Band) und H-032 (Low-Div): **Stundung/Haltefrist maximieren.** §23 ist der extremste
Fall (Steuer fällt komplett weg). Die drei PASS des Mandats sind EIN Satz:
je länger gehalten, desto besser nach deutscher Steuer — über Aktien-Rebalancing,
Dividendenvermeidung UND Krypto-Haltefrist hinweg identisch.


---

## Welle 13 (2026-07-09) — Läufe 107–109 (H-038 News-Sentiment, Sperrlisten-Override Hans), N=109. VERDICT: FAIL — SPERRLISTE BESTÄTIGT.

Sentiment-Panel: 163 Monate, median 385 Symbole/Monat (EODHD normalized, PIT T+1).
Fenster 2013–2026. Benchmarks: ETF-Pfad 546.380, EW-PIT-Sharpe 0,592.

| Kosten | final € | Sharpe |
|---|---|---|
| 5 bps | 226.018 | 0,456 |
| 10 bps | 213.639 | 0,430 |
| 20 bps | 182.272 | 0,357 |

Kriterien (10 bps): (1) < ETF ✗; (2) Sharpe 0,43 < EW 0,59 ✗; (3) DSR 0,166 ✗;
(5) 20 bps ✗. SPY-Korr 0,855. → **FAIL total.** **Der Sperrlisten-Grund §2.3 ist mit
eigenen sauberen PIT-Daten reproduziert:** Sentiment-Tilt liegt schon VOR Kosten
unter der No-Signal-Baseline (Sharpe 0,46@5bps < EW 0,59) und zerfällt monoton mit
Kosten — genau das dokumentierte Muster „kollabiert bei ~10 bps". Der Basket ist zudem
Mega-Cap-lastig (Sentiment-Coverage korreliert mit Aufmerksamkeit → hohe SPY-Korr).
News/Sentiment-Feld bestätigt zu (Override sauber durchgetestet, kein PASS).


---

## Welle 14 (2026-07-09) — Läufe 110–111 (H-039 Geopolitik-News Crisis-Alpha, Override Hans), N=111. VERDICT: FAIL.

Signal: 13.459 geopolitische EODHD-Artikel, tägliche Intensität-z (PIT), 482 Spikes
(z>1) im Fenster. Crisis-Basket EW(XLE/GLD/ITA).

**Test A (Event-Study) — die entscheidende Zahl:** Crisis-Basket-Überrendite vs SPY
NACH Spike: 5T +0,13 % (t=1,35), 20T +0,07 % (t=0,40), 60T +0,03 % (t=0,11).
→ **KEIN prädiktives Signal** (alle |t|<2, monoton zerfallend Richtung 0).
Test B (Rotation, entfällt formal, dokumentiert): hard 288.967 / tilt 283.117 —
beide UNTER ETF-Pfad 376.975 UND statische 50/50-Baseline 433.161; Sharpe 0,66/0,69
< SPY 0,88. → **FAIL total.**

**Bestätigt Fable Round 5 auf ZEITNÄHEREN Daten:** Auch mit echten News-Artikeln
(statt trägem GPR-Monatsindex) ist die geopolitische These bei Tagesauflösung tot —
schwaches, nicht-signifikantes 5-Tage-Zucken (+0,13 %, t=1,35), das über 20/60 T
komplett verpufft (Mean-Reversion). Der Crisis-Alpha-Move — wenn es ihn gibt — lebt
in den ersten Minuten (paid intraday, höchst-arbitragiert), NICHT auf EOD/monatlich.
Geopolitik-News-Feld doppelt bestätigt zu (GPR + EODHD-Artikel).

## Welle 15 (2026-07-09) — Läufe 112–116 (H-040 Low-Vol/BAB, H-041 Quality-Tilt), N=116. VERDICT: BEIDE FAIL.

Universum: survivorship-freies PIT-S&P-Verdict (1.167 Namen, 367 Rebalances,
1995–2026). EW-Band-50 %-Mechanik auf einer Faktor-Sub-Membership je Monatsende.
Kosten 10 bps, deutsche Steuern (inkl. Dividendensteuer aktiv — Low-Vol/Quality-
Namen zahlen mehr Div, der Drag gehört ehrlich rein).

**Baselines (gleiche Engine/gleiches Fenster):**
- EW-Band full-S&P: final **1.311.212**, Sharpe **0,545**, MaxDD −0,585
- ETF-Netto-Pfad (SPY, 18,5 % eff.): **1.593.150**
- SPY-Sharpe 0,603

**H-040 Low-Vol/BAB** (unterstes realized-Vol-Terzil, Familie 20/33/50 %):
| Variante | final | Sharpe | MaxDD |
|---|---|---|---|
| lowvol_20 | 562.187 | 0,484 | −0,437 |
| lowvol_33 (verdict) | 649.619 | 0,500 | −0,460 |
| lowvol_50 | 780.191 | 0,517 | −0,483 |

Kriterien (ALLE nötig): (1) Sharpe > full-EW-Band **✗** (0,50 < 0,545); (2) final >
ETF-Pfad **✗** (650k ≪ 1,59M); (3) DSR passes **✗** (Wahrscheinlichkeit 0,572 < 0,95);
(4) MaxDD besser als Baseline **✓** (−0,46 vs −0,585); (5) ≥60 % 2-J-Fenster Sharpe ≥
Baseline **✗** (34,5 %). → **FAIL (1/5).**

**H-041 Quality-Tilt** (oberstes ROE-Terzil, ROE-Coverage Median 552 Namen/Monat):
| Variante | final | Sharpe | MaxDD |
|---|---|---|---|
| quality_33 (verdict) | 568.052 | 0,506 | −0,369 |
| quality_50 | 547.811 | 0,498 | −0,367 |

Kriterien: Quality-EW-Band > ungescreent × 1,05 auf Sharpe UND final **✗** (0,506 <
0,545×1,05; 568k ≪ 1,31M) UND > ETF-Pfad **✗**. → **FAIL.** DSR-Wahrsch. 0,584 < 0,95.

**Learnings (ehrlich):**
1. **Beide Faktoren liefern ihr RISIKO-Versprechen, nicht das Rendite-Versprechen.**
   Low-Vol und Quality senken den MaxDD klar (−0,44…−0,46 bzw. −0,37 vs −0,585 der
   Baseline) — aber auf Sharpe UND Endvermögen liegen sie DEUTLICH darunter. Der Grund
   ist strukturell: das Ausschließen der High-Vol/Low-Quality-Namen entfernt gerade die
   größten Nach-Steuer-Gewinner 1995–2026 (die Vol-Extreme NVDA/TSLA/AMD-Klasse), plus
   der höhere Dividenden-Load der defensiven Namen erzeugt Steuer-Drag.
2. **Klassisches BAB braucht Hebel — Guardrail 4 verbietet ihn.** Frazzini-Pedersen
   BAB levert das Low-Beta-Bein auf Marktvol; nur so entsteht der Alpha-Claim. Die
   deployable, unhebelbare Long-Only-Variante (die einzige zulässige) liefert kein
   Netto-Rendite-Alpha. Der Faktor ist damit für dieses Mandat empirisch geschlossen.
3. **Neuer, wichtiger Baseline-Befund:** Selbst die volle-S&P-EW-Band (1,31M) LIEGT
   UNTER dem passiven ETF-Netto-Pfad (1,59M) — der thesaurierende ETF schlägt hier das
   turnover-reiche EW-Band-Buch nach deutscher Steuer. Konsistent mit dem Mandats-
   Kernbefund: die entscheidende Achse ist Turnover/Steuer-Stundung, nicht der Faktor.
   Die früheren PASS-Designs (H-024/H-032) gewannen gerade durch NIEDRIGEN Turnover +
   Stundung, nicht durch Faktor-Selektion.
4. **Muster nach 41 Hypothesen unverändert:** Kein Signal-/Faktor-Alpha überlebt
   DSR/PBO nach Kosten + deutscher Steuer + survivorship-frei. Es überlebt ausschließlich
   Low-Turnover-Steuerstundung (H-024, H-032, H-037). N=116.

## Welle 16a (2026-07-09) — Läufe 117–118 (H-043 Crisis-Alpha erste-Minuten INTRADAY, Auftrag Hans „alle Abo-Daten nutzen"), N=118. VERDICT: FAIL (netto) — aber interessantester FAIL.

Daten: EODHD 5m-Intraday XLE/GLD/ITA/SPY, 451.985 Bars, 2020-10-12 → 2026-07-08 (Abo-
Tiefe, 1 Regime). Signal = H-039-Geopolitik-Intensität-z (PIT shift 1). Event-Study:
Crisis-Basket EW(XLE/GLD/ITA) minus SPY, Session-Open → +5/15/30/60 min, Spike (z>1,
155 Tage) vs Baseline.

**Brutto-Signal REAL und robust:**
- 5m +0,061 % (t=2,20), 15m +0,087 % (t=2,59), 30m +0,086 % (t=2,21), 60m **+0,1245 %
  (t=2,70)** — monoton wachsend; Baseline (Nicht-Spike) ~0 → echte Spike-Konditionierung.
- **Drop-Top-K macht es STÄRKER, nicht schwächer:** drop-top1 t=3,46, top3 t=3,11, top5
  t=3,32, top10 t=2,55. NICHT von Extremtagen getrieben. Bestätigt H-039s Fluchtklausel
  („Move lebt in ersten Minuten") empirisch — der Effekt, den EOD nicht sehen konnte.

**Aber netto TOT — drei Killer:**
1. **Grösse vs 4-Bein-Kosten:** Trade = long 3 ETFs + short SPY. Brutto 12,45 bps @60min.
   Netto nach 6/10/14 bps all-in: 6,45 / 2,45 / **−1,55** bps. Nach dt. <1J-Steuer
   (26,375 %) bei 10 bps: **1,81 bps/Spike-Tag**. Intraday hat KEINE Low-Turnover-Variante
   → strukturell voller Steuer-/Kostenkeil.
2. **DSR vernichtend:** Spike-only-Strategie (27 Trades/J, 10 bps) → Ann-Sharpe 0,223,
   **DSR-Wahrscheinlichkeit 0,021 ≪ 0,95 (N=118) → FAIL.** Nicht von Best-of-N trennbar.
3. **Regime-abhängig:** 2020 +30,4 bps (t=2,61), 2022 +29,9 bps (t=1,93, Ukraine) — aber
   2021 +3,0 bps (t=0,56, NULL), 2025 +1,4 bps (t=0,22, NULL); 2023/24 kaum Spikes
   (Rolling-z durch 2022-Surge angehoben). Akut-Krisen-Effekt, kein All-Weather-Edge.

**Learning:** Der Wert des Intraday-Abos ist bewiesen — es macht den geopolitischen
Effekt SICHTBAR, den die EOD-Auflösung (H-039) nicht auflösen konnte. Der Verdict bleibt
das Mandats-Muster: **brutto real, netto tot** (hier: zu klein vs Multi-Bein-Intraday-
Kosten + voller <1J-Steuerkeil + Regime-Abhängigkeit + DSR-Deflation). Geopolitik-Feld
jetzt auf ALLEN Auflösungen geprüft (GPR-Monat → EODHD-News-Tag → 5m-Intraday) —
konsistent nicht deployable. N=118.

## Welle 16b (2026-07-09) — Lauf 119 (H-042 Overnight-Anomalie, Auftrag Hans „alle Daten nutzen"), N=119. VERDICT: FAIL — Dekomposition brutto glasklar, deployable tot.

Daten: EODHD RAW open/close SPY + 10 SPDR-Sektoren, 2000–2026 (6.666 Tage).

**Test 1 Dekomposition (kein Trial) — Anomalie REAL & stark (reproduziert Cooper/Cliff/
Gulen 2008):** gesamte Tagesprämie akkumuliert ÜBER NACHT.
- SPY: overnight **+2,24 bps/Tag** vs intraday +0,91.
- Sektoren extremer (overnight / intraday bps): XLU 3,37 / **−1,56**; XLE 3,46 / −0,52;
  XLI 3,95 / −0,29; XLK 3,38 / +0,08; XLF 2,78 / +0,17. Intraday für die meisten
  Sektoren flach/negativ.

**Test 2 deployable Overnight-Buch (kaufe Close, verkaufe Open) — katastrophal FAIL:**
schon bei 3 bps/Seite = 6 bps Round-Trip/Tag frisst der Kostenkeil die 2,24-bps-Prämie
→ −3,8 bps/Tag → 100k → **4.997 € netto** (vs ETF-Pfad 436.206 €), Sharpe_pretax −0,84.
252 Round-trips/J → Kosten allein töten es, Steuer nicht mal nötig. PASS=false.

**Learning:** Die berühmteste „wo lebt die Prämie"-Zerlegung ist brutto eindrucksvoll
bestätigt, aber 100 % untradeable — die Overnight-Prämie (2 bps/Nacht) ist strukturell
kleiner als tägliche Round-Trip-Kosten. Wieder: brutto real, netto tot. N=119.

## Welle 17 (2026-07-09) — Lauf 120 (H-045 Halloween / Sell-in-May), N=120. VERDICT: FAIL — schärft den Kernbefund.

In-Markt Nov–Apr, Cash Mai–Okt; jährliche Mai-Realisation (dt. Steuer, Verlusttopf),
5 bps/Switch. SPY + EW-SPDR-Sektoren, 2000–2026.

| Buch | final_net | ETF-Pfad | Sharpe_aktiv | Buy&Hold-Sharpe |
|---|---|---|---|---|
| SPY | 228.150 | 436.206 | 0,549 | 0,416 |
| EW-Sektoren | 208.932 | 373.678 | 0,526 | 0,384 |

Beide FAIL auf Endvermögen. Interessant: Sharpe IM investierten Fenster höher (Winter
hat real bessere risk-adjusted returns) — aber Endvermögen bricht ein, weil (a) ein halbes
Jahr Cash das Compounding opfert und (b) die jährliche Mai-Realisation die Steuer VORZIEHT
(SPY: 83k Steuer gezahlt, die der ETF bis zum Ende stundet).

**Learning (schärft den Kernbefund):** Low-Turnover allein GENÜGT NICHT. Die Mandats-
Gewinner (H-024 Band, H-032 Low-Div) blieben (a) VOLL investiert UND (b) minimierten die
Realisation. Halloween verletzt BEIDES (Cash-Phase + Jahres-Realisation) → verliert gegen
den voll-investierten, end-stundenden ETF. Präzisierte Formel des überlebenden Musters:
**voll investiert bleiben + Realisation maximal aufschieben.** N=120.

## Welle 18 (2026-07-09) — H-046 Covered-Call-Overlay („Aktien vermieten", Auftrag Hans), N=121. VERDICT: CONDITIONAL — erstes NICHT-stundungs-basiertes Ergebnis, das (annahmeabhängig) schlägt; ABER model-basiert & daten-gated.

**WICHTIG: kein echtes Options-Backtest** — EODHD-Optionsdaten NICHT im Plan (403/404
belegt). Prämien via Black-Scholes mit ANGENOMMENER IV (Grid). Modell, kein Verdict.

Struktur: Stock-Bein voll investiert/gestundet (ETF-artig, 18,5 % am Ende) + monatliches
cash-settled Short-Call-Overlay; positive Overlay-Monats-P&L SOFORT 26,375 % besteuert
(Stillhalterprämie), negative voll (konservativ, kein Offset). 3 bps/Monat Overlay-Kosten.
SPY 2000–2026. Buy&Hold-ETF-Netto-Pfad = 436.206.

Grid (Overlay-Netto-Beitrag | CC-Endwert):
| Strike | IV=realized | IV×1,15 | IV×1,3 |
|---|---|---|---|
| ATM | −283k \| 171k | −162k \| 291k | −43k \| 411k |
| 3 % OTM | −56k \| 397k | **+38k \| 492k** | **+136k \| 589k** |
| 5 % OTM | **+15k \| 469k** | **+85k \| 539k** | **+161k \| 615k** |

5/9 Zellen schlagen; sie clustern ökonomisch sinnvoll (OTM + Vol-Risk-Prämie), kein
Zufalls-Cherry-Pick. Bei historisch-typischer Annahme 5 % OTM/IV≈1,2× ≈ 540k vs 436k
(+24 %), Sharpe 0,73 vs 0,49, MaxDD −0,31 vs −0,52.

**Warum anders als H-040/041/042/045:** erntet die **Vol-Risk-Prämie** (implizite > realisierte
Vola, dokumentiert real), NICHT Steuerstundung → überlebt die Sofortbesteuerung, WENN OTM
geschrieben. ATM verliert immer (Bull-Upside-Cap + Sofort-Steuer, −283k bei fairer IV).

**Warum NICHT als PASS gebucht (ehrlich):**
1. Hängt komplett an der IV-Annahme; ohne echte Optionsdaten (403) nicht verifizierbar. Bei
   IV=realized schlägt nur 5 % OTM knapp (+15k) → modellabhängig.
2. Reale Frictions untertrieben: Bid/Ask, Vol-Skew (OTM-IV ≠ ATM-IV, nicht modelliert),
   Assignment/Pin/Roll.
3. Guardrail 4 (Derivat, Operator-Policy Hans) + Steuerberater (Stillhalter/Termingeschäft).

**Nächster echter Schritt:** echte Optionsdaten (EODHD-Options-Add-on / anderer Anbieter)
→ verdict-tauglicher Backtest. Erstes Kandidatenfeld seit H-024/032/037, das offen bleibt
statt geschlossen. N=121 (9 Grid-Zellen, explorativ/Modell).

## Welle 11-Nachlauf (2026-07-09) — Läufe 122–126 (H-035 Small-Cap-Momentum, H-036 Size-Prämie), N=126. VERDICT: BEIDE FAIL (H-036 = §2.5-Artefakt-Fang).

Universum endlich vollständig: survivorship-freies Small-Cap-Universum, 21.917 NYSE/NASDAQ
Common Stocks inkl. Delisted → Pre-Filter 15.101 je-handelbar, 6.899 Tage, 2000–2026. Band
per ADV60 (Dollar-Volumen-Proxy, keine Shares-Outstanding fürs Delisted-Universum). Small
median 2.288 Namen/Mo, Large 654. 30 bps.

**H-035 Small-Cap-Momentum (top30 × out{90,120,150}): sauberer FAIL.**
- Verdict out120: final 63.115 (aus 100k!), Sharpe 0,309, CAGR −1,8 %, MaxDD **−0,911**.
- EW-Band-Small-Kontrolle (SELBES Band, kein Signal): final 5,97M → Momentum-SELEKTION
  vernichtete >99 % ggü. „ganzes Band halten". PBO 0,886, DSR-p 0,071, Konsistenz 0,16.
- Small-Cap-Momentum ist netto ein Kapitalvernichter: Kosten + Momentum-Crash-DD + Delisting-
  Zwangsverkäufe fressen die akademische In-Sample-Prämie komplett. 4/5 Kriterien ✗.

**H-036 „Size-Prämie" (EW-Band Small vs Large): formal PASS, substanziell §2.5-ARTEFAKT.**
- Formal: Small 5,97M > Large 3,53M × 1,10 ✓ UND > ETF-Pfad 736k ✓ → Kriterien erfüllt.
- ABER Robustheit entlarvt es:
  - **Kosten-Sensitivität** (30/60/100/150 bps): überlebt (50%-Band = moderater Turnover) —
    also KEIN reines Fixkosten-Artefakt, aber auch kein Beweis für echte Prämie.
  - **Liquiditäts-Floor-Diskriminator** (ADV≥$1M/$10M/$50M, 60 bps): Endwert 5,47M → 5,01M →
    **1,90M**; Überschuss-über-ETF 4,78M → 1,21M = **~75 % verdampft bei echter Liquidität**.
    Bei $50M-Floor Sharpe 0,369 **< SPY 0,545** (risk-adjusted schlechter als SPY halten).
  - **MaxDD −0,957…−0,982** ($1-10M) bzw. −0,656 ($50M) = **uninvestierbarer Pfad**.
  - Modell berechnet WEDER Micro-Cap-Market-Impact NOCH Bid-Ask-Bounce (EW-Rebalancing erntet
    Phantom-Returns aus verrauschten Micro-Cap-Closes) → realer Rest noch kleiner.
- → **Kein deployabler Size-Edge.** Micro-Cap-Illiquiditäts-/Microstructure-Effekt, der bei
  Liquidität kollabiert. Genau der False-Positive, gegen den §2.5 existiert — gefangen.

**Learning:** Beide großen akademischen Small-Cap-Anomalien (Momentum, Size) sind auf
survivorship-freien Realdaten nach Kosten/Steuer NICHT deployabel — Momentum vernichtet
Kapital, „Size" ist ein Illiquiditäts-Artefakt mit uninvestierbarem Drawdown. Mandats-Muster
bestätigt. N=126.

## Steuer-Engine-Präzisierung (2026-07-10, keine neuen Trials) — Sparerpauschbetrag + Szenario-Klärung Hans

Auf Nachfrage Hans die Steuer-Engine re-auditiert. **Anleger-Szenario geklärt: Beamter mit
festem Einkommen** → daraus folgt autoritativ:
- **Satz 26,375 %** korrekt und UNVERÄNDERT: Soli gilt weiter auf Abgeltungsteuer (2021er
  Soli-Abschaffung betraf nur veranlagte ESt/Lohnsteuer, NICHT die Kapitalertragsteuer);
  Günstigerprüfung hilft nicht (Grenzsteuersatz > 25 %). Keine Kirchensteuer.
- **Grundfreibetrag (~12k €) gilt NICHT** für Hans' Kapitalgewinne — vom Gehalt aufgebraucht.
  (Die 12k gälten nur bei Anleger OHNE anderes Einkommen via Günstigerprüfung.)
- **Sparerpauschbetrag 1.000 €/Jahr** (§20 Abs. 9 EStG; 2.000 bei Zusammenveranlagung) ist der
  korrekte Freibetrag — war im Code bisher NICHT abgebildet (Steuer ab dem 1. €). **BUGFIX
  eingebaut:** TaxedPortfolio.set_date() armiert den Jahres-Freibetrag, sell() zieht ihn vom
  steuerbaren Gewinn nach Verlusttopf ab; verdict_engine.run_verdict + h011.run_variant rufen
  set_date(t) je Handelstag. Rückwärtskompatibel (ohne set_date == altes Verhalten).
- **Wirkung vernachlässigbar, KEINE Verdict-Änderung:** EW-Band-full 1,311M → 1,331M (+1,5 %),
  Low-Vol_33 +1,0 %, Quality_33 +4,2 % — alle weiterhin ≪ ETF-Pfad 1,59M. ETF-Benchmark bekommt
  den Freibetrag nicht (End-Realisation) → Korrektur begünstigt minimal die Strategien, FAILs
  damit eher robuster. Standalone-Steuer-Skripte (h037 Krypto §23, h042 Overnight, h045 Halloween,
  h046 Covered-Call) haben den 1k-Freibetrag NICHT — immateriell, ändert deren FAIL/CONDITIONAL nicht.
- Übrige Parameter bestätigt korrekt: FIFO, Aktien-Verlusttopf (Vortrag unbegrenzt), Kosten mindern
  Gewinn, Dividende 26,375 % am Ex-Tag (nicht gegen Verlusttopf), ETF-Teilfreistellung 30 % → 18,5 %.
  Nicht modelliert (bewusst, konservativ): Vorabpauschale (macht ETF-Latte minimal zu gut).

## Welle 19 (2026-07-10) — Läufe 127–128 (H-047 Net-Issuance/Buyback-Anomalie), N=128. VERDICT: FAIL.

Signal: PIT-Net-Issuance = −(FY-verwässerte-Aktienzahl_t / _{t−1} − 1) aus XBRL
(`WeightedAverageNumberOfDilutedSharesOutstanding`, `available_at`), Coverage Median 571/Mo.
Buyback-Terzil EW-Band 50 %, 10 bps, Div-Steuer aktiv.
- Buyback_33 (verdict): final **585.904**, Sharpe 0,499, MaxDD −0,432.
- Buyback_50: final 554.325, Sharpe 0,496, MaxDD −0,411.
- vs EW-Band-full-Baseline 1.337.689 / 0,548; ETF-Pfad 1.593.150; SPY-Sharpe 0,603.
- Kriterien: Sharpe>Base×1,05 ✗, final>Base×1,05 ✗, >ETF ✗, DSR-p 0,555 ✗. → **FAIL.**

**STRUKTURELLES META-MUSTER (jetzt 4-fach bestätigt: H-036/040/041/047):** Jeder Long-only-
Aktien-Screen (Size/Low-Vol/Quality/Buyback) SENKT den Drawdown, VERLIERT aber Endvermögen —
weil er strukturell die Mega-Compounder 1995–2026 ausschließt. Beim Buyback besonders sichtbar:
die größten Gewinner (Tech) VERWÄSSERN über Stock-Based-Comp → fallen aus dem Rückkäufer-Terzil.
Zudem lebt Net-Issuance-Alpha großteils im SHORT-Bein (Emittenten), das Guardrail 4 sperrt.
**Verallgemeinerte Erkenntnis:** Der ETF-Netto-Pfad ist schwer zu schlagen NICHT weil Faktoren
nicht wirken, sondern weil (a) Cap-Weighting die Mega-Gewinner voll mitnimmt, (b) der ETF ALLE
Steuer bis zum Ende stundet (18,5 % Teilfreistellung), (c) jeder aktive Screen die Gewinner
unterwichtet UND Steuer früher realisiert. Für long-only + unhebelbar + steuerpflichtig ist das
nahezu unüberwindbar. → Nächste Tests dürfen KEINE weiteren long-only-Faktor-Screens sein. N=128.

## Welle 20 (2026-07-10) — Läufe 129–131 (H-048 Direct-Indexing Tax-Loss-Harvesting vs ETF), N=131. VERDICT: FAIL — dt. TLH-Alpha real aber zu klein.

Kategorie-Wechsel (NICHT-Faktor-Screen, Steuer-Mechanik). DE: keine Wash-Sale-Regel → Verlust
ernten + sofort zurückkaufen (Exposure neutral), Verlust in Aktien-Verlusttopf. FAIRNESS:
End-Liquidation auf beiden Seiten (Endgewinne der Strategie zu 26,375 % minus Topf, ETF 18,5 %).
- ETF-Netto-Pfad: 2.335.072.
- Direct-Index EW ohne TLH (post-liq): 1.155.585, Steuer gezahlt 505.736.
- TLH-15 %: **1.217.851** (2.240 Ernten) → **Steuer-Alpha +62.267 (~+5 %)** vs no-TLH.
- TLH-30 %: 1.194.231 (1.169 Ernten). → **PASS=false.**

**Befund 1 — dt. TLH-Alpha real, aber strukturell klein:** Verluste ernten hilft messbar (+62k),
aber deutsches TLH ≪ US-TLH (kein Step-up, nur Aktiengewinn-Offset, senkt den SATZ nicht — reine
Stundung/Verrechnung). Reicht nicht gegen die ETF-Vorteile: Cap-Weighting + 18,5 %-Teilfreistellung
+ volle Stundung. Selbst ohne EW-vs-CW-Konfundierung ist das TLH-Alpha (~62k) < reiner Satz-Nachteil
(26,375 % vs 18,5 % auf den Endgewinn ≈ 150–180k). Direct-Indexing schlägt den thesaurierenden ETF
in DE NICHT. Frage endgültig beantwortet.

**Befund 2 — METHODISCH, verstärkt ALLE Verdicts:** Erstmals End-Liquidation angewandt. Das no-TLH-
Buch fällt von ~1,33M (alte mark-to-market-Basis früherer Verdicts) auf 1,156M post-liq. → Alle
bisherigen FAIL-Verdicts waren auf der STRATEGIE-FREUNDLICHEN mark-to-market-Basis gerechnet
(Endbuchgewinne ungesteuert geschenkt) und fallen mit ehrlicher Endsteuer noch klarer. Die PASS-
Designs (H-024/032) gewannen gerade WEIL sie kaum realisierten — mit Endsteuer bleibt ihr Vorteil
(niedrige Realisation) bestehen, ihr Abstand zum ETF wäre aber neu zu vermessen (offener Punkt).

**Gesamtbild jetzt sehr vollständig:** Nur Low-Turnover-Stundung eines KONZENTRIERTEN/gewinner-
getilteten Buchs (H-024/032) schlägt den ETF; TLH-Alpha zu klein; Faktor-Screens schließen die
Gewinner aus; Signale sterben an Kosten/DSR. Lokaler Testraum weitgehend gesättigt. N=131.

## ⚠️ KRITISCHE KORREKTUR (2026-07-10) — Reproduzierbarkeits-Bug + ehrliche End-Steuer revidieren die „Aktien-PASSes"

Bei der Re-Verifikation von H-032 unter End-Liquidation zeigte sich **Nicht-Determinismus**:
3 identische Läufe → low-div mtm 2,01M / 2,22M / 1,86M (±10 %), ETF-Vergleich kippte JA/NEIN.

**BUG 1 — Frozenset-Nichtdeterminismus (gefixt).** `load_membership`/`band_membership` liefern
`frozenset`s → `tradable = [s for s in members ...]` iteriert in prozess-abhängiger Reihenfolge
(PYTHONHASHSEED). Im Low-Div-Signal haben viele Titel Div=0 → Score-Gleichstand; `rank(method=
"first")` bricht Gleichstände nach dieser Zufallsreihenfolge → welche 50 Null-Div-Titel gewählt
werden variiert → ±10 % Swing. Betraf JEDE score_panel-Verdict-Hypothese (H-029/031/032/047,
H-035/036-Momentum in geringerem Maß). **Fix in verdict_engine.run_verdict:** `sorted(members)`
für `tradable` + `sorted(...)` für die Verkaufs-Sets (Steuer-Timing hängt an Sell-Reihenfolge via
Verlusttopf-Offset). Verifiziert: 2 Läufe byte-identisch (Anti-Pattern E-051).

**BUG 2 — strategie-freundliche mark-to-market-Basis (Methodik).** Alle früheren Verdicts nutzten
`final_value` = mark-to-market (End-Buchgewinne UNGESTEUERT), verglichen gegen den END-besteuerten
ETF-Pfad (18,5 %). Inkonsistent → begünstigte Strategien. Fix: `terminal_liquidation`-Option in
run_verdict (End-Realisation zu 26,375 % minus Verlusttopf).

**REVIDIERTER KERNBEFUND (deterministisch, End-Liquidation, Fenster 1996–2026):**
| Design | mtm | postliq | ETF-win-matched 1.610.149 |
|---|---|---|---|
| H-032 low-div | 2.005.510 | **1.589.963** | mtm schlägt / **postliq schlägt NICHT (−1,3 %)** |
| H-024 (=EW-Band-full) | 1.338.701 | 1.146.757 | beides NICHT |

- Die „PASSes" H-024/H-032/H-037 waren **relative Steuer-Mechanik-Demos** (Band>naiv; low-div>
  high-div; HODL>aktiv), NICHT „schlägt den passiven ETF". Frühere Formulierungen („2. Mandats-
  PASS, schlägt ETF") waren mtm-basis-abhängig und zu optimistisch — hiermit korrigiert.
- **Struktureller Grund:** Direktaktien zahlen 26,375 % Endsteuer, der Aktien-ETF nur 18,5 %
  (Teilfreistellung, §20 InvStG) — ein ~8-Pp-Wrapper-Vorteil + Cap-Weighting (fängt Mega-Gewinner)
  + volle Stundung. Für long-only, unhebelbar, steuerpflichtig faktisch unschlagbar.
- **Ehrlicher Gesamtbefund nach ~48 Hypothesen / 131 Trials:** KEINE deployable Aktien-Strategie
  schlägt den passiven thesaurierenden S&P-ETF netto nach ehrlicher deutscher Steuer. Low-div
  MATCHT den ETF bestenfalls (Gleichstand, kippt netto zum ETF). Die Steuer-Mechaniken (Stundung,
  Low-Turnover, Low-Distribution) sind real, reichen aber nicht über die ETF-Wrapper-Vorteile.
  Das EINZIGE echte Tax-Wedge-Feld mit 0 %-Potenzial bleibt Krypto-§23 (>1J steuerfrei) — aber
  reine Rückschau, kein Aktien-Ersatz. Covered-Call bleibt CONDITIONAL/daten-gated.

## Welle 21 (2026-07-10) — Läufe 132–133 (H-049 Konzentriertes Mega-Cap-Momentum, Copy-Trading-Archetyp), N=133. VERDICT: FAIL.

Aus Copy-Trading-Research (Auftrag Hans): meistkopierter eToro-Trader (Jeppe Kirk Bonde) + C2
„US Stock Momentum" konvergieren auf Archetyp = low-turnover, konzentriert in Mega-Cap-Gewinnern,
kein Hebel, minimale Frequenz. PIT-getestet (12-1-Momentum top_in {10,20}, top_out 40,
terminal_liquidation, Div-Steuer, deterministische Engine).
- Top-10: mtm 699.819 / postliq 658.885 / Sharpe 0,373 / MaxDD **−0,722**.
- Top-20: postliq 1.103.693 / Sharpe 0,445 / MaxDD −0,662.
- vs window-matched ETF 1.610.149, SPY-Sharpe 0,605. DSR-p 0,284. → **FAIL alle Kriterien.**
- Mehr Konzentration = SCHLECHTER (Top-10 < Top-20) → Konzentrations-Drawdown dominiert.

**Zentrale Erkenntnis (schließt die Copy-Trading-Frage):** PIT-Momentum-ROTATION ≠ „die Gewinner
halten". Jeppe hielt NVDA/META/TSLA DURCH = Hindsight; eine forward-safe Regel kennt die Gewinner
nicht vorher, der Momentum-Proxy rotiert in bereits-gelaufene Namen, wird von Momentum-Crashes
(2008/2022) zerlegt. Bestätigt den Vorab-Caveat: Copy-Trading-Track-Records sind survivorship-
selektiert, keine reproduzierbaren Regeln. **Tiefere Einsicht:** Der Edge des Cap-Index ist, die
Gewinner zu HALTEN OHNE sie WÄHLEN zu müssen (natürliche Konzentration, keine Prognose, kein
Selektionsrisiko, keine Rotation). Jeder aktive Winner-Selektionsversuch (Screen schließt sie aus;
Momentum rotiert durch sie) verliert. Erklärt strukturell, warum der passive ETF unschlagbar bleibt.
N=133.

**Zusatz: wikifolio-Check (2026-07-10, KEIN neuer Trial — analytisch dominiert).** Auf Anfrage Hans
geprüft. (a) Strategie-Archetypen der Top-wikifolios (Momentum/AlphaStars, Dividende/Dividendenstars,
diskretionäres Stockpicking/UMBRELLA/Haas) sind alle bereits getestet (H-032/H-049/Welle 1-2) oder
nicht regelbasiert; die Langläufer (Haas 13,2 %/J 13 J; Dividendenstars 13,9 %/J 12 J) MATCHEN nur den
Index brutto, Highflyer (AlphaStars 46 %/J seit 2022) sind kurz/survivorship. (b) Der wikifolio-Wrapper
ist STRUKTURELL schlechter als ein ETF: **keine Teilfreistellung → volle 26,375 % vs ETF 18,46 %**
(exakt der identifizierte ~8-Pp-Nachteil) + ~0,95 %/J Zertifikatgebühr + erfolgsabhängige Gebühr (bis
~30 %, HWM) + Emittentenrisiko (Inhaberschuldverschreibung L&S, kein Sondervermögen). Einziger Pluspunkt
(interne Umschichtung steuergestundet) ist genau der H-048-Effekt, der schon nicht gegen die Teilfrei-
stellung reicht. → wikifolio ist gegen den passiven ETF noch schwerer zu stellen als das widerlegte
Direct-Indexing; kein Backtest nötig. Bestätigt den Kernbefund ein weiteres Mal.

**Vertiefung „prüfe genauer" (2026-07-10, h050_wikifolio_hurdle.py — deterministische Rechnung):**
- **Survivorship belegt:** „Top-Performer"-Rangliste = gehebelte Extremjahre (Tech-Turbo-Madness
  +362 %/J). wikifolios werden bei **−30 % liquidiert** → Verlierer verschwinden. Die 62,8-%-
  „schlägt-den-Markt"-Zahl von wikifolio filterte explizit „max. 30 % Verlust zum Stichtag" raus
  + ist brutto/monatliche-Avg/vor Gebühren → survivorship-verzerrt, kein Netto-Beleg.
- **Gebühren exakt:** 0,95 %/J Zertifikategebühr + 5–30 % Performancegebühr auf Jahres-HWM (Reset
  31.12.) → ~3,5 %/J Drag bei 20 %-Fee/13 %-Rendite (ETF ~0,2 %).
- **HÜRDE (20 J, quantifiziert):** ein wikifolio braucht **~2–5 % Brutto-Alpha/Jahr**, nur um den
  ETF netto-nach-Steuer zu ERREICHEN (8 % Markt/20 % Fee → +3,4 %/J; bei Parität endet der Anleger
  −37 %). Mein Mandat fand: kein Ansatz liefert ~1 %/J dauerhaftes Netto-Alpha → Hürde 2–5× zu hoch.
- **Bester Langläufer (Haas 13,2 %/J/13 J, netto-Gebühr):** €100k→~€499k, nach 26,375 % = ~€394k
  vs ETF ~€407k → **ETF trotzdem vorne** (Teilfreistellung 18,46 % vs 26,375 % frisst Haas' Brutto-
  Vorsprung). Und Haas ist Survivor-Elite (ex-ante nicht wählbar) + Emittentenrisiko. → wikifolio
  strukturell dominiert; für einen deutschen Anleger schlägt der simple thesaurierende ETF es klar.

## Gross-vs-Net-Dekomposition (2026-07-10, h051_gross_vs_net.py — was kostet die Steuer?)

Auf Anfrage Hans: jede Kernstrategie brutto (Steuer=0) vs netto (volle Steuer + End-Liq), 31,5 J,
Kosten (10 bps) in beiden gleich → isoliert reinen Steuereffekt. CAGR brutto / netto / Steuer-Drag:
- ETF passiv: 11,21 % / 10,52 % / **−0,69 pp** (Steuer = 17,8 % vom Brutto; 18,46 % + volle Stundung)
- Low-Div (H-032): 11,17 % / 9,18 % / −2,00 pp (43,5 %)
- Momentum T20 (H-049): 10,90 % / 7,92 % / −2,98 pp (57,6 %)
- EW-Band (H-024): 10,80 % / 8,05 % / −2,75 pp (54,6 %)
- Low-Vol T33 (H-040): 9,83 % / 6,10 % / −3,73 pp (66,3 %)
- Quality T33 (H-041): 7,34 % / 5,41 % / −1,93 pp (43,6 %)

**Zwei Befunde:** (1) **Ohne Steuer gewinnt der ETF TROTZDEM** — sein Brutto-CAGR (11,21 %) ist der
höchste; low-div/momentum kommen nur ran, schlagen ihn nicht; quality/low-vol brutto schon zurück →
die Strategien haben KEIN Brutto-Alpha, Steuer ist nicht der primäre Killer. (2) Die Steuer ist der
zweite, größere Hammer, aber ungleich: ETF −0,69 pp/J, aktive Direktaktien −2…−3,7 pp/J (3–5×) wegen
keiner Teilfreistellung + Realisation durch Turnover. Steuer VERBREITERT den Abstand (0 → ~1,3 pp/J),
erzeugt ihn nicht. Kein neuer Trial (Dekomposition). N=133.

## Welle 22 (2026-07-11) — Läufe 134–135 (H-051 §23-Tax-Free-Asset-Sleeve), N=135. VERDICT: TEIL — nur Risk-adjusted robust (=Diversifikation); Absolut-„schlägt-ETF" ist Gold-Regime-Artefakt (Stresstest widerlegt). KORREKTUR meiner initialen PASS-Aussage.

Buy-and-Hold-to-terminal (§23-optimal: nie < 1 J → Gold/Krypto steuerfrei). Aktien-ETF 18,46 %,
§23-Sleeve 0 %. Kein Rebalance-Turnover.

**Fenster A (SPY+Gold, 2005–2026, 21,5 J) — ehrlich, kein Krypto-Hindsight:**
| Portfolio | Netto | CAGR | Sharpe | MaxDD |
|---|---|---|---|---|
| 100 % SPY | 767k | 9,94 % | 0,64 | −0,55 |
| 80/20 SPY/Gold | 788k | 10,07 % | 0,77 | −0,39 |
| 70/30 SPY/Gold | 798k | 10,14 % | 0,80 | −0,35 |
→ Gold-Sleeve verbessert ALLE 3 Kennzahlen — ABER siehe Stresstest.

**Fenster A-STRESS (Gold-Renditen ×0,3 ≈ Norm-Rückkehr) — DER ENTSCHEIDENDE TEST:**
| Portfolio | Netto | Sharpe | MaxDD |
|---|---|---|---|
| 100 % SPY | 767k | 0,64 | −0,55 |
| 80/20 SPY/Gold×0,3 | **655k** | 0,69 | −0,44 |
| 70/30 SPY/Gold×0,3 | **599k** | 0,71 | −0,38 |
→ Absolut-Endwert FÄLLT UNTER SPY (655k/599k < 767k) → der „schlägt-ETF-auf-Endvermögen" war ein
GOLD-REGIME-ARTEFAKT (Gold lief 2005–26 zufällig ~9–10 %/J = aktienähnlich, NICHT seine Norm).
Sharpe/MaxDD bleiben besser (Diversifikation, tax-unabhängig) — aber das ist kein Alpha/Steuer-Edge.

**Fenster B (+Krypto, 2016–2026) — Hindsight-FLAG:** 90/10 SPY/BTC 31,5 % CAGR (MaxDD −0,69) =
reine Rückschau (BTC ~290×). Haircut BTC×0,5: 90/10 = 17,7 % CAGR / Sharpe 1,04 vs SPY 13,5 %/0,88
(schlägt es selbst halbiert — aber BTC×0,5 über 2016–26 ist immer noch riesig, kein echter Stress).

**Ehrliche Dekomposition (kein Überverkauf):** (1) Sharpe/MaxDD-Gewinn ist real+robust, aber
GROSSTEILS Diversifikation (gilt für jeden schwach korrelierten Beimisch-Asset, tax-unabhängig);
der §23-TEIL (0 % statt 18,46 %) macht den Sleeve-Rendite-Beitrag zusätzlich steuerfrei (echt, kleiner).
(2) Golds ABSOLUT-Vorteil ist fenster-abhängig (Gold lief 2005–26 ~9–10 %/J, NICHT seine Norm ~0–1 %
real); bei Norm-Rückkehr bleibt Risk-adjusted-Vorteil, Absolut-Vorteil weg. (3) Krypto = stärkster
§23-Hebel, aber Forward-Rendite unbekannt + −69 % DD → Mechanismus echt, Rendite-Annahme nicht.
**Bedeutung (korrigiert nach Stresstest):** KEIN robustes „schlägt-den-ETF-auf-Endvermögen" — der
Absolut-Vorteil hing an Golds Ausnahme-Regime 2005–26. Robust ist nur die RISK-ADJUSTED-Verbesserung
(Sharpe/MaxDD), und das ist **Diversifikation** (Portfoliotheorie, tax-unabhängig, kein Alpha) + ein
kleiner §23-Bonus. Ergo: auf dem MANDATS-Ziel (Absolut-Rendite nach Steuer schlagen) auch hier FAIL;
der ehrliche Gewinn ist Risk-Management (bessere Sharpe/Drawdown), nicht Outperformance. Krypto-§23
ist der einzige echte 0 %-Hebel auf ein HIGH-Return-Asset — aber Forward-Rendite unwissbar (kein
deployable-Beleg). Bestätigt Kernbefund: kein reproduzierbarer Edge, der den ETF absolut schlägt.
N=135. OFFEN: Rebalance-Version (§23-1J-Disziplin); Silber; ob Risk-adjusted-Nutzen dem User reicht.

## Welle 23 (2026-07-11) — Läufe 136–139 (H-052 Global Tax-Aware Rebalancing, Weltmarkt + §23), N=139. VERDICT: Weltmarkt-Aktien FAIL; §23-Rebalancing-Prämie real aber KLEIN; Absolut-Alpha nein.

Auftrag Hans (Weltmarkt + Tax-Free-Rebalancing). Per-Sleeve-Basis, jährl. Rebalance, Aktien 18,46 %/
§23-Sleeve 0 %, End-Liq. Fenster A (SPY/EFA/EEM/GLD, 2005–26, 21,5 J):
| Portfolio | Netto | Sharpe | MaxDD |
|---|---|---|---|
| 100 % SPY BH | 777k | 0,64 | −0,55 |
| Global-Equity 60/25/15 | **605k** | 0,55 | −0,58 |
| Global-Equity rebal | 572k | 0,54 | −0,58 |
| US+Gold rebal 85/15 | 824k | 0,73 | −0,47 |
| US+Gold rebal, Gold×0,3 | 640k | 0,67 | −0,48 |

**Befund 1 — Weltmarkt (Geo-Diversifikation) SCHADETE:** US+EFA+EM (605k) ≪ US-only (777k); Rebalancing
noch schlechter (572k, rein in die Verlierer + Steuer). 2005–26 = US-Dominanz-Regime → Ex-US zog.
Direkte Antwort auf „bedien den Weltmarkt": auf Rendite half es NICHT.
**Befund 2 — §23-Rebalancing-Prämie real aber KLEIN:** US+Gold rebal schlug SPY auf allen 3 (824k/
0,73/−0,47), ABER Gold-Haircut ×0,3 → Absolut weg (640k < 777k), Sharpe/MaxDD nur knapp besser
(0,67 vs 0,64). Robust = etwas bessere Risk-adjusted (Diversifikation + steuerfreie Rebalancing-
Prämie), KEIN Absolut-Alpha. Fenster B (+Krypto 2016–26): US+Gold+BTC rebal Sharpe 1,33 / CAGR
24,7 % — spektakulär aber Hindsight-gated (Krypto-Forward unwissbar).
**Fazit (konsistent mit H-051):** kein reproduzierbares Absolut-Schlagen des Aktien-ETF nach Steuer;
der ehrliche, deployable Gewinn ist RISK-MANAGEMENT (ein §23-Gold-Sleeve + Rebalancing senkt Drawdown/
hebt Sharpe leicht, steuerfrei). Weltmarkt-Aktien-Diversifikation hilft NICHT auf Rendite (US-Regime).
Krypto = stärkster §23-Hebel, aber Wette. N=139.

## Welle 24 (2026-07-11) — Läufe 140–141 (H-053 §4.6.1 Insider-Patrone, BREIT survivorship-frei), N=141. VERDICT: FAIL — Patrone verschossen, Feld zu.

Opportunistische Insider-P-Käufe (Cohen-Malloy-Pomorski) + Cluster≥2 auf Small-Cap-Broad-Universum
(15.101 Namen) mit Handelbarkeits-Floor (Preis≥$5, ADV≥$1M), 30 bps, 12M-Halten. 23.211 P-Käufe →
22.925 opportunistisch (98,8 %), aber nur 723 Symbole abgedeckt.
- all-opp: final 1.459.702, CAGR 13,27 %, Sharpe 0,655, MaxDD **−55,4 %** (Median 51 Namen/Mo).
- cluster≥2: final 1.039.490, Sharpe 0,604, MaxDD −60,0 %.
- vs ETF-Pfad 772.823, SPY-Sharpe 0,642. **Schlägt den ETF absolut (1,46M) — einziger Signal-Test.**
- ABER: DSR-Wahrsch. 0,669 < 0,95 (FAIL); Sharpe nur hauchdünn > SPY (0,655 vs 0,642 = Beta/Risiko,
  kein Risk-adjusted-Alpha); MaxDD −55 % (< SPY); Fenster 3/6 (verliert 2013/21/25); PBO 0,257.
  crit1+2+4 ✓, crit3(DSR)+5(MaxDD) ✗ → **PASS=false.**

**Drei harte Vorbehalte (Ehrlichkeit):** (1) Nur 723 Symbole Form-4-Abdeckung (S&P-Historie-Ticker,
NICHT volles Small-Cap-Universum) → die eigentliche §4.6.1-These „Info-Gehalt in KLEINEN Firmen" ist
NICHT wirklich getestet (überwiegend größere Namen). (2) „Opportunistisch"-Filter filterte kaum
(98,8 %) → praktisch alle P-Käufe, nicht das diskriminierende Signal. (3) Mark-to-market + keine
Div-Steuer = pro-Strategie; mit End-Liquidation (26,375 %) schrumpft der ETF-Vorsprung. **Verdict:
FAIL** — konsistent mit H-031 (S&P) + Fable H1 (survivor). Hoher Absolut-Ertrag = Small-/Mid-Cap-
Risiko, kein robustes Insider-Alpha. Patrone verschossen, Insider-Feld endgültig zu. RESIDUAL-GAP:
echter Small-Cap-Test bräuchte Form-4 für tausende Small-Cap-CIKs (großer EDGAR-Pull; Prior FAIL). N=141.

## Welle 25 (2026-07-11) — Läufe 142–147 (H-054 Risk-Parity, H-055 Vol-Target, H-056 Monte Carlo), N=147. VERDICT: kein Absolut-Alpha; Risk-Ebene liefert ROBUST — MC zeigt erstmals verteilungs-robuste Dominanz einer Aufstellung.

Auftrag Hans („ohne Grenzen; Portfolio-Aufstellung; Monte Carlo; mit Risiko spielen"). Per-Sleeve-
Steuern (Aktien 18,46 %/§23 0 %/Bonds 26,375 %/Cash-Zins 26,375 %), End-Liq, kein Hebel. 2005–26.

**H-054 Inverse-Vol-Risk-Parity (SPY/EFA/GLD/TLT):** net 477–490k ≪ SPY 791k (unlevered RP = bond-
lastig, Guardrail-4 verbietet den Hebel, der RP tragfähig macht) — ABER Sharpe **0,81–0,82 vs 0,65**
und MaxDD **−0,24 vs −0,55**. Band-20 %-Variante spart kaum Steuer (66k vs 66k). +BTC 2016+: Sharpe
1,21, MaxDD −0,28. → Absolut FAIL / Risk-adjusted stark.
**H-055 Vol-Targeting (85/15 SPY/GLD, Ziel 10/15 %):** VT10 517k/Sharpe 0,785; VT15 674k/0,762 —
Referenz 85/15 ohne VT: 853k/0,733. → Sharpe-Gewinn MINIMAL, Endwert-Verlust MASSIV (Steuer-Drag
der De-Risking-Realisationen + verpasste Rallys). VT lohnt netto-nach-Steuer NICHT (De-Risking ist
in DE ein Steuer-Event — strukturell teurer als in US-Literatur).
**H-056 Monte Carlo (1.000 Bootstrap-Pfade à 21,5 J, Block 60T, Seed 42, statische Allokationen,
Terminal-Steuer):**
| Aufstellung | Median-Netto | 5 %-Quantil | P(schlägt SPY) | Median-MaxDD |
|---|---|---|---|---|
| 100 % SPY | 766k | 235k | — | −0,43 |
| **70/30 SPY/Gold** | **881k** | **332k** | **54 %** | **−0,33** |
| 60/40 SPY/TLT | 542k | 221k | 6 % | −0,31 |
| 50/30/20 SPY/GLD/TLT | 760k | 305k | 39 % | −0,27 |
| 40/25/20/15 +EFA | 649k | 259k | 26 % | −0,31 |

**Kernergebnis:** 70/30 SPY/Gold dominiert 100 % SPY VERTEILUNGSWEIT: Median +15 %, **5 %-Quantil
+41 %** (235k→332k — Sequence-Risk-Schutz), MaxDD −0,43→−0,33, P(besser)=54 %. Bonds (60/40) sind
nach dt. Steuer (26,375 % ohne Teilfreistellung) klar dominiert — der klassische 60/40 ist in DE
steuerlich strukturell benachteiligt. **Ehrlicher Caveat:** Bootstrap resampelt die 2005–26-Joint-
Verteilung → Golds heiße Sample-Mean ist eingebacken; der MEDIAN-Vorsprung hängt daran (vgl. H-051-
Haircut), der FLOOR-/DD-Vorteil (Diversifikation) ist der robuste Teil. MC statisch/Terminal-Steuer
(Rebalancing nicht simuliert — benannt).
**Fazit Welle 25:** Auch die Portfolio-/Risiko-Ebene erzeugt KEIN Absolut-Alpha (bestätigt Kernbefund
final), aber sie formt die VERTEILUNG zuverlässig: RP fast verdoppelt Netto-Sharpe; ein §23-Gold-
Sleeve hebt den Floor um ~40 % und senkt DD — der einzige robuste deployable Gewinn des Mandats.
VT ist in DE steuerlich kaputt. N=147.

## Welle 26 (2026-07-11) — Läufe 148–151 (H-057 Krisen-Rebalancing, H-058 Glide-Path MC, H-059 Sparplan MC), N=151. VERDICT: „Dumm schlägt clever" — statisches 70/30 schlägt zustandsabhängiges Timing; Sparplan bestätigt Aufstellungs-Urteil.

**H-057 Krisen-Rebalancing (Gold als steuerfreies Dry-Powder, historisch 2005–26):**
| Variante | Netto | Sharpe | MaxDD | Steuer |
|---|---|---|---|---|
| 100 % SPY | 767k | 0,64 | −0,55 | 151k |
| **statisch 70/30 jährl.** | **875k** | **0,814** | **−0,38** | 116k |
| crisis-rebal (revert@High) | 849k | 0,719 | −0,43 | 127k (1× §23-ST) |
| crisis-rebal (noRevert) | 769k | 0,725 | −0,44 | 186k (5× §23-ST!) |
→ **Das „clevere" zustandsabhängige Timing VERLIERT gegen stures jährliches Rebalancing.** Gründe:
(a) Revert-Verkäufe nach Erholung = Steuer-Events auf frische Gewinne; (b) schnelle Folge-Krisen →
Gold-Lots < 1 J → **§23-Kurzfrist-Steuer 44 %** frisst den Dry-Powder-Vorteil (5 Treffer in noRevert);
(c) Whipsaw. Dieselbe Lektion wie im Signal-Teil: jede zusätzliche Aktivität kostet in DE Steuer.

**H-058 Glide-Path (MC, 90/10→50/50):** Median 866k ≈ statisch 869k, Floor 319k < statisch 332k,
P(schlägt SPY) 61,6 % vs 54 %. → kein Mehrwert über statisch 70/30; leicht besserer P(beat) erkauft
mit leicht schlechterem Floor. Kein Grund für Glide-Komplexität.

**H-059 Sparplan (1.000 €/Monat, 258k eingezahlt, MC):** DCA-70/30 Median **924k** vs DCA-SPY 855k;
Floor (5 %-Q) **495k vs 396k (+25 %)**; P(besser) 55,5 %. → **Das Aufstellungs-Urteil (70/30 + §23-
Gold) überlebt die Sparplan-Realität** — für Hans' echten Fall (laufendes Beamten-Einkommen) gilt
dieselbe Empfehlung. Gold-Sample-Mean-Caveat gilt weiter für den Median-Teil; Floor-Teil robust.

**Fazit Welle 26:** Zeit-/Zustandsdimension erschöpft: Timing-Intelligenz (Krisen-Trigger, Glide)
fügt NICHTS hinzu — sie kostet Steuer (inkl. §23-Kurzfrist-Falle 44 % bei < 1 J Gold-Halten!) und
Whipsaw. Das robuste Endergebnis bleibt: statischer Aktien-ETF-Kern + §23-Gold-Sleeve + SELTENES
stures Rebalancing, identisch für Lump-Sum und Sparplan. N=151.

## Welle 27 (2026-07-11) — Läufe 152–153 (H-060 robuste Gold-Quote/Maximin, H-061 Rebalancing-Kadenz), N=153. VERDICT: Gold ist VERSICHERUNG, kein Wealth-Optimum (Maximin=0 %); Kadenz: 2-jährlich dominiert; Silber nein.

**H-060 Szenario-Sweep (Gold-Quote 0–50 % × Gold-Rendite ×{1,0/0,5/0,3/0,0}, jährl. Rebal, End-Liq):**
| Gold-Quote | ×1,0 | ×0,5 | ×0,3 | ×0,0 (flat) | min |
|---|---|---|---|---|---|
| 0 % | 767k | 767k | 767k | 767k | **767k** |
| 5 % | 775k | 733k | 719k | 698k | 698k |
| 10 % | 780k | 700k | 673k | 635k | 635k |
| 15 % | 784k | 667k | 629k | 576k | 576k |
| 20 % | **785k** | 635k | 587k | 523k | 523k |
| 30 % | 781k | 572k | 509k | 429k | 429k |

**MAXIMIN auf Endvermögen = 0 % Gold.** Selbst mit Rebalancing-Buy-the-Dip-Effekt drückt flaches
Gold das Endvermögen (5 % Quote → −9 % worst case; 15 % → −25 %). Historisches ×1,0-Optimum liegt
flach bei 15–25 % (nur +2 % über SPY). **Ehrliche Einordnung:** Der Gold-Sleeve ist KEINE Endvermögens-
Optimierung, sondern eine **VERSICHERUNG** (Floor/+41 %-MC-Quantil/DD −0,55→−0,35 aus W25/22) mit
bezifferbarer Prämie: im Flat-Gold-Worst-Case kostet 10 % Quote ~17 % Endvermögen. Kauf-Entscheidung
= Risikopräferenz Hans, nicht Dominanz. **Silber: NEIN** — 70/20/10 vs 70/30: Sharpe 0,733 < 0,781,
DD schlechter, mehr §23-ST-Treffer; Silber ist nur Vol ohne Diversifikationsgewinn.

**H-061 Kadenz (70/30, real):** never 798k/Sharpe 0,802 · annual 781k/0,811 (Steuer 200k!, 3 §23-ST-
Treffer — Kalender-Artefakt ~365T-Grenze, real mit 13-Monats-Kadenz vermeidbar) · **biennial 890k/
0,818/Steuer 120k/0 ST** · band20 862k/0,798. → **2-JÄHRLICH DOMINIERT** (bester Endwert UND Sharpe):
seltener = weniger Steuer-Drag, und der 2-J-Takt hält automatisch die §23-1-Jahres-Uhr ein. Jährlich
ist die SCHLECHTESTE aktive Kadenz (Steuer + ST-Falle). Bestätigt „stur+selten schlägt clever" final.

**Deployable-Empfehlung (Endfassung, ehrlich):** Aktien-ETF-Kern; OPTIONAL 10–15 % Xetra-Gold als
Versicherung (Prämie im Flat-Szenario ~10–20 % Endvermögen, dafür Floor/DD massiv besser); Rebalancing
alle 2 Jahre (nie jährlich); kein Silber, keine Bonds, kein VT, kein Timing. N=153.

## Welle 28 (2026-07-12) — Läufe 154–157 (H-062 VIX-Covered-Calls, H-063 EUR-Realität, H-064 Faktor-ETFs), N=157.

**H-062 Covered Calls mit ECHTER VIX-IV (2005–26, Skew-Haircut ehrlich):** Der H-046-Optimismus
SCHRUMPFT mit Daten-IV massiv. Overlay-Beitrag: ATM immer negativ (−85k…−286k). 3 % OTM: +82k nur
bei skew0 (unrealistisch), ~0 bei skew10, −79k bei skew20. 5 % OTM: +101k/+39k/−18k. Realistische
Skew-Zone (10–20 % IV-Abschlag für OTM-Calls) → **bester Fall +39k auf 454k Buy&Hold über 21,5 J
(~+8 %), schlechtester −18k.** → Covered-Call-Feld von CONDITIONAL auf **GRENZWERTIG-NULL**
herabgestuft: mit realer IV + realem Skew liegt der Netto-Beitrag um Null; echte Optionsdaten würden
vermutlich dazwischen landen. Die Tür ist analytisch fast zu (nur noch echte Preise könnten
überraschen). ATM-Desaster bestätigt.

**H-063 EUR-Denominierungs-Realität (2004–26):** Verdicts kippen NICHT — sie verstärken sich für
den EUR-Anleger: SPY in EUR 602k (besser als USD 530k, USD wertete auf); **Gold in EUR 962k netto
(steuerfrei!)**; 70/30 in EUR: 710k > 100 %-SPY 602k, Sharpe 0,692 vs 0,552, **MaxDD −0,26 vs −0,53**.
Für den EUR-Anleger hedgt der Gold-Sleeve zusätzlich das USD-Risiko → das Versicherungs-Argument
wird in EUR STÄRKER (DD mehr als halbiert). Gold-Regime-Caveat gilt unverändert.

**H-064 Faktor-ETFs im Steuer-Wrapper (fairer Test: 18,46 %, interne Umschichtung steuerfrei):**
| ETF | final | vs SPY | t(Excess) | beats |
|---|---|---|---|---|
| MTUM (Momentum, 2013+) | 607k | 413k | 1,41 | ✗ |
| USMV (MinVol) | 432k | 518k | −0,89 | ✗ |
| QUAL | 447k | 378k | 1,44 | ✗ |
| VLUE | 456k | 413k | 0,48 | ✗ |
| SCHD / NOBL (Div) | 524k / 313k | 518k / 378k | −0,07 / −0,75 | ✗ |
| **SPMO (S&P-Momentum, 2015+)** | **574k** | **320k** | **2,14** | formal ✓ |

**SPMO-Einordnung (ehrlich, KEIN PASS):** formal erfüllt (t=2,14>2), ABER (a) Familien-Widerspruch:
MTUM = derselbe Faktor mit LÄNGEREM Fenster failt (t=1,41) → klassischer Implementierungs-Pick;
(b) 10,7 J = 1 Regime (Mega-Cap-Momentum); (c) t=2,14 bei N=157 Trials weit unter DSR-Schwelle;
(d) **eigener 30-J-Kreuzbeleg:** h051-Gross-Lauf zeigt Momentum-Top-20 BRUTTO (=fondsintern, wie
ein ETF es intern hätte) CAGR 10,90 % < SPY 11,21 % — der Wrapper rettet keinen Faktor ohne
Brutto-Alpha. → Faktor-ETF-Familie: FAIL; SPMO = Watchlist-Kuriosität, kein Verdict. N=157.

## Welle 29 (2026-07-12) — Läufe 158–159 (H-065 Entnahmephase-MC, H-066 Rolling-Start), N=159. VERDICT: Gold-Sleeve DOMINIERT die Entnahmephase; ABER nur 32 % der 10-J-Fenster auf Endvermögen — Versicherungs-Charakter final bestätigt.

**H-065 Entnahme (500k, jährl. nominale Entnahme, MC 1.000 Pfade, Steuer auf Gewinnanteil):**
| Entnahme | 100 % SPY Ruin | 70/30 Ruin | Median-Rest SPY | Median-Rest 70/30 |
|---|---|---|---|---|
| 3,0 % | 0,0 % | 0,0 % | 3,41M | 3,64M |
| 4,0 % | 0,6 % | **0,0 %** | 2,95M | **3,24M** |
| 5,0 % | 2,0 % | **0,0 %** | 2,52M | **2,79M** |
→ In der ENTNAHMEPHASE dominiert 70/30 vollständig: Ruin-Risiko auf 0 selbst bei 5 %-Entnahme
(SPY: 2 %), UND höherer Median-Rest auf jeder Stufe. Sequence-Risk ist genau der Ort, wo die
DD-Dämpfung zählt. (Gold-Mean-Caveat im Bootstrap gilt.)

**H-066 Rolling-Start (139 10-J-Fenster, Starts 2005–2016, 2-J-Rebal, End-Liq):** 70/30 ≥ SPY in
nur **44/139 Fenstern (32 %)**. → Auf 10-J-ENDVERMÖGEN verliert 70/30 MEISTENS (2010er = US-Bull +
Gold flach 2011–19); der Full-Window-Vorsprung (21,5 J) hängt an den Gold-Bull-Enden (2005–11,
2022+). Ehrliches Gesamtbild damit final: **Gold-Sleeve = Versicherung, die in ~2/3 der Dekaden
Endvermögen kostet, dafür Drawdown IMMER dämpft und in der Entnahmephase Ruin eliminiert.**
Konsistent mit Maximin (W27). Für das MANDATS-Ziel (Markt auf 10 J schlagen): 70/30 ist KEIN
Gewinner — es ist Risikosteuerung. N=159.

## Welle 30 (2026-07-12) — Läufe 160–161 (H-067 BTC-Sizing/Kelly, H-068 Krisen-Replay EUR), N=161.

**H-067 BTC-Beimischung dimensioniert (statt geraten):** Beobachtet 2016–26: μ=69 %/J, σ=66 %.
Naives Kelly = 154 % (absurd = Hindsight-Beweis). Szenario-Kelly: ×0,25-Haircut → ½-Kelly 17,6 %;
×0,1 → **½-Kelly 5,7 %**. MC (500 Pfade, 10-J-Fenster): bei ×0,25 hebt 2–10 % BTC den FLOOR (q05
219k→243k) bei ~gleichem Median; bei ×0,0 (BTC tot) kosten 5–10 % BTC nur −3…−8 % Median, Floor ~flach.
→ **Robuste Empfehlung: ≤5 % BTC** (½-Kelly des Pessimist-Szenarios; im Tot-Szenario billig, sonst
substanzieller Beitrag). Ehrlich: Wette bleibt Wette — aber jetzt dimensioniert.

**H-068 Krisen-Replay in EUR (konkrete Versicherungs-Zahlen):**
| Krise | 100 % SPY (EUR) | 85/15 | 70/30 |
|---|---|---|---|
| GFC 2008 | **−50,0 %**, nicht erholt bis Ende 2010 | −36,8 %, 937 T | **−27,4 %, 833 T** |
| COVID 2020 | −33,7 %, 323 T | −29,2 % | −24,7 %, **191 T** |
| Inflation 2022 | −17,5 % | −13,4 % | −9,3 % |
→ Der Sleeve halbiert nahezu den GFC-Drawdown in EUR und verkürzt jede Erholung. Das ist die
Versicherung, konkret beziffert. N=161.

## Welle 31 (2026-07-12) — Lauf 162 (H-069 Cash-Flow-Rebalancing im Sparplan), N=162. VERDICT: PASS (klein aber sauber) — Null-Steuer-Rebalancing dominiert.

MC 1.000 Pfade, 1.000 €/Monat, 21,5 J, 70/30-Ziel:
| Modus | Median-Netto | Floor (q05) | Median-DD | Steuer |
|---|---|---|---|---|
| fixe Raten, kein Rebal | 923.937 | 495.249 | −0,234 | 0 |
| **Cash-Flow-Rebal (Rate→Untergewicht)** | **926.069** | **500.160** | **−0,231** | **0** |
| fixe Raten + 2-J-Verkaufs-Rebal | 918.045 | 499.046 | −0,236 | 8.011 |

→ **In der Ansparphase NIE verkaufen:** die frische Rate in den untergewichteten Sleeve lenken
liefert dieselbe Risiko-Kontrolle (End-Gewicht 0,68 vs 0,71, DD gleich) bei bestem Endwert, bestem
Floor und NULL Steuer — dominiert Verkaufs-Rebalancing strikt. Verkaufs-Rebalancing (2-J) ist erst
nötig, wenn keine Raten mehr fließen (Entnahme-/Haltephase). Fügt sich nahtlos ins Muster: jede
vermiedene Realisation gewinnt. N=162.

## Welle 32 (2026-07-12) — Lauf 163 (H-070 Integriertes Endportfolio, Synthese-MC), N=163. VERDICT: PASS (knapp, ehrlich) — 70/25/5 dominiert 70/30 im Basis-Szenario.

MC 1.000 Pfade (2016+-Fenster wegen BTC, 10,5 J), gleiche Pfade je Szenario, Terminal-Steuer:
| BTC-Szenario | 100 % SPY | 70/30 | **70/25/5** |
|---|---|---|---|
| ×1,0 (Hindsight) | 386k/q05 189k/DD −0,34 | 397k/221k/−0,24 | 1.039k/259k/**DD −0,46!** |
| **×0,25 (Basis ehrlich)** | 386k/189k/−0,34 | 397k/221k/−0,24 | **407k/224k/−0,24** |
| ×0,0 (tot) | 386k/189k/−0,34 | 397k/221k/−0,24 | 380k/213k/−0,25 |

→ Im ehrlichen Basis-Szenario (BTC×0,25) schlägt **70/25/5** das 70/30 auf Median UND Floor bei
gleichem DD → **Kriterien erfüllt.** Im Tot-Szenario kostet der 5 %-BTC-Sleeve nur −4 % Median
(nicht materiell). Volle Hindsight (×1,0) zeigt das Potenzial (2,6×), aber auch den Preis: MaxDD
−0,46 (BTC-Crashes schlagen durch — deshalb NICHT mehr als ~5 %). Konsistent mit H-067-Sizing.
**Damit ist das Endportfolio quantitativ fixiert: ~70 % Aktien-ETF / ~25 % Xetra-Gold / 0–5 % BTC
(§23-Disziplin >1 J), Cash-Flow-Rebalancing in der Ansparphase, 2-J-Verkaufs-Rebal danach.** N=163.

## Wellen 33–35 (2026-07-12) — TECHNISCHES INDIKATOR-LABOR: Daily (H-071, 75 Configs), Intraday (H-072, 7), WELT-Sweep (H-073, 950 Configs / 38 Assets). N=163→1195. VERDICT: GESAMTFAIL mit einer BTC-Fußnote.

**H-071 Daily-Batterie (SPY/GLD/BTC, 25 Signale je Asset: SMA/EMA/MACD/Donchian/RSI/Bollinger/
TS-Mom/Vol-Filter + AND/OR/Ensemble-Kombis; korrekte Steuer: ETF 18,46 % AUCH beim Timing, §23 f.
Gold/BTC; 5 bps; OOS=2. Hälfte):**
- **SPY: 0/25 schlagen B&H** (bester OR_SMA200_Mom12 620k vs B&H-OOS-Kriterium verfehlt; B&H 436k).
- **GLD: 0/25** (B&H §23-steuerfrei 844k unerreicht — Timing zerstört die Steuerfreiheit!).
- **BTC: 6/25 schlagen HODL** — ALLE slow-trend (TSmom_252: 47,8M vs 8,0M, Sharpe 0,98, OOS 0,94,
  21 Trades, Haltedauern >1 J → §23-frei, nur 361k Steuer). DSR-p 0,917 < 0,95 → formal FAIL.
**H-072 Intraday-Batterie (SPY 5m, 7 Strategien: ORB 30/60, SMA-Cross, RSI-Rev, Gap-Fade/Follow,
PrevDayMom, day-only, 4 bps):** **ALLE 7 BRUTTO NEGATIV** (−1,7 bis −10,1 bps/Tag; Sharpe −0,4 bis
−3,2). Nicht die Steuer killt — die SIGNALE existieren nicht (5m-SPY effizient + Kosten). Beste
Netto 88k vs B&H 191k. Intraday-Frage definitiv beantwortet: NEIN, und zwar vor Steuern.
**H-073 WELT-Sweep (Auftrag Hans „alle Welt-Daten": 6 Regionen-ETFs, 9 Sektor-ETFs, TLT, SLV, ETH,
20 EU-Blue-Chips × 25 Signale = 950 Configs, Steuerart je Klasse korrekt):**
- **11/950 (1,2 %) schlagen B&H (net+OOS) — WENIGER als Zufall unter Multiple Testing erwartet.**
- Regionen 0/150, Bonds 0/25, Silber 0/25, **ETH 0/25** (bestes 35,6M > HODL 21,1M aber OOS ✗),
  Sektoren 7/225 (verstreut, keine Familie), EU-Aktien 4/500 (verstreut).
- Bester „Gewinner" (RWE-SMA-Cross): **DSR-p 0,214** → Rauschen.
- **KRITISCH für die BTC-Fußnote: ETH repliziert NICHT** (0/25 vs BTC 6/25) → der BTC-slow-trend-
  Befund ist asset-idiosynkratisch (ein einziger sauberer Mega-Bull), keine übertragbare Regel.
  Bei N=1195 ist auch BTC-TSmom weit unter jeder DSR-Schwelle. → Fußnote geschlossen: FAIL.

**GESAMTVERDICT Indikator-Labor: die technische Analyse ist über 1.032 Configs, 41 Assets, 3 Asset-
Klassen-Steuerregime und 2 Zeitebenen hinweg TOT** — Trefferquote unter Zufallsniveau, kein Familien-
Muster außer dem nicht-replizierenden BTC-Fall, Intraday sogar brutto negativ. Der Mandats-Kernbefund
gilt jetzt mit maximaler Test-Breite. N=1195 (Ledger-Konvention: Groß-Sweeps zählen voll — die
DSR-Latte für ALLE künftigen Hypothesen liegt entsprechend hoch; das ist der ehrliche Preis des
„alles testen"). Reproduzierbar: h071/h072/h073_*.py + results-JSONs.

## Welle 36 (2026-07-12) — Läufe 1196–1197 (H-074 VIX-Regime-Gold-Quote), N=1197. VERDICT: FAIL — selbst der mildeste Timing-Fall verliert.

Gold-Quote regime-konditioniert (15 % normal / 35 % bei VIX>rolling-P80, Anpassung nur im 2-J-Raster,
§23-sicher) vs statisch 25 %: **regime 826k < statisch 874k** (Sharpe 0,76<0,79, DD −0,43<−0,39,
Steuer 14k>9k); robust unter Gold×0,5 (628k < 649k). Der steuer-schonendste, seltenste, kleinste
denkbare Timing-Eingriff verliert TROTZDEM (VIX-Spikes sind zu schnell für 2-J-Raster; und schnellere
Raster kosten Steuer — W26). **Timing ist damit auf JEDER Ebene geschlossen: Asset (W33-35), Zustand
(W26), Sleeve-Gewicht (W36).** N=1197.

## Welle 37 (2026-07-12) — Läufe 1198–1205 (H-075 Kalender, H-076 Sektor-Rotation), N=1205. VERDICT: alle FAIL.

**H-075 TOM/DoW (SPY 1993–2026):** TOM existiert BRUTTO schwach (in 6,61 vs out 1,48 bps/T, t=1,65 —
die Literatur-Anomalie ist als Residuum sichtbar), aber 12 Round-Trips/J fressen sie: netto 184k ≪
B&H 436k. TOM_2_2 148k. **Montag-Effekt INVERTIERT** in modernen Daten (DoW_TueFri: in 2,04 < out
8,17 bps — Montage sind heute die BESTEN Tage); DoW_WedOnly 15k (Karikatur). → Kalender-Klassiker
brutto (fast) tot, netto vollständig tot.
**H-076 Sektor-Rotation im Wrapper (9 SPDR, 12-1-Mom, Top-{1,3}, Puffer-Varianten):** 318–339k, alle
≪ B&H 436k, Sharpe 0,33–0,38 ≪ 0,42. Bestätigt Fable-sector_rotation-REJECT in der wrapper-korrekten
Monats-Variante. N=1205.

## Welle 39 (2026-07-12) — H-077 MEGA-STRATEGIE-SUCHE (609 Configs, 9 Stränge) + H-078 PORTFOLIO-LABOR (48 Konstruktionen × 1.000-Pfad-MC). N=1205→1862 (+Whale-Nachzügler). GUARDRAIL-4-RESEARCH-OVERRIDE Hans dokumentiert.

**Stufe-1-Screen (vektorisiert, Steuer-Approximation, KEIN Verdict) — Ergebnisse je Strang:**
| Strang | Configs | Stage-1-Survivors | Befund |
|---|---|---|---|
| INSIDER (Form-4-Grid: Fenster×Min-Insider×Wert×Rolle×Halten) | 162 | **0** | Feld tot, jetzt grid-breit |
| CONGRESS (Grid inkl. Kammer) | 108 | **0** | Feld tot, grid-breit |
| NEWS-Sentiment (Top/Bottom×Quantil×Halten) | 36 | **0** | bestätigt §2.3 |
| GEOPOLITIK (z×Horizont×Basket×Exit-Modus, inkl. Social-Proxy) | 96 | **0** | bestätigt W14/16a |
| SHORT/LS (Signale×Assets, borrow 3 %) | 24 | **0** | Short-Bein zahlt nie |
| FX-Majors (Trend/MR/LS, 44 %-§23-Kurzfrist) | 48 | 4 (trivial) | „Survivors" = ~0,6 %/J über Cash, ökonomisch tot |
| HEBEL (SPY/GLD/BTC×Signale×1,5–3×, Finanzierung 4 %) | 45 | 15 | ALLE BTC-Hindsight (LEV_BTC×2 = Absurd-Compounding); SPY/GLD-Hebel: 0 |
| OPTIONEN (VIX-Modell: CC/CSP/Collar×Strike×Skew×Fraktion) | 90 | 32 | **einziger substanzieller Cluster** — Vol-Risk-Prämie (CSP/CC), ABER modellabhängig (Skew-Annahme treibt), keine echten Preise → bleibt daten-gated wie H-046/062 |
**Kern-Lesart:** Über 609 zusätzliche Configs entsteht KEIN neues deployables Feld. Die einzigen
nicht-trivialen Survivor-Cluster sind (a) gehebeltes BTC (reine Hindsight, DSR-tot) und (b) das
Options-/Stillhalter-Feld — konsistent mit H-046/062: die Vol-Risk-Prämie ist der eine model-
positive Kandidat, verdict-fähig NUR mit echten Optionsdaten.

**H-078 Portfolio-Labor (48 Konstruktionen, ECHTES MC 1.000 Pfade × 2 Krypto-Szenarien, 2017+):**
Top nach FLOOR im ehrlichen ×0,25-Szenario: **65/25/5/5 SPY/Gold/BTC/ETH** (q05 215k, P>SPY 77 %,
DD −0,24), gefolgt von gold-lastigen Blends (40–60 % SPY + g75b25-Rest). Volle Tabelle im JSON.
→ Bestätigt die Endspezifikation über einen 48er-Sweep: Krypto-Sleeve 5–10 % aufgeteilt BTC/ETH
optimiert den Floor; Bonds/EW8/MaxCrypto alle dominiert. **Whale/13F-Nachzügler (CUSIP-Map, 22,8M Positionen, 668 Manager, 21.415 Symbole): 60 Configs,
0 Survivors** (beste c15-20-Konsens ~800-805k, OOS 0,54 — unter Stock-B&H-Bench). → ALLE Event-
Stränge grid-breit tot: Insider 162/0, Congress 108/0, Whale 60/0, News 36/0, Geo 96/0, Short 24/0.
N nach W39 komplett: 1.205+609+48+60=**1.922**.

## Welle 39c (2026-07-12) — H-079 Options-Cluster Stufe-2 (adversarial), N=1934. VERDICT: 0/12 überleben — Feld ist ANNAHME-GEBUNDEN.

Die 32 W39-Options-Survivors gestresst mit verkäufer-feindlichsten Annahmen (CC-OTM-IV = VIX×0,75;
CSP-Put-IV = VIX flat OHNE Skew-Bonus; Prämien-Haircut 10 % Bid/Ask; 3 bps/Mo):
**ALLE 12 repräsentativen Configs FAIL** (beste ADV_CC_otm5_f1.0 366k < B&H 454k; OOS-Sharpes 0,73–
0,89 bleiben > B&H-ähnlich, aber Netto-Endwert fällt durch). **Scharfe ehrliche Schlussfolgerung:**
Der GESAMTE Netto-Edge des Stillhalter-Felds liegt INNERHALB des Annahme-Bandes zwischen verkäufer-
freundlich (Skew-Bonus, volle Prämie → PASS, W39/H-046) und verkäufer-feindlich (→ FAIL). Die
Wahrheit liegt dazwischen und ist OHNE ECHTE OPTIONSPREISE nicht entscheidbar. Feld-Status final:
**UNENTSCHEIDBAR-modellbasiert, daten-gated** — weder PASS noch definitiv tot. Einzige Auflösung:
echte Optionsdaten (Operator-Entscheidung Hans). N=1934.

## Welle 40 (2026-07-12) — H-080 Rest-Dimensionen (30 Configs), N=1964. VERDICT: FAIL/exploratorisch — plus E-052-Artefakt-Fang.

**⚠️ Datenqualitäts-Fang (E-052, 2 Schichten):** (1) `month_panel` im Suchframework las prices_verdict
OHNE kanonische Hygiene-Trunkierung; (2) `pct_change()` paddete über NaN-Lücken delisteter Serien →
Ganz-Universum-Basket zeigte Fake-Endwert 7,5×10³⁰ (nach Fix 1: 10¹²; nach Fix 2: plausibel).
Beide Fixes im Framework (`fill_method=None` + |Monatsret|>100 %-Artefakt-Drop). Selektive W39-
Strand-Verdicts (0-Survivors) davon UNBERÜHRT (Inflation hätte sie nur begünstigt — sie failten
trotzdem); Anti-Pattern dokumentiert (docs/CLAUDE_CODING_ERRORS.md E-052).

**Ergebnisse (bereinigt; Bench = total-return-SPY-B&H 26,375 % ≈ 2,09M, Fenster 1995–2026):**
- **Insider-VERKÄUFE als Avoidance-Filter (12): 0 Survivors** — beste Filter-Baskets 1,85–1,95M
  < 2,09M; der Filter fügt dem Ganz-Universum-EW nichts hinzu (Insider-Sells sind mehrheitlich
  Diversifikation/Comp, kein Signal — konsistent mit Literatur).
- **Event×Technik (Insider-Kauf UND >SMA200-Proxy, 12): 0 Survivors** — beste 1,28M (OOS 1,1–1,2
  ordentlich, aber Niveau ≪ Bench). Konfirmations-Logik rettet das Insider-Feld nicht.
- **EU-Querschnitts-Momentum (6): 3 „Survivors" NUR vs EU-EW-Bench (828k > 675k)** — ABER Universum
  = 20 HEUTIGE Blue-Chips (survivorship-verzerrt), Familie klein, DSR bei N≈2000 chancenlos →
  exploratorisch, KEIN Verdict; konsistent mit H-028-GEM-FAIL. N=1964.

## Welle 41 (2026-07-12) — H-081 Stillhalter-VERDICT via echter CBOE-Historie (Vorschlag Hans), N=1967. **VERDICT: FAIL — die letzte offene Tür ist GESCHLOSSEN.**

Daten: CBOE-Original-Historien (echte gehandelte SPX-Optionspreise, realer Skew/Settlement):
BXMD 1986–2026 (38,4 J!), PUT (dichte Ära 2007+, 19,4 J), BXM 2002+ (24,3 J); SPX-TR via EODHD.
Daten-Fix unterwegs: PUT-CSV früh lückenhaft → Kalender-Annualisierung + Lücken-Maskierung (die
erste PUT-Zeile mit „SPXTR 20 %/J" war ein Annualisierungs-Artefakt — gefangen und korrigiert).

**Brutto (Form & Regime „mit eigenen Augen"):**
| Index | CAGR | SPXTR | Sharpe | SPXTR | MaxDD | SPXTR |
|---|---|---|---|---|---|---|
| BXMD (30Δ OTM, 38,4 J) | 10,90 % | 11,52 % | **0,903** | 0,821 | **−0,43** | −0,51 |
| PUT (19,4 J) | 7,08 % | 11,04 % | 0,689 | 0,754 | −0,33 | −0,51 |
| BXM (ATM, 24,3 J) | 6,11 % | 10,16 % | 0,609 | 0,724 | −0,36 | −0,51 |
**Dekaden-Regime:** BuyWrite gewinnt NUR in der Lost Decade (2000er: BXMD +2,8 vs −0,9; PUT +1,2 vs
−6,3) — verliert in jeder Bull-Dekade 4–6 pp/J (2010er/2020er); 1990er ~Gleichstand. → Die Prämie
ist real, aber sie ist eine VERSICHERUNGS-Prämie: sie zahlt in Seitwärts-/Bär-Regimen und kappt
Bullen. Selbst das beste Design (BXMD) trailt über 38 Jahre BRUTTO um 0,6 pp/J.

**Deutsches Steuer-Overlay (Monats-Dekomposition, Approximation benannt; Stillhalter-Asymmetrie:
positive Options-Monate sofort 26,375 %, negative nur Topf):** ALLE drei klar unter dem ETF-Pfad —
BXMD 3,56M vs 5,15M; PUT 359k vs 654k; BXM 459k vs 925k. Overlay-Beitrag überall NEGATIV.

**AUFLÖSUNG DES H-079-BANDES:** Die Realität (echter Skew, echte Preise, 40 J) liegt am ADVERSARIALEN
Ende. Die Modell-„PASSes" (H-046-Grid, W39-Cluster) waren die verkäuferfreundliche Bandkante; real
underperformt BuyWrite schon brutto in Bull-Regimen, und die asymmetrische deutsche Stillhalter-
Besteuerung verbreitert es. **Stillhalter-Feld: FAIL für den steuerpflichtigen deutschen Anleger —
GESCHLOSSEN.** Schluss-Synthese: BXMDs Sharpe/DD-Verbesserung ist dieselbe Versicherungs-Familie wie
der §23-Gold-Sleeve — aber der Gold-Sleeve liefert sie mit 0 % Steuer statt 26,375 % asymmetrisch →
**der Gold-Sleeve dominiert BuyWrite als Risiko-Werkzeug strikt.** Damit hat das Mandat KEINE offene
Alpha-Tür mehr auf erreichbaren Daten. N=1967.

## Welle 42 (2026-07-12) — H-082 Versicherungs-Duell: Protective Put vs §23-Gold-Sleeve (echte CBOE-Historie), N=1969. **VERDICT: Gold-Sleeve VERNICHTET Options-Versicherung.**

**PPUT (5 %-OTM-Protective-Put, 38,4 J echte Preise):** CAGR 8,03 vs SPXTR 11,52 % = **−3,5 pp/J
Versicherungskosten**; Sharpe SCHLECHTER (0,71 vs 0,82); DD nur −0,51→−0,39. Und der Killer: in der
Crash-Dekade (2000er), wo die Versicherung glänzen müsste, lieferte PPUT **−1,39 % vs SPX −0,95 %** —
die Prämien-Blutung fraß sogar den Crash-Schutz. CNDR: DD −0,19 aber CAGR 5,55 (Bond-Profil, netto
nutzlos: DE-Overlay 25k!). CLL: 6,95 vs 12,60 %. DE-Overlay 2005+: PPUT 545k vs ETF 806k.

**Duell-Tabelle (Fenster 2005–2026):**
| Versicherung | Netto-Endwert | MaxDD | Steuer auf Hedge |
|---|---|---|---|
| Keine (100 % SPY) | 767k | −0,55 | — |
| **70/30 §23-Gold (biennial)** | **890k (> ungesichert!)** | **−0,36** | **0 %** |
| Protective Put (PPUT, real) | ~545k | ~−0,39 | 26,375 % asym. |
| Collar / Condor | 405k / 25k | −0,22 / −0,19 | 26,375 % asym. |

**Endgültige Portfolio-Erkenntnis:** Der §23-Gold-Sleeve ist die einzige Absicherung mit POSITIVEM
Erwartungswert und 0 % Steuer — er dominiert jede optionsbasierte Versicherung (Put/Collar/Condor)
auf 40 Jahren echter Daten strikt: mehr Endvermögen als UNGESICHERT bei besserem Drawdown, während
Puts −3,5 pp/J kosten und im Crash-Jahrzehnt nicht mal halfen. Versicherungs-Frage GESCHLOSSEN. N=1969.

## Welle 43 (2026-07-12) — H-083 Einheitliche OOS-Re-Evaluation ALLER Strategien (Auftrag Hans; keine neuen Trials, N=1969).

**Teil A — Ernte gespeicherter OOS-Metriken (1.112 Configs aus 7 Familien-JSONs):** OOS-Mediane
0,24–0,86, alle unter SPY-B&H-OOS (0,78); die 13 „flagged" in h077 = Options/Hebel-Cluster (durch
H-079/H-081 aufgelöst), die 3 in h080 = EU-Momentum (survivorship-exploratorisch). → Kein Signal-
Survivor in den gespeicherten OOS-Hälften.

**Teil B — Einheitliches Rezenz-Holdout 2021-07→2026-07 (2022-Bär+Bull+2025-Vol; SPY: 11,69 %/
Sharpe 0,732/DD −0,25). 19 kanonische Strategien, gleiche Metrik, brutto inkl. Kosten:**
**4/19 schlagen SPY** (CAGR UND Sharpe):
| Gewinner | CAGR | Sharpe | Einordnung |
|---|---|---|---|
| GLD HODL (§23) | 17,65 % | 0,984 | Gold-Regime (läuft heiß — bekannter Caveat) |
| **70/30 SPY/GLD** | **14,09 %** | **1,02** (DD −0,20) | **die Endspez. — via Gold-Regime + Diversifikation** |
| 65/25/5/5 | 12,95 % | 0,852 | Endspez.-Variante |
| BTC TSmom252 | 36,2 % | 0,789 | bekannter idiosynkratischer Einzelfall (ETH repliziert nicht) |
**0/15 SIGNAL-Strategien schlagen SPY im Holdout:** SMA200/TSmom/OR-Combo (8–10,6 %), alle CBOE-
Options-Indizes (BXMD 10,5/PUT 9,5/BXM 8,5/PPUT 8,7/CNDR 3,1), TOM 2,0 %, Halloween 3,4 %,
SektorRot 3,7 %, 60/40 4,4 %, RP-IV3 (PIT-Gewichte) 6,4 %, BTC/ETH-HODL (Sharpe/DD).
**Fazit:** Auch im einheitlichen jüngsten Regime-Mix überlebt KEIN Signal; es gewinnen ausschließlich
die gold-haltigen Aufstellungen (Regime-getrieben, konsistent mit allen Caveats) — die Endspezifikation
bestätigt sich als beste real gelaufene Aufstellung des Holdouts (Sharpe 1,02, bester DD), aus
Diversifikation + §23, nicht aus Alpha. Ehrlichkeits-Rahmen: pseudo-OOS (Daten lagen bei Design vor).

## Welle 44 (2026-07-13) — H-084 Odd-Lot-Tender (neue Alpha-Landkarte, kapazitätsbeschränkt), N=1970.

**Stufe 1 (Drift-Proxy): FAIL** — 186 Events geharvestet (EDGAR SC TO-I, 61 ohne Preisdaten =
überwiegend non-traded REITs), 125 bepreist: Excess vs SPY +10BD +0,68 % (t=1,22), +30BD −0,22 %,
+45BD −0,39 %, Hit ~50 %. → KEIN anomaler Drift im Tender-Fenster; der Markt preist Tender im
Mittel effizient. **Methodisch ehrlich:** Der Proxy misst NICHT den Odd-Lot-Mechanismus selbst
(≤99 Stück unter Tender-Preis kaufen → zum Fixpreis andienen = Spread unsichtbar im Drift).
**Stufe 2 (echter Capture, geparste Tender-Preise): MECHANISMUS BESTÄTIGT, Magnitude unzuverlässig.**
189 Filings, 133 mit Preis geparst, 124 Captures: **60,5 % positiv, Median +3,73 %**, ~6,5 positive
Events/Jahr; Mean +64,9 % = klar parse-/entity-verrauscht (Regex-Fehlgriffe, non-traded-REIT-NAV-
Tender, Ticker-Mismatches — 0,2–5×-Filter zu lax). **Ehrliche Lesart:** Die Odd-Lot-Prämie existiert
richtungsmäßig (Mehrheit der Tender über Markt = konsistent mit Literatur), aber der backtest-basierte
Magnituden-Schätzer taugt nicht — die Strategie ist inhärent kuratiert (Fall-Prüfung nötig).
**Der entscheidende ehrliche Punkt — SKALIERUNG:** ≤99 Aktien × typ. $10–40 × ~3–5 % Capture ×
~6 Events/Jahr ≈ **~200–600 €/Jahr** Alpha-Obergrenze. Real, wiederholbar, risikoarm — aber per
Konstruktion Taschengeld, kein Vermögenshebel. Genau DAS ist kapazitätsbeschränktes Alpha: es
existiert, WEIL es winzig ist. Verdict: **REAL-ABER-KLEIN; Nutzen = Forward-Scanner (neue Filings
alerten, Fall manuell prüfen), kein Backtest-Feld.** Scanner-Infra steht (h084b, oddlot_captures.parquet).
Parallel: MEMO_ABFINDUNGSWERTE.md (deutsches Pendant, strukturell größer pro Fall) erstellt. N=1971.

## Welle 45 (2026-07-14) — H-085 Abfindungswerte-Watchlist (operativ, kein Trial), N=1971.

Inventar offener Fälle erstellt (ABFINDUNG_WATCHLIST.md, an Hans zugestellt): 6 Fälle — darunter
**Staatl. Mineralbrunnen: Kurs 45,12 UNTER Abfindung 46,00 = +1,95 % positiver Spread + Gratis-
Nachbesserungs-Option (Lehrbuch-Setup, HV Aug 2026, sehr illiquide)**; HHLA/capsensixx handeln mit
eingepreister Nachbesserungs-Prämie (−2,5/−4,1 %); Covestro/niiio Ticker-Lücken (manuell). Laufende
Verfahren bestätigen Basisrate (Volksfürsorge +26,2 %, Gauss +13 %). Radar-Prozess dokumentiert
(spruchverfahren-direkt/SpruchZ, Einstiegs-Trigger: HV-Beschluss + Kurs ≤ Abfindung). Damit ist die
„echtes Alpha"-Schiene operativ: Scanner (Odd-Lot US) + Watchlist (Abfindung DE) stehen.

## OPS-Ereignis (2026-07-14, kein Trial) — Paper-Pilot-Halt 07.–14.07. gelöst (Reconcile/Adoption)

Befund auf Nachfrage Hans („läuft die Paper-Engine?"): Infrastruktur GESUND (Scheduler+DMS seit
02.07., Heartbeat frisch, Zyklen täglich 21:30), aber **Trading seit 07.07. gehaltet**: Reconcile-
Fail-Closed (korrekt!) — GLD 19,588 @ 382,17 + TLT 87,573 @ 85,11 im Alpaca-Konto (manuell, ~14,9k $),
Ledger kannte sie nicht (cash_diff 14.941,52 $ > 100 $/10 bps). **Lösung (Operator-Wahl Hans:
Adoption):** neues One-off `scripts/ops_adopt_external_positions.py` (Dry-Run-first, offizielle
APIs ops.paper_ledger/broker_adapter read-only, atomarer Save) — BUY-Fills zu Broker-avg_entry
adoptiert → Ledger-Cash 87.302,61→72.362,87 vs Broker 72.361,08, **Rest-Diff +1,79 $** (avg_entry-
Rundung; weit unter BEIDEN Schwellen — Gate trippt bei >100 $ ODER >10 bps, run_live_paper:672/
_mismatch_exceeds_threshold:76, Review-Stage-3-Präzisierung). ack_halt mit
dokumentiertem Grund; Reconcile-Verify: 0 Positions-Diffs. **Pilot resumed — nächster Zyklus
14.07. 21:30.** Equity 87.033 = −0,96 % vs Baseline 87.875 (Stop 79.087 fern). Residual: LLY-Dust
(1e-9) beidseitig, unkritisch; Telegram-Credentials weiterhin unkonfiguriert (Alert nur Log).

## H-086 + E-051-Re-Runs (2026-07-22) — GESAMTBEWERTUNG P6 = W2 (Vorabpauschale) + W3 (Frozenset-Re-Validierung). Kein neuer Trial, N=1971.

**Teil 1 — H-086: Vorabpauschale (§18 InvStG) im ETF-Benchmark-Pfad (W2).** Script
`research/mandat/h086_vorabpauschale_etf_benchmark.py`, Ergebnis `results/h086_vorabpauschale.json`.
Fenster = H-032-window-matched (1996-02-07 → 2026-07-06), Kandidaten-Endwerte unverändert aus
h032b (post-E-051). Alt-Pfad exakt reproduziert (1.610.149 ✓).
- **Teilfreistellungs-Befund: KEINE zweite Lücke.** TFS 30 % war im Alt-Pfad bereits modelliert —
  als gerundete 18,5 % statt exakt 26,375 %×0,7 = 18,4625 % (Alt-Pfad besteuerte den ETF 0,0375 Pp
  zu HOCH). Beide Korrekturen (VP senkt ETF, exakte TFS hebt ETF) netto verrechnet, kein Rosinenpicken.
- **ETF alt 1.610.149 → neu 1.609.380** (VP ab 2018, exakte TFS, ohne Sparerpauschbetrag) =
  **netto −0,048 %** (Zerlegung: VP-Effekt −0,091 %, TFS-Rundung +0,043 %). Mit 1000 €/J SPB auf
  die VP: 1.610.944 (VP-Steuer fast vollständig absorbiert). VP-Steuern kumuliert nur 11.955 €
  (2019/20/23/24/25; 2018 Wertzuwachs negativ → VP=0; 2021/22 Basiszins 0; 2026 = Verkaufsjahr).
- **Warum so klein (korrigiert die Registry-Schätzung „3–6 % Endwert-Senkung" — stark überschätzt
  für dieses Fenster):** Die VP ist eine Steuer-VORAUSZAHLUNG — angesetzte VP mindern per §19
  InvStG den End-Veräußerungsgewinn zum IDENTISCHEN Satz (18,46 %). Netto bleibt nur der entgangene
  Zinseszins auf die vorgezogenen Zahlungen, und die großen Basiszins-Jahre (2023–25) liegen am
  Fensterende. Basiszins-Tabelle (BMF) im Script dokumentiert; 2026 = Carry-Forward 2,53 % (Annahme).
- **Verdicts: NICHTS kippt.** H-032 low-div 1.589.963: −1,25 % → **−1,21 %** vs ETF (bleibt UNTER
  ETF). H-024 (EW-Band-full) 1.146.757: −28,78 % → −28,75 %. Der Mandats-Kernbefund („keine
  Aktien-Strategie schlägt den ETF nach ehrlicher Steuer") wird durch die VP-Präzisierung BESTÄTIGT
  und ist jetzt benchmark-seitig sauber.
- Modellgrenzen: vor 2018 alte Rechtslage (ausschüttungsgleiche Erträge) NICHT modelliert → ETF-Pfad
  1996–2017 weiterhin eher zu stark (Richtung: pro Benchmark/gegen Kandidaten); VP-Zahlung via
  Anteilsverkauf am Jahresultimo, Mikro-Verkaufsgewinn unbesteuert (minimal pro ETF); Verkaufsjahr
  ohne VP. Deterministisch verifiziert (2 Läufe PYTHONHASHSEED 0/42 byte-identisch).

**Teil 2 — W3: E-051-Re-Runs der nie re-validierten FAIL-Screens** (je 2× PYTHONHASHSEED 0/42,
Byte-Vergleich der Result-JSONs; Alt-JSONs gesichert in `results/backup_pre_h086/`):
| Screen | alt (pre-Fix) | neu seed0 / seed42 | byte-identisch | Verdict |
|---|---|---|---|---|
| H-029 13F k10 | 752.752 | 662.222 / 685.998 | **NEIN** | FAIL bestätigt (beide Seeds: DSR 0,690/0,712 fail; crit1/2/4/5 ✓ wie alt; ETF 482.986) |
| H-031 Insider off10k | 1.295.845 | 1.078.459 / 1.167.489 | **NEIN** | FAIL bestätigt (beide Seeds: DSR 0,644/0,671 fail; crit5 jetzt ✓ statt ✗ — PASS bleibt false; ETF 772.823) |
| H-047 Buyback v33 | 585.904 | 573.906 (beide) | JA | FAIL bestätigt (alle 4 Kriterien false; Base-EW 1.338.701 = konsistent mit h032b) |
| H-035 SC-Momentum out120 | 63.115 | 249.995 (beide) | JA | FAIL bestätigt (Kriterien 1–4 false wie alt; weit unter ETF 736.459) |
| H-036 Size small/large | 5,97M / 3,53M | 6,73M / **1,00M** (beide) | JA | Script-„PASS" bestätigt; Gesamt-FAIL (Artefakt) bestätigt via h036c-Re-Run |
| H-036c ADV-Floor 1M/10M/50M | 5,47M/5,01M/1,90M | 6,05M/5,28M/1,93M (beide) | JA | Kollaps-Muster bestätigt (−68 % bei ADV≥$50M, DD −0,656) → Illiquiditäts-Artefakt-Deutung steht |
- **Ehrlicher Restbefund (E-051 nur TEILWEISE geschlossen):** H-029 und H-031 sind weiterhin
  NICHT deterministisch — beide nutzen script-EIGENE Portfolio-Loops (`run_consensus`/`run_insider`)
  mit Set-Iteration (`targets`, `held − keep − targets`, `fresh`) → Kauf-/Verkaufsreihenfolge bleibt
  PYTHONHASHSEED-abhängig (Verlusttopf-/Cash-Timing); der sorted()-Fix in verdict_engine greift dort
  nicht. Die Verdicts sind über die beobachtete Streuung (±4–8 % Endwert) hinweg STABIL (DSR-Fail
  in jedem Draw, Kriterienmuster identisch), aber „byte-identisch re-validiert" gilt für beide NICHT.
- Frozenset-Sensitivität war real und teils GROSS: H-035-Momentum-Endwerte ~4× höher als der
  Alt-Draw (68k→280k etc., Verdict unverändert FAIL), H-036-Large-Band-EW −72 % (3,53M→1,00M).
  Einzelne pre-Fix-Absolutwerte dieser Screens sind als Zufalls-Draws zu lesen; die Verdicts selbst
  kippen nirgends. **H-036b (Kosten-Sensitivität) wurde NICHT re-run** (nicht verdict-tragend;
  h036c war der Diskriminator). Kein Verdict-Flip in P6. N bleibt 1971 (Modell-Korrektur +
  Re-Validierung, keine neuen Trials).

## E-051-Determinismus-Fix H-029/H-031 (2026-07-25) — Rest-Nichtdeterminismus geschlossen. Kein neuer Trial, N=1971.

Follow-up aus GESAMTBEWERTUNG P6/W3 (Eintrag 2026-07-22): H-029 und H-031 waren als einzige
Screens NICHT byte-identisch über PYTHONHASHSEED 0/42 — Ursache script-eigene Set-Iterationen in
den Portfolio-Loops (`run_consensus`/`run_insider`); der sorted()-Fix in verdict_engine griff dort
nicht. Minimal-invasiv gefixt (nur sorted() an den Iterationsstellen, KEINE Logikänderung;
Kommentar „E-051-Determinismus-Fix 2026-07-24" je Stelle):
- `h029_13f_consensus.py`: Z.164 `for sym in sorted(held - keep - targets)` (Sell-Pending-Reihenfolge),
  Z.167 `entries = sorted(s for s in targets if s not in held)` (Buy-Pending-Reihenfolge).
- `h031_insider.py`: Z.132 `for s in sorted(fresh)` (hold_until-Insertion), Z.137
  `for sym in sorted(held)` (Sell-Pending-Reihenfolge), Z.141
  `entries = sorted(s for s in fresh if s not in held)` (Buy-Pending-Reihenfolge).
  Tie-Break = Ticker-Alphabet: deterministisch und fachlich neutral (EW-Slots, fixe Beträge).

**Verifikation (je Screen 2 volle Läufe, PYTHONHASHSEED=0 und =42, SHA256-Byte-Vergleich der
Result-JSONs; Kopien `results/h029_results.fix_seed0/42.json`, `h031_results.fix_seed0/42.json`):**
| Screen | byte-identisch | Verdict | Kriterienmuster | Endwert (deterministisch) |
|---|---|---|---|---|
| H-029 13F k10 | **JA** (91FCF1A6…) | FAIL unverändert | c1✓ c2✓ c3-DSR✗ (0,722) c4✓ c5✓ — identisch zu Re-Runs 2026-07-22 | 732.455 (Re-Run-Draws waren 662k/686k) |
| H-031 Insider off10k | **JA** (C563CD12…) | FAIL unverändert | c1✓ c2✓ c3-DSR✗ (0,679) c4✓ c5✓ — identisch zu Re-Runs 2026-07-22 | 1.133.711 (Re-Run-Draws waren 1.078k/1.167k) |
- Endwerte liegen in/nahe der 2026-07-22 beobachteten Seed-Streuung; kein Kriterium kippt,
  PASS bleibt in beiden Fällen false (DSR-Fail). E-051 damit für ALLE Verdict-Screens vollständig
  geschlossen: keine bekannte PYTHONHASHSEED-Abhängigkeit mehr. `h029_results.json`/
  `h031_results.json` = neuer deterministischer Stand (byte-gleich zu beiden Seed-Läufen).

---

## NACHTRAG 2026-08-08 — Mandat II im Ledger + Autoritaet der H-Nummernvergabe

**Warum dieser Eintrag existiert:** Der Ledger endete bei Welle 45 und kannte Mandat II nicht.
Genau daraus entstand am 2026-08-05 die **H-086-Kollision**: wer die naechste freie Nummer in
`registry.md` suchte, sah H-085 und uebersah das hier vergebene H-086 (Vorabpauschale). Die neue
Trendfilter-Hypothese musste nachtraeglich auf H-087 umbenannt werden.

**Regel ab jetzt (Autoritaet):** H-Nummern werden GEGEN BEIDE Dateien geprueft —
`grep -ohE "H-0[0-9]{2}" research/registry.md research/ledger.md | sort -u | tail` liefert die
hoechste vergebene Nummer. Vergeben sind bis heute: **H-001..H-088** (H-086 = Vorabpauschale,
2026-07-22; H-087 = Trendfilter-Langzeit; H-088 = Insider-DERA).

**Mandat II in Kurzform (Detail: `research/mandat2/ABSCHLUSS.md`, append-only dort):**
- P1–P13e, Wellen 46–48b, Trials 1.964 -> 3.539 kumuliert.
- Kein Kandidat bestand die Mehrfachtest-Korrektur. Holdout unangetastet.
- Felder geschlossen: Aktienauswahl (DSR), SPY-Trendfilter (DSR+PBO; H-087: verliert ohne
  Dauerkrise 0 von 338 Fenstern), Insider (H-088: unter Zufallskorb), Geopolitik/Akteurs-Posts
  (Welle 48/48b: Posttage = Zufallstage, Langfrist-Anti-Signal aus einer Episode).
- Sieben Befunde in ABSCHLUSS.md, darunter: Datenbasis traegt keinen SPY-Vergleich
  (Survivorship 2,36–2,90 pp bei 1,5 pp Marge); Ticker ist kein Schluessel (Befund 7).
- Neue Datenbestaende: SEC-DERA Form 4 (17.134 Emittenten, 2006–2026), CRSP-Marktreihe ab 1926,
  Truth-Social-Vollarchiv (40.631 Posts). Anti-Patterns E-107..E-131.

### Nachtrag 2026-08-11 — H-089 (Welle 49): Komposit-Kampagne abgeschlossen
Kampagne K1–K10 + Robustheit (research/strategie_n1/): Trials 3.559 → 6.269.
Verdikt: EIN bestätigter Kandidat (TS-Momentum-Dial/Trend-Chor, defensiv,
Index-Ebene) — Details Registry Welle 49. Negativ-Karte: Gold-Timing,
Einzelaktien-Timing, große Chöre, Offensiv-Geo (N1-GEO/Komposit W1–W6)
alle tot. Zwei ungültige Erstläufe offen dokumentiert (K10-Overflow
E-052-Klasse; Krypto-EODHD-Abbruch) — append-only, nicht zurückgeschrieben.

### Nachtrag 2026-08-16 — H-090 (Welle 50): Kurs-Exit-Familie auf Top-K-Momentum — FAIL, Feld zu
Quelle: extern zugelieferte Strategiespezifikation (User-Dokument). Getestet wurde
AUSSCHLIESSLICH die Kurs-Exit-Familie (Stop/Ziel/Trailing/RSI/5T-Tief/Zeit) auf der bereits
dreifach gefallenen Basisstrategie (H-011/012/049), PIT-Panel, Suchfenster 1995–2016.
Trials 6.269 → **6.277** (einmalig gebucht; zwei kontaminierte Laeufe — Zukunftspreis-Delisting/
Phantomtage in Phase 1, verschluckter Zero-Day-Trade im Phase-2-Replay — wurden nach
Review-BLOCKERn verworfen und OHNE Neubuchung wiederholt; Artefakte als
results/*.CONTAMINATED_run1.json quarantaenisiert).
Verdikt: **FAIL auf allen drei Sekundaerachsen** fuer alle 5 primaer bestandenen Varianten
(BASIS/V1/V4/V5/V6 — primaer nur gegen die V7-Nullmessung): Netto-PRIVAT_DE bester Kandidat
BASIS 687.867 < EW-Benchmark 727.364; DSR heterogen 5/5 durchgefallen (bestes p=0,37 gegen
0,95); PBO 71,4 % (trotz E-077-Verzerrung nach unten). Kernbefund: **beats_basis = false in
7/7** — KEINE Kurs-Exit-Variante schlaegt die nackte Basisstrategie, in beiden Haelften.
Stopp-Regel greift: Feld „Kurs-Exits auf Einzelaktien-Momentum" ZU; Wiederaufnahme nur mit
echtem OHLC ueber die volle Historie. Anti-Patterns E-152..E-155. Details: Registry Welle 50,
research/mandat2/results/h090_momentum_exits.json + h090_phase2_sekundaer.json.
