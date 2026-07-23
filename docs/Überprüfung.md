# Überprüfung: OOS und CAGR — gibt es einen tragfähigen Edge, oder nicht?

Erstellt: 2026-05-31
Modus: **Reine Recherche** (read-only, kein Code geändert, keine Backtests neu ausgeführt)
Thema: ausschließlich **Out-of-Sample (OOS)** und **CAGR** — präzise.
Autor dieser Recherche: Claude Code (agentische Durchleuchtung + externe Literatur)

---

## 0. Wie dieses Dokument zu lesen ist (Quellentrennung)

Jede inhaltliche Aussage trägt eine von drei Marken. Das ist die zentrale Disziplin dieser
Datei — keine Vermischung von Fakt, Zitat und Vermutung.

- **[V] = Verifiziert** — direkt im Code oder in einem Repo-Artefakt gelesen, mit `Datei:Zeile`
  oder Dateipfad belegt. Zahlen wurden **wörtlich** übernommen, nicht gerundet oder erfunden.
- **[Z] = Zitiert** — externe Literatur/Quelle, mit URL im Quellenverzeichnis (§9).
- **[H] = Hypothese / Spekulation** — meine eigene Schlussfolgerung oder Vermutung. Nicht belegt,
  ausdrücklich als unsicher markiert.

**Wichtige Ehrlichkeits-Hinweise vorab:**

1. Ich habe **keinen einzigen Backtest neu ausgeführt**. Alle CAGR-/Sharpe-/MaxDD-Zahlen unten
   sind **transkribiert** aus bereits existierenden Repo-Artefakten (`docs/results/*.md`,
   `docs/PROJEKT_ABSCHLUSS_2026_05.md`). Ich habe die Formeln, die diese Zahlen erzeugen,
   im Code verifiziert — aber nicht die Zahlen selbst reproduziert.
2. Diese Artefakte sind **nicht CI-bestätigt** und **nicht unabhängig repliziert**. Sie sind
   automatisch erzeugte Skript-Ausgaben.
3. **Das Projekt hat diese Frage bereits beantwortet.** Es existiert ein formelles
   Abschlussdokument `docs/PROJEKT_ABSCHLUSS_2026_05.md` (2026-05-29), dessen Verdikt mit dem
   übereinstimmt, was diese Recherche prüfen soll. Ich habe dieses Dokument nicht blind
   übernommen, sondern seine tragenden Aussagen gegen den Quellcode und die Ergebnisdokumente
   gegengeprüft. Diese Überprüfung **bestätigt** das Verdikt und ergänzt es um eine
   methodische Bias-Analyse und externe Literatureinordnung.

---

## 1. Ehrliches Gesamturteil (vorab, ohne Beschönigung)

**Auf dem aktuell verfügbaren Datensatz (Alpaca, ~75–194 überlebende Symbole, 2018–2025,
Large/Mid-Cap, long-only) und mit der aktuellen Methodik gibt es keinen tragfähigen,
risk-adjustierten Edge gegenüber einem passiven SPY-Investment.** [V] (OOS-Ergebnisse §3) + [H]

Das ist die ehrliche Antwort, und sie ist — wie vom Auftrag explizit zugelassen — ein
vollwertiges Ergebnis. Die Begründung ist nicht "schlechte Implementierung", sondern eine
Kombination aus drei strukturellen Gründen, die sich gegenseitig verstärken:

1. **Datensatz-Defekt (Survivorship-Bias):** Der Datensatz enthält nur heute noch handelbare
   Symbole. Delistings/Pleiten/Übernahmen fehlen. Das verzerrt **zugunsten** der Strategien —
   und trotzdem schlägt keine den Index. [V]
2. **Methodisches Multiple-Testing-Problem:** 9 Strategien (plus Parametervarianten) wurden
   getestet, ohne Deflated-Sharpe-Korrektur. Naive p-Werte/Sharpes sind dadurch wertlos. [V] + [Z]
3. **Ökonomische Realität:** Akademische Faktorprämien (Momentum, MAX-Anomalie, Mean-Reversion)
   replizieren auf einem kleinen Large-Cap-Long-Only-Universum schlecht und zerfallen nach
   Publikation/Kosten. Das ist in der Literatur breit dokumentiert. [Z]

**Gibt es einen Lösungsweg?** Ein *bedingter*, kein garantierter (§7). Der ehrlichste Satz ist:
Mit dem jetzigen Datensatz und Universum **nein**. Die einzigen nicht-falsifizierten Ideen
(Event/News-Intraday, Crypto-Carry) wurden nie sauber OOS-validiert und brauchen andere Daten
und Infrastruktur. Ein "echter" Versuch würde zuerst **bessere Daten** (survivorship-clean) und
**strengere Statistik** (DSR/CPCV) erfordern — beides liegt aktuell nicht vor. [V] + [H]

---

## 2. Was untersucht wurde (Vorgehen)

Agentische Durchleuchtung (vier parallele read-only Explorationsagenten) plus externe
Literaturrecherche (Web Search). Konkret abgedeckt:

- **Strategie-Inventar:** alle Strategien in `src/assembled_core/strategies/`,
  `src/assembled_core/signals/`, `src/assembled_core/events/` plus Wiring in
  `ops/paper_runner.py` und `configs/policy.yaml`. [V]
- **OOS-/Walk-Forward-Maschinerie:** `src/assembled_core/qa/walk_forward.py`,
  `qa/backtest_engine.py`, die `scripts/_oos_wf_*.py`-Runner, PIT-Guards, Kostenmodell,
  Survivorship-Behandlung. [V]
- **CAGR-/Sharpe-/MaxDD-Berechnung:** `qa/metrics.py` (kanonisch) und die Ad-hoc-Formeln in
  den OOS-Skripten. [V]
- **Dokumentierte Ergebnisse:** `docs/results/2026_05_*.md`, `docs/GO_LIVE_CHECKLIST.md`,
  `docs/audit/03/04/07e`, `docs/PROJEKT_ABSCHLUSS_2026_05.md`. [V]
- **Externe Literatur:** Backtest-Overfitting (PBO), Deflated Sharpe Ratio, Multiple Testing,
  Faktor-Decay post-Publikation, CAGR-Fallstricke, Momentum-Decay/Crowding. [Z]

---

## 3. Teil A — Verifizierte OOS-Ergebnisse (aus dem Code/Daten)

### 3.1 Die Ergebnistabelle (wörtlich transkribiert)

Quelle: `docs/PROJEKT_ABSCHLUSS_2026_05.md:59-69`, gegengeprüft gegen die einzelnen
`docs/results/2026_05_*_real_oos.md`. **Alle Zahlen verbatim.** [V]

| # | Strategie | Ø CAGR | Ø Sharpe | Ø MaxDD | SPY-Ref | Verdikt | Quelle |
|---|-----------|--------|----------|---------|---------|---------|--------|
| 1 | trend_baseline | **−6.1%** | −0.18 | −22.2% | +13.0% (Sh 0.95) | KEIN EDGE | `2026_05_trend_baseline_real_oos.md` |
| 2 | multifactor_v2 (TA-only) | +12.9% | 0.36 | −23.0% | +13.0% (Sh 0.95) | KEIN EDGE | `2026_05_multifactor_v2_real_oos.md` |
| 3 | multifactor_long_short (long-only) | **−19.5%** | −0.80 | ~−22% | +13.0% (Sh 0.95) | KEIN EDGE | `2026_05_multifactor_long_short_real_oos.md` |
| 4 | mfv2 + Altdata (full-stack) | +10.7% | 0.36 | −18.6% | +13.0% (Sh 0.95) | KEIN EDGE | `2026_05_mfv2_full_stack_real_oos.md` |
| 5 | vol_target_overlay | +8.8% | 0.88 | −8.4% | +14.5% (Sh 1.22) | OVERLAY BEHALTEN | `2026_05_vol_target_overlay_real_oos.md` |
| 6 | dual_momentum | +9.7% | 0.98 | −11.3% | +14.5% (Sh 1.26) | SCHWACH | `2026_05_dual_momentum_real_oos.md` |
| 7 | etf_pairs_meanrev (Full L/S, Ersatzpaare) | −0.3% | −0.49 | −2.0% | +19.7% (Sh 1.40) | KEIN EDGE | `2026_05_etf_pairs_meanrev_real_oos.md` |
| 8 | low_max_lottery (Bottom-Quintil) | +9.8% | 1.06 | −10.1% | SPY Sh 1.40 | KEIN EDGE | `2026_05_low_max_lottery_real_oos.md` |
| 9 | crypto_funding_carry | +4.5–6.7% APR | 4.40–5.64* | −$1.389–1.611 | n/a | MARGINAL / EXCHANGE-RISIKO | `2026_05_crypto_funding_carry_backtest.md` |

\* Sharpe für Crypto-Carry strukturell überhöht (eigener Caveat im Ergebnisdokument). [V]

**Hinweis zu den SPY-Referenzen:** Die SPY-Zahl unterscheidet sich zwischen den Dokumenten
(+13.0% / +14.5% / +19.7%), weil die Strategien **unterschiedliche Datums-Fenster und
Fold-Zahlen** haben. Die Vergleiche sind nur *innerhalb* je eines Dokuments fair. [V]

### 3.2 Die wichtigsten Einzelbefunde (nuanciert, nicht nur "alles rot")

- **trend_baseline** (aktive Paper-Pilot-Strategie): 0 von 10 Folds schlagen SPY,
  Ø CAGR −6.1% vs +13.0%, Sharpe negativ. Direkt verifiziert in
  `docs/results/2026_05_trend_baseline_real_oos.md:55-59`. [V] Das ist das GO_LIVE-Kriterium
  B1 — formal "erfüllt", inhaltlich **negativ** (`docs/GO_LIVE_CHECKLIST.md`, Status
  "[ERFÜLLT — Ergebnis negativ]"). [V]
- **multifactor_v2 (TA-only)** ist das einzige Aktienmodell, das den SPY-CAGR *nominal* fast
  erreicht (+12.9% vs +13.0%) und in 6/10 Folds schlägt — **aber** der Sharpe ist 0.36 vs 0.95,
  also risk-adjustiert ~2,6× schlechter (`2026_05_strategy_comparison.md:18`). [V] Und: dieser
  Lauf misst nur ~9 von 34 Faktoren, weil der Rest strukturell Null ist (§4.3). Es ist **kein
  Test der 34-Faktor-These**, sondern ein TA-Subset. [V]
- **mfv2 + Altdata** liefert **Sharpe-Delta = +0.00** gegenüber TA-only
  (`2026_05_mfv2_full_stack_real_oos.md:72`). Die teure Altdata-Integration bringt risk-adjustiert
  exakt nichts. [V]
- **multifactor_long_short** ist im **Long-Only-Modus** getestet — das bricht das
  Long-Short-Faktormodell fundamental; der negative Wert (−19.5%) ist daher kein fairer Test
  der Strategie, sondern ein Test einer kaputten Konfiguration. [V]
- **vol_target_overlay** ist **kein Alpha**, sondern ein Drawdown-Schutz: MaxDD −8.4% vs SPY
  −14.5%, im COVID-Crash 2020 −8.8% vs SPY −28.9%. Wird genau dafür behalten. [V]
- **dual_momentum** ist die "schwächste positive Evidenz" (+9.7%, Sharpe 0.98), aber nur 30,8%
  der Folds schlagen SPY, und SPYs eigener Sharpe (1.26) liegt darüber. Kein robuster Edge. [V]
- **low_max_lottery:** Der MAX-Effekt (Bali et al. 2011) ist auf diesem Large-Cap-Universum
  **abwesend oder umgekehrt** — High-MAX (+47.6% CAGR) schlägt Low-MAX (+9.8%) deutlich. [V]
- **etf_pairs_meanrev:** Mit **Ersatzpaaren** getestet (Originalpaare fehlten im Cache),
  laut Dokument "informational only — nicht mit der Originalspezifikation vergleichbar". [V]
- **news_alpha** (event-driven, "die eigentliche Crisis-Alpha-Idee") und **crisis_alpha** (live):
  **Kein dokumentiertes reales OOS-Ergebnis.** crisis_alpha hat nur eine *synthetische*
  COVID-Validierung (`2026_04_crisis_alpha_covid2020_validation.md`, ausdrücklich "All signal
  data is synthetic"). [V]

---

## 4. Teil A — Wie OOS und CAGR im Code wirklich funktionieren

### 4.1 CAGR — verifizierte Formeln und ein subtiler Defekt

**Kanonisch** (`src/assembled_core/qa/metrics.py:232-262`, direkt gelesen): [V]

```python
periods_per_year = _get_periods_per_year(freq)   # 252 für "1d"
if periods < periods_per_year:                   # Guard: < 1 Jahr → None
    return None
years = periods / periods_per_year
total_return = end_value / start_value
cagr = (total_return ** (1.0 / years)) - 1.0
```

- **Basis: 252 Handelstage** (nicht 365 Kalendertage). Kein 365-Tage-CAGR existiert irgendwo. [V]
- **Form: geometrisch**, Endwert/Startwert hoch (1/Jahre). Korrekt. [V]
- **Schutz: gibt `None` zurück bei < 1 Jahr** — verhindert das Annualisieren kurzer Fenster. [V]

**Die OOS-Skripte benutzen aber NICHT diese kanonische Funktion.** Jede gemeldete OOS-CAGR-Zahl
stammt aus einer **Inline-Formel** in `scripts/_oos_wf_*.py`, z.B.
`scripts/_oos_wf_trend_baseline.py:207-212`: [V]

```python
n_years = len(eq_test) / 252
cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
```

**Divergenz [V]:** Der kanonische 1-Jahres-Guard **fehlt** in der Reporting-Variante; stattdessen
`max(n_years, 0.01)`. In den Produktionskonfigurationen ist das Testfenster `TEST_WINDOW_DAYS = 252`
(≈1 Handelsjahr), daher ist die Divergenz im Normalfall **inert**. Aber: ein verkürztes/letztes
Fold würde trotzdem annualisiert; `max(n_years, 0.01)` könnte ein 2-Bar-Fold in die 100. Potenz
heben. Das ist ein latentes Risiko, kein aktiver Fehler in den vorliegenden Zahlen. [V] + [H]

**`BacktestResult.metrics` hat kein `cagr`/`maxdd`-Key** (`qa/backtest_engine.py:104-139`,
`pipeline/portfolio.py:271` → dict ist nur `{final_pf, sharpe, trades}`). [V] Konsequenz: das
Crisis-Compare-Skript `scripts/_crisis_alpha_backtest_compare.py:109` macht `metrics.get("cagr")`
→ **immer `None`** → CAGR wird dort still weggelassen (toter Key, kein falscher Wert). [V]

### 4.2 OOS-/Walk-Forward-Maschinerie — rigoros genug?

**Verifizierte Stärken** [V]:

- **10 saubere, nicht-überlappende Jahres-Folds** (252/252/252 train/test/step, anchored
  2018-01-01), durch Regressionstest abgesichert (`tests/test_walk_forward_no_leakage.py`).
- **Purging und Embargo sind implementiert** (López-de-Prado-Stil) in
  `qa/walk_forward.py:126-136`, inkl. Guard, der bei `purge_days < max_label_horizon` wirft.
- **PIT-Guard mit echtem Test:** `tests/test_trend_baseline_pit_safety.py` multipliziert alle
  Zukunfts-Bars ×5 bzw. ~0 und prüft Byte-Identität des as_of-Signals (`< 1e-10`). Ein
  `center=True`/Future-Leak-Bug würde diesen Test brechen. Nicht-trivial, lastentragend.
- **Signal-Pfad korrekt getrennt:** `compute_signals` ist "latest-bar-only" (für Live), der
  Backtest ruft korrekt `generate_trend_signals_from_prices` (volle Zeitreihe). Bestätigt:
  `target_qty=0.0` ist Absicht; die OOS-Skripte multiplizieren `target_weight × capital`. [V]
- **Kosten werden real abgezogen:** `commission_bps=10`, Spread, Slippage in
  `pipeline/portfolio.py:164-190`, verifiziert subtrahiert. [V]
- **SPY-Benchmark ist konservativ:** kostenlos + ohne Dividenden-Reinvest
  (`2026_05_trend_baseline_real_oos.md:72`) → der Vergleich ist **gegen** die Strategie
  verzerrt. Die Strategie verliert also trotz Handicap *für* sie. [V]

**Verifizierte Schwächen / Bias-Richtung** [V] (mit Richtung):

| Problem | Datei:Zeile | Richtung |
|---------|-------------|----------|
| Statische `watchlist.txt`, **keine** PIT-Universe-Anbindung (Survivorship) | `_oos_wf_*.py` (kein Aufruf von `get_universe_members_pit`) | **optimistisch** |
| Signal und Fill auf demselben Bar-Close | `backtest_engine.py:415`, `portfolio.py:176` | optimistisch (mild) |
| mfv_long_short: Winsorize-Clip über **ganzes** Fenster gepoolt | `multifactor_signal.py:60-86` | optimistisch (klein, nur diese Strategie) |
| `purge_days=0` im Headline-Lauf | `test_walk_forward_no_leakage.py:181` | neutral (zustandslose Regel) |
| SPY-Benchmark kostenlos + ohne Dividenden | `_oos_wf_trend_baseline.py:240-261` | **pessimistisch** (konservativ) |
| `MaxDD%`/Calmar über **globalen** Peak statt Peak-to-date | `qa/metrics.py:215-217` (Audit F-1) | **optimistisch** (MaxDD% sieht besser aus) |

**Kernschluss zur Methodik** [V] + [H]: Die wichtigste Eigenschaft ist, dass **alle materiellen
Verzerrungen die Strategie schmeicheln** (Survivorship, Same-Bar-Fills, optimistisches MaxDD%),
während der Benchmark ein konservatives Handicap trägt. Daraus folgt:

- **Ein NEGATIVES Ergebnis ist belastbar — sogar konservativ.** Eine Strategie, die *hier*
  SPY nicht schlägt, würde unter sauberen PIT-Daten und realistischen Fills **noch deutlicher**
  verlieren. Das trend_baseline-Verdikt (0/10 Folds) ist robust. [V] + [H]
- **Ein POSITIVES Ergebnis wäre zu bezweifeln**, solange (a) das PIT-Universe nicht
  angebunden ist und (b) der MaxDD%-Global-Peak-Bug und (c) der mfv_long_short-Winsorize-Leak
  nicht gefixt sind. [V] + [H]

Das deckt sich mit dem internen Audit `docs/audit/03_lookahead_correctness.md` (2026-05-30):
"Die negativen OOS-Ergebnisse STEHEN." [V] Der mfv_long_short-Defekt macht die negative
Schlussfolgerung **robuster, nicht schwächer** (er ist optimistisch — die wahre Performance
ist also noch schlechter). [V]

### 4.3 Daten-Defizite, die jeden "Edge" untergraben

- **Survivorship-Bias** ist explizit anerkannt (`PROJEKT_ABSCHLUSS_2026_05.md:154-163`): nur
  überlebende Symbole, Alpaca Free Tier liefert keine delisted-Daten. Verzerrt High-MAX und
  Momentum **zugunsten** der Strategien. [V]
- **Tote/genullte Faktoren in mfv2:** `insider_activity_score = 0.00` (59.506 Zeilen, 100%
  "unknown", `multifactor_v2.py:266`), `congress_activity = 0.00` (keine Datendateien,
  `:282`). Im OOS-Lauf sind **19 von 34 Faktoren auf 0.0 degradiert**
  (`2026_05_mfv2_altdata_diagnostik.md`). [V] Solche Null-Faktoren erzeugen keinen Null-Beitrag,
  sondern **"lautlosen Drag" durch factor dilution** (`PROJEKT_ABSCHLUSS_2026_05.md:185-186`). [V]
- **News-Sentiment-Daten** decken nur 2025-12-22 bis 2026-05-21 ab — die **gesamte OOS-Periode
  2018–2025 ist für diese Faktoren unbewertet** (`2026_05_mfv2_altdata_diagnostik.md:167-169`). [V]
- **Cross-sectional z-Score kollabiert** bei kleinem Universum (Std ≈ 0) → earnings_surprise_z,
  sector_rotation_bias strukturell Null (`PROJEKT_ABSCHLUSS_2026_05.md:182-184`). [V]
- **Kein DSR/CPCV angewandt:** "Kein der hier berechneten Ergebnisse wurde DSR-korrigiert. Das
  ist ein bekanntes Defizit dieser Forschungsphase" (`PROJEKT_ABSCHLUSS_2026_05.md:172-173`).
  CPCV ist implementiert, aber **nie ausgeführt** (GO_LIVE B2 OFFEN). [V]

---

## 5. Teil B — Externe Literatur (zitiert)

Diese Quellen liefern den theoretischen Rahmen dafür, *warum* solche Strategien typischerweise
scheitern und *wie* man echten von eingebildetem Edge unterscheidet. Sie bestätigen unabhängig
die Defizite, die das Projekt selbst benannt hat.

### 5.1 Backtest-Overfitting & Deflated Sharpe Ratio — Bailey & López de Prado [Z]

- **Probability of Backtest Overfitting (PBO):** Bailey, Borwein, López de Prado, Zhu führen eine
  Cross-Validation-Technik ein, die misst, ob die *Strategieauswahl* zu Overfitting neigt — im
  Sinne, dass ausgewählte Strategien out-of-sample **unter den Median der Versuche** fallen. [Z]
- **Deflated Sharpe Ratio (DSR):** korrigiert den Sharpe für (a) **Selektionsbias unter Multiple
  Testing** (wer viele Varianten probiert und die beste behält, bläht den Sharpe auf — selbst
  wenn alle Kandidaten reines Rauschen sind) und (b) **nicht-normale Renditen** (Skew, Fat Tails,
  Vol-Clustering). [Z]
- **Direkte Relevanz hier [V→Z-Brücke]:** Das Projekt hat **9 Strategien + Parametervarianten**
  getestet und **keine DSR-Korrektur** angewandt. Genau das ist der Fall, vor dem diese
  Literatur warnt: der naive Sharpe einzelner Konfigurationen ist nach so vielen Versuchen
  statistisch wertlos. Selbst der "beste" Wert (mfv2 TA-only, Sharpe 0.36) liegt **weit unter**
  jeder DSR-Signifikanzschwelle und unter SPY (0.95).
- **Minimum Backtest Length:** López de Prado zeigt, dass die nötige Stichprobenlänge mit der
  Anzahl der getesteten Konfigurationen wächst; ein kurzer Backtest mit vielen Trials erzeugt
  fast garantiert einen scheinbar guten, aber überangepassten "Edge". [Z]

### 5.2 Multiple Testing — Harvey, Liu & Zhu (2016) [Z]

- "…and the Cross-Section of Expected Returns" (Review of Financial Studies 29(1)): Ein
  Großteil der hunderten publizierten "Faktoren" sind wahrscheinlich **False Discoveries** durch
  Data-Mining. [Z]
- **Neue Hürde:** Ein neuer Faktor braucht einen **t-Wert > 3.0** (nicht 2.0), gerade weil so
  viele Faktoren getestet wurden. [Z]
- **Relevanz hier [H]:** Die mfv2-These mit 34 Faktoren ist faktisch ein Faktor-Zoo. Auf einem
  75–194-Titel-Universum mit ~9 lebenden Faktoren ist die Wahrscheinlichkeit hoch, dass jeder
  scheinbare Beitrag innerhalb des Rauschens liegt. Ohne t>3.0-Disziplin ist "es funktioniert
  in einigen Folds" kein Signal.

### 5.3 Faktor-Decay nach Publikation — McLean & Pontiff (2016) [Z]

- "Does Academic Research Destroy Stock Return Predictability?" (Journal of Finance): Untersuchung
  von 97 Predictor-Variablen. Renditen sind **out-of-sample 26% niedriger** und
  **post-publication 58% niedriger** als in-sample. [Z]
- Der Post-Publication-Rückgang ist **größer für Predictors mit höherer in-sample-Rendite**, und
  Renditen konzentrieren sich auf Aktien mit **hohem idiosynkratischem Risiko und niedriger
  Liquidität** — also gerade **nicht** Large-Caps. [Z]
- **Relevanz hier [V→Z-Brücke]:** Die getesteten akademischen Effekte (MAX-Anomalie, Momentum)
  stammen aus Studien auf breiten Small-Cap-inklusiven Universen. Auf einem **Large/Mid-Cap-
  Long-Only-Universum** sind genau die Renditequellen (Small-Cap, illiquide, hohe Idio-Vol)
  abgeschnitten. Das Projekt beobachtet exakt das: "Effekt abwesend" (low_max),
  "negativ" (Momentum) — `PROJEKT_ABSCHLUSS_2026_05.md:190-199`. Die Literatur erklärt, *warum*
  das zu erwarten war.

### 5.4 CAGR-Fallstricke [Z]

- **Annualisierung kurzer Fenster amplifiziert Rauschen:** +10% in einem Monat → ~213% CAGR,
  völlig unrealistisch. Schon Fenster < 3–6 Monate sind irreführend. [Z]
- **Volatility Drag:** Compounding bestraft Verluste stärker als Gewinne. −20% dann +25% =
  arithmetisch +2,5%, geometrisch **0,0%**. Höhere Vol bei gleichem Mittel = weniger Endkapital. [Z]
- **Pfadunabhängigkeit:** CAGR ignoriert zwischenzeitliche Drawdowns, Varianz und Tail-Risk.
  15% CAGR bei 60% Vol ist nicht 12% CAGR bei 15% Vol. [Z]
- **Relevanz hier [V]:** Das Projekt verwendet CAGR korrekt geometrisch (252-Basis), aber die
  Lehre bleibt: **CAGR allein ist kein Edge-Maß.** Genau deshalb ist der Sharpe-Vergleich
  (0.36 vs 0.95) aussagekräftiger als der CAGR-Vergleich (12.9% vs 13.0%), und genau deshalb
  ist mfv2s "nominaler CAGR-Gleichstand" eine Falle, kein Erfolg. Die 1-Jahres-Folds des
  Projekts vermeiden zwar die schlimmste Kurzfrist-Annualisierung — aber 10 Ein-Jahres-Folds
  sind statistisch dünn ("keine Signifikanzaussage möglich", `2026_05_strategy_comparison.md:97`).

### 5.5 Momentum-Decay, Crowding, Universumsgröße [Z]

- Momentum hat **signifikant decayed**; ohne Filter ist die Profitabilität vieler Events
  **unter den Transaktionskosten**. [Z]
- **Crowding** führt zu Alpha-Decay und erhöht Tail-Risk; bei gecrowdetem Momentum reverten die
  Renditen stark. [Z]
- **Universumsgröße:** Die Performance ist **nicht homogen** über die Universumsgröße —
  **große Firmen behindern** die Momentum-Performance. [Z]
- **Relevanz hier [V→Z-Brücke]:** Das ist die direkte ökonomische Erklärung für trend_baseline
  (−6.1%) und multifactor (long-only) auf 75–194 Large-Caps: ein gecrowdeter, large-cap-lastiger,
  long-only, nach-Kosten-Bereich ist genau der, in dem Momentum laut Literatur am schwächsten ist.

---

## 6. Teil C — Synthese: Warum die Strategien scheitern

Verknüpfung von [V] (Repo-Evidenz) und [Z] (Literatur). Die Schlussfolgerungen sind [H], wo sie
über die direkte Evidenz hinausgehen.

1. **Die Edge-Quellen passen nicht zum Universum.** [V]+[Z]+[H] Die getesteten Faktoren leben
   laut Literatur in Small-Caps, illiquiden, hoch-idiosynkratischen Aktien (McLean/Pontiff) und
   leiden bei großen Firmen (Momentum-Universumseffekt). Das Projekt-Universum ist das Gegenteil:
   Large/Mid-Cap, liquide, long-only. Ergebnis: Effekte "abwesend oder umgekehrt" — exakt wie
   beobachtet.
2. **Multiple Testing ohne Korrektur entwertet jeden scheinbaren Treffer.** [V]+[Z] 9 Strategien
   + Varianten, kein DSR. Selbst die besten Sharpes (0.36–1.06) sind ohne Deflationierung nicht
   interpretierbar — und liegen ohnehin unter SPY.
3. **Daten-Defekte verzerren zugunsten der Strategien — und sie verlieren trotzdem.** [V]+[H]
   Survivorship-Bias, Same-Bar-Fills und der optimistische MaxDD%-Bug machen die Ergebnisse
   *besser* als die Realität. Dass selbst dann kein Edge erscheint, ist das stärkste Argument,
   dass keiner existiert.
4. **Tote Faktoren erzeugen Drag, kein Alpha.** [V] mfv2s 34-Faktor-These wurde nie wirklich
   getestet (nur ~9 lebende Faktoren); die genullten Faktoren verdünnen das Signal.
5. **CAGR-Nominalgleichstand ≠ Edge.** [V]+[Z] mfv2s +12.9% vs +13.0% verführt — aber der Sharpe
   (0.36 vs 0.95) zeigt: das ist mehr Risiko für dieselbe Rendite, kein Vorteil.

---

## 7. Teil D — Gibt es einen Lösungsweg?

Ehrliche, abgestufte Antwort. Ich trenne, was **verifiziert nötig** wäre, was **plausibel**
ist, und was **reine Spekulation** bleibt.

### 7.1 Was definitiv nötig wäre, bevor man überhaupt von "Edge" reden darf [V]+[Z]

Diese drei sind keine Optionen, sondern Voraussetzungen — und keine ist aktuell erfüllt:

1. **Survivorship-bereinigte Daten** (CRSP, Sharadar o.ä.). Das Projekt benennt das selbst als
   offenen Punkt (`PROJEKT_ABSCHLUSS_2026_05.md:212-213`). Ohne das ist **jedes positive**
   Ergebnis unglaubwürdig. [V]
2. **DSR + CPCV statt naivem Sharpe.** Die Methodik (DSR-Korrektur, CPCV-Lauf) ist teils
   implementiert, aber nicht angewandt (B2 OFFEN). Ohne Multiple-Testing-Korrektur ist kein
   gemeldeter Sharpe vertrauenswürdig. [V]+[Z]
3. **Methodische Fixes**, die den Code bereits als Schuld ausweist: PIT-Universe an die
   OOS-Skripte anbinden, MaxDD%-Global-Peak-Bug fixen, mfv_long_short-Winsorize-Leak +
   fehlenden PIT-Test ergänzen. (Nur als Befund genannt — **kein Auftrag, keine Umsetzung
   in dieser Recherche.**) [V]

### 7.2 Wo *vielleicht* noch etwas zu holen ist (geparkte, nicht falsifizierte Ideen)

Diese wurden **nie sauber OOS-validiert** — sie sind also weder bewiesen noch widerlegt: [V]

- **News-/Event-Alpha (intraday, directional).** `src/assembled_core/events/news_alpha/`
  existiert, hat aber **kein dokumentiertes OOS-Ergebnis**. Konzeptionell anders als alles
  Getestete (Hormuz → Öl-Long binnen Stunden). Bräuchte **Intraday-Daten** und einen schnellen
  Execution-Pfad (`PROJEKT_ABSCHLUSS_2026_05.md:205-207`). [V] — **[H]:** Das ist die
  interessanteste offene Spur, weil sie nicht im selben gecrowdeten, decay-anfälligen
  Cross-Section-Momentum-Raum spielt. Aber ohne Intraday-Daten und einen kostenrealistischen
  Event-Backtest ist jede Erwartung Spekulation.
- **Crypto-Funding-Carry.** Realer Edge (+4.5–6.7% APR nach Fees), aber Sharpe strukturell
  überhöht und **Exchange-Gegenparteirisiko nicht modelliert** (FTX-Szenario). Nur als kleiner
  Portfolio-Baustein mit crypto-nativer Infrastruktur sinnvoll. [V]
- **PEAD (Post-Earnings-Announcement-Drift).** Als Research-Modul implementiert, im Code selbst
  als "research use only — not suitable for live production" markiert (`pead_strategy.py:11-14`);
  PIT-Safety nicht vollständig verifiziert. [V]

### 7.3 Die ehrliche Gesamtaussage zum Lösungsweg [H]

- **Auf dem jetzigen Datensatz/Universum: Nein.** Klassische Cross-Section-Faktoren (Momentum,
  MAX, Mean-Reversion, Multifaktor) sind hier mit hoher Wahrscheinlichkeit **strukturell
  chancenlos**, weil Universum, Long-Only-Constraint und Daten-Defekte genau die Renditequellen
  abschneiden, die die Literatur als Quelle des Effekts identifiziert. Mehr Faktoren, mehr
  Tuning oder ein neuer Multifaktor-Mix würde nur das Multiple-Testing-Problem verschärfen.
- **Mit anderem Daten-/Problemraum: Vielleicht, aber unbewiesen.** Der einzige glaubwürdige Pfad
  zu echtem Edge führt über (a) **bessere Daten** (survivorship-clean, ggf. breiter/illiquider
  oder Long-Short statt Long-Only) **und** (b) einen **anderen Effekt** (Event/Intraday, Carry),
  der nicht im gleichen ausarbitrierten Raum liegt — **und** (c) **strenge Statistik** (DSR/CPCV).
  Ob *dann* ein Edge erscheint, ist offen und kann ohne die Daten nicht beantwortet werden.
- **Was definitiv keinen Edge bringt:** Schönrechnen. Kürzere Fenster annualisieren, MaxDD% über
  globalen Peak, Sharpe ohne DSR, Cherry-Picking guter Folds — das produziert nur scheinbaren
  Edge, der out-of-sample verschwindet (genau das warnt §5.1–5.3). Der Auftrag, nicht
  schönzurechnen, ist hier methodisch goldrichtig.

---

## 8. Teil E — Explizit als Spekulation markiert [H]

Diese Punkte gehen über die Evidenz hinaus und sind **meine Hypothesen**, nicht belegt:

- **[H]** Selbst mit survivorship-clean Daten würde ich für die klassischen Faktoren auf einem
  Large-Cap-Long-Only-Universum **netto nach Kosten keinen robusten Edge** erwarten — McLean/Pontiff
  und der Momentum-Universumseffekt deuten klar in diese Richtung. Der ehrlichste Erwartungswert
  ist "ehrlicher, aber nicht besser" (so formuliert es auch das Abschlussdokument selbst).
- **[H]** Die plausibelste Quelle eines *echten* Edges in diesem Projekt ist **nicht** Querschnitts-
  Aktien-Faktor-Investing, sondern **Event-getriebenes, schnelles, directional Trading** auf
  einem Daten-/Latenz-Vorteil — also news_alpha-artig. Das ist aber genau der Teil, der am
  wenigsten getestet ist und die höchsten Infrastruktur-Anforderungen hat (Intraday, Execution).
- **[H]** Falls die Alpha-Suche je wieder aufgenommen wird, wäre der wissenschaftlich ehrlichste
  erste Schritt **nicht** eine neue Strategie, sondern: einen survivorship-clean Datensatz
  beschaffen, **eine** klar vorab-spezifizierte Hypothese aufstellen, und sie **ein einziges Mal**
  mit DSR/CPCV testen — um das Multiple-Testing-Problem nicht weiter zu vergrößern.

---

## 9. Quellenverzeichnis (extern, [Z])

Backtest-Overfitting & Deflated Sharpe:
- Bailey, Borwein, López de Prado, Zhu — *The Probability of Backtest Overfitting*:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253 ·
  PDF: https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf
- Bailey & López de Prado — *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
  Overfitting and Non-Normality*:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 ·
  PDF: https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
- Minimum Backtest Length / Deflated SR (Aufbereitung, Stefan Jansen, ML4T):
  https://stefan-jansen.github.io/machine-learning-for-trading/08_ml4t_workflow/01_multiple_testing/

Multiple Testing / Faktor-Zoo:
- Harvey, Liu, Zhu — *…and the Cross-Section of Expected Returns* (RFS 2016):
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2249314 ·
  NBER: https://www.nber.org/papers/w20592 ·
  PDF: https://people.duke.edu/~charvey/Research/Published_Papers/P118_and_the_cross.PDF

Faktor-Decay nach Publikation:
- McLean & Pontiff — *Does Academic Research Destroy Stock Return Predictability?* (J. Finance 2016):
  https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365 ·
  SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2156623 ·
  PDF: https://www.hec.ca/finance/Fichier/McLean.pdf

CAGR-Fallstricke:
- *The CAGR Myth That Could Cost You Thousands* (Quantlake):
  https://www.quantlake.com/post/the-cagr-myth-that-could-cost-you-thousands
- Annualized Return (CAGR) — Definition & Beispiele (Ryan O'Connell, CFA):
  https://ryanoconnellfinance.com/annualized-return/

Momentum-Decay / Crowding / Universumsgröße:
- *Breaking Down Momentum Strategies* (Quant Arb):
  https://www.algos.org/p/breaking-down-momentum-strategies
- *Momentum universe shrinkage effect in price momentum* (arXiv 1211.6517):
  https://arxiv.org/pdf/1211.6517
- *Decomposing momentum: The forgotten component* (ScienceDirect):
  https://www.sciencedirect.com/science/article/abs/pii/S0378426624002061

---

## 10. Interne Quellen (Repo, [V])

- `docs/PROJEKT_ABSCHLUSS_2026_05.md` — formelles Abschlussdokument, OOS-Tabelle :59-69, Lessons :154-213
- `docs/results/2026_05_trend_baseline_real_oos.md` — B1-Headline, Metriken :53-59, Limitationen :67-72
- `docs/results/2026_05_multifactor_v2_real_oos.md`, `…_mfv2_full_stack_real_oos.md`,
  `…_multifactor_long_short_real_oos.md`, `…_dual_momentum_real_oos.md`,
  `…_vol_target_overlay_real_oos.md`, `…_etf_pairs_meanrev_real_oos.md`,
  `…_low_max_lottery_real_oos.md`, `…_crypto_funding_carry_backtest.md`
- `docs/results/2026_05_strategy_comparison.md` — Headline :78, Sharpe-Caveat :18, Signifikanz :97
- `docs/results/2026_05_mfv2_altdata_diagnostik.md` — tote Faktoren, News-Datenlücke :167-169
- `docs/GO_LIVE_CHECKLIST.md` — B1 "[ERFÜLLT — Ergebnis negativ]"
- `docs/audit/03_lookahead_correctness.md` — "negative OOS-Ergebnisse STEHEN", mfv_long_short-Leak
- `docs/audit/04_numeric_verification.md` — F-1 MaxDD%-Global-Peak-Bias
- `docs/audit/07e_strategy_feature_ml.md` — STR-001..009, PIT-Befunde
- `src/assembled_core/qa/metrics.py:232-262` (CAGR), `:125-151` (Sharpe), `:198-229` (MaxDD)
- `src/assembled_core/qa/walk_forward.py:126-136,210-381` (Folds/Purge/Embargo)
- `src/assembled_core/qa/backtest_engine.py:104-139` (BacktestResult), `pipeline/portfolio.py:164-190` (Kosten)
- `scripts/_oos_wf_trend_baseline.py:207-212` (Ad-hoc-CAGR), `:240-261` (SPY-Benchmark)
- `src/assembled_core/strategies/multifactor_v2.py:266,282` (genullte Faktoren)
- `tests/test_trend_baseline_pit_safety.py`, `tests/test_walk_forward_no_leakage.py` (PIT-Guards)

---

_Status: Reine Recherche, abgeschlossen 2026-05-31. Kein Code geändert, keine Backtests neu
ausgeführt, keine Zahlen erfunden. Interne Zahlen transkribiert (CI-unbestätigt, nicht
unabhängig repliziert); externe Aussagen zitiert (§9); Hypothesen als [H] markiert._

---

## Korrektur-Nachtrag (2026-07-23) — Vorzeichen der SPY-Dividenden-Zeile (Z. ~210)

Die Tabellenzeile „SPY-Benchmark kostenlos + ohne Dividenden → **pessimistisch (konservativ)**"
ist im Dividenden-Teil **falsch vorzeichenbehaftet**. Richtig ist:

- **SPY ohne Dividenden senkte die Latte** für die Strategien: Der Benchmark wurde um die
  Dividendenrendite (~1,5–2 % p.a.) untertrieben, war also LEICHTER zu schlagen — für die
  Strategie ist das **optimistisch**, nicht pessimistisch. (Nur der Teil „SPY kostenlos"
  wirkt pessimistisch/konservativ.)
- **Konsequenz für die Verdicts:** Die negativen Verdicts werden dadurch **STÄRKER**, nicht
  schwächer — eine Strategie, die schon den dividenden-losen (zu niedrigen) SPY nicht schlug,
  verliert gegen den echten Total-Return-SPY noch deutlicher. Der Kernschluss („negatives
  Ergebnis ist belastbar") bleibt damit uneingeschränkt gültig und wird verstärkt.
- Kontext (Befund 2026-06-01): `output/aggregates/daily.parquet` `close` ist total-return-
  adjustiert — die Strategieseite hatte Dividenden implizit drin, der SPY-Vergleich in
  `_oos_wf_trend_baseline.py:240-261` nicht.

Der ursprüngliche Text oben wird absichtlich **nicht still geändert** (Audit-Artefakt).
