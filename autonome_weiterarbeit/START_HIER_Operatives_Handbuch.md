# START_HIER — Operatives Handbuch

**Zweck:** Die drei Audit-Teile haben ~4500 Zeilen. Dieses Dokument ist die Kurzfassung für morgen früh am Laptop. Kein Hintergrund, keine Begründung, nur "tue genau das."

Wenn du dich überwältigt fühlst von den Audit-Dokumenten: fang hier an. Arbeite dich Woche für Woche durch. Die Begründung für jeden Schritt steht in Teil 1, 2, oder 3 — du kannst jederzeit nachschlagen.

---

## Woche 1 — Hygiene-Sprint (Null-Risiko, sofortiges Ergebnis)

**Ziel:** 150+ Files aus dem Repo raus, ohne dass irgendwas kaputt geht.

### Tag 1 — Scratch und Legacy-Files löschen

```bash
# Alle Scratch/Debug/Legacy-Artefakte am Root
git rm notes/scratch.ps1 notes/scratch.txt
git rm oos_debug_log.txt
git rm review_bundle.txt
git rm "uninstaller für automatische ausführung sprint_5.txt"
git rm 000_seed_project.ps1.disabled
git rm README_INTEGRATION.txt
git rm README_ONECLICK.md
git rm PROJECT_STATUS.txt
git rm run_all.ps1 run_all_sprint2.ps1 run_sprint2.ps1
git rm scripts/tools/fix_all_project.ps1.bak scripts/tools/fix_indent.ps1.bak

git commit -m "chore: remove scratch/legacy/backup artifacts from repo"
```

**Kontrolle:** `python scripts/cli.py info` läuft noch? Gut, weiter.

### Tag 2 — Wave-Wiring-Tests löschen

```bash
git rm tests/test_wave*_wiring.py
git commit -m "test: remove 147 wave-wiring smoke tests (observability-only)"
```

**Kontrolle:** `pytest -m fast -q` läuft ohne Fehler? Gut, weiter.

### Tag 3 — Redundante Audit/Status-Docs löschen

```bash
cd docs/
git rm CODE_QUALITY_AUDIT.md CODE_QUALITY_FINAL_REPORT.md \
       CODE_QUALITY_FIXES_APPLIED.md CODE_QUALITY_FIXES_SUMMARY.md \
       CODE_QUALITY_FULL_AUDIT.md CODE_QUALITY_SUMMARY.md \
       DEEP_AUDIT_REPORT.md FULL_PROJECT_AUDIT.md \
       FULL_SYSTEM_AUDIT_OUTPUT.md FINAL_CODE_REVIEW_FINDINGS.md \
       REVIEW_AUDIT_SPRINT13_EVIDENCE_PACK.md \
       FINAL_DOWNLOAD_SUMMARY.md FINAL_IMPROVEMENTS_APPLIED.md \
       FINAL_STATUS_REPORT.md FULL_SYSTEM_RUN_REPORT.md \
       B3_COMPLETION_NOTES.md MERGE_GATE_SPRINT13.md \
       RELEASE_NOTES_SPRINT13.md ROADMAP_STATUS_SPRINT13.md \
       SPRINT_7_ACCEPTANCE.md SPRINT_C1_COMPLETION_SUMMARY.md \
       SPRINT11_BENCHMARKS.md SPRINT11_E1_VECTORIZE_PLAN.md \
       SPRINT4_CORPORATE_ACTIONS_PLAN.md ROADMAP_NR3_STATUS.md \
       TEST_SUMMARY_FINAL.md DATA_DOWNLOAD_STATUS.md \
       DOWNLOAD_STATUS_REPORT.md
cd ..

git commit -m "docs: remove 27 redundant audit/status/sprint-completion docs"
```

**Nach Woche 1:** Repo ist ~185 Files kleiner, keine funktionale Änderung.

---

## Woche 2 — Namenskonflikte killen

**Ziel:** Die gefährlichsten Duplikate beseitigen, bevor sie Daten korrumpieren.

### Tag 1 — `config.py`-Duplikat

```bash
# Prüfe, welche Version häufiger importiert wird
grep -rln "from src.assembled_core.config import\|from src.assembled_core import config" src/ scripts/ tests/ | wc -l
grep -rln "from src.assembled_core.config.config import" src/ scripts/ tests/ | wc -l
```

Die Root-Version hat 35 Importer, die Package-Version 7. Entscheidung: **Root-Version behalten, Package-Version löschen.**

```bash
git rm src/assembled_core/config/config.py
# Update __init__.py um die Exports weiter bereitzustellen:
# Add to src/assembled_core/config/__init__.py:
#   from src.assembled_core.config import OUTPUT_DIR, SUPPORTED_FREQS, get_output_path, get_base_dir
```

Teste:
```bash
pytest -k "test_config" -v
python -c "from src.assembled_core.config import OUTPUT_DIR; print(OUTPUT_DIR)"
```

Commit: `fix: resolve config.py naming conflict (package vs root module)`

### Tag 2 — `logging_config.py`-Duplikat

Der Root-`logging_config.py` (180 Zeilen, 10 Importer) ist der etablierte. Der Package-`config/logging_config.py` (54 Zeilen, 6 Importer) ist ein kleinerer Neubau.

Prüfe Unterschied:
```bash
diff src/assembled_core/logging_config.py src/assembled_core/config/logging_config.py
```

Entscheide: welcher bleibt? Die kleinere Version hat vielleicht sinnvolle Neuerungen. Nach Prüfung einen löschen und Importer umbiegen.

### Tag 3 — `stat_arb.py`-Duplikat

```bash
# Der flache stat_arb.py ist observability-wired, der Ordner ist real
git rm src/assembled_core/strategies/stat_arb.py
# Wenn Tests den import brauchen, in __init__.py des Ordners re-exportieren
```

---

## Woche 3 — Observability-Schicht in Archive

**Ziel:** Die ~215 observability-wired Files aus `src/assembled_core/` in `archive/` verschieben.

### Vorbereitung

Erstelle den Target-Ordner:
```bash
mkdir -p archive/observability_graveyard_2026q2/
cd archive/observability_graveyard_2026q2/
echo "# Observability Graveyard 2026 Q2

Files moved out of src/assembled_core/ because they were only wired
for observability (import + config instantiation + meta-dict entry)
without actually influencing trading decisions.

See Audit Teil 2 for full rationale.

If you need to reactivate any of these, move them back and ensure
they are actually used in a signal/sizing/risk/execution decision.
" > README.md
cd ../..
```

### Move-Liste (aus Teil 2 extrahiert)

Arbeite Modul für Modul durch. Pro Modul ein Commit:

**ml/ (52 observability-Files — 16.000+ Zeilen)**
```bash
mkdir -p archive/observability_graveyard_2026q2/ml/
# Liste aus Teil 2, Sektion 2.2:
# maml, gnn_stocks, bayesian_nn, rl_portfolio, rl_execution, tda_regime,
# symbolic_regression, temporal_attention, causal_inference, copula_models,
# evt_models, gaussian_process, graph_models, online_gradient_boosting,
# online_hmm_regime, online_hpo, online_learning, feature_clustering,
# feature_importance_tracker, feature_selection, conformal, conformal_prediction,
# stacking, stacking_ensemble, nested_meta_labeling, triple_barrier,
# quantile_models, factor_timing, adversarial_validation (prüfen — möglicherweise REAL), 
# automl, bayesian_ensemble, combined_regime, cpcv (prüfen), feedback_loop (prüfen — groß),
# garch_models, hyperopt, lime_explainer, model_monitoring, retraining_scheduler,
# signal_correlation, signal_decay_tracker, calibration_monitor

# Verschiebe (NICHT löschen!):
for f in maml gnn_stocks bayesian_nn rl_portfolio rl_execution tda_regime \
         symbolic_regression temporal_attention causal_inference copula_models \
         evt_models gaussian_process graph_models online_gradient_boosting \
         online_hmm_regime online_hpo online_learning feature_clustering \
         feature_importance_tracker; do
  git mv src/assembled_core/ml/$f.py archive/observability_graveyard_2026q2/ml/
done

# Danach: grep, welche trading_cycle.py-Step-Blöcke jetzt kaputt sind
grep -n "from src.assembled_core.ml.maml\|from src.assembled_core.ml.gnn_stocks\|..." src/assembled_core/pipeline/trading_cycle.py

# Entweder: Die betroffenen Step-Blöcke aus trading_cycle.py komplett löschen
# Oder: Die Imports auskommentieren und die Step-Blöcke ins meta schreiben "archived"

git commit -m "refactor(ml): archive 19 observability-only ML modules"
```

Wiederhole für `risk/`, `features/`, `portfolio/`, `signals/`, `execution/`, `intel/`, `data/`, `events/` nach derselben Methode.

**Wichtig:** Nach jedem Modul-Commit: `pytest -m fast -q` laufen lassen. Wenn Tests brechen, haben die Tests echten Bezug zu den verschobenen Modulen — prüfen, ob die Tests selbst Wiring-Tests sind (dann löschen) oder echte Tests (dann die Verschiebung reverten).

---

## Woche 4-6 — `trading_cycle.py` zerlegen

**Ziel:** Der 10.544-Zeilen-Monolith wird in 7 Funktionen à ≤500 Zeilen.

### Ziel-Struktur

Eine neue Datei `src/assembled_core/pipeline/trading_cycle_v2.py`:

```python
def run_trading_cycle(context: CycleContext) -> CycleResult:
    data = ingest_data(context)
    features = build_features(data, context)
    signals = generate_signals(features, context)
    targets = size_positions(signals, context)
    checked = check_risk(targets, context)
    orders = route_orders(checked, context)
    fills = book_fills(orders, context)
    return CycleResult(...)
```

### Vorgehen

1. **Tag 1:** Erstelle leere Stubs der 7 Funktionen. Lass die alte `trading_cycle.py` intakt.
2. **Tag 2–5:** Für jede der 309 Steps in der alten Datei:
   - Ist der Step real genutzt (nicht nur observability)? Wenn nein: löschen.
   - Wenn ja: in welche der 7 Funktionen gehört er? Verschieben.
3. **Tag 6–8:** Tests schreiben für jede der 7 neuen Funktionen (keine Wiring-Tests, echte Funktionstests).
4. **Tag 9:** Umschalter: `run_trading_cycle` zeigt jetzt auf die neue Version. Alte Datei als `_legacy_trading_cycle.py` parken.
5. **Tag 10:** Nach 2 Wochen Stabilität: `_legacy_trading_cycle.py` löschen.

### Regel für alle Step-Blöcke

Ein Step bleibt nur dann im neuen Code, wenn **alle** Bedingungen erfüllt sind:

- Der Step beeinflusst ein `result`-Feld, das später von einem anderen Step oder vom Caller gelesen wird
- Der Step hat einen Test, der sein Verhalten mit konkreten Werten (nicht nur Existenz) prüft
- Der Step hat keine `log.debug("skipped: ...")`-Struktur

Wenn ein Step nur `result.meta["xyz"] = {"available": True}` schreibt und niemand `result.meta["xyz"]` liest — **raus**.

---

## Woche 7 — Eine Strategie End-to-End

**Ziel:** Reale Performance-Zahlen, die du in 3 Monaten wieder reproduzieren kannst.

### Setup

1. Entscheide: **eine** Strategie. Vorschlag: EMA-Trend (das was der Default-Backtest macht) + News-Overlay (weil News-Pipeline dein stärkstes Asset ist).

2. Parameter einfrieren:
   - Universum: `watchlist.txt` (29 US-Large-Caps)
   - Zeitraum: 2020-01-01 bis heute (mindestens 5 Jahre)
   - Kosten: 10 bps Commission + 5 bps Slippage + 50 bps Borrow für Shorts
   - Position-Sizing: Vol-Target 15% annualisiert, max. 5% pro Position
   - Kein Hebel

3. Reproduzierbarer Run:
```bash
python scripts/run_backtest_strategy.py \
    --strategy trend_baseline \
    --freq 1d \
    --start-date 2020-01-01 \
    --end-date 2026-04-01 \
    --universe watchlist.txt \
    --start-capital 100000 \
    --with-costs \
    --generate-report
```

4. Erwartung: Sharpe realistisch zwischen 0.3 und 1.2, MDD zwischen 15% und 35%. Falls Sharpe > 2.0 oder MDD < 10%: Kosten-Modell prüfen, da stimmt was nicht.

5. Ergebnis festhalten: `docs/results/2026_04_trend_baseline_5y.md` mit Metriken und Charts.

---

## Woche 8 — Crisis-Alpha gegen 2020 backtesten

**Ziel:** Beantworten: "Hätte unser Crisis-Mode auf COVID-März-2020 getriggert?"

### Vorgehen

1. Historische News von Februar/März 2020 laden (oder simulieren)
2. Crisis-Alpha-State-Machine darauf laufen lassen
3. Tracken: wann WATCH → ACTIVE, wann ACTIVE → COOLDOWN
4. Simulation: Wenn getriggert, wurde GLD/TLT gekauft, SPY geshortet?
5. Ergebnis: P&L des Crisis-Sub-Portfolios vs. Baseline

Wenn Crisis-Alpha im März 2020 **nicht** getriggert hat: das Gates-System ist falsch kalibriert.

Wenn es getriggert hat und das P&L negativ war: Basket-Definition oder Sizing ist falsch.

Wenn es getriggert hat und positiv war: du hast ein **validiertes** Signal. Das ist Gold wert — dokumentieren, festnageln, nicht anfassen.

---

## Wochen 9-12 — News → Signal schließen

**Ziel:** Die Pipeline-Schicht und die Trading-Entscheidungen verbinden.

Konkret:
- `signals/news_signal_bridge.py` aktiv in `rules_trend.py` einbauen
- `features/news_features.py` zum Factor-Store hinzufügen
- `signals/intel_signal_adapter.py` real integrieren (nicht nur trading_cycle_only)

Sobald das steht, ist die News-Pipeline nicht nur "gute Infrastruktur", sondern "aktive Alpha-Quelle".

---

## Regeln für alle Wochen

### Die "No-new-feature"-Regel

Bis Woche 12 **keine neuen Features** mehr. Keine neuen ML-Module, keine neuen Strategien, keine neuen Data-Sources. Konsolidierung first.

### Die "Kein wave-N mehr"-Regel

Keine neuen "Wave"-Commits. Wenn Claude Code einen vorschlägt, ablehnen.

### Die Commit-Regel

Jeder Commit muss einen der drei Typen haben:
- `refactor: ...` (Code-Qualität, keine Funktionsänderung)
- `fix: ...` (Bug-Fix)
- `feat: ...` (nur für neue Tests, oder nach Woche 12 für neue Features)

Keine `feat(wiring): wave-N ...` Commits mehr.

### Die Test-Regel

Jeder neue Test muss mit konkreten Werten prüfen, nicht nur Existenz.

```python
# Schlecht (Wave-Wiring-Stil):
def test_compute_something_importable():
    assert compute_something is not None

def test_compute_something_empty():
    result = compute_something([])
    assert result == []

# Gut:
def test_compute_something_with_real_inputs():
    # Given
    data = pd.DataFrame({"close": [100, 102, 101, 103, 105]})
    # When
    result = compute_something(data, window=3)
    # Then
    assert result.iloc[-1] == pytest.approx(103.0, rel=0.01)
    assert len(result) == len(data)
    assert not result.isna().all()
```

---

## Nach 12 Wochen

Das Repo sollte dann haben:
- ~1000 statt ~2000 Files
- Eine Strategie mit reproduzierbaren 5-Jahres-Metriken
- Validiertes Crisis-Alpha gegen COVID 2020
- News-Signale fließen in Trading-Entscheidungen
- `trading_cycle.py` ≤ 500 Zeilen pro Funktion
- Null Wave-Wiring-Tests
- Null observability-wired Module in `src/`
- Ein sauberes `archive/` für zukünftige Reaktivierung

Das ist der Zustand, in dem du **ernsthaft** über Richtung B (News-Signal-Service) oder Richtung A (Personal-Quant-Platform) nachdenken kannst.

---

## Wenn du stecken bleibst

Wenn in Woche 3 oder 4 Tests brechen, von denen du nicht weißt warum:

1. Schau in `KNOWN_ISSUES.md`, Teil 1 §1, Teil 2-Modul-Tabelle für die betroffene Datei
2. Wenn Test nur Importability prüft: Test löschen
3. Wenn Test echte Funktionalität prüft: Verschiebung reverten, alternative Strategie suchen

Wenn du in einer Woche zwei Commits hintereinander brichst: **Stopp**. Reverte beide Commits. Rufe Claude Code und bitte um einen frischen Analysegang für nur diesen einen Sub-Bereich.

---

## Was dieses Handbuch nicht ersetzt

- Die detaillierten Befunde in Teil 1 (z.B. Punkte zu Regulatorik, Datenquellen, Wissenschaftlichkeit)
- Die Datei-Verdicts in Teil 2 und Teil 3
- Dein eigenes Urteil: "Wird das dem System gerecht, was wir bauen wollen?"

Dieses Handbuch ist die **Reihenfolge**. Das **Warum** steht in den drei Audit-Teilen.

---

**Viel Erfolg, Hans.**

Wenn du in einer Woche wieder hier bist und Woche-1 abgeschlossen hast, zeig mir den Diff. Wir können dann Woche 2 gemeinsam planen.
