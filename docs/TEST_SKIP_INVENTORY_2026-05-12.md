# Test-Skip-Inventur (KNOWN_ISSUES §8.14, Wave 19 Follow-On)

**Datum:** 2026-05-12
**Scope:** 10 Test-Dateien aus `KNOWN_ISSUES.md §8.14`.
**Methode:** Jeder `pytest.skip` / `pytest.importorskip` / `pytest.mark.skip*` /
`xfail`-Marker einzeln klassifiziert nach: **(L)egitim**, **(S)tale**,
**(B)uggy**.

## Klassifikationen

| Kategorie | Bedeutung | Aktion |
|-----------|-----------|--------|
| **L** | Optionale Dep oder echter Slow-/Env-Conditional | Beibehalten |
| **S** | Test referenziert ein Modul, das archiviert wurde — Test skipt universell | Test mit-archivieren *oder* in xfail überführen — strukturelle Entscheidung, Wave 19 lässt offen |
| **B** | Falsch implementierter Skip (z. B. `skipif(True)` für eine Bedingung, die runtime-prüfbar wäre) | In dieser Sitzung fixen |

## Datei-für-Datei

### 1. `tests/test_qa_numba_kernels.py` — **L**

Ein `@pytest.mark.skipif(not _NUMBA_AVAILABLE, ...)`. `numba` ist
absichtlich nicht im Default-venv (siehe Performance-Migrationsplan
§8.2). Skip ist berechtigt und sollte bleiben, bis numba aktiviert wird.

### 2. `tests/test_ml_signals_intel.py` — **S** (7×) + **L** (3×)

- 7× `pytest.importorskip("src.assembled_core.intel.multichannel_propagation")`
  → Modul liegt in `archive/intel_research_2026q2/intel/multichannel_propagation.py`.
  Tests skippen universell.
- 3× `@pytest.mark.skipif(not _scipy_available(), ...)` — legitim (scipy optional).

### 3. `tests/test_ml_features.py` — **S** (17×) + **L** (4×) + **L** (1× arch-conditional)

- 5× `importorskip intel.weaponized_interdependence` →
  archive/observability_graveyard_2026q2/intel/weaponized_interdependence.py
- 5× `importorskip intel.scenario_trees` →
  archive/intel_research_2026q2/intel/scenario_trees.py
- 5× `importorskip portfolio.barbell_strategy` →
  archive/observability_graveyard_2026q2/portfolio/barbell_strategy.py
- 2× `importorskip features.volatility_features` →
  archive/observability_graveyard_2026q2/features/volatility_features.py
- 3× scipy skipifs **L**.
- 1× `pytest.skip("arch is installed")` — bewusst-inverter Skip-Pattern
  (Test gilt nur für die *Abwesenheit* von `arch`). **L**.

### 4. `tests/test_ml_foundation.py` — **L** (9× existierender Pfad) + **S** (12×) + **L** (8× scipy/arch)

- 9× `importorskip src.assembled_core.ml.purged_cv` → **Modul EXISTIERT**.
  Tests laufen normal, skip-Mechanismus ist defensiv.
- 5× `importorskip src.assembled_core.ml.copula_models` → **Modul EXISTIERT**.
  Tests laufen normal.
- 6× `importorskip src.assembled_core.ml.garch_models` →
  archive/observability_graveyard_2026q2/ml/garch_models.py
- 6× `importorskip src.assembled_core.ml.evt_models` →
  archive/observability_graveyard_2026q2/ml/evt_models.py
- 3× scipy skipifs, 3× arch skipifs **L**.

### 5. `tests/test_intel_to_signal.py` — **S** (1× file-level)

- Modul-level `importorskip src.assembled_core.signals.intel_signal_adapter`
  → das Modul existiert noch im src-Tree (Shim), aber die Klasse
  `IntelSignalAdapter` wurde archiviert →
  `archive/observability_graveyard_2026q2/signals/intel_signal_adapter.py`.
- Eine zusätzliche `@pytest.mark.skip(reason="IntelSignalAdapter class
  archived to observability_graveyard_2026q2/signals/")` auf der gesamten
  Testklasse — schon korrekt dokumentiert.

### 6. `tests/test_competitive_analysis_impl.py` — **B** (Zeile 1312, in dieser Sitzung gefixt)

- Zeile 1312 hatte `@pytest.mark.skipif(True, reason="hmmlearn not
  installed in test env")` — das `True` skipt unconditional, auch wenn
  `hmmlearn` doch installiert ist. **Gefixt:** Skip in Runtime-Check via
  `HMMLEARN_AVAILABLE`-Flag aus `assembled_core.ml.regime_hmm`
  überführt. Begleit-Test bei Zeile 1325/1335 nutzt diesen Pfad bereits
  korrekt.

### 7. `tests/test_automl.py` — **S** (file-level)

- Modul-level `importorskip src.assembled_core.ml.automl` →
  archive/observability_graveyard_2026q2/ml/automl.py. Skipt komplett.
- 3× sklearn-conditional skipifs würden greifen, wenn die Datei nicht
  ohnehin universell skippte. **L** nominal, aber unerreichbar.

### 8. `tests/test_trading_cycle_regression_daily.py` — **L** (xfail)

- Zeile 181: `@pytest.mark.xfail(reason="Legacy vs cycle path qty
  divergence: legacy filters to last row per symbol before feature
  computation, ...")`. Dokumentierte Bekannt-Schuld; tracked.

### 9. `tests/test_trading_cycle_backtest_snapshot_equivalence.py` — **L** (xfail)

- Zeile 60: `@pytest.mark.xfail(reason="Snapshot mode filters prices
  differently than history-slice mode, ...")`. Dokumentierte Tech-Debt.

### 10. `tests/test_backtest_vs_two_eod_cycles.py` — **L** (xfail)

- Zeile 155: `@pytest.mark.xfail(reason="E0.1 bit-identical
  backtest↔paper parity requires position-evolution threading
  (known tech-debt, sunset 2026-07-01) ...")`. Dokumentierte Tech-Debt
  mit Sunset-Datum.

## Konsolidierte Befunde

| Kategorie | Anzahl Marker | Aktion |
|-----------|--------------:|--------|
| Legitim (optionale Deps, slow, env-conditional, dokumentierter xfail) | ~25 | Behalten |
| Stale (Modul archiviert) | ~30 | Strukturelle Entscheidung offen |
| Buggy | 1 | Wave 19 gefixt |

## Empfehlung — strukturelle Stale-Tests

5 Testdateien bestehen praktisch nur aus `importorskip`-Stubs für
archivierte Module:

- `tests/test_ml_signals_intel.py` (multichannel_propagation)
- `tests/test_ml_features.py` (weaponized_interdependence, scenario_trees,
  barbell_strategy, volatility_features)
- `tests/test_intel_to_signal.py` (IntelSignalAdapter archived class)
- `tests/test_automl.py` (ml.automl)
- Teile von `tests/test_ml_foundation.py` (garch_models, evt_models)

Drei Pfade möglich:

1. **Test-Dateien mit-archivieren** — verschieben nach
   `archive/observability_graveyard_2026q2/tests/`. Reduziert Test-
   Collection-Noise. Risiko: wenn die Module später re-aktiviert
   werden, müssen die Tests manuell zurückgeholt werden.
2. **In file-level skip mit klarem reason überführen** — z. B.
   `pytestmark = pytest.mark.skip(reason="targets archived module ...")`
   am Anfang jeder betroffenen Datei. Sichtbarer als die `importorskip`-
   Falle.
3. **Status quo lassen** — Tests verursachen keine Laufzeit-Kosten,
   sondern nur Collection-Noise.

Wave 19 lässt die strukturelle Entscheidung **offen** — sie ist eine
saubere User-Entscheidung, kein autonomes Risk-Item. Diese Inventur
ist die zugehörige Vorarbeit.

## Geprüfte aber nicht angefasste Marker

- 3 dokumentierte xfails — alle mit Sunset-Datum oder klarer Tech-Debt-
  Begründung. Nichts zu tun.
- scipy / sklearn / arch / numba / hmmlearn skipifs — alle legitim,
  jede Dep ist absichtlich nicht im Default-venv.
