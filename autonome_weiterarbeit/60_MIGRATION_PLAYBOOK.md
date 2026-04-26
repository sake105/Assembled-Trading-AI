# 60 — Migration-Playbook: Monolith → Plugin-Architektur

**Zweck:** Schrittweise Anleitung, wie du vom aktuellen Repo-Zustand (10.499-Zeilen-`trading_cycle.py`, 147 Wave-Tests, 52+ ungenutzte ML-Module, +41 Scripts seit Audit) zu der im Plan beschriebenen Plugin-Architektur kommst — **ohne dass die existierende Pipeline wärend der Migration bricht**.

**Scope:** Konkret für dieses Repo. Keine generischen Best-Practices, keine Microservices (du willst einen Monolith mit sauberer interner Modularität, kein Service-Mesh). Der Plan-Ziel ist ein Single-Process-Python-System mit Plugin-Registry.

**Zeithorizont:** 4-6 Monate bei 10-15h/Woche. Ehrlicherweise: eher 6.

---

## 0. Grundprinzipien

### 0.1 Strangler-Fig statt Big-Bang

**Prinzip (Martin Fowler, 2004):** Man wickelt den alten Monolithen in eine Fassade, die Traffic wahlweise an alt oder neu routet. Neue Funktionen werden außerhalb gebaut, alte schrittweise ersetzt, bis der Monolith leer ist.

**Für dein Repo konkret:** Die "Fassade" ist nicht ein HTTP-Proxy (du hast kein Microservice-Setup), sondern eine **Dispatcher-Schicht** in Python, die entscheidet ob ein Signal-Request an den alten `trading_cycle.py`-Pfad oder an die neue Plugin-Registry geht.

```python
# src/assembled_core/dispatcher.py (neu)
from enum import Enum

class Pipeline(Enum):
    LEGACY = "legacy"      # ruft trading_cycle.py
    MODERN = "modern"      # ruft Plugin-Registry
    SHADOW = "shadow"      # ruft beide, vergleicht, nutzt LEGACY

class SignalDispatcher:
    def __init__(self, mode: Pipeline, registry, legacy_fn):
        self.mode = mode
        self.registry = registry
        self.legacy = legacy_fn
    
    def run(self, inputs):
        if self.mode == Pipeline.LEGACY:
            return self.legacy(inputs)
        elif self.mode == Pipeline.MODERN:
            return self._run_plugins(inputs)
        elif self.mode == Pipeline.SHADOW:
            legacy_result = self.legacy(inputs)
            try:
                modern_result = self._run_plugins(inputs)
                self._record_diff(inputs, legacy_result, modern_result)
            except Exception as e:
                log.error(f"Modern pipeline failed: {e}")
            return legacy_result  # Legacy wins bis Cutover
```

**Die drei Zustände** werden pro Signal-Typ (nicht global!) umgeschaltet. Ein einzelnes Feature-Flag reicht nicht — du brauchst ein Flag pro Migrations-Einheit.

**Die Warnung aus der Heureka-Praxis:** "A half-finished Strangler Fig becomes a Frankenstein – not an improvement. Don't half-strangle. Kill cleanly, or don't bother." Setze dir Zeitlimit pro Migrations-Einheit. Wenn nach 4 Wochen nicht fertig, entweder fertig machen oder zurückrollen.

### 0.2 Characterization-Tests VOR dem Refactor

**Prinzip (Michael Feathers, "Working Effectively with Legacy Code"):** Bevor du eine Zeile Legacy-Code änderst, schreibst du Tests, die das **aktuelle Verhalten** dokumentieren — egal ob das Verhalten richtig ist. Du friest den Ist-Zustand ein, änderst dann, und siehst bei jedem Diff ob du was versehentlich kaputt gemacht hast.

**Für dein Repo konkret:** Equity-Kurve ist dein Characterization-Output. Du lässt `trading_cycle.py` auf einem fixen Datenbereich (z.B. 2024-01-01 bis 2024-12-31, 50 fixe Ticker) laufen, speicherst das Ergebnis als "approved", und prüfst bei jeder Änderung den Diff.

### 0.3 Die "Remove before Refactor"-Regel

**Prinzip:** Jeder Refactor ist einfacher, wenn der zu refactorierende Code vorher kleiner ist. Lösche toten Code, bevor du lebenden Code umbaust.

**Für dein Repo konkret:**
- 147 Wave-Tests → weg, bevor du Tests restrukturierst
- 52+ observability-wired ML-Module → archivieren, bevor du ML-Pipeline neubaust
- `trading_cycle.py`-Scheinintegrationen → entfernen, bevor du die echten Integrations bewegst

Das ist kein Luxus, das ist die entscheidende Reihenfolge.

### 0.4 AI-Disziplin

**Beobachtung aus mehreren Quellen:** AI-Assistenten wie Claude haben ein systematisches Muster — sie bauen eher neu als umbauen. Aus dem InfoWorld-Artikel: "AI implements prompts directly without considering refactoring opportunities, architectural patterns, or maintainability trade-offs. It just adds what you asked for." Dein Audit zeigt genau dieses Muster: 699 Commits, 0 Refactor-Commits, aber +41 Scripts seit Audit trotz Plan auf dem Tisch.

**Regel für dich:** Jeder AI-assistierte Commit in einer Migration-Session muss eine von zwei Kategorien erfüllen:
- **REMOVE**: Dateien/Zeilen werden gelöscht. `git show --stat <commit>` zeigt negative Nettosumme.
- **REPLACE**: Dateien werden atomar ausgetauscht. Alte weg, neue rein, gleicher LoC-Bereich.

Was **nicht erlaubt** ist während der Migrations-Phase:
- Neue Stub-Module "für später"
- Zusätzliche `observability`-Wiring-Einträge
- Parallel-Implementierungen mit `_v2`-Suffix

Wenn dir Claude einen Commit vorschlägt, der neue Dateien hinzufügt und alte stehen lässt: ablehnen. Du bist in Migrations-Modus, nicht in Feature-Entwicklungs-Modus.

---

## 1. Tool-Stack für die Migration

### 1.1 Pflicht-Tools (vor Start installieren)

```bash
# Dead-Code-Detection
uv pip install vulture==2.14            # AST-basiert, Konfidenz-Scores
uv pip install deadcode==2.5.0          # Alternative mit --fix-Option

# Module-Boundaries & Dependency-Enforcement
uv pip install tach==0.29.0             # Rust-Backend, schnell, interaktives Setup
uv pip install import-linter==2.1       # Contract-Types (forbidden, layers, independence)
uv pip install deptry==0.20.0           # Unused Dependencies erkennen

# Komplexitäts-Metriken
uv pip install radon==6.0.1             # Cyclomatic Complexity, Maintainability Index
uv pip install wily==1.25.0             # Komplexitäts-Trends über Git-History
uv pip install lizard==1.17.13          # Alternative CC-Tool, unterstützt mehr Sprachen

# Characterization-Tests / Approval-Testing
uv pip install approvaltests==14.4.0    # Python-Port von ApprovalTests
uv pip install syrupy==4.7.2            # Alternative Snapshot-Tester (pytest-native)

# Pre-Commit + Enforcement
uv pip install pre-commit==4.0.1
uv pip install ruff==0.8.4              # Linter + Formatter, schneller als flake8+black
```

**Versions-Stand:** April 2026. `uv` ist der moderne Paketmanager (von Astral, 10-100x schneller als pip). Falls du noch pip nutzt, jetzt wechseln:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1.2 Optional, aber hilfreich

```bash
uv pip install pydeps==3.0.1            # Dependency-Graph visualisieren
uv pip install pylint==3.3.2            # tiefere Analyse als ruff
uv pip install mypy==1.13.0             # Type-Checker, findet Import-Probleme
```

### 1.3 Keine Empfehlung

- **GitHub Copilot / Cursor Composer** für Refactoring: die Erfahrung zeigt, dass diese Tools **neuen Code** gut produzieren, aber **Refactoring** (im Sinne von "weniger, klarer, dasselbe tun") schlecht. Nutze Claude im Chat-Modus mit expliziten Anti-Pattern-Prompts (siehe §8).
- **Automatische Refactoring-Tools** wie Refact.ai: nett für Greenfield, gefährlich bei 10k-Zeilen-Files.

---

## 2. Baseline-Messung (Woche 0, vor Start)

**Bevor du irgendetwas änderst, misst du den Ist-Zustand.** Sonst weißt du am Ende nicht, ob du wirklich besser bist.

### 2.1 Das Baseline-Script

```bash
# scripts/migration/baseline.sh
#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="docs/migration/baseline_$(date +%Y-%m-%d)"
mkdir -p "$OUT_DIR"

echo "=== LOC Statistics ==="
find src/assembled_core -name "*.py" | xargs wc -l | tail -1 > "$OUT_DIR/loc_total.txt"
find tests -name "*.py" | xargs wc -l | tail -1 > "$OUT_DIR/loc_tests.txt"
find scripts -name "*.py" | xargs wc -l | tail -1 > "$OUT_DIR/loc_scripts.txt"

echo "=== File Counts ==="
find src/assembled_core -name "*.py" | wc -l > "$OUT_DIR/files_src.txt"
find tests -name "test_wave*_wiring.py" | wc -l > "$OUT_DIR/files_wave_tests.txt"

echo "=== Cyclomatic Complexity ==="
radon cc src/assembled_core --min C -s --total-average > "$OUT_DIR/cc.txt"

echo "=== Maintainability Index ==="
radon mi src/assembled_core --min C -s > "$OUT_DIR/mi.txt"

echo "=== Dead Code (min-confidence 80) ==="
vulture src/assembled_core --min-confidence 80 > "$OUT_DIR/vulture_80.txt" || true

echo "=== Dead Code (min-confidence 60) ==="
vulture src/assembled_core --min-confidence 60 > "$OUT_DIR/vulture_60.txt" || true

echo "=== Unused Dependencies ==="
deptry . > "$OUT_DIR/deptry.txt" || true

echo "=== Biggest Files (top 20) ==="
find src/assembled_core -name "*.py" -exec wc -l {} \; | sort -rn | head -20 > "$OUT_DIR/biggest.txt"

echo "=== trading_cycle.py specifics ==="
FILE="src/assembled_core/pipeline/trading_cycle.py"
{
  echo "Lines: $(wc -l < $FILE)"
  echo "try blocks: $(grep -c 'try:' $FILE)"
  echo "except Exception: $(grep -c 'except Exception' $FILE)"
  echo "inline imports: $(grep -c '^    from \|^    import ' $FILE)"
  echo "Step comments: $(grep -cE '# Step [0-9]+\.' $FILE)"
  echo "log.debug calls: $(grep -c 'log.debug' $FILE)"
} > "$OUT_DIR/trading_cycle_stats.txt"

echo "Baseline saved to $OUT_DIR"
```

### 2.2 Erwartete Baseline-Werte (aus Audit-Daten)

| Metrik | Wert |
|---|---|
| `trading_cycle.py` Zeilen | ~10.499 |
| `trading_cycle.py` Cyclomatic Complexity (Gesamt) | vermutlich >500 |
| Vulture Dead-Code (min-conf 80) | >200 Funde erwartet |
| Wave-Test-Files | 147 |
| ML-Folder Files | 110 |
| Scripts Files | 136 |
| Workflows | 34 |

### 2.3 Das Ziel nach Migration

| Metrik | Ziel |
|---|---|
| Größte Datei LoC | < 500 |
| Max. Cyclomatic Complexity pro Funktion | < 10 (Radon-Grade A-B) |
| Maintainability Index (alle Files) | ≥ 20 (Grade A) |
| Dead-Code (min-conf 80) | 0 Funde |
| Tests | > 80% echte Funktionalitätstests, 0 Wave-Wiring |
| Plugin-Count | 15-25 BaseSignals |
| Module-Boundary-Violations (tach check) | 0 |

**Wichtig:** Das Ziel ist **nicht** weniger LoC um jeden Preis. Ein 30-Zeilen-Plugin, das eine Strategie kapselt, ist besser als eine 150-Zeilen-Funktion in einem Monolith. Die richtige Metrik ist **Modular-Dichte** (Funktionalität pro LoC in einer klar abgegrenzten Einheit).

---

## 3. Die vier Migrations-Phasen

```
Phase 1 (Woche 1-3):   DELETE        →  Sterile Umgebung schaffen
Phase 2 (Woche 4-6):   CHARACTERIZE  →  Legacy-Verhalten einfrieren
Phase 3 (Woche 7-16):  STRANGLE      →  Plugins bauen, Traffic umleiten
Phase 4 (Woche 17-20): CLEANUP       →  Legacy-Reste entfernen
```

Jede Phase hat ein klares Go/No-Go-Kriterium am Ende.

---

## 4. Phase 1 — DELETE (Woche 1-3)

**Ziel:** Ballast weg, bevor Strukturarbeit beginnt. **Keine Architektur-Änderung.**

### 4.1 Woche 1: Root-Level und Wave-Tests

**Tag 1-2: Root-Level-Datei-Audit**

```bash
# Diese Dateien aus dem Audit waren im Root und gehören weg:
git rm review_bundle.txt                    # 5.7 MB Müll
git rm oos_debug_log.txt                    # Log-Datei im Root
git rm missing_symbols.txt                  # gehört in structured data
git rm README_INTEGRATION.txt               # verdoppelt README
git rm README_ONECLICK.md                   # verdoppelt README
git rm PROJECT_STATUS.txt                   # widerspricht PROJEKT_STATUS.md
git rm 000_seed_project.ps1.disabled       # disabled = weg

git commit -m "chore: remove root-level cruft (7 files, -5.7MB)"
```

**Tag 3-5: Wave-Tests löschen**

Die 147 `test_wave*_wiring.py` sind laut Audit "Scheintests" (prüfen nur `assert modul is not None`). Das ist Safety-theater, kein Testing.

```bash
# Vor Löschung: Baseline-Test-Run
pytest tests/ --ignore=tests/test_wave -q > /tmp/tests_before.txt
# erwarte: die nicht-wave tests laufen weiterhin

# Löschung
cd tests
for f in test_wave*_wiring.py; do
    git rm "$f"
done
cd ..

# Post-Löschung Test-Run
pytest tests/ -q > /tmp/tests_after.txt

# Diff: sollte keine Änderung in den echten Tests zeigen
diff /tmp/tests_before.txt /tmp/tests_after.txt

git commit -m "chore: remove 147 wave_wiring scheintests

These tests only asserted 'modul is not None' and added zero
functional coverage. Removed per audit 2026-04-23.

Deleted: tests/test_wave*_wiring.py (147 files, ~8000 LoC)
Remaining tests: <N> pass, <M> fail (unchanged from pre-deletion)"
```

**Entscheidendes Detail:** Du committest das in **einem** Commit, nicht 147 einzelnen. Grund: der Revert-Fall ist "zurück zum Vor-Zustand", nicht "147 einzelne Wave-Tests zurückholen".

### 4.2 Woche 2: ML-Module archivieren

Die 52+ observability-wired ML-Module werden **nicht gelöscht**, sondern **archiviert**. Grund: einige könnten später doch nützlich sein (z.B. `garch_models.py`), aber sie stören aktuell das Mental-Modell und den Dead-Code-Scan.

**Vorgehen:**

```bash
# Verzeichnis für Archiv anlegen
mkdir -p archive/ml_observability_graveyard_2026q2

# Liste aus Audit: ML-Module ohne echten Decision-Path
cat > /tmp/ml_to_archive.txt <<EOF
src/assembled_core/ml/maml.py
src/assembled_core/ml/gnn_stocks.py
src/assembled_core/ml/bayesian_nn.py
src/assembled_core/ml/rl_portfolio.py
src/assembled_core/ml/rl_execution.py
src/assembled_core/ml/tda_regime.py
src/assembled_core/ml/symbolic_regression.py
src/assembled_core/ml/temporal_attention.py
src/assembled_core/ml/causal_inference.py
src/assembled_core/ml/copula_models.py
src/assembled_core/ml/gaussian_process.py
src/assembled_core/ml/graph_models.py
src/assembled_core/ml/online_gradient_boosting.py
src/assembled_core/ml/online_hmm_regime.py
src/assembled_core/ml/online_hpo.py
src/assembled_core/ml/online_learning.py
src/assembled_core/ml/feature_clustering.py
src/assembled_core/ml/feature_selection.py
src/assembled_core/ml/conformal.py
src/assembled_core/ml/stacking.py
src/assembled_core/ml/stacking_ensemble.py
src/assembled_core/ml/nested_meta_labeling.py
src/assembled_core/ml/quantile_models.py
src/assembled_core/ml/factor_timing.py
src/assembled_core/ml/adversarial_validation.py
src/assembled_core/ml/automl.py
src/assembled_core/ml/bayesian_ensemble.py
src/assembled_core/ml/feedback_loop.py
src/assembled_core/ml/hyperopt.py
src/assembled_core/ml/lime_explainer.py
src/assembled_core/ml/news_ml_bridge.py
src/assembled_core/ml/nlp_sentiment.py
src/assembled_core/ml/regime_model_router.py
src/assembled_core/ml/regime_weight_trainer.py
src/assembled_core/ml/retraining_scheduler.py
src/assembled_core/ml/signal_correlation.py
src/assembled_core/ml/signal_decay_tracker.py
src/assembled_core/ml/evt_models.py
EOF

# Verschieben mit git mv (behält History)
while read -r file; do
    if [ -f "$file" ]; then
        git mv "$file" "archive/ml_observability_graveyard_2026q2/$(basename $file)"
    fi
done < /tmp/ml_to_archive.txt

# Jetzt: trading_cycle.py-Imports auf archivierte Module finden
grep -rn "from src.assembled_core.ml.\(maml\|gnn_stocks\|bayesian_nn\|rl_portfolio\)" src/ tests/
```

**Wichtig — wenn `trading_cycle.py` diese Module importiert:** Du musst **beide** Seiten fixen.

```python
# VORHER in trading_cycle.py (Beispiel, Step 8.62):
try:
    from src.assembled_core.ml.maml import MAMLConfig
    _maml_cfg = MAMLConfig()
    result.meta["maml"] = {"available": True, ...}
except Exception as _maml_exc:
    log.debug("[MAML] maml skipped: %s", _maml_exc)

# NACHHER: einfach weg, nicht in archive/-Pfad umlenken
# (Grund: der Try-Block tut sowieso nichts Sinnvolles)
```

Jedes Mal, wenn du einen Step entfernst, reduziert sich `trading_cycle.py` um 15-25 Zeilen. 40 Steps × 20 Zeilen = ~800 Zeilen weg, ohne dass irgendwas an echter Funktionalität verloren geht.

**Commit-Struktur:**

```bash
# Commit 1: Dateien verschieben
git commit -m "chore: archive 38 observability-wired ML modules

These modules had zero decision-path integration per audit 2026-04-23.
Moved to archive/ml_observability_graveyard_2026q2/ to preserve git history.
Can be restored individually if needed later.

Files: src/assembled_core/ml/{maml,gnn_stocks,bayesian_nn,...}.py"

# Commit 2: trading_cycle.py-Steps entfernen, die nur darauf verwiesen
git commit -m "refactor: remove 38 observability-only steps from trading_cycle.py

Removes try/except blocks in Steps 8.62, 8.63, 8.65, ... that imported
archived ML modules and only wrote to result.meta without affecting decisions.

trading_cycle.py: 10,499 → ~9,700 lines (-800, -7.6%)"
```

**Go/No-Go Check nach Woche 2:**

```bash
# Test-Suite muss grün bleiben (ohne Wave-Tests)
pytest tests/ -q
# erwarte: Gleiche Anzahl passes/fails wie Baseline

# Equity-Invarianz: Backtest auf identischen Daten produziert identische Kurve
python scripts/migration/golden_equity_check.py  # kommt in Woche 4 (Phase 2)
# für jetzt: manueller Check mit einem bekannten Ticker-Set
```

Wenn Tests rot werden: du hast Module archiviert, die doch benutzt wurden. Zurückholen mit `git mv archive/ml_observability_graveyard_2026q2/<file>.py src/assembled_core/ml/` und neu evaluieren.

### 4.3 Woche 3: Scripts und Workflows ausdünnen

**Scripts:** 136 Dateien, davon viele Sprint-Inseln (`sprint9_backtest.py`, `sprint9_execute.py`, `sprint10_portfolio.py`) und Legacy-Runner.

**Regel:** Wenn ein Script in den letzten 60 Tagen nicht ausgeführt wurde und keinen Referenten in der Dokumentation hat → Archiv.

```bash
# Sprint-Scripts identifizieren
ls scripts/ | grep -E "sprint[0-9]+" > /tmp/sprint_scripts.txt
# erwarte: 15-20 Dateien

# PowerShell-Scripts: das meiste davon ist Legacy aus dem Windows-Start
ls scripts/ | grep "\.ps1$" > /tmp/ps1_scripts.txt
# erwarte: ~50 Dateien

# Archiv-Zielstruktur:
mkdir -p archive/scripts_legacy_2026q2/{sprint_phases,powershell_legacy,one_off}

# Verschieben
while read -r f; do
    git mv "scripts/$f" "archive/scripts_legacy_2026q2/sprint_phases/$f"
done < /tmp/sprint_scripts.txt

# PowerShell-Scripts: nur die behalten, die du aktiv nutzt
# Rest archivieren
```

**Workflows:** 34 GitHub-Actions-Workflows ist 2× zu viel. Typisches Trading-System-Repo hat 5-8 Workflows (test, lint, deploy, nightly, release, docker-build, coverage).

```bash
# Liste analysieren
ls .github/workflows/ | sort > /tmp/workflows.txt

# Gruppieren in: ESSENTIAL, MERGEABLE, ARCHIVE
# Beispiele:
# ESSENTIAL: test.yml, lint.yml, deploy-prod.yml
# MERGEABLE: test-windows.yml + test-linux.yml → test.yml mit matrix
# ARCHIVE: sprint9_ci.yml, experimental_*.yml
```

**Go/No-Go Check nach Woche 3:**

```bash
# Baseline neu laufen
./scripts/migration/baseline.sh

# Erwartung:
# - LoC-Gesamt: -15% bis -25% vs Tag 0
# - Files: -300 bis -400
# - trading_cycle.py: ~8500 Zeilen
# - Tests grün wie Baseline
```

Wenn du diese Zahlen nicht erreichst, hast du zu zögerlich gelöscht. Härter werden.

---

## 5. Phase 2 — CHARACTERIZE (Woche 4-6)

**Ziel:** Aktuelles Verhalten des Monolithen als automatisierten Test einfrieren. **Erst dann** kannst du sicher refactoren.

### 5.1 Woche 4: Golden-Equity-Baseline

Der wichtigste Characterization-Test für ein Trading-System ist die **Equity-Kurve**. Gleicher Input → gleiche Equity-Kurve → Refactor war safe.

```python
# tests/characterization/test_golden_equity.py
import pytest
import pandas as pd
from pathlib import Path
from approvaltests.approvals import verify

from src.assembled_core.pipeline.trading_cycle import run_trading_cycle

FIXTURES = Path("tests/characterization/fixtures")

@pytest.mark.characterization
def test_golden_equity_trend_baseline_2024():
    """
    Läuft die trend_baseline Strategie auf AAPL/MSFT/NVDA (2024-01-01 → 2024-06-30).
    Output: daily equity curve als CSV-String.
    
    Dieser Test friert das Verhalten des alten Systems ein.
    JEDE Änderung erzeugt einen Diff, den du manuell approve'st.
    """
    bars = pd.read_parquet(FIXTURES / "bars_3tickers_h1_2024.parquet")
    config = {
        "strategy": "trend_baseline",
        "universe": ["AAPL", "MSFT", "NVDA"],
        "initial_equity": 100_000,
        "start": "2024-01-01",
        "end": "2024-06-30",
    }
    result = run_trading_cycle(bars, config)
    
    # Equity-Kurve als stabilisierter String
    equity_str = result.equity_curve.round(2).to_csv(index=True, date_format="%Y-%m-%d")
    
    verify(equity_str, options=fixture_path(FIXTURES / "golden_equity_trend_baseline_2024"))
```

**Wie ApprovalTests arbeitet:**

1. Erster Lauf: Test schlägt fehl (keine `.approved`-Datei). 
2. Du prüfst die `.received`-Datei manuell. Macht sie Sinn?
3. Wenn ja: `mv *.received *.approved` und in Git committen.
4. Fortan: jede Änderung produziert `.received`, Test schlägt fehl bis du manuell approved hast.

**Fixture-Daten:**

Die `bars_3tickers_h1_2024.parquet` ist ein **fester Snapshot**. Nicht aus Live-API. Nicht aus yfinance zur Test-Zeit. Ein committeter Parquet-File mit einem handverlesenen Datenbereich.

```python
# scripts/migration/build_fixture.py
# Lauft einmalig, committet den Parquet
import yfinance as yf
import pandas as pd

tickers = ["AAPL", "MSFT", "NVDA"]
data = yf.download(tickers, start="2024-01-01", end="2024-06-30", interval="1h")
data.to_parquet("tests/characterization/fixtures/bars_3tickers_h1_2024.parquet")
```

Der Vorteil: **deterministische Tests**. Keine Netzwerk-Abhängigkeit, keine Flakiness, keine API-Rate-Limits.

### 5.2 Woche 5: Weitere Characterization-Tests

Die Equity-Kurve ist die Headline. Aber es gibt 4 weitere Outputs, die du einfrieren solltest:

**Test 2: Feature-Vector-Stability**

```python
@pytest.mark.characterization
def test_golden_features_2024_q1():
    """Feature-Ausgabe für einen fixen Tag muss stabil sein."""
    bars = pd.read_parquet(FIXTURES / "bars_3tickers_h1_2024.parquet")
    features = compute_features_for_day(bars, date="2024-03-15", ticker="AAPL")
    
    features_str = features.round(6).to_string()
    verify(features_str, options=fixture_path(FIXTURES / "golden_features_AAPL_20240315"))
```

**Test 3: Order-Sequence-Invariance**

```python
@pytest.mark.characterization
def test_golden_order_sequence():
    """Gleiche Signale → gleiche Order-Sequenz."""
    signals = pd.read_parquet(FIXTURES / "signals_fixture_2024.parquet")
    orders = convert_signals_to_orders(signals)
    
    order_str = orders.to_csv(index=False, date_format="%Y-%m-%d %H:%M:%S")
    verify(order_str, options=fixture_path(FIXTURES / "golden_orders_2024"))
```

**Test 4: Regime-Classification-Stability**

```python
@pytest.mark.characterization
def test_golden_regime_classification():
    """Regime-Classifier-Output pro Tag muss stabil sein."""
    bars = pd.read_parquet(FIXTURES / "bars_SPY_daily_2020_2024.parquet")
    regimes = classify_regimes(bars)
    
    regime_str = regimes.to_csv(index=True)
    verify(regime_str, options=fixture_path(FIXTURES / "golden_regimes_SPY_2020_2024"))
```

**Test 5: Risk-Metrics-Reproducibility**

```python
@pytest.mark.characterization
def test_golden_risk_metrics():
    """Sharpe, Sortino, MaxDD für einen festen Backtest."""
    equity = pd.read_parquet(FIXTURES / "equity_fixture.parquet")
    metrics = compute_risk_metrics(equity)
    
    metrics_str = f"""Sharpe: {metrics['sharpe']:.4f}
Sortino: {metrics['sortino']:.4f}
MaxDD:   {metrics['max_drawdown']:.4f}
Calmar:  {metrics['calmar']:.4f}
"""
    verify(metrics_str)
```

### 5.3 Woche 6: Scenario-Tests für historische Krisen

Zusätzlich zum Golden-Equity: vier Szenario-Fixtures, die typische Marktereignisse abdecken.

```python
SCENARIOS = {
    "gfc_2008":     ("2008-01-01", "2009-06-30"),
    "covid_2020":   ("2020-01-01", "2020-12-31"),
    "rate_2022":    ("2022-01-01", "2022-12-31"),
    "calm_2017":    ("2017-01-01", "2017-12-31"),  # Vergleichsbaseline
}

@pytest.mark.characterization
@pytest.mark.parametrize("scenario", SCENARIOS.keys())
def test_golden_equity_scenarios(scenario):
    start, end = SCENARIOS[scenario]
    bars = pd.read_parquet(FIXTURES / f"bars_SPY_daily_{scenario}.parquet")
    config = {"strategy": "trend_baseline", "start": start, "end": end, ...}
    result = run_trading_cycle(bars, config)
    equity_str = result.equity_curve.round(2).to_csv()
    verify(equity_str, options=fixture_path(FIXTURES / f"golden_equity_{scenario}"))
```

**Go/No-Go nach Phase 2:**

- Alle 5+4=9 Characterization-Tests grün (mit approved-Dateien im Repo)
- Die approved-Dateien sind `git diff`-freundlich (Text, nicht Binary)
- Ein erzwungener Bug-Inject erzeugt tatsächlich einen Test-Fail (Sanity-Check des Frameworks)

```bash
# Sanity-Check: künstlichen Bug einbauen
sed -i 's/np.mean/np.median/' src/assembled_core/signals/rules_trend.py
pytest tests/characterization/ -v
# erwarte: mehrere Tests rot
# dann: git checkout src/assembled_core/signals/rules_trend.py
```

Wenn der Bug-Inject **keinen** Test-Fail produziert, ist dein Characterization-Framework nicht streng genug.

---

## 6. Phase 3 — STRANGLE (Woche 7-16)

**Ziel:** Plugin-Architektur aufbauen, Signal-für-Signal aus dem Monolithen rausziehen, Traffic schrittweise umleiten.

### 6.1 Woche 7: Plugin-Infrastruktur

**Das `BaseSignal`-Interface:**

```python
# src/assembled_core/signals/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

@dataclass
class SignalOutput:
    symbol: str
    score: float                # [-1, +1]
    confidence: float           # [0, 1]
    features_used: list[str]
    horizon_days: int
    computed_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

class BaseSignal(ABC):
    """Interface für alle Signal-Plugins.
    
    Subclasses registrieren sich via pyproject.toml [project.entry-points."ata.signals"].
    """
    
    # Pflicht-Attribute
    name: str = "base"           # eindeutig im System
    version: str = "0.0.0"       # Semver
    
    # Empfohlen
    horizon_days: int = 5
    required_features: list[str] = []
    required_data: list[str] = []   # z.B. ["bars_daily", "news"]
    
    @abstractmethod
    def compute(self, symbol: str, feature_store, now: datetime) -> SignalOutput | None:
        """Haupt-Methode: berechnet Signal für einen Ticker zu einem Zeitpunkt.
        
        Returns None wenn nicht anwendbar (fehlende Daten, außerhalb Universum).
        """
        ...
    
    def healthcheck(self) -> tuple[bool, str]:
        """Prüft, ob Signal aktuell funktionsfähig ist.
        
        Returns (healthy, reason). Wenn False, wird Signal im Dispatcher deaktiviert.
        """
        return True, "ok"
    
    def describe(self) -> dict:
        """Für Feature-Catalog (siehe §REVIEW_GAPS Lücke #6)."""
        return {
            "name": self.name,
            "version": self.version,
            "horizon_days": self.horizon_days,
            "required_features": self.required_features,
            "required_data": self.required_data,
            "docstring": self.__class__.__doc__ or "",
        }
```

**Die Registry:**

```python
# src/assembled_core/signals/registry.py
import logging
import sys
from importlib.metadata import entry_points
from .base import BaseSignal

log = logging.getLogger(__name__)

class SignalRegistry:
    """Sammelt alle registrierten Signal-Plugins via entry_points."""
    
    def __init__(self):
        self._signals: dict[str, BaseSignal] = {}
        self._load_errors: dict[str, str] = {}
    
    def load_all(self) -> None:
        """Lädt alle Plugins aus der 'ata.signals' Entry-Point-Gruppe.
        
        Fehler in einzelnen Plugins dürfen System nicht killen.
        """
        if sys.version_info < (3, 10):
            raise RuntimeError("Python 3.10+ required for selectable entry points")
        
        eps = entry_points(group="ata.signals")
        
        for ep in eps:
            try:
                cls = ep.load()
                if not issubclass(cls, BaseSignal):
                    raise TypeError(f"{ep.name} does not inherit from BaseSignal")
                
                instance = cls()
                
                if instance.name in self._signals:
                    raise ValueError(f"Duplicate signal name: {instance.name}")
                
                healthy, reason = instance.healthcheck()
                if not healthy:
                    log.warning(f"Signal {instance.name} unhealthy: {reason}")
                    continue
                
                self._signals[instance.name] = instance
                log.info(f"Loaded signal: {instance.name} v{instance.version} from {ep.value}")
            
            except Exception as e:
                self._load_errors[ep.name] = str(e)
                log.error(f"Failed to load signal {ep.name}: {e}")
    
    def all(self) -> list[BaseSignal]:
        return list(self._signals.values())
    
    def get(self, name: str) -> BaseSignal | None:
        return self._signals.get(name)
    
    def errors(self) -> dict[str, str]:
        return dict(self._load_errors)
```

**Registration in pyproject.toml:**

```toml
# pyproject.toml
[project.entry-points."ata.signals"]
trend_baseline = "assembled_core.signals.builtin.trend_baseline:TrendBaselineSignal"
# weitere kommen in späteren Wochen dazu
```

**Das erste Plugin — Trend-Baseline als Pilot:**

```python
# src/assembled_core/signals/builtin/trend_baseline.py
from datetime import datetime
from ..base import BaseSignal, SignalOutput

class TrendBaselineSignal(BaseSignal):
    """EMA-20/50-Crossover als erste Referenz-Implementation.
    
    Das ist NICHT die finale Strategie, sondern dient als Blaupause
    und Regression gegen die Characterization-Tests.
    """
    name = "trend_baseline"
    version = "1.0.0"
    horizon_days = 10
    required_features = ["ema_20", "ema_50"]
    required_data = ["bars_daily"]
    
    def compute(self, symbol, feature_store, now) -> SignalOutput | None:
        features = feature_store.get(symbol, now, ["ema_20", "ema_50"])
        if features is None or features["ema_20"] is None:
            return None
        
        ema_fast = features["ema_20"]
        ema_slow = features["ema_50"]
        
        score = 1.0 if ema_fast > ema_slow else -1.0 if ema_fast < ema_slow else 0.0
        
        return SignalOutput(
            symbol=symbol,
            score=score,
            confidence=0.3,   # low confidence, it's a baseline
            features_used=["ema_20", "ema_50"],
            horizon_days=self.horizon_days,
            computed_at=now,
        )
```

### 6.2 Woche 8: Dispatcher und Shadow-Mode

**Das Dispatcher-Interface:**

```python
# src/assembled_core/dispatcher.py
from enum import Enum
from dataclasses import dataclass
import json
import logging

log = logging.getLogger(__name__)

class Mode(Enum):
    LEGACY_ONLY = "legacy"      
    SHADOW = "shadow"            # beide, Legacy wins, Diff wird geloggt
    MODERN_ONLY = "modern"       # Plugin-Registry

@dataclass
class DispatcherConfig:
    trend_baseline: Mode = Mode.LEGACY_ONLY
    residual_momentum: Mode = Mode.LEGACY_ONLY
    pead_sue: Mode = Mode.LEGACY_ONLY
    liquidity_index: Mode = Mode.LEGACY_ONLY
    # weitere kommen wöchentlich dazu

class SignalDispatcher:
    def __init__(self, config: DispatcherConfig, registry, legacy_engine):
        self.config = config
        self.registry = registry
        self.legacy = legacy_engine
    
    def run_signal(self, signal_name: str, inputs: dict):
        mode = getattr(self.config, signal_name, Mode.LEGACY_ONLY)
        
        if mode == Mode.LEGACY_ONLY:
            return self.legacy.run(signal_name, inputs)
        
        elif mode == Mode.MODERN_ONLY:
            plugin = self.registry.get(signal_name)
            if plugin is None:
                raise RuntimeError(f"Signal {signal_name} not in registry, but mode=MODERN_ONLY")
            return self._run_plugin(plugin, inputs)
        
        elif mode == Mode.SHADOW:
            legacy_result = self.legacy.run(signal_name, inputs)
            
            plugin = self.registry.get(signal_name)
            if plugin is not None:
                try:
                    modern_result = self._run_plugin(plugin, inputs)
                    self._record_diff(signal_name, inputs, legacy_result, modern_result)
                except Exception as e:
                    log.error(f"Shadow-run for {signal_name} failed: {e}")
            
            return legacy_result  # Legacy gewinnt in SHADOW
    
    def _record_diff(self, name, inputs, legacy_out, modern_out):
        diff = {
            "signal": name,
            "inputs_hash": hash(json.dumps(inputs, sort_keys=True, default=str)),
            "legacy_score": legacy_out.get("score"),
            "modern_score": modern_out.score if modern_out else None,
            "legacy_confidence": legacy_out.get("confidence"),
            "modern_confidence": modern_out.confidence if modern_out else None,
            "delta_score": abs(legacy_out.get("score", 0) - (modern_out.score if modern_out else 0)),
        }
        log.info(f"SHADOW_DIFF {json.dumps(diff)}")
        
        # Auch in Postgres für spätere Analyse
        # db.insert("shadow_diffs", diff)
```

**Shadow-Mode-Analyse-Script:**

```python
# scripts/migration/analyze_shadow_diffs.py
"""
Nach 48-72h SHADOW-Mode: Stimmen Legacy und Plugin im Score überein?

Erwartung: delta_score=0 für deterministische Signale.
Toleranz: 1e-6 für Float-Rounding.
"""
import pandas as pd

diffs = pd.read_sql("SELECT * FROM shadow_diffs WHERE signal='trend_baseline'", conn)
mismatches = diffs[diffs["delta_score"] > 1e-6]

print(f"Total shadow runs: {len(diffs)}")
print(f"Mismatches: {len(mismatches)} ({len(mismatches)/len(diffs):.1%})")

if len(mismatches) > 0:
    print("\nExamples of mismatches:")
    print(mismatches.head(10))
```

**Cutover-Regel:** Wenn `mismatches` < 0.1% über 72h Shadow, und Characterization-Tests weiterhin grün sind, kannst du auf `MODERN_ONLY` umstellen. Dann verschwindet auch der Legacy-Pfad für dieses Signal.

### 6.3 Woche 9-12: Signal-für-Signal migrieren

**Die Reihenfolge ist entscheidend. Start mit kleinsten, endest mit komplexesten.**

| Woche | Signal | Alt-Quelle | Neu in | Aufwand |
|---|---|---|---|---|
| 9 | `trend_baseline` | `pipeline/signals.py::compute_ema_signals` | `builtin/trend_baseline.py` | S (~150 LoC) |
| 10 | `residual_momentum` | `strategies/multifactor_v1.py` | `builtin/residual_momentum.py` | M (~300 LoC) |
| 11 | `liquidity_index` | `features/liquidity_proxy.py` + `trading_cycle.py` Step 8.71 | `builtin/liquidity_index.py` | M (~200 LoC) |
| 12 | `regime_hmm` | `ml/regime_hmm.py` (real genutzt!) | `builtin/regime_hmm.py` | M (~250 LoC) |

**Wichtig:** Reihenfolge vermeidet Dependencies. `regime_hmm` wird von `residual_momentum` benötigt, daher muss `residual_momentum` eine eigene Regime-Version haben oder auf das aktive Regime-Signal zugreifen können. In Phase 3 ist die Reihenfolge so gewählt, dass jedes neue Plugin **auf Features**, aber **nicht auf andere Plugins** angewiesen ist.

**Pro Signal folgt der gleiche Prozess:**

1. Identify: Wo liegt die Logik im Monolithen?
2. Extract: Kopiere relevante Funktionen in ein neues Plugin
3. Refactor: mache Inputs/Outputs klar, eliminiere globale State-Zugriffe
4. Register: in `pyproject.toml` unter `ata.signals`
5. Shadow-Test: `dispatcher.config.<signal> = Mode.SHADOW` für 72h
6. Analyse: Shadow-Diff-Report prüfen
7. Cutover: wenn sauber, `Mode.MODERN_ONLY`
8. Cleanup: Legacy-Code-Pfad entfernen

**Pro Signal geplanter Aufwand:** 2-5 Tage. Mit 10-15h/Woche = 1 Signal pro Woche.

### 6.4 Woche 13-14: Execution-Layer-Migration

Die `accounting/` und `execution/`-Module existieren schon, aber sind observability-wired. Jetzt werden sie aktiviert.

**Aus dem Repo-Check:**
- `execution/intent_store.py` — existiert, ungenutzt
- `execution/order_lifecycle.py` — existiert, ungenutzt
- `execution/kill_switch.py` — existiert, wird teilweise genutzt
- `execution/pre_trade_checks.py` — existiert, ungenutzt
- `accounting/tax_lots.py` — existiert, ungenutzt
- `accounting/reconciliation.py` — existiert, ungenutzt

**Reuse-Entscheidung (nicht neu bauen):**

Alle genannten Module sind **inhaltlich konzeptionell richtig** (das hat der Repo-Check gezeigt). Sie werden nicht neugeschrieben, sondern aktiviert. Konkret:

```python
# src/assembled_core/execution/engine.py (NEU, dünn)
from .intent_store import IntentStore
from .order_lifecycle import OrderLifecycle
from .kill_switch import KillSwitch
from .pre_trade_checks import PreTradeChecks

class ExecutionEngine:
    """Dünne Orchestrator-Klasse.
    
    Statt 2696 Zeilen in unified_paper_engine.py: 
    100 Zeilen Orchestrator + existierende Module.
    """
    def __init__(self, broker, db, config):
        self.intents = IntentStore(db)
        self.lifecycle = OrderLifecycle(db)
        self.kill_switch = KillSwitch(db)
        self.checks = PreTradeChecks(config)
        self.broker = broker
    
    def submit(self, signal_output):
        if not self.kill_switch.guard_entry():
            return "blocked_by_kill_switch"
        
        intent = self.intents.create(signal_output)
        
        passed, reason = self.checks.run(intent)
        if not passed:
            self.intents.reject(intent, reason)
            return f"rejected: {reason}"
        
        order = self.lifecycle.submit(intent, self.broker)
        return order.status
```

**Dann:** `unified_paper_engine.py` (2696 Zeilen) wird deprecated. Code-Pfade, die darauf zeigen, werden auf `ExecutionEngine` umgeroutet. Nach 2 Wochen sauberem Shadow-Run wird `unified_paper_engine.py` gelöscht.

### 6.5 Woche 15-16: `trading_cycle.py` endgültig zerlegen

Nach Woche 14 sollte `trading_cycle.py` von 10.499 auf ~3000-4000 Zeilen geschrumpft sein (Wave-Removals + Signal-Extraktionen). Jetzt der finale Cut.

**Analyse:** Was ist noch drin?

```bash
wc -l src/assembled_core/pipeline/trading_cycle.py
# erwarte ~3500 Zeilen nach Phase 3 Teil 1

# Grobe Kategorisierung mit radon
radon cc src/assembled_core/pipeline/trading_cycle.py -s > /tmp/tc_cc.txt
```

**Typische verbliebene Kategorien:**

1. Daten-Ladung (Bars, News, Makros) — **→** nach `src/assembled_core/data/loader.py`
2. Feature-Berechnung — **→** nach `src/assembled_core/features/pipeline.py`
3. Signal-Aggregation — **→** nach `src/assembled_core/signals/composite.py`
4. Order-Generierung — **→** in `ExecutionEngine`
5. Logging / Monitoring — **→** nach `src/assembled_core/observability/`

Jede Kategorie ist ein Extract-Refactoring à la Feathers (Method Object, Move Method).

**Ziel-Größe für `trading_cycle.py` am Ende:** 100-200 Zeilen, nur noch Orchestrator:

```python
# src/assembled_core/pipeline/trading_cycle.py (nach Phase 3)
def run_trading_cycle(inputs, config):
    data = data_loader.load(inputs)
    features = feature_pipeline.compute(data)
    signals = composite_signal.aggregate(features, registry=signal_registry)
    orders = execution_engine.submit_all(signals)
    return RunResult(data, features, signals, orders)
```

**Go/No-Go nach Phase 3:**

- Alle Characterization-Tests weiterhin grün
- `trading_cycle.py` < 500 Zeilen
- Signal-Registry enthält ≥ 5 Plugins
- Shadow-Mode-Diff < 0.1% für alle aktivierten Signale
- `tach check` passes (keine illegalen Import-Abhängigkeiten mehr)

---

## 7. Phase 4 — CLEANUP (Woche 17-20)

**Ziel:** Legacy-Reste endgültig entfernen, CI-Gates härten.

### 7.1 Woche 17: Archive-Bereinigung

Die `archive/`-Ordner aus Phase 1 sind jetzt 3-4 Monate alt. Was davon wurde nie wieder angefasst?

```bash
# Letzte Git-Aktivität pro archivierter Datei
for f in $(find archive/ -name "*.py"); do
    last=$(git log -1 --format="%ai" -- "$f")
    echo "$last  $f"
done | sort

# Alles, was seit Archivierung (Woche 2) nicht mehr angefasst wurde → löschen
```

Das ist schwer, weil "vielleicht später" ein seduktiver Gedanke ist. Strikte Regel: **3 Monate nicht berührt = löschen**. Wenn du es brauchst, ist es in git-History.

### 7.2 Woche 18: Import-Linter + Tach-Gates

Jetzt, wo die Architektur steht, enforced sie.

**`tach.toml`:**

```toml
# tach.toml
[[modules]]
path = "assembled_core.signals"
depends_on = ["assembled_core.features", "assembled_core.data", "assembled_core.common"]

[[modules]]
path = "assembled_core.execution"
depends_on = ["assembled_core.accounting", "assembled_core.common"]

[[modules]]
path = "assembled_core.accounting"
depends_on = ["assembled_core.common"]

[[modules]]
path = "assembled_core.pipeline"
depends_on = ["assembled_core.signals", "assembled_core.execution", "assembled_core.data"]
```

**`.importlinter`:**

```ini
[importlinter]
root_package = assembled_core

[importlinter:contract:layers]
name = Strict Layer Architecture
type = layers
layers =
    assembled_core.pipeline
    assembled_core.signals
    assembled_core.features
    assembled_core.data
    assembled_core.common

[importlinter:contract:forbidden_legacy]
name = Nobody imports trading_cycle except orchestrator
type = forbidden
source_modules = 
    assembled_core.signals
    assembled_core.features
    assembled_core.execution
    assembled_core.data
forbidden_modules = 
    assembled_core.pipeline.trading_cycle
```

**CI-Integration:**

```yaml
# .github/workflows/architecture.yml
name: Architecture Check

on: [push, pull_request]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install tach import-linter deptry vulture radon
      - run: tach check
      - run: lint-imports
      - run: deptry .
      - run: vulture src/ --min-confidence 90
      - run: radon cc src/ --min C --total-average
```

### 7.3 Woche 19: Pre-Commit-Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  
  - repo: https://github.com/tach-org/tach
    rev: v0.29.0
    hooks:
      - id: tach
  
  - repo: https://github.com/seddonym/import-linter
    rev: v2.1
    hooks:
      - id: import-linter
  
  - repo: https://github.com/fpgmaas/deptry
    rev: 0.20.0
    hooks:
      - id: deptry
  
  - repo: local
    hooks:
      - id: no-large-files
        name: Reject files > 500 lines
        entry: bash -c 'for f in "$@"; do lines=$(wc -l < "$f"); if [ "$lines" -gt 500 ]; then echo "$f has $lines lines (max 500)"; exit 1; fi; done'
        language: system
        files: \.py$
      
      - id: no-wave-wiring
        name: Forbid wave_wiring test files
        entry: bash -c '[ -z "$(ls tests/ | grep wave_wiring)" ]'
        language: system
        pass_filenames: false
      
      - id: commit-must-delete
        name: Migration commit must have net-negative LoC
        entry: bash scripts/check_net_negative.sh
        language: system
        stages: [commit-msg]
```

Der `no-large-files`-Hook ist ein Sicherungsmechanismus: keine Datei > 500 Zeilen mehr zulässig. Wenn Claude in einer zukünftigen AI-Session versucht, wieder ein 10k-Zeilen-File zu bauen, wird der Hook das ablehnen.

### 7.4 Woche 20: Post-Migration-Review

**Das End-Baseline:**

```bash
./scripts/migration/baseline.sh
diff docs/migration/baseline_2026-05-01/ docs/migration/baseline_$(date +%Y-%m-%d)/
```

**Erwartete Diffs:**

| Metrik | Vorher | Nachher | Δ |
|---|---|---|---|
| LoC `src/` | ~155.000 | ~60.000 | -60% |
| Files `src/` | 551 | ~200 | -63% |
| `trading_cycle.py` LoC | 10.499 | ~200 | -98% |
| Wave-Tests | 147 | 0 | -100% |
| Max File LoC | 10.499 | < 500 | — |
| Cyclomatic-Grade C+ Functions | viele | 0 | -100% |
| Vulture Dead-Code (conf 90) | >200 | < 20 | -90% |
| Plugins registriert | 0 | 10-15 | +∞ |

Wenn du nur 30% der Ziel-Reduktion erreichst, ist das **immer noch ein Erfolg**. Perfekt ist der Feind von gut.

---

## 8. AI-Disziplin-Spezifika

### 8.1 Der Prompt-Rahmen für Migrations-Sessions

Jede Claude-Session während der Migration beginnt mit diesem Kontext:

```markdown
## Context für diese Session

Ich arbeite an der Migration von Assembled-Trading-AI (ein Python-Trading-System) 
von einem 10k-Zeilen-Monolith zu einer Plugin-Architektur. Wir sind in 
Phase X (STRANGLE / CLEANUP / ...).

## Regeln für diese Session

1. **REMOVE oder REPLACE, nicht ADD**: Wenn du mir einen Commit vorschlägst, 
   muss er entweder Zeilen löschen oder atomar ersetzen. Keine neuen Stub-Dateien.

2. **Keine observability-wired Pattern**: Wenn du einen try/except mit log.debug 
   einbauen willst, widerstehe. Entweder die Funktion funktioniert, oder sie 
   wird ganz entfernt.

3. **Ein Signal pro Session**: Wir migrieren genau ein Signal (z.B. trend_baseline). 
   Keine Erweiterungen auf andere Signale "quick win".

4. **Characterization-Tests sind heilig**: Wenn eine Änderung einen Test rot macht,
   wird sie rückgängig gemacht oder der Test wird bewusst re-approved.

5. **Neue Dateien brauchen Rechtfertigung**: Wenn du eine neue Datei erzeugen 
   willst, erkläre WARUM sie nicht in eine bestehende Datei gehört.

6. **LoC-Budget**: Diese Session hat ein Netto-Negative-LoC-Budget. Am Ende der
   Session muss `git diff --shortstat` zeigen: mehr Deletions als Additions.

## Aktueller Task

<konkret, 1-2 Sätze>

## Was NICHT geändert werden soll

<Liste von Dateien, die diese Session nicht anfassen>
```

### 8.2 Red Flags bei AI-generiertem Refactoring-Code

Diese Muster zeigen: die AI hat das "Refactoring"-Ziel nicht verstanden.

**Red Flag 1: Neue `_v2`-Dateien**

```python
# Ablehnen:
# neue Datei: src/assembled_core/signals/trend_baseline_v2.py
# während: src/assembled_core/signals/trend_baseline.py weiter existiert
```

Refactoring ersetzt, es parallelisiert nicht.

**Red Flag 2: "Enhanced" Wrapper-Funktionen**

```python
# Ablehnen:
def enhanced_compute_signal(symbol, features):
    """Wrapper around compute_signal with additional features."""
    base = compute_signal(symbol, features)
    # ... mehr Code, der die Basis erweitert
```

Das ist Addition getarnt als Refactoring.

**Red Flag 3: Try-Except um neue Imports**

```python
# Ablehnen:
try:
    from .new_module import do_thing
    _HAS_NEW_MODULE = True
except ImportError:
    _HAS_NEW_MODULE = False
    def do_thing(*args): pass
```

Das ist neues Wave-Wiring. Entweder der Import funktioniert (dann kein Try), oder er funktioniert nicht (dann repariere das Problem).

**Red Flag 4: `# TODO: implement later`-Kommentare**

Jeder TODO-Kommentar in neuem Code ist ein Riecher. Entweder fertig oder gar nicht.

**Red Flag 5: Erweiterung der `requirements.txt` um Libraries "die nützlich sein könnten"**

Jede neue Library kostet Maintenance-Budget. Nur hinzufügen, wenn sie **heute** gebraucht wird.

### 8.3 Der Sanity-Check nach jeder AI-Session

```bash
# 1. LoC-Diff
git diff --shortstat main
# erwarte: Deletions >= Additions * 1.2 (20% Netto-Reduktion)

# 2. Neue Dateien?
git status --porcelain | grep "^A " 
# erwarte: max 1-2 neue Dateien, jede mit klarer Existenzberechtigung

# 3. Neue Try-Except-Blöcke?
git diff | grep -c "^+.*except Exception"
# erwarte: 0 bis 1

# 4. Neue Stubs?
git diff | grep -c "pass$"
# erwarte: sehr wenig

# 5. Characterization-Tests?
pytest tests/characterization/ -v
# erwarte: alle grün

# 6. Tach-Check?
tach check
# erwarte: passes
```

Wenn eine dieser Checks fehlschlägt: nicht committen, Session ist gescheitert, zurückrollen.

---

## 9. Risiken und Mitigations

### 9.1 Risiko: "Ich refactore 6 Monate und produziere keinen Value"

**Mitigation:**
- Parallel-Running: das alte System läuft weiter im Paper-Modus. Kein Value-Verlust.
- Wöchentliche Metric-Updates (LoC, Complexity, Dead-Code-Rate). Das ist auch ohne Feature-Output Progress.
- Nach Phase 3 (Woche 16) sind neue Features (Plugin-Entry-Points) trivial hinzufügbar. Value-Anstieg ist hinten.

### 9.2 Risiko: "Neuer Code ist heimlich schlechter als alter"

**Mitigation:**
- Characterization-Tests zeigen Verhaltens-Abweichungen.
- Shadow-Mode zeigt Score-Abweichungen in Live-Daten.
- Pre-Commit-Hooks (radon min-grade, vulture, tach) verhindern Re-Regression.

### 9.3 Risiko: "Live-Paper-System bricht während Migration"

**Mitigation:**
- Phase 1 ist nur DELETE — kann den Live-Pfad nur verbessern, nicht verschlechtern (wenn Tests grün).
- Phase 2 baut nur Tests, berührt Live nicht.
- Phase 3 Shadow-Mode default: Legacy wins. Modern läuft parallel, aber hat keinen P&L-Effekt.
- Nur der Cutover auf `MODERN_ONLY` pro Signal ist riskant, und nur nach 72h sauberem Shadow.

### 9.4 Risiko: "AI baut während Migration wieder Wave-Wiring"

**Mitigation:**
- `.pre-commit-config.yaml` mit `no-wave-wiring`-Hook (siehe §7.3).
- Prompt-Rahmen explizit (§8.1).
- Wöchentlicher Sanity-Check auf Commit-Muster (`git log --stat --since="1 week ago"`).

### 9.5 Risiko: "Wir vergessen, warum etwas so gebaut war"

**Mitigation:**
- `docs/decisions/YYYY-MM-DD_<titel>.md` nach jedem nicht-trivialen Architektur-Entscheidung (aus Gap-Analyse Neu-3).
- Commit-Messages mit Kontext, nicht nur "refactor X".
- Post-Mortem nach jeder Phase.

### 9.6 Risiko: "Ich habe kein Zeit mehr und breche ab"

**Mitigation:**
- Jede Phase produziert einen **abgeschlossenen Zustand**. Wenn du nach Phase 1 abbrichst, hast du schon ein deutlich kleineres Repo.
- Phase 3 ist die längste, aber jedes migrierte Signal ist einzeln wertvoll. Du musst nicht alle migrieren.
- Die Plugin-Infrastruktur aus Woche 7-8 ist auch dann nützlich, wenn du nur 2 Signale migrierst. Sie erlaubt dir **zukünftige** Signale sauber zu bauen.

---

## 10. Metriken-Dashboard

### 10.1 Wöchentliches Tracking

```bash
# scripts/migration/weekly_metrics.sh
#!/usr/bin/env bash
# Lauft jeden Sonntag. Output in docs/migration/weekly.csv

WEEK=$(date +%Y-W%V)
METRICS_CSV="docs/migration/weekly_metrics.csv"

# Headers wenn nicht existiert
if [ ! -f "$METRICS_CSV" ]; then
    echo "week,loc_src,files_src,trading_cycle_loc,wave_tests,vulture_conf80,tach_violations,plugins_registered" > "$METRICS_CSV"
fi

LOC_SRC=$(find src/ -name "*.py" -exec wc -l {} \; | awk '{sum+=$1} END {print sum}')
FILES_SRC=$(find src/ -name "*.py" | wc -l)
TC_LOC=$(wc -l < src/assembled_core/pipeline/trading_cycle.py)
WAVE=$(find tests/ -name "test_wave*_wiring.py" | wc -l)
VULTURE=$(vulture src/ --min-confidence 80 2>/dev/null | wc -l)
TACH=$(tach check 2>&1 | grep -c "violation" || echo 0)
PLUGINS=$(python -c "from importlib.metadata import entry_points; print(len(list(entry_points(group='ata.signals'))))")

echo "$WEEK,$LOC_SRC,$FILES_SRC,$TC_LOC,$WAVE,$VULTURE,$TACH,$PLUGINS" >> "$METRICS_CSV"
```

### 10.2 Visualisierung

```python
# scripts/migration/plot_progress.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("docs/migration/weekly_metrics.csv")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0,0].plot(df["week"], df["trading_cycle_loc"])
axes[0,0].set_title("trading_cycle.py LoC")
axes[0,0].axhline(500, color="g", linestyle="--", label="Target")
axes[0,0].axhline(10499, color="r", linestyle="--", label="Start")
axes[0,0].legend()

axes[0,1].plot(df["week"], df["wave_tests"])
axes[0,1].set_title("Wave-Wiring Tests")

axes[1,0].plot(df["week"], df["vulture_conf80"])
axes[1,0].set_title("Dead-Code (Vulture conf 80)")

axes[1,1].plot(df["week"], df["plugins_registered"])
axes[1,1].set_title("Plugins Registered")

plt.tight_layout()
plt.savefig("docs/migration/progress.png", dpi=100)
```

Das PNG wird Teil der wöchentlichen Selbst-Kontrolle. Wenn die Kurven flach sind, hast du eine Woche verloren.

---

## 11. Quellen

Diese Empfehlungen basieren auf:

**Strangler-Fig-Pattern:**
- Martin Fowler (2004): [Strangler Fig Application](https://martinfowler.com/bliki/StranglerFigApplication.html)
- Chris Richardson, microservices.io: [Pattern: Strangler application](https://microservices.io/patterns/refactoring/strangler-application.html)
- Heurekadevs (2025): [How to Kill a Monolith Without Killing Yourself](https://www.heurekadevs.com/how-to-kill-a-monolith-without-killing-yourself-the-strangler-fig-pattern-in-practice) — besonders die "Don't half-strangle"-Warnung
- MaibornWolff (2026): [Strangler Fig Pattern](https://www.maibornwolff.de/en/know-how/strangler-pattern/)

**Characterization / Approval Testing:**
- Michael Feathers, "Working Effectively with Legacy Code" (2004)
- Nicolas Carlo, understandlegacycode.com: [Approval Tests](https://understandlegacycode.com/approval-tests/), [Difference Characterization vs Approval](https://understandlegacycode.com/blog/characterization-tests-or-approval-tests/)
- Mario Cervera (2025): [Characterization testing](https://mariocervera.com/characterization-testing-adding-tests-to-legacy-code)
- ApprovalTests.Python auf PyPI
- Qodo (2025): [Automate Approval Testing](https://www.qodo.ai/blog/automate-approval-testing/)

**Dead-Code-Detection:**
- [Vulture](https://github.com/jendrikseipp/vulture) — Jendrik Seipp, aktiv, 2026 maintained
- [deadcode](https://github.com/albertas/deadcode) — Albertas, EuroPython 2024 Presentation
- CleanAI.pro (2026): [How to Find Dead Code in Python with Vulture](https://www.cleanai.pro/blog/find-dead-code-python-vulture)

**Module-Boundaries:**
- [Tach](https://github.com/tach-org/tach) — Rust-implementiert, empfohlen für große Codebases
- [Import-Linter](https://import-linter.readthedocs.io/) — Kedro-Projekt nutzt es in Production

**Plugin-Architektur:**
- [Python Packaging User Guide: Creating and discovering plugins](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/)
- John Raines (2023): [Level Up in Python with Dependency Inversion and Entry Points](https://johndanielraines.medium.com/level-up-in-python-with-dependency-inversion-and-entry-points-c648a1b087ee)
- [setuptools Entry Points](https://setuptools.pypa.io/en/stable/userguide/entry_point.html)

**Komplexitäts-Metriken:**
- [Radon](https://radon.readthedocs.io/) — Standard-Tool für CC/MI in Python
- [Wily](https://wily.readthedocs.io/) — Git-History-basierte Complexity-Trends
- Penify.dev (2025): [Python Code Complexity Checkers Comparison](https://blogs.penify.dev/docs/python-code-complexity-checkers-comparison.html)

**AI-Code-Anti-Patterns:**
- InfoWorld (2025): [Refactoring AI code](https://www.infoworld.com/article/3610521/refactoring-ai-code-the-good-the-bad-and-the-weird.html)
- SoftwareSeni (2025): [Anti-Patterns and Quality Degradation in AI-Generated Code](https://www.softwareseni.com/understanding-anti-patterns-and-quality-degradation-in-ai-generated-code/) — "AI implements prompts directly without considering refactoring opportunities"
- Atlassian (2025): [How to effectively utilise AI to enhance large-scale refactoring](https://www.atlassian.com/blog/developer/how-to-effectively-utilise-ai-to-enhance-large-scale-refactoring) — 1400+ Files Fallstudie
- getdx.com (2025): [Enterprise AI Refactoring Best Practices](https://getdx.com/blog/enterprise-ai-refactoring-best-practices/) — Quality-Gates reduzieren Post-Deployment-Issues um 70%

---

## 12. Zusammenfassung auf einer Seite

**Dein Ausgangspunkt (2026-04-24):**
- 702 Commits, 0 Refactor-Commits
- `trading_cycle.py`: 10.499 Zeilen, 309 Steps
- 147 Wave-Wiring-Tests (Scheintests)
- 52+ observability-wired ML-Module
- Repo wächst noch, statt zu schrumpfen

**Dein Ziel in 4-6 Monaten:**
- `trading_cycle.py` < 500 Zeilen (nur Orchestrator)
- 0 Wave-Wiring-Tests
- 10-15 Plugins via entry_points registriert
- Alle Characterization-Tests grün über 2008/2020/2022-Szenarien
- CI-Gates verhindern Re-Regression

**Die vier Phasen:**
1. **DELETE** (Woche 1-3): Wave-Tests, observability-wired ML-Module, tote Scripts
2. **CHARACTERIZE** (Woche 4-6): Golden-Equity + 5 weitere Approval-Tests, Scenario-Tests
3. **STRANGLE** (Woche 7-16): Plugin-Infrastruktur + Signal-für-Signal-Migration + Shadow-Mode
4. **CLEANUP** (Woche 17-20): Archive leeren, Tach/Import-Linter/Pre-Commit-Gates

**Die drei Regeln:**
1. REMOVE oder REPLACE — niemals ADD während Migration
2. Characterization-Tests sind heilig — jede Änderung, die einen Test rot macht, ist zu begründen oder zurückzurollen
3. Shadow-Mode vor Cutover — 72h Parallelbetrieb, dann erst Legacy abschalten

**Ehrlicher Hinweis:**
Das Playbook ist ambitioniert. Wenn du 70% davon schaffst, bist du trotzdem in einer komplett anderen Liga. Die wichtigsten drei Dinge, die du auf keinen Fall auslassen darfst:
- **Characterization-Tests** (sonst refactorst du blind)
- **Shadow-Mode** (sonst Live-Risiko)
- **Pre-Commit-Hooks** (sonst entsteht das Chaos neu)

Alles andere ist optimizable. Diese drei nicht.
