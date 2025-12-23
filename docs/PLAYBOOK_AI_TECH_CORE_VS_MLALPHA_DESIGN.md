# AI/Tech Strategy Playbook - Core vs Core+ML vs ML Only

**Phase:** Advanced Analytics & Factor Labs  
**Status:** Design Phase  
**Last Updated:** 2025-01-XX

---

## 1. Overview & Goals

**Ziel:** Standardisierte Experimente fur AI/Tech-Strategien:

- **Core-Only** vs. **Core+ML** vs. **ML-Only** Setups systematisch vergleichen
- Optional Walk-Forward-Analyse fur Out-of-Sample-Validierung
- Optional Regime-Performance-Analyse (Bull/Bear/Crisis)
- Factor Exposures & Deflated Sharpe aus Risk-/Factor-Reports aggregieren
- Kein Eingriff in Finanzlogik, nur Orchestrierung & Auswertung

**Motivation:**

AI/Tech-Strategien werden typischerweise in drei Varianten evaluiert:

1. **Core-Only**: Nur klassische Faktoren (Momentum, Trend Strength, Volatility, etc.)
2. **Core+ML**: Core-Faktoren kombiniert mit ML-Alpha-Faktoren (z.B. Ridge/Lasso/Random Forest Predictions)
3. **ML-Only**: Nur ML-Alpha-Faktoren (reine ML-Signale)

Das Playbook orchestriert Batch-Backtests, Risk-Reports, Walk-Forward-Analysen und Regime-Analysen fur alle drei Setups und aggregiert die Ergebnisse in einer ubersichtlichen Zusammenfassung.

**Bezug zu bestehenden Modulen:**

- Baut auf `scripts/batch_backtest.py` auf (Batch-Backtest-Orchestrierung)
- Nutzt `scripts/generate_risk_report.py` (Risk-Reports mit Factor-Exposures)
- Optional: `scripts/run_walk_forward_analysis.py` (Walk-Forward-Analyse)
- Optional: `scripts/generate_tca_report.py` (TCA-Reports)
- Reuse von Config-Strukturen aus `docs/WORKFLOWS_BATCH_BACKTESTS_AND_PARALLELIZATION.md`

**Datenbasis:** Alle Analysen basieren auf **lokalen Backtest-Outputs** und Factor-Returns. Keine Live-APIs.

---

## 2. Experiment Types

Das Playbook unterstutzt mindestens folgende Szenarien:

### E1: Plain Backtest (IS+OOS) fur alle drei Setups

**Ziel:** Standard-Backtests fur Core, Core+ML und ML-Only Setups uber denselben Zeitraum (z.B. 2015-2020).

**Outputs:**

- Drei separate Backtest-Verzeichnisse (pro Setup)
- Performance-Reports (Sharpe, Sortino, Max Drawdown, Total Return, etc.)
- Risk-Reports (optional: mit Factor-Exposures)
- TCA-Reports (optional)

**Aggregation:**

- Vergleichstabelle: Core vs. Core+ML vs. ML-Only (Metriken, Deflated Sharpe, Factor Exposures)
- Markdown-Report mit Zusammenfassung

### E2: Walk-Forward-Analyse (Rolling, z.B. 1Y Train, 3M Test)

**Ziel:** Out-of-Sample-Validierung durch zeitfenster-basierte Train/Test-Splits.

**Configuration:**

- Train-Window: z.B. 1 Jahr (252 Trading Days)
- Test-Window: z.B. 3 Monate (63 Trading Days)
- Rolling vs. Expanding Window
- Step-Size: z.B. 3 Monate (test_size)

**Outputs:**

- Walk-Forward-Results pro Setup (IS/OOS-Metriken pro Split)
- Aggregierte Metriken uber alle Splits (Mean, Std, Min/Max)
- Equity-Kurven pro Split (optional)
- Vergleich: Core vs. Core+ML vs. ML-Only (OOS-Performance)

**Integration:**

- Nutzt `scripts/run_walk_forward_analysis.py` oder programmatische API
- Verweise auf `docs/WALK_FORWARD_AND_REGIME_B3_DESIGN.md`

### E3: Regime-Performance-Analyse (Bull/Bear/Crisis)

**Ziel:** Evaluation der Strategie-Performance in verschiedenen Marktregimen.

**Inputs:**

- Equity-Kurven aus Backtests
- Regime-State-Datei (z.B. aus `scripts/generate_risk_report.py --enable-regime-analysis`)
- Optional: Benchmark-Symbol oder Benchmark-File

**Outputs:**

- Metriken pro Regime (Sharpe, Sortino, Max DD, Win Rate, etc.)
- Anzahl Perioden pro Regime
- Relative Performance vs. Benchmark (optional)
- Vergleich: Core vs. Core+ML vs. ML-Only (Regime-spezifische Performance)

**Integration:**

- Nutzt Regime-Analyse aus Risk-Reports oder `docs/WALK_FORWARD_AND_REGIME_B3_DESIGN.md`
- Benchmark-Klassifikation (z.B. SPY fur Bull/Bear/Crisis)

### E4: Factor Exposures & Deflated Sharpe aus Risk-/Factor-Reports

**Ziel:** Attribution der Strategie-Returns auf Factor-Returns und Multiple-Testing-Adjustment.

**Inputs:**

- Risk-Reports mit Factor-Exposures (aus `scripts/generate_risk_report.py --enable-factor-exposures`)
- Factor-Returns-Datei (z.B. `output/factor_returns/factor_returns.parquet`)
- Optional: Factor-Analysis-Reports (IC/IR, Deflated Sharpe)

**Outputs:**

- Top-5-Faktor-Exposures pro Setup (Mean Beta, Std Beta, R2)
- Deflated Sharpe Ratios (bereinigt um Multiple Testing)
- Vergleich: Core vs. Core+ML vs. ML-Only (Faktor-Profil, Robustheit)

**Integration:**

- Nutzt Factor-Exposures aus `docs/SIGNAL_API_AND_FACTOR_EXPOSURES_A2_DESIGN.md`
- Nutzt Deflated Sharpe aus `docs/DEFLATED_SHARPE_B4_DESIGN.md`

---

## 3. Inputs & Config Contracts

### 3.1 Batch-Config (YAML/JSON)

Das Playbook erwartet eine Batch-Config-Datei im Format von `docs/WORKFLOWS_BATCH_BACKTESTS_AND_PARALLELIZATION.md`.

**Beispiel-Struktur:**

```yaml
batch_name: ai_tech_core_vs_ml_2015_2020
description: "Compare AI/Tech core bundle vs. ML bundles on 2015-2020"
output_root: "output/batch_backtests"

defaults:
  freq: "1d"
  data_source: "local"
  strategy: "multifactor_long_short"
  rebalance_freq: "M"
  max_gross_exposure: 1.0
  start_capital: 100000.0
  generate_report: true
  generate_risk_report: true
  generate_tca_report: false
  symbols_file: "config/universe_ai_tech_tickers.txt"

runs:
  - id: "core_2015_2020"
    bundle_path: "config/factor_bundles/ai_tech_core_bundle.yaml"
    start_date: "2015-01-01"
    end_date: "2020-12-31"
    generate_risk_report: true
    generate_tca_report: false

  - id: "core_ml_2015_2020"
    bundle_path: "config/factor_bundles/ai_tech_core_ml_bundle.yaml"
    start_date: "2015-01-01"
    end_date: "2020-12-31"
    generate_risk_report: true
    generate_tca_report: false

  - id: "ml_only_2015_2020"
    bundle_path: "config/factor_bundles/ai_tech_ml_alpha_bundle.yaml"
    start_date: "2015-01-01"
    end_date: "2020-12-31"
    generate_risk_report: true
    generate_tca_report: false
```

**Wichtige Felder:**

- **batch_name**: Kurzname des Batches
- **description**: Freitext-Beschreibung
- **output_root**: Basis-Output-Verzeichnis
- **runs**: Liste von Run-Configs (mindestens 3: core, core_ml, ml_only)
- **run.id**: Eindeutiger Run-Name (wird zur Identifikation verwendet)

**Run-Identifikation:**

- Runs werden anhand der `id` identifiziert
- Empfohlene Naming-Convention: `{setup}_{start_date}_{end_date}` (z.B. `core_2015_2020`, `core_ml_2015_2020`, `ml_only_2015_2020`)
- Das Playbook erwartet mindestens drei Runs: einen mit "core", einen mit "core_ml" oder "ml", einen mit "ml_only" oder "mlalpha"

### 3.2 Playbook-Config (Python Dataclass)

Zusatzlich zur Batch-Config erwartet das Playbook eine Playbook-spezifische Config:

**Pfade:**

- `batch_config`: Pfad zur Batch-Config-Datei (YAML/JSON)
- `factor_returns_file`: Pfad zur Factor-Returns-Datei (fur Factor-Exposures)
- `benchmark_symbol`: Benchmark-Symbol (z.B. "SPY") oder `None`
- `benchmark_file`: Pfad zur Benchmark-Datei (CSV/Parquet) oder `None`
- `output_root`: Basis-Output-Verzeichnis fur Playbook-Outputs (default: `output/ai_tech_playbook/`)

**Flags:**

- `enable_walk_forward`: Boolean (default: False)
- `enable_regime_analysis`: Boolean (default: False)
- `enable_factor_exposures`: Boolean (default: True, wenn `factor_returns_file` gesetzt)

**Walk-Forward-Config (optional):**

- `walk_forward_train_size_days`: Integer (default: 252)
- `walk_forward_test_size_days`: Integer (default: 63)
- `walk_forward_step_size_days`: Integer (default: 63, kann auch None sein fur Rolling)
- `walk_forward_min_train_periods`: Integer (default: 252)
- `walk_forward_mode`: "rolling" oder "expanding" (default: "rolling")

**Regime-Config (optional):**

- `regime_file`: Pfad zur Regime-State-Datei (optional, wird aus Risk-Reports gelesen falls vorhanden)
- `enable_regime_benchmark`: Boolean (default: True, wenn `benchmark_symbol` oder `benchmark_file` gesetzt)

---

## 4. Playbook API Design

### 4.1 Dataclasses

```python
@dataclass
class AiTechPlaybookConfig:
    """Configuration for AI/Tech Strategy Playbook.
    
    Attributes:
        batch_config: Path to batch config file (YAML/JSON)
        factor_returns_file: Optional path to factor returns file (for factor exposures)
        benchmark_symbol: Optional benchmark symbol (e.g., "SPY")
        benchmark_file: Optional path to benchmark file (CSV/Parquet with timestamp, returns/close)
        output_root: Output root directory for playbook outputs (default: output/ai_tech_playbook/)
        enable_walk_forward: Enable walk-forward analysis (default: False)
        enable_regime_analysis: Enable regime-based performance analysis (default: False)
        enable_factor_exposures: Enable factor exposure analysis (default: True if factor_returns_file is set)
        walk_forward_train_size_days: Training window size in days (default: 252)
        walk_forward_test_size_days: Test window size in days (default: 63)
        walk_forward_step_size_days: Step size for rolling window (default: 63, None for non-overlapping)
        walk_forward_min_train_periods: Minimum training periods (default: 252)
        walk_forward_mode: "rolling" or "expanding" (default: "rolling")
        regime_file: Optional path to regime state file (will be read from risk reports if available)
        enable_regime_benchmark: Enable benchmark-based regime classification (default: True if benchmark provided)
    """
    
    batch_config: Path
    factor_returns_file: Path | None = None
    benchmark_symbol: str | None = None
    benchmark_file: Path | None = None
    output_root: Path = Path("output/ai_tech_playbook/")
    enable_walk_forward: bool = False
    enable_regime_analysis: bool = False
    enable_factor_exposures: bool = True
    walk_forward_train_size_days: int = 252
    walk_forward_test_size_days: int = 63
    walk_forward_step_size_days: int | None = 63
    walk_forward_min_train_periods: int = 252
    walk_forward_mode: Literal["rolling", "expanding"] = "rolling"
    regime_file: Path | None = None
    enable_regime_benchmark: bool = True


@dataclass
class SetupResult:
    """Result for a single setup (core, core_ml, ml_only).
    
    Attributes:
        setup_name: Name of setup ("core", "core_ml", "ml_only")
        run_id: Run ID from batch config
        backtest_dir: Path to backtest output directory
        risk_report_dir: Path to risk report directory (if available)
        tca_report_dir: Path to TCA report directory (if available)
        metrics: Dictionary with performance metrics (from risk report or backtest)
        deflated_sharpe: Deflated Sharpe ratio (if available)
        factor_exposures_summary: DataFrame with factor exposures summary (if available)
        walk_forward_results: Optional WalkForwardResult (if enabled)
        regime_performance: Optional DataFrame with regime performance (if enabled)
    """
    
    setup_name: str
    run_id: str
    backtest_dir: Path
    risk_report_dir: Path | None = None
    tca_report_dir: Path | None = None
    metrics: dict[str, float | int] | None = None
    deflated_sharpe: float | None = None
    factor_exposures_summary: pd.DataFrame | None = None
    walk_forward_results: Any | None = None  # WalkForwardResult from walk_forward module
    regime_performance: pd.DataFrame | None = None


@dataclass
class PlaybookResult:
    """Aggregated result of AI/Tech playbook execution.
    
    Attributes:
        timestamp: Execution timestamp
        batch_name: Name of batch config
        setup_results: List of SetupResult (one per setup: core, core_ml, ml_only)
        experiment_summary: DataFrame with comparison of all setups
        output_dir: Output directory for playbook results
    """
    
    timestamp: pd.Timestamp
    batch_name: str
    setup_results: list[SetupResult]
    experiment_summary: pd.DataFrame
    output_dir: Path
```

### 4.2 Core Function

```python
def run_ai_tech_playbook(cfg: AiTechPlaybookConfig) -> PlaybookResult:
    """
    Run AI/Tech Strategy Playbook.
    
    Orchestrates:
    1. Batch backtests (via batch_backtest.py or programmatic API)
    2. Risk reports (with optional factor exposures and regime analysis)
    3. Optional walk-forward analysis
    4. Aggregates results into summary
    
    Args:
        cfg: AiTechPlaybookConfig instance
        
    Returns:
        PlaybookResult with aggregated results
    """
    # Step 1: Load batch config
    from scripts.batch_backtest import load_batch_config
    batch_cfg = load_batch_config(cfg.batch_config)
    
    # Step 2: Run batch backtests (or read existing results)
    # Option A: Run via CLI (subprocess)
    # Option B: Run programmatically (call batch_backtest.run_batch)
    
    # Step 3: For each run, generate risk reports (if not already generated)
    # - Enable factor exposures if cfg.enable_factor_exposures
    # - Enable regime analysis if cfg.enable_regime_analysis
    
    # Step 4: Optional walk-forward analysis (if enabled)
    # - Run walk-forward for each setup
    # - Aggregate OOS metrics
    
    # Step 5: Aggregate results
    # - Load metrics from risk reports
    # - Load factor exposures summaries
    # - Load deflated Sharpe (if available in factor reports)
    # - Create comparison DataFrame
    
    # Step 6: Generate summary reports
    # - experiment_summary.csv
    # - experiment_summary.md
    
    # Step 7: Return PlaybookResult
    pass
```

**Implementation Details:**

- **Batch-Backtest-Execution**: Kann via CLI (subprocess) oder programmatisch erfolgen
- **Risk-Report-Generation**: Nutzt `scripts/generate_risk_report.py` mit Flags fur Factor-Exposures und Regime-Analyse
- **Walk-Forward-Integration**: Nutzt `scripts/run_walk_forward_analysis.py` oder programmatische API
- **Result-Aggregation**: Liest Metriken aus Risk-Reports, Factor-Exposures-Summaries, Walk-Forward-Results
- **Summary-Generation**: Erstellt CSV und Markdown-Reports mit Vergleichstabellen

---

## 5. CLI Design

### 5.1 Standalone Script

**Neues Script:** `scripts/run_ai_tech_playbook.py`

**CLI-Parameter:**

```bash
python scripts/run_ai_tech_playbook.py \
    --batch-config configs/batch_backtests/ai_tech_core_vs_ml_2015_2020.yaml \
    --output-dir output/ai_tech_playbook/experiment_20250101 \
    --factor-returns-file output/factor_returns/factor_returns.parquet \
    --benchmark-symbol SPY \
    --with-walk-forward \
    --walk-forward-train-size 252 \
    --walk-forward-test-size 63 \
    --with-regime \
    --with-factor-exposures
```

**Argumente:**

- `--batch-config`: Pfad zur Batch-Config-Datei (required)
- `--output-dir`: Output-Verzeichnis fur Playbook-Results (default: `output/ai_tech_playbook/<timestamp>/`)
- `--factor-returns-file`: Pfad zur Factor-Returns-Datei (optional, fur Factor-Exposures)
- `--benchmark-symbol`: Benchmark-Symbol (z.B. "SPY", optional)
- `--benchmark-file`: Pfad zur Benchmark-Datei (optional, Alternative zu `--benchmark-symbol`)
- `--with-walk-forward`: Aktiviert Walk-Forward-Analyse (flag)
- `--walk-forward-train-size`: Train-Window-Groesse in Tagen (default: 252)
- `--walk-forward-test-size`: Test-Window-Groesse in Tagen (default: 63)
- `--walk-forward-step-size`: Step-Size in Tagen (default: 63, None fur non-overlapping)
- `--walk-forward-mode`: "rolling" oder "expanding" (default: "rolling")
- `--with-regime`: Aktiviert Regime-Analyse (flag)
- `--regime-file`: Pfad zur Regime-State-Datei (optional, wird aus Risk-Reports gelesen falls vorhanden)
- `--with-factor-exposures`: Aktiviert Factor-Exposures-Analyse (flag, default: True wenn `--factor-returns-file` gesetzt)
- `--skip-batch-run`: Uberspringt Batch-Backtest-Execution (nutzt existierende Ergebnisse)

### 5.2 CLI Integration

**Integration in `scripts/cli.py` als Subcommand `ai_tech_playbook`:**

```bash
python scripts/cli.py ai_tech_playbook \
    --batch-config configs/batch_backtests/ai_tech_core_vs_ml_2015_2020.yaml \
    --with-walk-forward \
    --with-regime
```

**Subcommand-Handler:**

```python
def ai_tech_playbook_subcommand(args: argparse.Namespace) -> int:
    """Run AI/Tech Strategy Playbook subcommand."""
    from scripts.run_ai_tech_playbook import run_ai_tech_playbook_from_args
    
    try:
        return run_ai_tech_playbook_from_args(args)
    except Exception as e:
        logger.error(f"AI/Tech playbook failed: {e}", exc_info=True)
        return 1
```

---

## 6. Outputs

### 6.1 Output-Struktur

**Basis-Pfad:** `output/ai_tech_playbook/<timestamp or label>/`

**Dateien:**

- `experiment_summary.csv`: Vergleichstabelle mit Metriken, Deflated Sharpe, Regime-Metriken pro Setup
- `experiment_summary.md`: Lesbare Zusammenfassung mit Tabellen, Grafiken-Links, Interpretationen
- `playbook_config.json`: Serialisierte Playbook-Config (fur Reproduzierbarkeit)
- `playbook_result.json`: Serialisierte PlaybookResult (optional, fur programmatische Weiterverarbeitung)

**Sub-Verzeichnisse:**

- `backtests/`: Symlinks oder Verweise auf Batch-Backtest-Output-Verzeichnisse
- `risk_reports/`: Symlinks oder Verweise auf Risk-Report-Verzeichnisse
- `walk_forward/`: Walk-Forward-Results (wenn aktiviert)
- `regime_analysis/`: Regime-Performance-Reports (wenn aktiviert)

### 6.2 experiment_summary.csv

**Spalten:**

- `setup`: "core", "core_ml", "ml_only"
- `run_id`: Run ID aus Batch-Config
- `total_return`: Total Return (pct)
- `cagr`: CAGR (pct)
- `sharpe_ratio`: Sharpe Ratio (annualized)
- `sortino_ratio`: Sortino Ratio (annualized)
- `max_drawdown_pct`: Max Drawdown (pct)
- `volatility`: Annualized Volatility (pct)
- `deflated_sharpe`: Deflated Sharpe Ratio (wenn verfugbar)
- `n_tests`: Number of tests fur Deflated Sharpe (wenn verfugbar)
- `mean_factor_r2`: Mean R2 von Factor-Exposures (wenn verfugbar)
- `top_factor_1`: Top Factor Exposure (wenn verfugbar)
- `top_factor_1_beta`: Top Factor Beta (wenn verfugbar)
- `oos_sharpe_mean`: Mean OOS Sharpe (Walk-Forward, wenn aktiviert)
- `oos_sharpe_std`: Std OOS Sharpe (Walk-Forward, wenn aktiviert)
- `bull_sharpe`: Sharpe in Bull-Regime (wenn aktiviert)
- `bear_sharpe`: Sharpe in Bear-Regime (wenn aktiviert)
- `crisis_sharpe`: Sharpe in Crisis-Regime (wenn aktiviert)

### 6.3 experiment_summary.md

**Struktur:**

1. **Header**: Experiment-Name, Timestamp, Config-Info
2. **Executive Summary**: Kurze Zusammenfassung (Top-Performer, Hauptunterschiede)
3. **Performance Comparison**: Tabelle mit Metriken (aus CSV)
4. **Factor Exposures**: Top-5-Faktoren pro Setup (wenn verfugbar)
5. **Deflated Sharpe Analysis**: Vergleich der Deflated Sharpe Ratios (wenn verfugbar)
6. **Walk-Forward Results**: OOS-Performance-Ubersicht (wenn aktiviert)
7. **Regime Performance**: Performance nach Regime (wenn aktiviert)
8. **Links**: Verweise auf Backtest-Verzeichnisse, Risk-Reports, Walk-Forward-Results

**Formatierung:**

- Markdown-Tabellen fur Metriken-Vergleiche
- Code-Blocks fur Pfade
- Links zu detaillierten Reports

---

## 7. Implementation Plan

### P1: API & Config + Skeleton-Modul

**Tasks:**

- [ ] Erstelle `src/assembled_core/qa/ai_tech_playbook.py` mit Dataclasses (`AiTechPlaybookConfig`, `SetupResult`, `PlaybookResult`)
- [ ] Implementiere `load_ai_tech_playbook_config()` (CLI-Args zu Config)
- [ ] Implementiere Skeleton `run_ai_tech_playbook()` (noch ohne konkrete Orchestrierung)
- [ ] Erstelle `scripts/run_ai_tech_playbook.py` mit Argument-Parsing
- [ ] Basis-Tests fur Config-Loading

**Deliverables:**

- Skeleton-Modul mit Dataclasses
- CLI-Script mit Argument-Parsing
- Basis-Tests

### P2: Orchestrierung von Batch-Backtest + Risk-Reports + Factor-Exposures

**Tasks:**

- [ ] Integriere Batch-Backtest-Execution (via `scripts/batch_backtest.py` oder programmatisch)
- [ ] Fur jeden Run: Risk-Report-Generierung mit Factor-Exposures (wenn aktiviert)
- [ ] Lade Metriken aus Risk-Reports
- [ ] Lade Factor-Exposures-Summaries
- [ ] Erstelle `experiment_summary.csv` und `experiment_summary.md` (ohne Walk-Forward/Regime)
- [ ] Tests fur Basis-Orchestrierung

**Deliverables:**

- Funktionsfahige Basis-Orchestrierung (ohne Walk-Forward/Regime)
- Summary-Reports (CSV/Markdown)
- Tests

### P3: Optional Walk-Forward/Regime-Integration

**Tasks:**

- [ ] Integriere Walk-Forward-Analyse (via `scripts/run_walk_forward_analysis.py` oder programmatisch)
- [ ] Integriere Regime-Analyse (aus Risk-Reports oder separate Regime-Datei)
- [ ] Erweitere `experiment_summary.csv` und `experiment_summary.md` um Walk-Forward/Regime-Metriken
- [ ] Tests fur Walk-Forward/Regime-Integration

**Deliverables:**

- Vollstandige Orchestrierung mit Walk-Forward und Regime-Analyse
- Erweiterte Summary-Reports
- Tests

### P4: CLI-Integration + Tests + Docs

**Tasks:**

- [ ] Integriere `ai_tech_playbook` als Subcommand in `scripts/cli.py`
- [ ] Erganze `info_subcommand` um `ai_tech_playbook`
- [ ] Umfassende Tests (Unit-Tests, Integration-Tests)
- [ ] Aktualisiere Dokumentation:
  - `docs/ADVANCED_ANALYTICS_FACTOR_LABS.md` (neuer Eintrag fur Playbook)
  - `docs/WORKFLOWS_BATCH_BACKTESTS_AND_PARALLELIZATION.md` (Hinweis auf Playbook)
  - `README.md` (Quick Link zu Playbook)

**Deliverables:**

- CLI-Integration
- Vollstandige Test-Suite
- Dokumentation

---

## 8. Risks & Limitations

### 8.1 Laufzeit (Batch + Walk-Forward kann schwergewichtig sein)

**Problem:** Batch-Backtests mit Walk-Forward-Analyse konnen sehr lange dauern (Stunden bei groBen Zeitraumen und vielen Splits).

**Mitigation:**

- Dokumentiere erwartete Laufzeit pro Experiment-Typ
- Option `--skip-batch-run` erlaubt Nutzung existierender Ergebnisse
- Walk-Forward kann optional deaktiviert werden
- Empfehlung: Zuerst Plain Backtests, dann optional Walk-Forward

### 8.2 Abhangigkeit von existierenden Configs (AI/Tech-Setup muss existieren)

**Problem:** Das Playbook erwartet Batch-Configs mit spezifischen Run-IDs (core, core_ml, ml_only) und Factor-Bundle-Pfaden.

**Mitigation:**

- Klare Dokumentation der erwarteten Batch-Config-Struktur
- Validation der Batch-Config beim Start (mindestens 3 Runs mit erwarteten IDs)
- Fehlermeldungen mit Hinweisen auf Beispiel-Configs

### 8.3 Kein Live-Trading, nur Research

**Problem:** Das Playbook ist ausschliesslich fur Research/Backtesting gedacht, nicht fur Live-Trading.

**Mitigation:**

- Klare Dokumentation: "Research Tool Only"
- Keine Integration mit Live-Trading-APIs
- Alle Analysen basieren auf lokalen Daten

### 8.4 Factor-Returns-Datei muss existieren (fur Factor-Exposures)

**Problem:** Factor-Exposures-Analyse erfordert eine Factor-Returns-Datei.

**Mitigation:**

- Factor-Exposures-Analyse ist optional (kann deaktiviert werden)
- Dokumentation: Wie man Factor-Returns-Datei erstellt
- Fallback: Risk-Reports ohne Factor-Exposures sind trotzdem nutzbar

### 8.5 Walk-Forward erfordert ausreichend historische Daten

**Problem:** Walk-Forward-Analyse erfordert genug Daten fur Train/Test-Splits.

**Mitigation:**

- Validation der Zeitraume (start_date, end_date) gegen Walk-Forward-Config
- Klare Fehlermeldungen, wenn nicht genug Daten vorhanden
- Empfehlung: Mindestens 2-3 Jahre Daten fur Walk-Forward

---

## 9. Success Criteria

### 9.1 Standardisierte Experimente fur Core vs. Core+ML vs. ML-Only

**Kriterium:** Ein einziger CLI-Befehl fuhrt alle drei Setups aus und generiert Vergleichs-Reports.

**Messung:**

- `python scripts/cli.py ai_tech_playbook --batch-config <config>` generiert vollstandige Vergleichs-Reports
- `experiment_summary.csv` enthalt Metriken fur alle drei Setups
- `experiment_summary.md` ist lesbar und informativ

### 9.2 Integration mit Walk-Forward und Regime-Analyse

**Kriterium:** Walk-Forward und Regime-Analyse konnen optional aktiviert werden und werden korrekt in Summary-Reports integriert.

**Messung:**

- Walk-Forward-Results erscheinen in `experiment_summary.csv` und `experiment_summary.md`
- Regime-Performance-Metriken erscheinen in Summary-Reports
- Alle Metriken sind konsistent mit einzelnen Backtest-/Risk-Reports

### 9.3 Factor Exposures & Deflated Sharpe Integration

**Kriterium:** Factor Exposures und Deflated Sharpe Ratios werden korrekt aus Risk-/Factor-Reports gelesen und in Summary-Reports angezeigt.

**Messung:**

- `experiment_summary.csv` enthalt `deflated_sharpe` und Factor-Exposure-Metriken (wenn verfugbar)
- `experiment_summary.md` zeigt Top-5-Faktoren pro Setup
- Deflated Sharpe Ratios sind konsistent mit Factor-Reports

### 9.4 Unit-Tests und Integrationstests grun

**Kriterium:** Alle Tests sind grun.

**Messung:**

- `pytest tests/test_qa_ai_tech_playbook.py` - alle Tests grun
- Integration-Test: Playbook mit echten Batch-Configs - alle Checks funktionieren
- Mindestens 80% Code-Coverage

---

## 10. References

**Design Documents:**

- [Batch Backtests & Parallelization P4 Design](WORKFLOWS_BATCH_BACKTESTS_AND_PARALLELIZATION.md) - Batch-Config-Struktur
- [Walk-Forward & Regime Analysis B3 Design](WALK_FORWARD_AND_REGIME_B3_DESIGN.md) - Walk-Forward-API
- [Signal API & Factor Exposures A2 Design](SIGNAL_API_AND_FACTOR_EXPOSURES_A2_DESIGN.md) - Factor-Exposures-Format
- [Deflated Sharpe B4 Design](DEFLATED_SHARPE_B4_DESIGN.md) - Deflated Sharpe Ratios
- [Advanced Analytics Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md) - Gesamte Roadmap

**Code References:**

- `scripts/batch_backtest.py` - Batch-Backtest-Orchestrierung
- `scripts/generate_risk_report.py` - Risk-Report-Generierung mit Factor-Exposures
- `scripts/run_walk_forward_analysis.py` - Walk-Forward-Analyse
- `config/factor_bundles/ai_tech_*.yaml` - AI/Tech Factor Bundles
- `research/playbooks/ai_tech_multifactor_mlalpha_regime_playbook.py` - Bestehendes Playbook (Referenz)

**Workflows:**

- [Multi-Factor Long/Short Strategy Workflows](WORKFLOWS_STRATEGIES_MULTIFACTOR.md) - Multi-Factor-Strategie-Workflows
- [Risk Metrics & Attribution Workflows](WORKFLOWS_RISK_METRICS_AND_ATTRIBUTION.md) - Risk-Report-Workflows

---

