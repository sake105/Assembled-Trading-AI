# Sprint A3 - Operations & Monitoring (Design)

**Phase A3** - Advanced Analytics & Factor Labs

**Status:** Design Phase  
**Last Updated:** 2025-01-XX

---

## 1. Overview & Goals

**Ziel:** Einmal pro Tag schnell sehen, ob Backend & Pipelines "gesund" sind.

**Fokus:** Read-only Health-Checks, keine Trades / keine Schreibzugriffe auf Daten.

**Motivation:**
Nach einem Tag mit EOD-Runs, Backtests, Risk-Reports, TCA-Reports und Factor-Exposures-Analysen braucht das Operations-Team eine schnelle Uebersicht:

- Wurden die erwarteten Runs durchgefuehrt?
- Sind die Output-Dateien vorhanden und plausibel?
- Gibt es ungewoehnliche Abweichungen in Metriken (Drawdown, Volatility, Sharpe)?
- Haben sich Factor-Exposures drastisch geaendert?
- Ist die Strategie noch im erwarteten Performance-Band?

**Scope:**
- Existenz-Checks: Letzte Runs, wichtige Output-Dateien
- Plausibilitaets-Checks: Metriken im erwarteten Band
- Status-Level: OK, WARN, CRITICAL
- Read-only: Script darf nur lesen, loggen, Reports schreiben

**Abgrenzung:**
- Nicht: EOD-Pipeline ausfuehren (dafuer: `run_daily`, `run_eod_pipeline`)
- Nicht: Backtests neu starten (dafuer: `run_backtest`, `batch_backtest`)
- Nicht: Risk/TCA-Reports neu generieren (dafuer: `generate_risk_report`, `generate_tca_report`)
- Sondern: Bestehende Artefakte pruefen und zusammenfassen

---

## 2. Current State

### 2.1 Existierende EOD-/Backtest-/Batch-Workflows

**EOD Pipeline:**
- `scripts/run_daily.py`: EOD-MVP Runner (Order-Generierung)
- `scripts/run_eod_pipeline.py`: Vollstaendige Pipeline (Execute, Backtest, Portfolio, QA)
- CLI: `python scripts/cli.py run_daily --freq 1d`

**Backtest-Workflows:**
- `scripts/run_backtest_strategy.py`: Einzelner Strategy-Backtest
- `scripts/batch_backtest.py`: Batch-Backtests aus Config-File
- CLI: `python scripts/cli.py run_backtest --freq 1d --strategy trend_baseline`

**Report-Generierung:**
- `scripts/generate_risk_report.py`: Risk-Report aus Backtest-Outputs
- `scripts/generate_tca_report.py`: TCA-Report aus Backtest-Outputs
- CLI: `python scripts/cli.py risk_report --backtest-dir output/backtests/experiment_123/`

**Existierende Health-Checks:**
- `src/assembled_core/qa/health.py`: Basis-Health-Checks (prices, orders, portfolio)
- `aggregate_qa_status()`: Aggregiert QA-Checks fuer eine Frequenz
- API-Endpoint: `GET /api/v1/qa/status?freq=1d`

### 2.2 Typische Artefakte nach einem Tag

**Backtest-Verzeichnisse:**
- `output/backtests/experiment_*/` (experiment_id oder timestamp)
  - `equity_curve.csv` oder `equity_curve.parquet`
  - `performance_report.md`
  - Optional: `trades.csv`, `positions.csv`

**Risk-Reports:**
- `output/risk_reports/` oder `output/backtests/experiment_*/risk_report.md`
  - `risk_summary.csv`
  - `exposure_timeseries.csv`
  - `risk_by_regime.csv` (optional)
  - `risk_by_factor_group.csv` (optional)
  - `factor_exposures_detail.csv` (optional, wenn --enable-factor-exposures)
  - `factor_exposures_summary.csv` (optional)

**TCA-Reports:**
- `output/tca_reports/` oder `output/backtests/experiment_*/tca_report.md`
  - `tca_trades.csv`
  - `tca_summary.csv`
  - `tca_risk_summary.csv` (optional)

**ML-Validation-Outputs:**
- `output/ml_validation/` oder aehnliche Verzeichnisse
  - Model-Zoo-Summaries (CSV/Markdown)
  - Factor-Validation-Reports

**EOD-Outputs:**
- `output/orders_YYYYMMDD.csv` (SAFE-Bridge Orders)
- `output/portfolio_equity_{freq}.csv`
- `output/portfolio_report.md`

---

## 3. Health-Check Scope & Requirements

### 3.1 Existenz-Checks

**Letzte Runs:**
- Finde neuesten Backtest-Run in `output/backtests/` (nach timestamp oder experiment_id)
- Pruefe, ob Risk-Report existiert (in Backtest-Dir oder `output/risk_reports/`)
- Pruefe, ob TCA-Report existiert (in Backtest-Dir oder `output/tca_reports/`)
- Pruefe, ob letzte EOD-Orders existieren (`output/orders_YYYYMMDD.csv`)

**Wichtige Output-Dateien:**
- Equity-Curve (CSV/Parquet)
- Risk-Report-Markdown
- Risk-Summary-CSV
- TCA-Report-Markdown (optional)
- Factor-Exposures-Summary-CSV (optional, wenn Factor-Exposures aktiviert)

### 3.2 Plausibilitaets-Checks

**Performance-Metriken (aus Risk-Report oder Equity-Curve):**
- **Max Drawdown**: Erwartetes Band (z.B. -5% bis -30% fuer moderate Strategien)
  - WARN: Drawdown < -30% oder > -5% (ungewöhnlich konservativ)
  - CRITICAL: Drawdown < -50%
- **Daily Volatility**: Erwartetes Band (z.B. 0.01 bis 0.05 fuer daily)
  - WARN: Volatility < 0.005 oder > 0.10
- **Sharpe Ratio**: Erwartetes Band (z.B. 0.5 bis 3.0)
  - WARN: Sharpe < 0.0 oder > 5.0
  - CRITICAL: Sharpe < -1.0

**Trading-Activity:**
- **Turnover**: Vergleiche mit historischem Range
  - WARN: Turnover ploetzlich < 50% oder > 200% des Durchschnitts
- **Anzahl Trades**: Vergleiche mit historischem Range
  - WARN: Anzahl Trades ploetzlich < 50% oder > 200% des Durchschnitts

**Benchmark-Korrelation (optional):**
- **Correlation zum Benchmark**: Erwartetes Band (z.B. 0.3 bis 0.8)
  - WARN: Correlation < 0.0 oder > 0.95 (zu stark korreliert)
  - CRITICAL: Negative Correlation wenn erwartet positiv

**Factor Exposures (optional, wenn Factor-Exposures aktiviert):**
- **Mean Beta pro Faktor**: Pruefe, ob im erwarteten Range (z.B. -1.0 bis 1.0 fuer die meisten Faktoren)
  - WARN: Mean Beta ausserhalb [-2.0, 2.0]
  - CRITICAL: Mean Beta ausserhalb [-3.0, 3.0]
- **Beta-Stabilitaet**: Pruefe std_beta (niedrige std_beta = stabil)
  - WARN: std_beta > 0.5 (instabile Exposure)
- **Factor-Profil-Konsistenz**: Pruefe, ob Top-3-Faktoren sich geaendert haben
  - WARN: Top-3-Faktoren haben sich drastisch geaendert (z.B. Momentum wurde zu Value)

**Zeitliche Plausibilitaet:**
- **Letzte Aktualisierung**: Pruefe, ob Equity-Curve/Reports weniger als N Tage alt sind
  - WARN: Letzte Aktualisierung > 2 Tage alt
  - CRITICAL: Letzte Aktualisierung > 7 Tage alt

### 3.3 Status-Level

**OK**: Alle Checks bestanden, keine Warnungen
**WARN**: Mindestens ein Check hat Warnung, aber keine Critical
**CRITICAL**: Mindestens ein Check ist kritisch

**Aggregierter Gesamtstatus:**
- Wenn mindestens ein CRITICAL: Gesamtstatus = CRITICAL
- Sonst: Wenn mindestens ein WARN: Gesamtstatus = WARN
- Sonst: Gesamtstatus = OK

### 3.4 Read-only Requirement

**Das Script darf NICHT:**
- Trades ausfuehren
- Datenbanken schreiben
- Backtests neu starten
- Reports neu generieren (falls fehlend, nur warnen)

**Das Script darf:**
- Dateien lesen
- Metriken berechnen (aus bestehenden Dateien)
- Logs schreiben
- Health-Reports schreiben

---

## 4. Health-Check API & CLI Design

### 4.1 Python-API

**Dataclass: HealthCheckResult**

```python
@dataclass
class HealthCheckResult:
    """Result of a health check run.
    
    Attributes:
        overall_status: Overall status ("ok", "warn", "critical")
        timestamp: Timestamp when check was run (UTC)
        checks: List of individual check results
        summary: Summary statistics (counts per status)
        metadata: Optional metadata (lookback_days, backtest_dir, etc.)
    """
    overall_status: Literal["ok", "warn", "critical"]
    timestamp: pd.Timestamp
    checks: list[HealthCheck]
    summary: dict[str, int]  # {"ok": 5, "warn": 2, "critical": 0}
    metadata: dict[str, Any] | None = None
```

**Dataclass: HealthCheck**

```python
@dataclass
class HealthCheck:
    """Result of a single health check.
    
    Attributes:
        name: Check name (e.g., "max_drawdown", "factor_exposures_momentum")
        status: Check status ("ok", "warn", "critical")
        value: Actual value (float, int, str, None)
        expected_range: Expected range (tuple, dict, or None)
        details: Optional dictionary with additional details
        last_updated_at: Optional timestamp when underlying data was last updated
    """
    name: str
    status: Literal["ok", "warn", "critical"]
    value: float | int | str | None
    expected_range: tuple[float, float] | dict[str, Any] | None = None
    details: dict[str, Any] | None = None
    last_updated_at: pd.Timestamp | None = None
```

**Funktion: run_health_check**

```python
def run_health_check(
    backtests_root: Path | str = "output/backtests/",
    days: int = 60,
    benchmark_symbol: str | None = None,
    benchmark_file: Path | None = None,
    output_dir: Path | None = None,
) -> HealthCheckResult:
    """Run comprehensive health check on backend operations.
    
    Args:
        backtests_root: Root directory containing backtest outputs (default: "output/backtests/")
        days: Lookback window in days for historical comparison (default: 60)
        benchmark_symbol: Optional benchmark symbol (e.g., "SPY") for correlation checks
        benchmark_file: Optional path to benchmark file (CSV/Parquet with timestamp, returns/close)
        output_dir: Optional output directory for health reports (default: "output/health/")
    
    Returns:
        HealthCheckResult with overall status and list of checks
    
    Note:
        This function is read-only: it only reads existing files and computes metrics.
        It does not generate new backtests or reports.
    """
```

### 4.2 CLI Design

**Neues Script: scripts/check_health.py**

```python
def main() -> int:
    """CLI entry point for health check."""
    parser = argparse.ArgumentParser(
        description="Check backend health status (read-only)"
    )
    parser.add_argument(
        "--backtests-root",
        type=Path,
        default=Path("output/backtests/"),
        help="Root directory containing backtest outputs (default: output/backtests/)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="Lookback window in days for historical comparison (default: 60)"
    )
    parser.add_argument(
        "--benchmark-symbol",
        type=str,
        default=None,
        help="Benchmark symbol (e.g., 'SPY') for correlation checks"
    )
    parser.add_argument(
        "--benchmark-file",
        type=Path,
        default=None,
        help="Path to benchmark file (CSV/Parquet with timestamp, returns/close)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for health reports (default: output/health/)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    
    args = parser.parse_args()
    
    result = run_health_check(
        backtests_root=args.backtests_root,
        days=args.days,
        benchmark_symbol=args.benchmark_symbol,
        benchmark_file=args.benchmark_file,
        output_dir=args.output_dir,
    )
    
    if args.format == "json":
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print_health_summary(result)
    
    return 0 if result.overall_status == "ok" else 1
```

**CLI-Subcommand in scripts/cli.py:**

```python
def check_health_subcommand(args: argparse.Namespace) -> int:
    """Check backend health status subcommand.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 = ok, 1 = warn/critical)
    """
    from scripts.check_health import run_health_check
    
    backtests_root = args.backtests_root
    if not backtests_root.is_absolute():
        backtests_root = ROOT / backtests_root
    backtests_root = backtests_root.resolve()
    
    output_dir = None
    if args.output_dir:
        output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
        output_dir = output_dir.resolve()
    
    benchmark_file = None
    if args.benchmark_file:
        benchmark_file = args.benchmark_file if args.benchmark_file.is_absolute() else ROOT / args.benchmark_file
        benchmark_file = benchmark_file.resolve()
    
    result = run_health_check(
        backtests_root=backtests_root,
        days=args.days,
        benchmark_symbol=args.benchmark_symbol,
        benchmark_file=benchmark_file,
        output_dir=output_dir,
    )
    
    # Print summary to stdout
    if args.format == "json":
        import json
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        from scripts.check_health import print_health_summary
        print_health_summary(result)
    
    return 0 if result.overall_status == "ok" else 1
```

**CLI-Argumente registrieren:**

```python
# In scripts/cli.py, bei den anderen Subcommand-Definitionen:
health_parser = subparsers.add_parser(
    "check_health",
    help="Check backend health status (read-only)",
    description="Runs comprehensive health checks on backend operations, including existence checks, plausibility checks, and status reporting.",
)

health_parser.add_argument(
    "--backtests-root",
    type=Path,
    default=Path("output/backtests/"),
    metavar="DIR",
    help="Root directory containing backtest outputs (default: output/backtests/)"
)

health_parser.add_argument(
    "--days",
    type=int,
    default=60,
    metavar="N",
    help="Lookback window in days for historical comparison (default: 60)"
)

health_parser.add_argument(
    "--benchmark-symbol",
    type=str,
    default=None,
    metavar="SYMBOL",
    help="Benchmark symbol (e.g., 'SPY') for correlation checks"
)

health_parser.add_argument(
    "--benchmark-file",
    type=Path,
    default=None,
    metavar="FILE",
    help="Path to benchmark file (CSV/Parquet with timestamp, returns/close)"
)

health_parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    metavar="DIR",
    help="Output directory for health reports (default: output/health/)"
)

health_parser.add_argument(
    "--format",
    type=str,
    choices=["text", "json"],
    default="text",
    help="Output format: 'text' for human-readable summary, 'json' for machine-readable (default: text)"
)

health_parser.set_defaults(func=check_health_subcommand)
```

---

## 5. Output & Reporting

### 5.1 Output-Dateien

**output/health/health_summary.json:**
- JSON-Serialisierung von HealthCheckResult
- Maschinenlesbar, fuer Integration in Dashboards/Alerting

**output/health/health_summary.md:**
- Lesbarer Markdown-Report
- Format:
  ```markdown
  # Health Check Summary
  
  **Overall Status:** OK | WARN | CRITICAL
  **Timestamp:** 2025-01-XX 10:00:00 UTC
  **Lookback Days:** 60
  
  ## Summary
  - OK: 5 checks
  - WARN: 2 checks
  - CRITICAL: 0 checks
  
  ## Checks
  
  | Check Name | Status | Value | Expected Range | Details |
  |------------|--------|-------|----------------|---------|
  | max_drawdown | OK | -12.5% | [-30%, -5%] | Within expected range |
  | daily_volatility | WARN | 0.08 | [0.01, 0.05] | Above expected range |
  | sharpe_ratio | OK | 1.25 | [0.5, 3.0] | Within expected range |
  ...
  ```

**output/health/health_checks.csv (optional):**
- Eine Zeile pro Check
- Spalten: name, status, value, expected_range_min, expected_range_max, details_json, last_updated_at

### 5.2 Struktur fuer jede Check-Zeile

**CSV-Spalten:**
- `name`: Check-Name (z.B. "max_drawdown", "factor_exposures_momentum_beta")
- `status`: "ok", "warn", "critical"
- `value`: Tatsaechlicher Wert (float, int, str)
- `expected_range_min`: Erwartetes Minimum (float, None)
- `expected_range_max`: Erwartetes Maximum (float, None)
- `details`: JSON-String mit zusaetzlichen Details
- `last_updated_at`: Timestamp der letzten Aktualisierung der zugrundeliegenden Daten

---

## 6. Integration Points

### 6.1 EOD-/Daily-Runs

**Option 1: Am Ende von run_daily / run_eod_pipeline:**
- Optionaler Health-Check nach Pipeline-Abschluss
- Nur wenn `--check-health` Flag gesetzt ist
- Loggt Health-Status, aber stoppt Pipeline nicht bei WARN/CRITICAL

**Option 2: Per Cron / Scheduled Task:**
- Separater Cron-Job, der taeglich `python scripts/cli.py check_health` ausfuehrt
- Unabhaengig von EOD-Pipeline

**Empfehlung:** Option 2 (separater Cron), da Health-Check read-only ist und nicht in den kritischen Pfad gehoert.

### 6.2 Risk-Report / TCA / Factor-Exposure-Outputs

**Wiederverwendung bestehender Artefakte:**
- Health-Check liest `risk_summary.csv` aus Risk-Report-Verzeichnis
- Health-Check liest `factor_exposures_summary.csv` aus Risk-Report-Verzeichnis (falls vorhanden)
- Health-Check liest `tca_summary.csv` aus TCA-Report-Verzeichnis (falls vorhanden)
- **NICHT:** Health-Check generiert Reports neu (nur Warnung, wenn fehlend)

### 6.3 Optional: Integration in profile_jobs.py

**Zukuenftige Integration:**
- Neuer Job-Typ: `OPERATIONS_HEALTH_CHECK`
- Taeglicher Job, der `check_health` ausfuehrt
- Profiling-Integration: Misst Ausfuehrungszeit, Loggt Status

**Nicht in A3.1-A3.3:**
- Diese Integration ist optional und kann spaeter hinzugefuegt werden

---

## 7. Implementation Plan A3.1-A3.3

### A3.1: Design & HealthCheck-Dataclasses + Skeleton-Implementation

**Tasks:**
- [ ] Design-Dokument erstellen (dieses Dokument)
- [ ] Neue Datei: `scripts/check_health.py`
- [ ] Dataclasses: `HealthCheckResult`, `HealthCheck`
- [ ] Skeleton-Funktion: `run_health_check()` (noch ohne konkrete Checks)
- [ ] Helper-Funktionen: `find_latest_backtest()`, `load_risk_summary()`, etc.

**Deliverables:**
- Design-Dokument (dieses Dokument)
- Skeleton-Implementation mit Dataclasses
- Basis-Struktur fuer Checks

### A3.2: Konkrete Checks + CLI-Integration + Tests

**Tasks:**
- [ ] Implementiere Existenz-Checks:
  - Finde neuesten Backtest-Run
  - Pruefe Equity-Curve-Existenz
  - Pruefe Risk-Report-Existenz
  - Pruefe TCA-Report-Existenz (optional)
- [ ] Implementiere Plausibilitaets-Checks:
  - Max Drawdown Check
  - Daily Volatility Check
  - Sharpe Ratio Check
  - Turnover Check (historischer Vergleich)
  - Anzahl Trades Check (historischer Vergleich)
  - Benchmark-Correlation Check (optional)
  - Factor-Exposures-Checks (optional)
- [ ] Implementiere Report-Generierung:
  - `write_health_summary_json()`
  - `write_health_summary_markdown()`
  - `write_health_checks_csv()` (optional)
- [ ] CLI-Integration:
  - `scripts/check_health.py` main()
  - `scripts/cli.py` check_health_subcommand()
  - Argument-Parsing
- [ ] Tests: `tests/test_check_health.py`
  - Test Existenz-Checks
  - Test Plausibilitaets-Checks
  - Test Report-Generierung
  - Test CLI-Integration

**Deliverables:**
- Funktionsfaehige Health-Check-Implementation
- CLI-Integration
- Unit-Tests (alle gruen)

### A3.3: Runbook docs/OPERATIONS_BACKEND.md + Doku-Updates

**Tasks:**
- [ ] Erstelle Runbook: `docs/OPERATIONS_BACKEND.md`
  - Uebersicht: Was ist Operations & Monitoring?
  - Taeglicher Workflow: Health-Check ausfuehren
  - Interpretation: Was bedeuten OK/WARN/CRITICAL?
  - Troubleshooting: Was tun bei WARN/CRITICAL?
  - Best Practices: Erwartete Baender anpassen
- [ ] Update README.md:
  - Neuer Quick-Link: "Operations & Monitoring" unter Usage & Workflows
  - Kurzbeschreibung: Health-Check-Script
- [ ] Update docs/ADVANCED_ANALYTICS_FACTOR_LABS.md:
  - A3: Operations & Monitoring als Completed markieren
  - Kurzbeschreibung der Health-Check-Funktionalitaet

**Deliverables:**
- Runbook-Dokumentation
- Aktualisierte README.md
- Aktualisierte Factor Labs Doku

---

## 8. Risks & Limitations

### 8.1 Falsch-positive WARN/CRITICAL bei zu engen Baendern

**Problem:** Wenn erwartete Baender zu eng sind, gibt es haeufig false positives.

**Mitigation:**
- Erwartete Baender sollten konservativ sein (breit genug, um normale Schwankungen zu erlauben)
- Baender sollten konfigurierbar sein (via Config-File oder CLI-Parameter, Future Work)
- WARN sollte nicht als Blocking behandelt werden (nur Information)

### 8.2 Abhaengigkeit von Output-Dateien

**Problem:** Wenn Risk-Report/TCA-Report/Factor-Exposures fehlen, kann Health-Check nicht alle Checks durchfuehren.

**Mitigation:**
- Existenz-Checks sollten klar als WARN signalisieren, wenn Dateien fehlen
- Health-Check sollte trotzdem andere Checks durchfuehren (graceful degradation)
- Dokumentation: Welche Checks optional sind vs. erforderlich

### 8.3 Historischer Vergleich erfordert ausreichend Daten

**Problem:** Historischer Vergleich (Turnover, Anzahl Trades) funktioniert nur, wenn genuegend historische Daten vorhanden sind.

**Mitigation:**
- Wenn historische Daten fehlen, sollten Checks als "OK" behandelt werden (kein WARN ohne Datenbasis)
- Lookback-Window sollte konfigurierbar sein (default: 60 Tage)
- Logging: Klar dokumentieren, wenn historischer Vergleich nicht moeglich ist

### 8.4 Performance: Viele Dateien lesen

**Problem:** Health-Check liest viele Dateien (Equity-Curves, Risk-Reports, etc.), kann langsam sein.

**Mitigation:**
- Caching: Wenn moeglich, Cache-Last-Modified-Times
- Parallelisierung: Mehrere Checks parallel ausfuehren (Future Work)
- Timeout: Health-Check sollte nicht laenger als 30 Sekunden dauern (Future Work: Timeout-Mechanismus)

---

## 9. Success Criteria

### 9.1 Taegliche Health-Uebersicht aus einem einzigen Command

**Kriterium:** `python scripts/cli.py check_health` liefert in < 10 Sekunden eine vollstaendige Health-Uebersicht.

**Messung:**
- Ausfuehrungszeit < 10 Sekunden (bei typischer Anzahl Backtest-Verzeichnisse)
- Output ist klar und lesbar (Markdown-Report oder JSON)

### 9.2 Metriken & Status konsistent mit Backtests und Risk-Reports

**Kriterium:** Health-Check-Metriken stimmen mit Metriken aus Risk-Reports ueberein.

**Messung:**
- Max Drawdown aus Health-Check = Max Drawdown aus Risk-Report (bis auf Rundungsfehler)
- Sharpe Ratio aus Health-Check = Sharpe Ratio aus Risk-Report (bis auf Rundungsfehler)
- Factor-Exposures aus Health-Check = Factor-Exposures aus Risk-Report (falls vorhanden)

### 9.3 Unit-Tests und Integrationstests gruen

**Kriterium:** Alle Tests sind gruen.

**Messung:**
- `pytest tests/test_check_health.py` - alle Tests gruen
- Integration-Test: Health-Check mit echten Backtest-Verzeichnissen - alle Checks funktionieren

**Tests:**
- Mindestens 80% Code-Coverage
- Edge Cases abgedeckt (fehlende Dateien, ungueltige Metriken, etc.)

---

## 10. References

**Design Documents:**
- [Risk 2.0 D2 Design](RISK_2_0_D2_DESIGN.md) - Risk-Report-Struktur
- [Signal API & Factor Exposures A2 Design](SIGNAL_API_AND_FACTOR_EXPOSURES_A2_DESIGN.md) - Factor-Exposures-Format
- [Advanced Analytics Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md) - Gesamte Roadmap

**Code References:**
- `scripts/generate_risk_report.py` - Risk-Report-Generierung, Output-Struktur
- `scripts/generate_tca_report.py` - TCA-Report-Generierung, Output-Struktur
- `src/assembled_core/qa/health.py` - Basis-Health-Checks (prices, orders, portfolio)
- `src/assembled_core/qa/metrics.py` - Metriken-Berechnung (Sharpe, Drawdown, etc.)

**Workflows:**
- [Risk Metrics & Attribution Workflows](WORKFLOWS_RISK_METRICS_AND_ATTRIBUTION.md) - Risk-Report-Workflow
- [EOD Pipeline Workflows](WORKFLOWS_EOD_AND_QA.md) - EOD-Pipeline-Workflow

