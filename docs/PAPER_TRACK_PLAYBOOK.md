# Paper Track Playbook - Backtest to Paper to Live

**Phase:** Advanced Analytics & Factor Labs - Track A (Anwendung & Produktisierung)  
**Status:** Active Documentation  
**Last Updated:** 2025-01-XX

---

## 1. Overview & Scope

**Ziel:** Dieser Playbook definiert den standardisierten Prozess, um Trading-Strategien vom Backtest uber den Paper-Track (Simulation mit realen Marktdaten) in Richtung Live-Trading zu bringen.

**Workflow-Ubersicht:**

```
Backtest (Research) → Paper Track (Simulation) → Live Trading (Production)
     Phase 0              Phase 1-2                Phase 3+
```

**Wichtig:** Dieser Playbook ist eine **Prozess- und Governance-Definition**, kein Trading-Tutorial. Er beschreibt:

- **Qualitatskriterien** (Wann darf eine Strategie in den Paper-Track?)
- **Operationale Prozesse** (Wie lauft der Paper-Track ab?)
- **Monitoring & Evaluation** (Welche Metriken werden beobachtet?)
- **Gate-Decisions** (Wann ist eine Strategie bereit fur Live-Trading?)

**Datenbasis:** Alle Phasen basieren auf **lokalen Daten** und **SAFE-Bridge-Outputs** (CSV/JSON). Keine Live-APIs in Phase 0-2.

**Abgrenzung:**

- **Phase 0-2 (Backtest + Paper)**: Vollstandig im Backend-Repo abgebildet
- **Phase 3+ (Live)**: Erfordert zusatzliche professionelle & regulatorische Checks (ausserhalb dieses Playbooks)

---

## 2. Phase 0 - Candidate-Backtest

### 2.1 Gate-Kriterien: "Wann darf eine Strategie in den Paper-Track?"

Eine Strategie muss die folgenden **mindestanforderungen** erfullen, um in den Paper-Track aufgenommen zu werden:

#### 2.1.1 Mindest-Backtestdauer

- **Empfehlung:** >= 5 Jahre historische Daten
- **Alternativ:** Mehrere vollstandige Marktregime (z.B. Bull, Bear, Crisis, Sideways)
- **Begrundung:** Strategie muss verschiedene Marktphasen erlebt haben

#### 2.1.2 Regime-Coverage (B3)

- **Requirement:** Strategie-Performance muss in **mindestens 3 verschiedenen Regimen** evaluiert sein
- **Regime-Definition:** Bull, Bear, Crisis, Sideways, Reflation (basierend auf `docs/WALK_FORWARD_AND_REGIME_B3_DESIGN.md`)
- **Kriterium:** Strategie sollte nicht in allen Regimen komplett versagen
- **Messung:** Regime-Performance aus Risk-Reports (`risk_by_regime.csv`)

#### 2.1.3 Deflated Sharpe Ratio (B4)

- **Requirement:** Deflated Sharpe Ratio >= 0.5 (konservativer Schwellenwert)
- **Begrundung:** Adjustiert um Multiple Testing (siehe `docs/DEFLATED_SHARPE_B4_DESIGN.md`)
- **Interpretation:**
  - DSR < 0.5: Gimmick, nicht weiterverfolgen
  - 0.5 <= DSR < 1.0: Nur mit zusatzlicher Evidenz weiterverfolgen
  - DSR >= 1.0: Ernstzunehmender Kandidat
- **Messung:** Deflated Sharpe aus Factor-Reports oder ML-Validation-Reports

#### 2.1.4 Maximaler Backtest-Drawdown

- **Requirement:** Max Drawdown <= -30% (anpassbar je nach Strategie-Typ und Risikoprofil)
- **Alternative:** Max Drawdown sollte nicht deutlich schlechter sein als ein relevanter Benchmark
- **Messung:** Max Drawdown aus Risk-Reports (`risk_summary.csv`)

#### 2.1.5 Point-in-Time (PIT) Sicherheit (B2)

- **Requirement:** PIT-Checks mussen bestehen (keine Look-Ahead-Violations)
- **Runtime-Guards:** Signal-API nutzt PIT-Checks aus `docs/POINT_IN_TIME_AND_LATENCY.md`
- **Validierung:** `validate_signal_frame()` sollte keine PIT-Violations melden
- **Messung:** PIT-Checks aus QA-Reports oder explizite Validierung

#### 2.1.6 Weitere Qualitatskriterien (Optional, aber empfohlen)

- **Walk-Forward-Analyse:** Out-of-Sample-Performance sollte konsistent sein
- **Factor Exposures:** Strategie sollte ein klares Faktor-Profil haben (nicht zufallig)
- **Turnover:** Angemessen (nicht zu hoch, da sonst Kosten zu stark ins Gewicht fallen)

### 2.2 Backtest → Paper Gate Checklist

| Kriterium | Requirement | Messung | Status |
|-----------|-------------|---------|--------|
| Backtest-Dauer | >= 5 Jahre oder mehrere Regime | Zeitraum aus Backtest-Config | ☐ Erfullt |
| Regime-Coverage | >= 3 verschiedene Regime evaluiert | `risk_by_regime.csv` aus Risk-Report | ☐ Erfullt |
| Deflated Sharpe | >= 0.5 (konservativ) | Deflated Sharpe aus Factor-/ML-Reports | ☐ Erfullt |
| Max Drawdown | <= -30% (anpassbar) | `max_drawdown_pct` aus Risk-Report | ☐ Erfullt |
| PIT-Sicherheit | Keine Look-Ahead-Violations | PIT-Checks aus QA-Report oder Validierung | ☐ Erfullt |
| Walk-Forward (opt.) | OOS-Performance konsistent | Walk-Forward-Results (wenn verfugbar) | ☐ Erfullt |
| Factor Exposures (opt.) | Klares Faktor-Profil | `factor_exposures_summary.csv` | ☐ Erfullt |
| QA-Status | QA-Gates bestanden | QA-Report (`qa_gate_result`) | ☐ Erfullt |
| Dokumentation | Backtest-Report vorhanden | `performance_report.md`, `risk_report.md` | ☐ Erfullt |
| Model Card (opt.) | Beschreibung der Strategie | Model-Card-Dokument (wenn ML-basiert) | ☐ Erfullt |

**Gate-Entscheidung:**

- **Alle kritischen Kriterien (Dauer, Regime, DSR, MaxDD, PIT) erfullt:** ✅ Gate PASSED → Phase 1
- **Kritische Kriterien nicht erfullt:** ❌ Gate FAILED → Weiterer Research erforderlich
- **Optionale Kriterien fehlen:** ⚠️ WARN → Kann trotzdem weiter, aber dokumentieren

---

## 3. Phase 1 - Paper-Track Setup

### 3.1 Erforderliche Artefakte

Vor dem Start des Paper-Tracks mussen folgende Artefakte vorhanden sein:

#### 3.1.1 Backtest-Report

- **Performance-Report:** `performance_report.md` mit Metriken (Sharpe, Sortino, Max DD, Total Return, etc.)
- **Risk-Report:** `risk_report.md` mit erweiterten Risk-Metriken, Regime-Attribution, Factor-Exposures (optional)
- **TCA-Report (optional):** `tca_report.md` mit Transaktionskosten-Analyse

#### 3.1.2 Model Card (wenn ML-basiert)

- **Beschreibung:** Strategie-Typ, verwendete Faktoren, ML-Modell (falls zutreffend)
- **Trainingsdaten:** Zeitraum, Universe, Feature-Set
- **Performance-Metriken:** IC, Rank-IC, Sharpe, Deflated Sharpe
- **Limitations:** Bekannte Schwachen, Regime-Abhangigkeiten

#### 3.1.3 Configs & Parameter

- **Factor-Bundle-Config:** YAML-Datei mit Faktor-Gewichtungen (z.B. `config/factor_bundles/ai_tech_core_bundle.yaml`)
- **Backtest-Config:** Parameter aus Backtest (Rebalance-Frequenz, Max Exposure, Start-Kapital)
- **Universe-Definition:** Liste der Symbole (z.B. `config/universe_ai_tech_tickers.txt`)

### 3.2 Technisches Setup

#### 3.2.1 EOD-Pipeline Integration

- **Script:** `scripts/run_daily.py` oder `scripts/run_eod_pipeline.py`
- **Output:** SAFE-Bridge-Orders (`output/orders_YYYYMMDD.csv`)
- **QA-Reports:** Automatische QA-Checks nach jedem Run

#### 3.2.2 SAFE-Bridge-Outputs

- **Orders CSV:** `output/orders_{freq}.csv` (Format: timestamp, symbol, side, qty, price)
- **Orders JSON (optional):** Maschinenlesbares Format fur spateren API-Export
- **Portfolio-Equity:** `output/portfolio_equity_{freq}.csv` (Equity-Kurve aus Paper-Trading)

#### 3.2.3 QA & Risk-Reports (Periodisch)

- **Daily QA:** Automatische QA-Checks nach jedem EOD-Run
- **Weekly Risk-Report:** Wochentlicher Risk-Report mit aktuellen Metriken
- **Monthly Review:** Monatlicher Vergleich Paper-Track vs. Backtest-Performance

### 3.3 Rollen & Tasks

#### 3.3.1 Researcher (Strategie-Entwickler)

**Tasks:**

- Erstellt Backtest-Report und Model Card
- Dokumentiert Strategie-Logik und Parameter
- Prasentiert Strategie bei Backtest → Paper Gate
- Uberwacht Paper-Track-Performance (wochentlich)

**Artefakte:**

- Backtest-Reports
- Model Card
- Strategie-Dokumentation

#### 3.3.2 Operator (EOD-Pipeline-Betreiber)

**Tasks:**

- Fuhrt EOD-Pipeline taglich aus (`run_daily.py`)
- Uberwacht QA-Status (OK/WARN/CRITICAL)
- Generiert wochentliche Risk-Reports
- Dokumentiert Incidents (z.B. fehlgeschlagene Runs)

**Artefakte:**

- QA-Reports
- Risk-Reports
- Run-Logs

#### 3.3.3 Risk/QA (Qualitatssicherung)

**Tasks:**

- Prut Backtest → Paper Gate-Kriterien
- Uberwacht Paper-Track-Metriken (wochentlich)
- Entscheidet uber WARN/CRITICAL-Szenarien
- Fuhrt monatliche Reviews durch

**Artefakte:**

- Gate-Decision-Logs
- Monitoring-Reports
- Review-Dokumentation

---

## 4. Phase 2 - Paper-Track Laufzeit

### 4.1 Mindestdauer & Mindestanzahl Trades

- **Mindestdauer:** 6-12 Monate (empfohlen: 12 Monate)
- **Alternativ:** Mindestens 250 Trading-Tage (ca. 1 Jahr)
- **Mindestanzahl Trades:** >= 100 Trades (um statistische Signifikanz zu erreichen)
- **Begrundung:** Strategie muss verschiedene Marktphasen und Szenarien erlebt haben

### 4.2 Paper Track Runner - Daily Execution

#### 4.2.1 Running Paper Track Days

**CLI Command (recommended):**
```bash
# Run for single day
python scripts/cli.py paper_track \
    --config-file configs/paper_track/strategy_core_ai_tech.yaml \
    --as-of 2025-01-15

# Run for date range
python scripts/cli.py paper_track \
    --config-file configs/paper_track/strategy_core_ai_tech.yaml \
    --start-date 2025-01-15 \
    --end-date 2025-01-20

# Dry run (no files written, useful for testing)
python scripts/cli.py paper_track \
    --config-file configs/paper_track/strategy_core_ai_tech.yaml \
    --as-of 2025-01-15 \
    --dry-run

# Fail fast on errors
python scripts/cli.py paper_track \
    --config-file configs/paper_track/strategy_core_ai_tech.yaml \
    --start-date 2025-01-15 \
    --end-date 2025-01-20 \
    --fail-fast
```

**Standalone Script (alternative):**
```bash
# Run for single day
python scripts/run_paper_track.py \
    --config-file configs/paper_track/strategy_core_ai_tech.yaml \
    --as-of 2025-01-15

# Run for date range
python scripts/run_paper_track.py \
    --config-file configs/paper_track/strategy_core_ai_tech.yaml \
    --start-date 2025-01-15 \
    --end-date 2025-01-20
```

**Python API (for automation/custom workflows):**
```python
from src.assembled_core.paper.paper_track import (
    PaperTrackConfig,
    run_paper_day,
    save_paper_state,
    write_paper_day_outputs,
)
from scripts.run_paper_track import load_paper_track_config
import pandas as pd
from pathlib import Path

# Load config from YAML/JSON
config = load_paper_track_config(Path("configs/paper_track/strategy_core_ai_tech.yaml"))

# Run for a specific date
as_of = pd.Timestamp("2025-01-15", tz="UTC")
state_path = config.output_root / "state" / "state.json"

result = run_paper_day(config, as_of, state_path)
write_paper_day_outputs(result, config.output_root)
save_paper_state(result.state_after, state_path)
```

#### 4.2.2 Output Artefacts (per Day)

After running a paper track day, the following files are created in `output/paper_track/{strategy_name}/runs/{YYYYMMDD}/`:

- **`equity_snapshot.json`**: Portfolio snapshot (equity, cash, positions value, total return)
- **`positions.csv`**: Current positions (symbol, qty)
- **`orders_today.csv`**: Orders executed today (timestamp, symbol, side, qty, price, fill_price, costs)
- **`trades_today.csv`**: Same as orders_today.csv (alias for compatibility)
- **`daily_summary.json`**: Complete daily summary (JSON format)
- **`daily_summary.md`**: Human-readable daily summary (Markdown format)

**State File:**
- **`state/state.json`**: Persistent portfolio state (positions, cash, equity, last_run_date)

#### 4.2.3 Aggregated Outputs

Over time, aggregated outputs are generated:
- **`equity_curve.csv`**: Complete equity curve (timestamp, equity, cash, positions_value)
- **`trades_all.csv`**: All trades across all days
- **`positions_history.csv`**: Historical positions snapshots

### 4.3 Metriken im Paper-Track

Die folgenden Metriken werden **kontinuierlich** im Paper-Track beobachtet:

#### 4.2.1 Performance-Metriken

- **Sharpe Ratio (annualized):** Risiko-adjustierte Rendite
- **Deflated Sharpe Ratio:** Adjustiert um Multiple Testing (wenn mehrere Strategien gleichzeitig getestet werden)
- **Sortino Ratio:** Downside-Risk-adjustierte Rendite
- **Total Return / CAGR:** Gesamtrendite und annualisierte Rendite
- **Max Drawdown:** Maximaler Verlust vom Hochststand

#### 4.2.2 Trade-Metriken

- **Hit-Rate / Win-Rate:** Anteil profitabler Trades
- **Average Win / Average Loss:** Durchschnittliche Gewinne vs. Verluste
- **Profit Factor:** Verhaltnis von Gesamtgewinnen zu Gesamtverlusten
- **Turnover:** Portfolio-Umschlagshaufigkeit
- **Number of Trades:** Gesamtanzahl ausgefuhrter Trades

#### 4.2.3 Kosten-Metriken

- **Slippage:** Unterschied zwischen erwartetem und tatsachlichem Fill-Preis
- **Transaction Costs:** Gesamtkosten (Commission + Spread + Impact)
- **Cost-Adjusted Returns:** Rendite nach Abzug von Transaktionskosten

#### 4.2.4 Risk-Metriken

- **Volatility (annualized):** Standardabweichung der Returns
- **Value at Risk (VaR):** Maximaler erwarteter Verlust (z.B. 95% VaR)
- **Expected Shortfall (ES):** Durchschnittlicher Verlust bei VaR-Uberschreitung
- **Skewness / Kurtosis:** Verteilung der Returns (Tail-Risiken)

#### 4.2.5 Factor & Regime-Metriken

- **Factor Exposures:** Rolling Regression der Strategie-Returns vs. Factor-Returns
- **Regime-Performance:** Performance in verschiedenen Marktregimen (Bull/Bear/Crisis)
- **Factor R2:** Erklarungsgehalt der Factor-Exposures

### 4.3 Abweichungen vs. Backtest: Akzeptable Schwellen

**Wichtig:** Die folgenden Schwellen sind **Beispiele** und sollten je nach Strategie-Typ und Risikoprofil angepasst werden.

#### 4.3.1 Performance-Abweichungen

- **Sharpe Ratio:** Paper-Track Sharpe sollte nicht mehr als **30% schlechter** sein als Backtest-Sharpe
  - Beispiel: Backtest Sharpe = 1.5 → Paper-Track Sharpe >= 1.05 ist akzeptabel
- **Total Return:** Paper-Track Return sollte innerhalb von **±50%** des Backtest-Returns liegen
  - Beispiel: Backtest Return = 20% → Paper-Track Return zwischen 10% und 30% ist akzeptabel
- **Max Drawdown:** Paper-Track MaxDD sollte nicht mehr als **50% schlechter** sein als Backtest-MaxDD
  - Beispiel: Backtest MaxDD = -20% → Paper-Track MaxDD <= -30% ist akzeptabel

#### 4.3.2 Trade-Metriken-Abweichungen

- **Hit-Rate:** Paper-Track Hit-Rate sollte innerhalb von **±10 Prozentpunkten** des Backtest liegen
  - Beispiel: Backtest Hit-Rate = 55% → Paper-Track Hit-Rate zwischen 45% und 65% ist akzeptabel
- **Turnover:** Paper-Track Turnover sollte nicht mehr als **50% hoher** sein als Backtest
  - Beispiel: Backtest Turnover = 2.0 → Paper-Track Turnover <= 3.0 ist akzeptabel

#### 4.3.3 Factor-Exposures-Abweichungen

- **Top-3-Faktoren:** Sollten stabil bleiben (keine drastischen Anderungen in den wichtigsten Exposures)
- **Factor R2:** Sollte nicht mehr als **20 Prozentpunkte** niedriger sein als im Backtest
  - Beispiel: Backtest R2 = 0.6 → Paper-Track R2 >= 0.4 ist akzeptabel

### 4.4 WARN- und CRITICAL-Szenarien

#### 4.4.1 WARN-Szenarien (Uberwachung verscharfen)

**Trigger:**

- Eine oder mehrere Metriken uberschreiten akzeptable Schwellen (siehe 4.3)
- Deflated Sharpe fällt unter 0.3 (aber noch >= 0.0)
- Max Drawdown uberschreitet Backtest-MaxDD um > 30% (aber noch <= 50%)

**Actions:**

1. **Erhohte Uberwachung:** Tagliche Review der Metriken (statt wochentlich)
2. **Parameter-Review:** Uberprufung, ob Strategie-Parameter angepasst werden mussen
3. **Regime-Analyse:** Uberprufung, ob aktuelle Marktregime die Abweichung erklaren
4. **Dokumentation:** WARN-Status in Monitoring-Report dokumentieren
5. **Review-Meeting:** Wochentliches Review-Meeting zwischen Researcher, Operator und Risk/QA

**Dauer:** WARN-Status kann 2-4 Wochen bestehen bleiben, solange keine weiteren Verschlechterungen auftreten

#### 4.4.2 CRITICAL-Szenarien (Pause oder Kill)

**Trigger:**

- Deflated Sharpe < 0.0 (negative risikoadjustierte Rendite)
- Max Drawdown uberschreitet Backtest-MaxDD um > 50%
- Mehrere kritische Metriken gleichzeitig ausserhalb der Schwellen
- PIT-Violations entdeckt (Look-Ahead-Bias)
- Systematische Fehler in der Strategie-Logik entdeckt

**Actions:**

1. **Sofort-Pause:** EOD-Pipeline fur diese Strategie pausieren (keine weiteren Orders generieren)
2. **Root-Cause-Analysis:** Detaillierte Analyse der Ursachen (Datenprobleme? Parameter-Drift? Regime-Wechsel?)
3. **Parameter-Review:** Vollstandige Uberprufung aller Strategie-Parameter
4. **Backtest-Review:** Uberprufung, ob Backtest korrekt war (Overfitting? Datenprobleme?)
5. **Decision:** 
   - **Fix & Resume:** Wenn behebbares Problem identifiziert → Fix implementieren, Backtest wiederholen, Paper-Track neu starten
   - **Kill:** Wenn Problem fundamental ist (z.B. PIT-Violation, systematischer Fehler) → Strategie beenden, Lessons Learned dokumentieren

**Dokumentation:** CRITICAL-Status muss in Decision-Log dokumentiert werden mit Root-Cause-Analysis und Entscheidungsgrundlage

---

## 5. Phase 3 - Go/No-Go Richtung Live

### 5.1 Klarstellung: Live-Betrieb erfordert zusatzliche Checks

**Wichtig:** Dieser Playbook deckt nur die **technischen und quantitativen** Aspekte ab. Live-Trading erfordert zusatzlich:

- **Regulatorische Compliance:** Erfullung aller relevanten Finanzmarktregulierungen
- **Risk-Management-Framework:** Professionelles Risk-Management (Position-Limits, Exposure-Limits, etc.)
- **Operational Excellence:** Hochverfugbare Systeme, Disaster Recovery, Monitoring & Alerting
- **Legal & Compliance Review:** Rechtsabteilung, Compliance-Abteilung
- **Capital Allocation:** Zuteilung von Trading-Kapital

**Dieser Playbook ist keine Ersetzung fur professionelle Due-Diligence.**

### 5.2 Go/No-Go-Kriterien (Beispiele, spater mit Profis verfeinern)

#### 5.2.1 Quantitative Kriterien

- **Paper-Track-Dauer:** >= 12 Monate erfolgreich
- **Paper-Track-Performance:** Sharpe >= Backtest-Sharpe * 0.8 (konservativ)
- **Paper-Track MaxDD:** <= Backtest-MaxDD * 1.2 (maximal 20% schlechter)
- **Deflated Sharpe:** >= 0.5 (konservativ)
- **Hit-Rate:** >= Backtest-Hit-Rate - 5 Prozentpunkte
- **Regime-Performance:** Strategie sollte in mindestens 2 von 3 Hauptregimen (Bull/Bear/Crisis) profitabel sein

#### 5.2.2 Qualitative Kriterien

- **Konsistenz:** Paper-Track-Performance sollte konsistent mit Backtest sein (keine systematischen Abweichungen)
- **Robustheit:** Strategie sollte robust gegen Parameter-Anderungen sein (Sensitivitatsanalyse)
- **Transparenz:** Vollstandige Dokumentation (Model Card, Backtest-Report, Paper-Track-Report)
- **Operational Readiness:** EOD-Pipeline lauft stabil (keine haufigen CRITICAL-Status)

#### 5.2.3 Dokumentations-Anforderungen

- **Aktualisierte Model Card:** 
  - Paper-Track-Performance-Metriken
  - Vergleich Paper-Track vs. Backtest
  - Identifizierte Limitationen und Risiken
- **Paper-Track-Report:**
  - Zusammenfassung der Paper-Track-Performance
  - Metriken-Vergleich (Paper-Track vs. Backtest)
  - Regime-Performance-Analyse
  - Factor-Exposures-Analyse
- **Entscheidungs-Log:**
  - Alle WARN/CRITICAL-Events dokumentiert
  - Root-Cause-Analysen dokumentiert
  - Entscheidungsgrundlage fur Go/No-Go

### 5.3 Go/No-Go-Entscheidung

**Go-Entscheidung:**

- Alle quantitativen Kriterien erfullt
- Qualitative Kriterien erfullt
- Vollstandige Dokumentation vorhanden
- **Aber:** Finale Entscheidung liegt bei Risk-Management und Compliance

**No-Go-Entscheidung:**

- Quantitative Kriterien nicht erfullt
- Oder: Qualitative Bedenken (z.B. zu hohe Volatilitat, zu hohe Kosten)
- Oder: Regulatory/Compliance-Bedenken
- Oder: Operational Readiness nicht gegeben

---

## 6. Checklisten & Templates

### 6.1 Backtest → Paper Gate Checklist

| Kriterium | Requirement | Messung | Status | Kommentare |
|-----------|-------------|---------|--------|------------|
| Backtest-Dauer | >= 5 Jahre oder mehrere Regime | Zeitraum aus Backtest-Config | ☐ | |
| Regime-Coverage | >= 3 verschiedene Regime evaluiert | `risk_by_regime.csv` aus Risk-Report | ☐ | |
| Deflated Sharpe | >= 0.5 (konservativ) | Deflated Sharpe aus Factor-/ML-Reports | ☐ | |
| Max Drawdown | <= -30% (anpassbar) | `max_drawdown_pct` aus Risk-Report | ☐ | |
| PIT-Sicherheit | Keine Look-Ahead-Violations | PIT-Checks aus QA-Report | ☐ | |
| Walk-Forward (opt.) | OOS-Performance konsistent | Walk-Forward-Results | ☐ | |
| Factor Exposures (opt.) | Klares Faktor-Profil | `factor_exposures_summary.csv` | ☐ | |
| QA-Status | QA-Gates bestanden | QA-Report | ☐ | |
| Dokumentation | Backtest-Report vorhanden | `performance_report.md`, `risk_report.md` | ☐ | |
| Model Card (opt.) | Beschreibung der Strategie | Model-Card-Dokument | ☐ | |

**Gate-Entscheidung:** ☐ PASSED → Phase 1 | ☐ FAILED → Weiterer Research

**Entscheider:** _________________ **Datum:** _________________

---

### 6.2 Paper → Live Gate Checklist

| Kriterium | Requirement | Messung | Status | Kommentare |
|-----------|-------------|---------|--------|------------|
| Paper-Track-Dauer | >= 12 Monate | Start-Datum vs. aktuelles Datum | ☐ | |
| Paper-Track Sharpe | >= Backtest-Sharpe * 0.8 | Sharpe aus Paper-Track-Report | ☐ | |
| Paper-Track MaxDD | <= Backtest-MaxDD * 1.2 | MaxDD aus Paper-Track-Report | ☐ | |
| Deflated Sharpe | >= 0.5 | Deflated Sharpe aus Paper-Track | ☐ | |
| Hit-Rate | >= Backtest-Hit-Rate - 5pp | Hit-Rate aus Paper-Track-Report | ☐ | |
| Regime-Performance | Profitabel in >= 2/3 Hauptregimen | Regime-Performance-Analyse | ☐ | |
| Konsistenz | Keine systematischen Abweichungen | Vergleich Paper-Track vs. Backtest | ☐ | |
| Operational Readiness | EOD-Pipeline stabil | QA-Status-Historie | ☐ | |
| Dokumentation | Aktualisierte Model Card vorhanden | Model-Card-Dokument | ☐ | |
| Paper-Track-Report | Vollstandiger Report vorhanden | Paper-Track-Report-Dokument | ☐ | |
| Entscheidungs-Log | WARN/CRITICAL-Events dokumentiert | Decision-Log-Dokument | ☐ | |

**Gate-Entscheidung:** ☐ GO → Live-Trading-Vorbereitung | ☐ NO-GO → Weiterer Paper-Track oder Kill

**Entscheider:** _________________ **Datum:** _________________

**Zusatzliche Checks (ausserhalb dieses Playbooks):**

- ☐ Regulatory Compliance Review
- ☐ Legal Review
- ☐ Risk-Management-Framework Review
- ☐ Operational Excellence Review
- ☐ Capital Allocation Decision

---

## 7. References

**Design Documents:**

- [Point-in-Time & Latency (B2)](POINT_IN_TIME_AND_LATENCY.md) - PIT-Sicherheit und Look-Ahead-Bias-Prevention
- [Walk-Forward & Regime Analysis (B3)](WALK_FORWARD_AND_REGIME_B3_DESIGN.md) - Regime-Detection und Regime-Performance-Analyse
- [Deflated Sharpe Ratio (B4)](DEFLATED_SHARPE_B4_DESIGN.md) - Multiple-Testing-Adjustment und Deflated Sharpe
- [Signal API & Factor Exposures (A2)](SIGNAL_API_AND_FACTOR_EXPOSURES_A2_DESIGN.md) - Factor-Exposure-Analyse
- [Advanced Analytics Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md) - Gesamte Roadmap

**Workflow Documentation:**

- [Risk Metrics & Attribution Workflows](WORKFLOWS_RISK_METRICS_AND_ATTRIBUTION.md) - Risk-Report-Generierung und Factor-Exposures
- [Operations Backend Runbook](OPERATIONS_BACKEND.md) - Daily Health-Checks und Monitoring
- [Use Cases & Roles](USE_CASES_AND_ROLES_A1.md) - Rollen-Definitionen (Researcher, Operator, Risk/QA)
- [Factor Analysis Workflows](WORKFLOWS_FACTOR_ANALYSIS.md) - Factor-Analyse-Workflows

**Code References:**

- `scripts/run_daily.py` - EOD-Pipeline fur Paper-Track
- `scripts/generate_risk_report.py` - Risk-Report-Generierung mit Regime-Analyse und Factor-Exposures
- `scripts/check_health.py` - Health-Checks fur Paper-Track-Monitoring
- `src/assembled_core/qa/point_in_time_checks.py` - PIT-Sicherheits-Checks
- `src/assembled_core/risk/factor_exposures.py` - Factor-Exposure-Berechnung

---

