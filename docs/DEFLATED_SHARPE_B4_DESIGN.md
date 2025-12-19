# Deflated Sharpe Ratio & Factor-Zoo Protection (B4)

**Phase B4** - Advanced Analytics & Factor Labs

**Status:** Design Phase  
**Last Updated:** 2025-01-XX

---

## 1. Overview & Goals

**Ziel:** Schutz vor False Discovery Rate (FDR) und Multiple Testing Bias bei der Faktor- und Modell-Evaluation durch Deflated Sharpe Ratio (DSR) nach Bailey & Lopez de Prado (2014).

**Problem:**
- Bei vielen getesteten Faktoren/Parametern/Modellen steigt die Wahrscheinlichkeit, dass zufaellig hohe Sharpe Ratios durch "Factor Zoo" / "Multiple Testing" entstehen
- Klassische Sharpe Ratio beruecksichtigt nicht, wie viele Varianten getestet wurden
- Harvey/Liu/Zhu (2016) zeigen: Bei 1000 getesteten Faktoren ist ein Sharpe von 2.0 nicht mehr signifikant

**Loesung:**
- Deflated Sharpe Ratio: Adjustiert Sharpe Ratio um erwarteten Maximum-Sharpe unter Null-Hypothese
- Beruecksichtigt: Anzahl Tests (n_tests), Anzahl Beobachtungen (n_obs), Verteilungs-Eigenschaften (Skewness, Kurtosis)
- Integration in alle Experiment-/Ranking-Reports: n_tests und sharpe_deflated als zusaetzliche Spalten

**Bezug zu bestehenden Modulen:**
- Baut auf `qa/metrics.py` auf (compute_sharpe_ratio)
- Erweitert `qa/factor_analysis.py` (bereits compute_deflated_sharpe_ratio vorhanden, aber noch nicht vollstaendig integriert)
- Integration in `qa/factor_ranking.py` (bereits ls_deflated_sharpe verwendet)
- Integration in ML-Validation-Reports (`research/ml/model_zoo_factor_validation.py`)
- Integration in Factor-Analysis-Reports (`scripts/run_factor_analysis.py`)

**Datenbasis:** Alle Analysen basieren auf lokalen Backtest-Outputs. Keine Live-APIs.

---

## 2. API Design

### 2.1 Kernfunktion

**Modul:** `src/assembled_core/qa/metrics.py`

```python
def deflated_sharpe_ratio(
    sharpe_annual: float,
    n_obs: int,
    n_tests: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Compute Deflated Sharpe Ratio (DSR) to adjust for multiple testing.
    
    The Deflated Sharpe Ratio adjusts the observed Sharpe Ratio for:
    - Multiple testing (False Discovery Rate)
    - Non-normal return distributions (skewness, kurtosis)
    
    Formula (Bailey & Lopez de Prado 2014):
        DSR = (SR - E[max_SR]) / std(SR)
        where:
        - E[max_SR] = expected maximum Sharpe under null (multiple testing)
        - std(SR) = standard deviation of Sharpe (distribution adjustment)
    
    Args:
        sharpe_annual: Observed annualized Sharpe Ratio
        n_obs: Number of return observations (time periods, not years)
            Example: 252 daily returns = n_obs=252 (not n_obs=1 year)
        n_tests: Effective number of tests (factors x parameter combinations x models)
            Example: 50 factors x 3 parameter sets x 2 models = n_tests=300
        skew: Skewness of returns (default: 0.0, assumes normal)
        kurtosis: Kurtosis of returns (default: 3.0, assumes normal)
            Excess kurtosis = kurtosis - 3.0
    
    Returns:
        Deflated Sharpe Ratio (float)
        - Positive DSR: Significant Sharpe after adjustment
        - Negative DSR: Sharpe may be due to luck/multiple testing
        - NaN: If inputs invalid (n_obs < 2, sharpe is NaN/Inf)
    
    Properties:
        - sharpe_deflated <= sharpe_annual (always)
        - For n_tests=1 and large n_obs: sharpe_deflated ≈ sharpe_annual
        - For growing n_tests (fixed sharpe): sharpe_deflated decreases
    
    References:
        Bailey, D. H., & Lopez de Prado, M. (2014). The deflated Sharpe ratio:
        Correcting for selection bias, backtest overfitting and non-normality.
        Journal of Portfolio Management, 40(5), 94-107.
        
        Harvey, C. R., Liu, Y., & Zhu, H. (2016). ... and the cross-section
        of expected returns. Review of Financial Studies, 29(1), 5-68.
    """
```

**Hinweis:** Es existiert bereits eine aehnliche Funktion `compute_deflated_sharpe_ratio` in `qa/factor_analysis.py` mit Parameter `n_trials` statt `n_tests`. Diese sollte konsolidiert/vereinheitlicht werden.

### 2.2 Convenience-Funktion

```python
def deflated_sharpe_ratio_from_returns(
    returns: pd.Series,
    n_tests: int,
    scale: Literal["daily", "monthly", "annual"] = "daily",
    risk_free_rate: float = 0.0,
    skew: float | None = None,
    kurtosis: float | None = None,
) -> float:
    """Compute Deflated Sharpe Ratio directly from returns.
    
    Convenience function that:
    1. Computes annualized Sharpe Ratio from returns
    2. Computes skewness/kurtosis if not provided
    3. Calls deflated_sharpe_ratio()
    
    Args:
        returns: Series of returns (daily, monthly, or annual)
        n_tests: Effective number of tests
        scale: Scale of returns ("daily", "monthly", "annual")
            Used for annualization of Sharpe
        risk_free_rate: Risk-free rate (annualized, default: 0.0)
        skew: Optional skewness (if None, computed from returns)
        kurtosis: Optional kurtosis (if None, computed from returns)
    
    Returns:
        Deflated Sharpe Ratio (float)
    """
```

---

## 3. Parameter-Definitionen

### 3.1 n_obs (Number of Observations)

**Definition:** Anzahl Return-Beobachtungen (Zeitperioden), nicht Jahre.

**Beispiele:**
- 252 taegliche Returns → n_obs = 252
- 60 monatliche Returns → n_obs = 60
- 5 jaehrliche Returns → n_obs = 5

**Wichtig:** n_obs ist die Anzahl der Return-Perioden, nicht die Anzahl der Jahre.

### 3.2 n_tests (Number of Tests)

**Definition:** Effektive Anzahl getesteter Varianten.

**Berechnung:**
- **Factor Analysis:** Anzahl Faktoren × Anzahl Parameter-Kombinationen
  - Beispiel: 50 Faktoren × 1 Parameter-Set = n_tests = 50
  - Beispiel: 20 Faktoren × 3 Lookback-Windows = n_tests = 60
- **ML Model Zoo:** Anzahl Modelle × Anzahl Hyperparameter-Kombinationen
  - Beispiel: 5 Modelle × 2 Alpha-Werte = n_tests = 10
  - Beispiel: 3 Modelle × 4 Hyperparameter-Sets = n_tests = 12
- **Combined:** Faktoren × Modelle × Parameter
  - Beispiel: 30 Faktoren × 3 Modelle × 2 Parameter-Sets = n_tests = 180

**Hinweis:** Bei korrelierten Tests (z.B. aehnliche Faktoren) kann n_tests effektiv niedriger sein. Fuer erste Implementierung verwenden wir die einfache Multiplikation.

---

## 4. Integration Plan

### 4.1 Experiment-/Ranking-Summary-Tables

**Folgende Reports bekommen Spalten `n_tests` und `sharpe_deflated`:**

1. **Factor Analysis Reports** (`scripts/run_factor_analysis.py`):
   - `factor_analysis_*_portfolio_summary.csv`: 
     - Neue Spalten: `n_tests`, `sharpe_deflated`
     - `n_tests`: Anzahl getesteter Faktoren (aus factor_set)
   - Markdown-Report: Erweitert um Deflated Sharpe Spalte

2. **ML Model Zoo Summary** (`research/ml/model_zoo_factor_validation.py`):
   - `ml_model_zoo_summary.csv`:
     - Neue Spalten: `n_tests`, `ls_sharpe_deflated` (fuer L/S Sharpe)
     - `n_tests`: Anzahl Modelle im Zoo (len(model_configs))
   - Markdown-Report: Erweitert um Deflated Sharpe

3. **ML Validation Reports** (`scripts/run_ml_factor_validation.py`):
   - `ml_portfolio_metrics_*.csv`:
     - Neue Spalten: `n_tests`, `ls_sharpe_deflated`
     - `n_tests`: 1 (single model) oder Anzahl Modelle bei Model-Zoo-Integration

4. **Factor Ranking** (`qa/factor_ranking.py`):
   - Bereits teilweise implementiert (`ls_deflated_sharpe`)
   - Erweitern um `n_tests` Spalte in Ranking-Output

### 4.2 Zentrale Metriken-Funktion

**Erweiterung von `qa/metrics.py`:**

- `PerformanceMetrics` Dataclass: Optional `deflated_sharpe_ratio: float | None` hinzufuegen
- `compute_all_metrics()`: Optional `n_tests: int = 1` Parameter hinzufuegen
  - Wenn `n_tests > 1`: Berechne `deflated_sharpe_ratio` automatisch
  - Wenn `n_tests = 1`: `deflated_sharpe_ratio = sharpe_ratio` (keine Adjustment)

---

## 5. Eigenschaften & Tests

### 5.1 Erwartete Eigenschaften

1. **Monotonie:** `sharpe_deflated <= sharpe_annual` (immer)
2. **n_tests=1:** Fuer `n_tests=1` und ausreichend grosse Stichprobe: `sharpe_deflated ≈ sharpe_annual`
3. **n_tests wachsend:** Fuer wachsendes `n_tests` bei fixem Sharpe: `sharpe_deflated` faellt
4. **n_obs wachsend:** Fuer wachsendes `n_obs` bei fixem Sharpe und `n_tests`: `sharpe_deflated` steigt (mehr Daten = mehr Signifikanz)

### 5.2 Test-Cases

**Unit Tests (`tests/test_qa_deflated_sharpe.py`):**

1. `test_deflated_sharpe_monotonicity`: Verifiziere `sharpe_deflated <= sharpe_annual`
2. `test_deflated_sharpe_n_tests_one`: Fuer `n_tests=1` sollte `sharpe_deflated ≈ sharpe_annual` (mit Toleranz)
3. `test_deflated_sharpe_increases_with_n_obs`: Fuer wachsendes `n_obs` steigt `sharpe_deflated`
4. `test_deflated_sharpe_decreases_with_n_tests`: Fuer wachsendes `n_tests` faellt `sharpe_deflated`
5. `test_deflated_sharpe_handles_skew_kurtosis`: Test mit non-normal returns (skew != 0, kurtosis != 3)
6. `test_deflated_sharpe_edge_cases`: NaN/Inf inputs, n_obs < 2, n_tests = 0

**Integration Tests:**

1. `test_factor_analysis_includes_deflated_sharpe`: Factor-Analysis-Report enthaelt `sharpe_deflated`
2. `test_model_zoo_includes_deflated_sharpe`: Model-Zoo-Summary enthaelt `ls_sharpe_deflated`
3. `test_factor_ranking_includes_n_tests`: Factor-Ranking enthaelt `n_tests` Spalte

---

## 6. Implementation Plan

### B4.1: Kernfunktion konsolidieren

**Tasks:**
1. Pruefe bestehende `compute_deflated_sharpe_ratio` in `qa/factor_analysis.py`
2. Verschiebe/dupliziere nach `qa/metrics.py` mit einheitlicher Signatur (`n_tests` statt `n_trials`)
3. Erstelle `deflated_sharpe_ratio_from_returns()` Convenience-Funktion
4. Unit Tests fuer beide Funktionen

### B4.2: Integration in Factor Analysis

**Tasks:**
1. Erweitere `run_factor_analysis.py`:
   - Berechne `n_tests` aus factor_set (Anzahl Faktoren)
   - Berechne `sharpe_deflated` fuer jeden Faktor in Portfolio-Summary
   - Schreibe `n_tests` und `sharpe_deflated` in CSV
2. Erweitere Markdown-Report um Deflated Sharpe Spalte
3. Integration Tests

### B4.3: Integration in ML Validation

**Tasks:**
1. Erweitere `run_ml_factor_validation.py`:
   - Berechne `n_tests` (1 fuer single model, oder Anzahl Modelle bei Model-Zoo)
   - Berechne `ls_sharpe_deflated` in Portfolio-Metriken
   - Schreibe in `ml_portfolio_metrics_*.csv`
2. Erweitere `model_zoo_factor_validation.py`:
   - Berechne `n_tests = len(model_configs)`
   - Berechne `ls_sharpe_deflated` pro Modell
   - Schreibe in `ml_model_zoo_summary.csv`
3. Integration Tests

### B4.4: PerformanceMetrics Erweiterung

**Tasks:**
1. Erweitere `PerformanceMetrics` Dataclass um `deflated_sharpe_ratio: float | None`
2. Erweitere `compute_all_metrics()` um optional `n_tests: int = 1` Parameter
3. Automatische Berechnung von `deflated_sharpe_ratio` wenn `n_tests > 1`
4. Rueckwaerts-Kompatibilitaet: Wenn `n_tests=1` oder nicht gesetzt, `deflated_sharpe_ratio=None`

---

## 7. References & Guidelines

**Research Guidelines:**
- Siehe `docs/RESEARCH_ROADMAP.md` fuer Guidelines zur Faktor-Evaluation
- Empfehlung: Nur Faktoren/Modelle mit `sharpe_deflated > 0.5` als signifikant betrachten
- Bei `n_tests > 100`: Deflated Sharpe wird sehr konservativ (erwartetes Maximum steigt)

**Literatur:**
- Bailey, D. H., & Lopez de Prado, M. (2014). The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting and non-normality. Journal of Portfolio Management, 40(5), 94-107.
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). ... and the cross-section of expected returns. Review of Financial Studies, 29(1), 5-68.

---

## 8. Known Limitations

1. **Korrelierte Tests:** Formel nimmt Unabhaengigkeit an. Bei korrelierten Faktoren (z.B. aehnliche Momentum-Faktoren) ist n_tests effektiv niedriger.
2. **Vereinfachte Formel:** Implementierung verwendet vereinfachte Approximation. Vollstaendige Formel erfordert scipy.stats (optional dependency).
3. **Skewness/Kurtosis:** Standardmaessig normal angenommen (skew=0, kurtosis=3). Fuer genaue Ergebnisse sollten diese aus Returns berechnet werden.

---

## 9. Success Criteria

- [ ] `deflated_sharpe_ratio()` Funktion in `qa/metrics.py` implementiert
- [ ] `deflated_sharpe_ratio_from_returns()` Convenience-Funktion implementiert
- [ ] Unit Tests (6+ Tests) alle passing
- [ ] Factor Analysis Reports enthalten `n_tests` und `sharpe_deflated`
- [ ] ML Model Zoo Summary enthaelt `n_tests` und `ls_sharpe_deflated`
- [ ] `PerformanceMetrics` erweitert um `deflated_sharpe_ratio`
- [ ] Integration Tests (3+ Tests) alle passing
- [ ] Dokumentation aktualisiert (Workflows, README)

---

## 10. Implementation Notes

**Bestehende Implementierung:**
- `compute_deflated_sharpe_ratio()` existiert bereits in `qa/factor_analysis.py`
- Verwendet `n_trials` statt `n_tests` (konsolidieren)
- Wird bereits in `qa/factor_ranking.py` verwendet (`ls_deflated_sharpe`)
- Muss in `qa/metrics.py` verschoben/konsolidiert werden

**Migration:**
- Alte Funktion in `factor_analysis.py` als Deprecated markieren
- Neue Funktion in `metrics.py` als Primary API
- Alte Aufrufe schrittweise migrieren

