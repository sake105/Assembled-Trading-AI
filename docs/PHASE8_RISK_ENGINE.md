# Phase 8: Risk Engine & Scenario Analysis

## Übersicht

Phase 8 fügt einen umfassenden Risk-Engine-Layer hinzu, der Portfolio-Risiko-Analysen, Szenario-Simulationen und Shipping-/Systemic-Risk-Bewertungen ermöglicht.

**Komponenten:**
- **Portfolio Risk Metrics** (`qa.risk_metrics`): VaR, ES, Volatilität
- **Scenario Engine** (`qa.scenario_engine`): Stress-Tests und Szenario-Simulationen
- **Shipping Risk** (`qa.shipping_risk`): Shipping-Exposure und Systemic-Risk-Flags

## Portfolio Risk Metrics

### Modul: `src/assembled_core/qa/risk_metrics.py`

**Funktion:** `compute_portfolio_risk_metrics(equity, freq)`

Berechnet Portfolio-Risiko-Kennzahlen aus Equity-Kurven:

- **daily_vol**: Tägliche Volatilität (Standardabweichung der täglichen Returns)
- **ann_vol**: Annualisierte Volatilität
- **max_drawdown**: Maximaler Drawdown (wiederverwendet aus `qa.metrics`)
- **var_95**: Value at Risk bei 95% Konfidenz (historisch, in absoluten Werten)
- **es_95**: Expected Shortfall bei 95% Konfidenz (historisch, in absoluten Werten)

**Beispiel:**
```python
from src.assembled_core.qa.risk_metrics import compute_portfolio_risk_metrics
import pandas as pd

equity = pd.Series([10000, 10100, 10050, 10200, 10100])
risk_metrics = compute_portfolio_risk_metrics(equity, freq="1d")

print(f"Daily Vol: {risk_metrics['daily_vol']}")
print(f"VaR 95%: {risk_metrics['var_95']}")
print(f"ES 95%: {risk_metrics['es_95']}")
```

**Integration in Reports:**
Die Risk-Metriken werden automatisch in QA-Reports eingefügt, wenn `equity` DataFrame bereitgestellt wird.

## Scenario Engine

### Modul: `src/assembled_core/qa/scenario_engine.py`

**Szenario-Definition:**
```python
@dataclass
class Scenario:
    name: str
    shock_type: Literal["equity_crash", "vol_spike", "shipping_blockade"]
    shock_magnitude: float
    shock_start: datetime | None = None
    shock_end: datetime | None = None
```

**Unterstützte Szenarien:**

1. **equity_crash**: Prozentualer Preisrückgang
   - `shock_magnitude`: Prozentuale Änderung (z.B. -0.20 für -20%)
   - Preise werden ab `shock_start` um den Shock skaliert

2. **vol_spike**: Erhöhte Volatilität
   - `shock_magnitude`: Multiplikator auf Return-Volatilität (z.B. 2.0 für 2x)
   - Returns im Shock-Fenster werden multipliziert

3. **shipping_blockade**: Stärkerer Shock für Shipping-exponierte Symbole
   - Basis-Shock für alle Symbole
   - Zusätzlicher Shock für Symbole mit "SHIP" im Namen

**Beispiel:**
```python
from src.assembled_core.qa.scenario_engine import Scenario, apply_scenario_to_prices
from datetime import datetime, timezone

scenario = Scenario(
    name="Market Crash 2024",
    shock_type="equity_crash",
    shock_magnitude=-0.20,  # -20%
    shock_start=datetime(2024, 3, 15, tzinfo=timezone.utc)
)

shocked_prices = apply_scenario_to_prices(prices_df, scenario)
```

**Helper-Funktion:**
```python
from src.assembled_core.qa.scenario_engine import run_scenario_on_equity

results = run_scenario_on_equity(equity_series, scenario, freq="1d")
print(f"Baseline VaR: {results['baseline_metrics']['var_95']}")
print(f"Shocked VaR: {results['shocked_metrics']['var_95']}")
print(f"Delta: {results['delta_metrics']['var_95']}")
```

## Shipping Risk

### Modul: `src/assembled_core/qa/shipping_risk.py`

**Funktion:** `compute_shipping_exposure(portfolio_positions, shipping_features)`

Berechnet Portfolio-gewichtete Shipping-Exposure-Metriken:

- **avg_shipping_congestion**: Portfolio-gewichteter Durchschnitt der Congestion-Scores
- **high_congestion_weight**: Summe der Gewichte für Positionen mit Congestion > Schwellwert
- **top_routes**: Liste der häufigsten Route-IDs im Portfolio
- **exposed_symbols**: Liste der Symbole mit hoher Congestion

**Funktion:** `compute_systemic_risk_flags(shipping_exposure)`

Generiert Systemic-Risk-Flags:

- **high_shipping_risk**: True wenn avg_congestion > Schwellwert oder high_exposure > Schwellwert
- **exposed_to_blockade_routes**: True wenn exponiert zu Blockade-Routen
- **risk_level**: "LOW", "MEDIUM", oder "HIGH"
- **risk_reason**: Human-readable Grund für Risk-Level

**Beispiel:**
```python
from src.assembled_core.qa.shipping_risk import (
    compute_shipping_exposure,
    compute_systemic_risk_flags
)

portfolio = pd.DataFrame({
    "symbol": ["AAPL", "MSFT", "GOOGL"],
    "weight": [0.4, 0.3, 0.3]
})

shipping_features = pd.DataFrame({
    "symbol": ["AAPL", "MSFT", "GOOGL"],
    "shipping_congestion_score": [80.0, 45.0, 30.0]
})

exposure = compute_shipping_exposure(portfolio, shipping_features)
flags = compute_systemic_risk_flags(exposure)

print(f"Avg Congestion: {exposure['avg_shipping_congestion']:.2f}")
print(f"Risk Level: {flags['risk_level']}")
```

**Integration in Reports:**
Shipping-Risk-Metriken werden automatisch in QA-Reports eingefügt, wenn `portfolio_positions` und `shipping_features` bereitgestellt werden.

## Beispiel-Flows

### Flow 1: Backtest → Equity → Risk Metrics

```python
from src.assembled_core.qa.backtest_engine import run_portfolio_backtest
from src.assembled_core.qa.risk_metrics import compute_portfolio_risk_metrics

# Run backtest
result = run_portfolio_backtest(...)
equity = result.equity["equity"]

# Compute risk metrics
risk_metrics = compute_portfolio_risk_metrics(equity, freq="1d")
print(f"VaR 95%: {risk_metrics['var_95']}")
```

### Flow 2: Backtest + Szenario → Vergleich vor/nach

```python
from src.assembled_core.qa.scenario_engine import Scenario, run_scenario_on_equity

# Run backtest
result = run_portfolio_backtest(...)
equity = result.equity["equity"]

# Define scenario
scenario = Scenario(
    name="Market Crash",
    shock_type="equity_crash",
    shock_magnitude=-0.20
)

# Compare before/after
results = run_scenario_on_equity(equity, scenario, freq="1d")
print(f"Baseline VaR: {results['baseline_metrics']['var_95']}")
print(f"Shocked VaR: {results['shocked_metrics']['var_95']}")
print(f"Delta: {results['delta_metrics']['var_95']}")
```

### Flow 3: Portfolio + Shipping-Features → Shipping Risk

```python
from src.assembled_core.qa.shipping_risk import (
    compute_shipping_exposure,
    compute_systemic_risk_flags
)
from src.assembled_core.features.shipping_features import add_shipping_features

# Get portfolio positions (from backtest or manual)
portfolio = pd.DataFrame({
    "symbol": ["AAPL", "MSFT"],
    "weight": [0.6, 0.4]
})

# Get shipping features (from prices with features)
prices_with_features = add_shipping_features(prices_df, shipping_events)
shipping_features = prices_with_features[
    ["symbol", "shipping_congestion_score"]
].drop_duplicates()

# Compute exposure and flags
exposure = compute_shipping_exposure(portfolio, shipping_features)
flags = compute_systemic_risk_flags(exposure)

print(f"Risk Level: {flags['risk_level']}")
print(f"Exposed Symbols: {exposure['exposed_symbols']}")
```

## Integration in QA Reports

Alle Risk-Engine-Komponenten können optional in QA-Reports integriert werden:

```python
from src.assembled_core.reports.daily_qa_report import generate_qa_report

report_path = generate_qa_report(
    metrics=metrics,
    gate_result=gate_result,
    strategy_name="my_strategy",
    freq="1d",
    equity=equity_df,  # For risk_metrics
    portfolio_positions=portfolio_df,  # For shipping_risk
    shipping_features=shipping_features_df  # For shipping_risk
)
```

## Weitere Informationen

- **Risk Metrics**: `src/assembled_core/qa/risk_metrics.py`
- **Scenario Engine**: `src/assembled_core/qa/scenario_engine.py`
- **Shipping Risk**: `src/assembled_core/qa/shipping_risk.py`
- **Tests**: `tests/test_qa_risk_metrics.py`, `tests/test_qa_scenario_engine.py`, `tests/test_qa_shipping_risk.py`

