# Transaction Cost Analysis & Execution Constraints Design (E4)

**Phase E4** – Advanced Analytics & Factor Labs

**Status:** 📋 Design Phase  
**Last Updated:** 2025-12-13

---

## Overview

**Ziel:** Einfache Transaction Cost Analysis (TCA) für Backtest-Strategien mit Schätzung von Execution-Kosten und deren Auswirkung auf Net-Returns und Performance-Metriken.

Dieses Design-Dokument beschreibt die geplante Erweiterung für E4:
- **Per-Trade Cost Estimation**: Commission + Spread/2 + Slippage
- **Round-Trip Cost Calculation**: Gesamtkosten pro Trade und pro Strategie
- **Net-Return Analysis**: Cost-adjusted Returns, Net-Sharpe, Cost vs. Gross-Return
- **Cost Attribution**: Optional nach Faktor-Gruppen oder Regimes

**Bezug zu bestehenden Modulen:**
- Baut auf `costs.py` auf (CostModel für Backtests)
- Nutzt `risk/risk_metrics.py` für Net-Return-Metriken
- Integriert mit `generate_risk_report.py` (optionaler TCA-Teil)
- Verwendet Trades aus `backtest_engine.py` (BacktestResult.trades)

**Datenbasis:** Alle Analysen basieren auf **lokalen Backtest-Outputs** (trades.csv/parquet). Keine Live-Market-Microstructure-Daten.

**Wichtig:** Dieses Modul fokussiert auf **einfache, approximative TCA** für Backtests, nicht auf hochfrequente Live-Execution-Analysen.

---

## Scope E4

### 1. Per-Trade Cost Estimation

**Simple Cost Model:**
```
cost_per_trade = commission + spread/2 + slippage
```

**Komponenten:**
- **Commission**: Fixe Gebühr pro Trade (z.B. 0.5 bps, bereits in CostModel vorhanden)
- **Spread Cost**: Half-spread Approximation (`spread_cost = price * (spread_bps / 2) / 10000`)
  - Spread kann aus Vol/Liq-Faktoren approximiert werden (z.B. `rv_20`, `turnover` falls vorhanden)
  - Fallback: Fixer Spread (z.B. 5-10 bps je nach Asset-Klasse)
- **Slippage**: Market Impact Approximation
  - Kann aus Trade-Größe relativ zum durchschnittlichen Volumen approximiert werden
  - Fallback: Fixer Slippage (z.B. 2-5 bps)

**Eingaben:**
- `trades_df`: DataFrame mit `timestamp`, `symbol`, `side`, `qty`, `price`
- Optional: `factor_panel_df` mit Vol/Liq-Faktoren für Spread-Approximation
- Optional: `price_panel_df` für Volume-Daten (falls verfügbar)

**Ausgabe:**
- `cost_per_trade`: Series mit geschätzten Kosten pro Trade (in absoluten Werten)

### 2. TCA für Trades

**Berechnung:**
- Für jeden Trade: Kosten berechnen und zu `trades_df` hinzufügen
- Round-Trip Costs: Für jede vollständige Position (Long Entry + Long Exit oder Short Entry + Short Exit)
- Realized Net PnL: Gross PnL - Costs

**Eingaben:**
- `trades_df`: DataFrame mit Trades
- `cost_per_trade`: Series, float (konstante Kosten) oder dict (pro Symbol/Side)

**Ausgabe:**
- `tca_trades_df`: DataFrame mit zusätzlichen Spalten:
  - `cost_commission`: Commission-Kosten
  - `cost_spread`: Spread-Kosten
  - `cost_slippage`: Slippage-Kosten
  - `cost_total`: Gesamtkosten
  - `realized_pnl_gross`: Gross PnL (ohne Kosten)
  - `realized_pnl_net`: Net PnL (nach Kosten)

### 3. TCA Summary & Aggregation

**Zeitbasierte Aggregation:**
- Tägliche/Wöchentliche/Monatliche Kosten
- Kosten vs. Gross-Returns
- Cost Ratio (Costs / Gross-Return)

**Strategie-Level Aggregation:**
- Total Costs, Total Gross PnL, Total Net PnL
- Average Cost per Trade
- Round-Trip Cost Statistics
- Cost Impact auf Sharpe, Sortino, CAGR

**Eingaben:**
- `tca_trades_df`: DataFrame mit Cost-annotated Trades
- `freq`: Aggregations-Frequenz ("D", "W", "M")

**Ausgabe:**
- `tca_summary_df`: DataFrame mit aggregierten Metriken pro Periode

### 4. Cost-Adjusted Risk Metrics

**Net-Return Metriken:**
- Berechne Net-Returns aus Equity-Kurve + Costs
- Neuberechnung von Sharpe, Sortino, Max Drawdown mit Net-Returns
- Vergleich: Gross vs. Net Performance

**Eingaben:**
- `returns`: Series mit Gross-Returns (aus Equity-Kurve)
- `costs`: Series mit täglichen Kosten (aus TCA)
- `freq`: Frequenz für Annualisierung

**Ausgabe:**
- Dictionary mit Net-Performance-Metriken und Vergleichs-Metriken

### 5. Optional: Cost Attribution

**Nach Faktor-Gruppen:**
- Kosten pro Faktor-Kategorie (Trend, Vol/Liq, Earnings, etc.)
- Kann ähnlich wie `compute_risk_by_factor_group` implementiert werden

**Nach Regimes:**
- Kosten in verschiedenen Markt-Phasen (Bull, Bear, etc.)
- Identifikation von Regime-spezifischen Cost-Profilen

---

## Data Contracts

### Inputs

**1. `trades_df` (Required):**
```python
# DataFrame mit Spalten:
# - timestamp: pd.Timestamp (UTC)
# - symbol: str
# - side: str ("BUY" oder "SELL")
# - qty: float (positive Anzahl)
# - price: float (Trade-Preis)
```

**2. `positions_df` (Optional):**
```python
# DataFrame mit Spalten:
# - timestamp: pd.Timestamp (UTC)
# - symbol: str
# - weight: float (Portfolio-Gewicht) oder qty: float
```

**3. `price_panel_df` (Optional):**
```python
# DataFrame mit Spalten:
# - timestamp: pd.Timestamp (UTC)
# - symbol: str
# - close: float
# - volume: float (optional, für Slippage-Schätzung)
```

**4. `factor_panel_df` (Optional):**
```python
# DataFrame mit Spalten:
# - timestamp: pd.Timestamp (UTC)
# - symbol: str
# - rv_20: float (Realized Volatility, für Spread-Approximation)
# - turnover_20d: float (optional, für Spread/Liquidity-Schätzung)
```

### Outputs

**1. `tca_trades_df`:**
```python
# DataFrame mit allen Spalten aus trades_df plus:
# - cost_commission: float
# - cost_spread: float
# - cost_slippage: float
# - cost_total: float
# - realized_pnl_gross: float (optional)
# - realized_pnl_net: float (optional)
```

**2. `tca_summary_df`:**
```python
# DataFrame mit Spalten:
# - timestamp: pd.Timestamp (UTC)
# - total_cost: float
# - total_gross_pnl: float (optional)
# - total_net_pnl: float (optional)
# - n_trades: int
# - avg_cost_per_trade: float
# - cost_ratio: float (cost / gross_return, optional)
```

**3. `tca_by_factor_df` (Optional):**
```python
# DataFrame mit Spalten:
# - factor_group: str
# - total_cost: float
# - n_trades: int
# - avg_cost_per_trade: float
```

**4. `tca_by_regime_df` (Optional):**
```python
# DataFrame mit Spalten:
# - regime: str
# - total_cost: float
# - n_trades: int
# - avg_cost_per_trade: float
```

---

## Planned Functions

### Core Functions

**1. `estimate_per_trade_cost()`**
```python
def estimate_per_trade_cost(
    trades: pd.DataFrame,
    method: str = "simple",
    commission_bps: float = 0.5,
    spread_bps: float | None = None,
    slippage_bps: float = 3.0,
    factor_panel_df: pd.DataFrame | None = None,
    price_panel_df: pd.DataFrame | None = None,
    **kwargs
) -> pd.Series:
    """
    Schätzt Kosten pro Trade.
    
    Args:
        trades: DataFrame mit Trades (timestamp, symbol, side, qty, price)
        method: "simple" (fixed costs) oder "adaptive" (basierend auf Faktoren)
        commission_bps: Commission in Basis-Punkten
        spread_bps: Spread in Basis-Punkten (None = aus Faktoren schätzen)
        slippage_bps: Slippage in Basis-Punkten
        factor_panel_df: Optional DataFrame mit Vol/Liq-Faktoren
        price_panel_df: Optional DataFrame mit Prices/Volume
    
    Returns:
        Series mit geschätzten Kosten pro Trade (Index = trades.index)
    """
    pass
```

**2. `compute_tca_for_trades()`**
```python
def compute_tca_for_trades(
    trades: pd.DataFrame,
    cost_per_trade: pd.Series | float | dict,
    *,
    price_col: str = "price",
    compute_pnl: bool = False,
) -> pd.DataFrame:
    """
    Berechnet TCA für alle Trades.
    
    Args:
        trades: DataFrame mit Trades
        cost_per_trade: Series (pro Trade), float (konstant) oder dict (pro Symbol/Side)
        price_col: Name der Preis-Spalte
        compute_pnl: Ob PnL berechnet werden soll (erfordert Round-Trip-Matching)
    
    Returns:
        DataFrame mit zusätzlichen Cost- und PnL-Spalten
    """
    pass
```

**3. `summarize_tca()`**
```python
def summarize_tca(
    tca_trades: pd.DataFrame,
    freq: str = "D",
    equity_curve: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Aggregiert TCA-Metriken über Zeit.
    
    Args:
        tca_trades: DataFrame mit Cost-annotated Trades
        freq: Aggregations-Frequenz ("D", "W", "M")
        equity_curve: Optional Equity-Kurve für Return-Berechnung
    
    Returns:
        DataFrame mit aggregierten Metriken pro Periode
    """
    pass
```

**4. `compute_cost_adjusted_risk_metrics()`**
```python
def compute_cost_adjusted_risk_metrics(
    returns: pd.Series,
    costs: pd.Series,
    freq: Literal["1d", "5min"] = "1d",
    risk_free_rate: float = 0.0,
) -> dict[str, float | None]:
    """
    Berechnet Risk-Metriken mit Cost-adjusted Returns.
    
    Args:
        returns: Series mit Gross-Returns
        costs: Series mit täglichen Kosten (muss zu returns passen)
        freq: Frequenz für Annualisierung
        risk_free_rate: Risk-free Rate (annualisiert)
    
    Returns:
        Dictionary mit Net-Performance-Metriken und Vergleichs-Metriken
    """
    pass
```

---

## Integration

### Option A: Eigenes Script (Recommended)

**Neues Script:** `scripts/generate_tca_report.py`

**CLI-Argumente:**
- `--backtest-dir`: Pfad zu Backtest-Output-Verzeichnis
- `--output-dir`: Output-Verzeichnis (default: backtest-dir)
- `--commission-bps`: Commission in Basis-Punkten (default: 0.5)
- `--spread-bps`: Spread in Basis-Punkten (optional, sonst aus Faktoren geschätzt)
- `--factor-panel-file`: Optional Factor-Panel für Spread-Approximation
- `--regime-file`: Optional Regime-File für Cost-Attribution

**Outputs:**
- `tca_trades.csv/parquet`
- `tca_summary.csv/parquet`
- `tca_report.md`
- Optional: `tca_by_factor.csv`, `tca_by_regime.csv`

### Option B: Integration in Risk Report

**Erweiterung von:** `scripts/generate_risk_report.py`

**Neue Flag:** `--with-tca`

Wenn gesetzt:
- Lädt `trades.csv/parquet` aus Backtest-Dir
- Führt TCA durch
- Ergänzt Risk-Report um TCA-Sektion
- Speichert zusätzliche TCA-Dateien

---

## Implementation Steps

### E4.1: Core Module (`transaction_costs.py`)

**Aufgaben:**
1. Implementiere `estimate_per_trade_cost()` mit "simple" method
2. Implementiere `compute_tca_for_trades()` mit Basic-PnL-Matching
3. Implementiere `summarize_tca()` mit täglicher Aggregation
4. Implementiere `compute_cost_adjusted_risk_metrics()` mit Net-Return-Berechnung

**Dependencies:**
- Nutzt `risk/risk_metrics.py` für Net-Return-Metriken
- Nutzt `costs.py` für CostModel-Parameter (falls vorhanden)

### E4.2: Script (`generate_tca_report.py`)

**Aufgaben:**
1. CLI-Argumente definieren
2. Daten-Loading (trades, optional factor_panel, optional regime)
3. TCA-Funktionen aufrufen
4. Markdown-Report generieren
5. CSV/Parquet-Outputs speichern

### E4.3: Tests

**Test-Dateien:**
- `tests/test_risk_transaction_costs.py`

**Tests:**
- `test_estimate_per_trade_cost_simple()`: Fixed costs
- `test_compute_tca_for_trades()`: Cost annotation
- `test_summarize_tca()`: Aggregation
- `test_compute_cost_adjusted_risk_metrics()`: Net-Return-Metriken
- `test_tca_report_integration()`: End-to-End mit Dummy-Trades

### E4.4: Documentation

**Updates:**
- Workflow-Dokument: `docs/WORKFLOWS_RISK_METRICS_AND_ATTRIBUTION.md` → TCA-Sektion
- `docs/ADVANCED_ANALYTICS_FACTOR_LABS.md` → E4 Status

---

## Limitations & Future Enhancements

**Limitations (Scope E4):**
- Einfache Cost-Modelle (keine intraday-Microstructure)
- Approximative Spread/Slippage-Schätzungen
- Round-Trip-Matching ist einfach (First-In-First-Out)

**Future Enhancements:**
- Adaptive Cost-Modelle basierend auf Volatility/Liquidity-Faktoren
- Regime-spezifische Cost-Schätzungen
- Market-Impact-Modelle (basierend auf Trade-Größe relativ zu Volume)
- Integration mit Live-Execution-Layer (Post-Trade TCA)

---

## References

- **Cost Model:** `src/assembled_core/costs.py` (CostModel, get_default_cost_model)
- **Risk Metrics:** `src/assembled_core/risk/risk_metrics.py` (compute_basic_risk_metrics)
- **Backtest Engine:** `src/assembled_core/qa/backtest_engine.py` (BacktestResult.trades)
- **Risk Report:** `scripts/generate_risk_report.py`
- **Factor Labs Roadmap:** `docs/ADVANCED_ANALYTICS_FACTOR_LABS.md`

