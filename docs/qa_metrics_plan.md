# Plan: Zentrales Metrik-Modul `src/assembled_core/qa/metrics.py`

## Analyse bestehender Implementierungen

### Aktuelle Metriken-Verteilung

**1. `pipeline/backtest.py::compute_metrics()`:**
- `final_pf`: Final Performance Factor (equity[-1] / equity[0])
- `sharpe`: Sharpe Ratio (einfache Berechnung, **ohne Annualisierung**)
- `rows`, `first`, `last`: Metadaten

**2. `pipeline/portfolio.py::simulate_with_costs()`:**
- `final_pf`: Final Performance Factor
- `sharpe`: Sharpe Ratio (**mit Annualisierung** basierend auf `freq`)
- `trades`: Anzahl Trades

**3. `api/routers/risk.py::get_risk_summary()`:**
- `sharpe_ratio`: Sharpe Ratio (mit Annualisierung)
- `max_drawdown`: Maximaler Drawdown (absolut)
- `max_drawdown_pct`: Maximaler Drawdown (in Prozent)
- `current_drawdown`: Aktueller Drawdown
- `volatility`: Volatilität (annualisiert)
- `var_95`: Value at Risk (95% Konfidenz, historisch)

### Identifizierte Probleme

1. **Inkonsistente Sharpe-Berechnung:**
   - `pipeline/backtest.py`: Keine Annualisierung
   - `pipeline/portfolio.py`: Mit Annualisierung (252 für "1d", 252*78 für "5min")
   - `api/routers/risk.py`: Mit Annualisierung (gleiche Logik wie portfolio.py)

2. **Duplizierte Logik:**
   - Sharpe-Berechnung in 3 verschiedenen Stellen
   - Drawdown-Berechnung nur in `risk.py`, nicht in `compute_metrics()`

3. **Fehlende Metriken:**
   - **CAGR** (Compound Annual Growth Rate)
   - **Sortino Ratio** (Downside-Risk-adjusted Return)
   - **Hit Rate** (Win Rate, benötigt Trades-DataFrame)
   - **Turnover** (Portfolio-Turnover, benötigt Trades-DataFrame)
   - **Calmar Ratio** (Return / Max Drawdown)
   - **Profit Factor** (benötigt Trades-DataFrame)

4. **Keine einheitliche API:**
   - Verschiedene Return-Typen (dict vs. Pydantic-Model)
   - Verschiedene Input-Formate (Equity-DataFrame vs. Trades-DataFrame)

---

## Vorschlag: Zentrales Metrik-Modul

### Ziel

Ein zentrales Modul `src/assembled_core/qa/metrics.py`, das:
- Alle Performance-Metriken konsistent berechnet
- Einheitliche API für Equity- und Trades-basierte Metriken bietet
- Type-safe Output (Dataclass) liefert
- Bestehende Funktionen schrittweise ersetzen kann

---

## API-Design

### Input-Typen

**1. Equity-DataFrame:**
```python
pd.DataFrame mit Spalten:
- timestamp: pd.Timestamp (UTC)
- equity: float
- (optional) daily_return: float (wenn vorhanden, wird verwendet, sonst berechnet)
```

**2. Trades-DataFrame:**
```python
pd.DataFrame mit Spalten:
- timestamp: pd.Timestamp (UTC)
- symbol: str
- side: str ("BUY" oder "SELL")
- qty: float
- price: float
```

**3. Zusätzliche Parameter:**
```python
- start_capital: float (für CAGR, Turnover)
- freq: str ("1d" oder "5min") (für Annualisierung)
- risk_free_rate: float (optional, default: 0.0, für Sharpe/Sortino)
```

### Output-Typ: Dataclass

```python
@dataclass
class PerformanceMetrics:
    """Performance metrics from equity curve."""
    
    # Performance
    final_pf: float  # Final Performance Factor
    total_return: float  # Total Return (final_pf - 1.0)
    cagr: float | None  # Compound Annual Growth Rate (None if < 1 year)
    
    # Risk-Adjusted Returns
    sharpe_ratio: float | None  # Sharpe Ratio (annualisiert)
    sortino_ratio: float | None  # Sortino Ratio (annualisiert)
    calmar_ratio: float | None  # Calmar Ratio (CAGR / |max_drawdown_pct|)
    
    # Risk Metrics
    max_drawdown: float  # Maximaler Drawdown (absolut, negativ)
    max_drawdown_pct: float  # Maximaler Drawdown (in Prozent, negativ)
    current_drawdown: float  # Aktueller Drawdown (absolut)
    volatility: float | None  # Volatilität (annualisiert)
    var_95: float | None  # Value at Risk (95% Konfidenz)
    
    # Trade Metrics (None wenn keine Trades-DataFrame übergeben)
    hit_rate: float | None  # Win Rate (Anteil gewinnender Trades)
    profit_factor: float | None  # Profit Factor (Gewinne / Verluste)
    avg_win: float | None  # Durchschnittlicher Gewinn pro Trade
    avg_loss: float | None  # Durchschnittlicher Verlust pro Trade
    turnover: float | None  # Portfolio-Turnover (annualisiert)
    total_trades: int | None  # Anzahl Trades
    
    # Metadata
    start_date: pd.Timestamp  # Erster Timestamp
    end_date: pd.Timestamp  # Letzter Timestamp
    periods: int  # Anzahl Perioden
    start_capital: float  # Startkapital
    end_equity: float  # End-Equity
```

### Hauptfunktionen

```python
def compute_all_metrics(
    equity: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    start_capital: float = 10000.0,
    freq: str = "1d",
    risk_free_rate: float = 0.0
) -> PerformanceMetrics:
    """Compute all performance metrics from equity curve and optional trades.
    
    Args:
        equity: DataFrame with columns: timestamp, equity (and optionally daily_return)
        trades: Optional DataFrame with columns: timestamp, symbol, side, qty, price
        start_capital: Starting capital (for CAGR, turnover)
        freq: Frequency string ("1d" or "5min") for annualization
        risk_free_rate: Risk-free rate (annualized) for Sharpe/Sortino
    
    Returns:
        PerformanceMetrics dataclass with all computed metrics
    """

def compute_equity_metrics(
    equity: pd.DataFrame,
    start_capital: float = 10000.0,
    freq: str = "1d",
    risk_free_rate: float = 0.0
) -> PerformanceMetrics:
    """Compute metrics from equity curve only (no trade-level metrics).
    
    Args:
        equity: DataFrame with columns: timestamp, equity
        start_capital: Starting capital
        freq: Frequency string ("1d" or "5min")
        risk_free_rate: Risk-free rate (annualized)
    
    Returns:
        PerformanceMetrics (trade metrics will be None)
    """

def compute_trade_metrics(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    start_capital: float = 10000.0,
    freq: str = "1d"
) -> dict[str, float | int]:
    """Compute trade-level metrics (hit_rate, profit_factor, turnover, etc.).
    
    Args:
        trades: DataFrame with columns: timestamp, symbol, side, qty, price
        equity: DataFrame with columns: timestamp, equity (for turnover calculation)
        start_capital: Starting capital
        freq: Frequency string ("1d" or "5min")
    
    Returns:
        Dictionary with trade metrics:
        - hit_rate: float
        - profit_factor: float
        - avg_win: float
        - avg_loss: float
        - turnover: float
        - total_trades: int
    """
```

### Helper-Funktionen

```python
def compute_sharpe_ratio(
    returns: pd.Series,
    freq: str = "1d",
    risk_free_rate: float = 0.0
) -> float | None:
    """Compute annualized Sharpe ratio.
    
    Args:
        returns: Series of returns
        freq: Frequency string ("1d" or "5min")
        risk_free_rate: Risk-free rate (annualized)
    
    Returns:
        Sharpe ratio (annualized) or None if insufficient data
    """

def compute_sortino_ratio(
    returns: pd.Series,
    freq: str = "1d",
    risk_free_rate: float = 0.0
) -> float | None:
    """Compute annualized Sortino ratio (downside deviation only).
    
    Args:
        returns: Series of returns
        freq: Frequency string ("1d" or "5min")
        risk_free_rate: Risk-free rate (annualized)
    
    Returns:
        Sortino ratio (annualized) or None if insufficient data
    """

def compute_drawdown(equity: pd.Series) -> tuple[pd.Series, float, float, float]:
    """Compute drawdown series and metrics.
    
    Args:
        equity: Series of equity values
    
    Returns:
        Tuple of (drawdown_series, max_drawdown, max_drawdown_pct, current_drawdown)
    """

def compute_cagr(
    start_value: float,
    end_value: float,
    periods: int,
    freq: str = "1d"
) -> float | None:
    """Compute Compound Annual Growth Rate.
    
    Args:
        start_value: Starting value
        end_value: Ending value
        periods: Number of periods
        freq: Frequency string ("1d" or "5min")
    
    Returns:
        CAGR (annualized) or None if < 1 year of data
    """

def compute_turnover(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    start_capital: float,
    freq: str = "1d"
) -> float:
    """Compute annualized portfolio turnover.
    
    Args:
        trades: DataFrame with columns: timestamp, symbol, side, qty, price
        equity: DataFrame with columns: timestamp, equity
        start_capital: Starting capital
        freq: Frequency string ("1d" or "5min")
    
    Returns:
        Annualized turnover ratio
    """
```

---

## Metriken-Definitionen

### Performance-Metriken

1. **Final Performance Factor (final_pf):**
   - `equity[-1] / equity[0]`
   - Bereits implementiert

2. **Total Return:**
   - `final_pf - 1.0`
   - Einfache Ableitung

3. **CAGR (Compound Annual Growth Rate):**
   - `((end_value / start_value) ^ (1 / years)) - 1`
   - `years = periods / periods_per_year`
   - Nur berechnen wenn `periods >= periods_per_year`

### Risk-Adjusted Returns

4. **Sharpe Ratio (annualisiert):**
   - `(mean_return - risk_free_rate) / (std_return * sqrt(periods_per_year))`
   - `periods_per_year = 252` für "1d", `252 * 78` für "5min"
   - Bereits implementiert, aber inkonsistent

5. **Sortino Ratio (annualisiert):**
   - `(mean_return - risk_free_rate) / (downside_std * sqrt(periods_per_year))`
   - `downside_std`: Standardabweichung nur negativer Returns
   - **Neu**

6. **Calmar Ratio:**
   - `CAGR / |max_drawdown_pct|`
   - **Neu**

### Risk-Metriken

7. **Max Drawdown:**
   - Bereits in `risk.py` implementiert
   - `drawdown = equity - expanding_max(equity)`
   - `max_drawdown = min(drawdown)`

8. **Max Drawdown %:**
   - `(max_drawdown / peak_equity) * 100`
   - Bereits in `risk.py` implementiert

9. **Current Drawdown:**
   - `drawdown[-1]`
   - Bereits in `risk.py` implementiert

10. **Volatility (annualisiert):**
    - `std_return * sqrt(periods_per_year)`
    - Bereits in `risk.py` implementiert

11. **VaR (95% Konfidenz):**
    - `percentile(returns, 5) * current_equity`
    - Bereits in `risk.py` implementiert

### Trade-Metriken (benötigen Trades-DataFrame)

12. **Hit Rate (Win Rate):**
    - `winning_trades / total_trades`
    - Gewinnender Trade: Trade mit positivem P&L
    - **Neu**

13. **Profit Factor:**
    - `total_gains / total_losses` (wenn total_losses > 0)
    - **Neu**

14. **Average Win:**
    - `total_gains / winning_trades`
    - **Neu**

15. **Average Loss:**
    - `total_losses / losing_trades`
    - **Neu**

16. **Turnover (annualisiert):**
    - `(total_notional / avg_equity) * periods_per_year`
    - `total_notional = sum(abs(qty * price))`
    - `avg_equity = mean(equity)`
    - **Neu**

17. **Total Trades:**
    - `len(trades)`
    - Bereits in `portfolio.py` implementiert

---

## Migrations-Plan

### Phase 1: Neues Modul erstellen
1. Erstelle `src/assembled_core/qa/metrics.py`
2. Implementiere `PerformanceMetrics` Dataclass
3. Implementiere Helper-Funktionen (Sharpe, Sortino, Drawdown, CAGR, Turnover)
4. Implementiere `compute_all_metrics()` und `compute_equity_metrics()`
5. Implementiere `compute_trade_metrics()`
6. Erstelle Tests (`tests/test_qa_metrics.py`)

### Phase 2: Bestehende Funktionen erweitern
1. `pipeline/backtest.py::compute_metrics()`:
   - Nutze `qa.metrics.compute_equity_metrics()` intern
   - Behalte Rückwärtskompatibilität (dict-Output)
   - Optional: Deprecation-Warning für direkte Nutzung

2. `pipeline/portfolio.py::simulate_with_costs()`:
   - Nutze `qa.metrics.compute_all_metrics()` wenn Trades verfügbar
   - Behalte Rückwärtskompatibilität

3. `api/routers/risk.py::get_risk_summary()`:
   - Nutze `qa.metrics.compute_equity_metrics()` intern
   - Behalte Pydantic-Model-Output

### Phase 3: Integration in Backtest-Engine
1. `qa/backtest_engine.py::run_portfolio_backtest()`:
   - Nutze `qa.metrics.compute_all_metrics()` statt `pipeline.backtest.compute_metrics()`
   - Erweitere `BacktestResult.metrics` um neue Metriken

### Phase 4: Dokumentation
1. Aktualisiere `docs/BACKEND_MODULES.md`
2. Aktualisiere `docs/ARCHITECTURE_BACKEND.md`
3. Erstelle Beispiel-Notebook oder Doku für Metriken-Interpretation

---

## Beispiel-Verwendung

```python
from src.assembled_core.qa.metrics import compute_all_metrics
from src.assembled_core.pipeline.io import load_prices, load_orders

# Equity und Trades laden
equity = load_prices("1d")  # oder aus Backtest
trades = load_orders("1d")

# Alle Metriken berechnen
metrics = compute_all_metrics(
    equity=equity,
    trades=trades,
    start_capital=10000.0,
    freq="1d",
    risk_free_rate=0.02  # 2% risk-free rate
)

# Metriken ausgeben
print(f"Final PF: {metrics.final_pf:.4f}")
print(f"CAGR: {metrics.cagr:.2%}" if metrics.cagr else "CAGR: N/A (< 1 year)")
print(f"Sharpe: {metrics.sharpe_ratio:.4f}" if metrics.sharpe_ratio else "Sharpe: N/A")
print(f"Sortino: {metrics.sortino_ratio:.4f}" if metrics.sortino_ratio else "Sortino: N/A")
print(f"Max DD: {metrics.max_drawdown_pct:.2f}%")
print(f"Hit Rate: {metrics.hit_rate:.2%}" if metrics.hit_rate else "Hit Rate: N/A")
print(f"Turnover: {metrics.turnover:.2f}x" if metrics.turnover else "Turnover: N/A")
```

---

## Offene Fragen

1. **Risk-Free Rate:**
   - Soll `risk_free_rate` als Parameter übergeben werden oder aus Config/Env-Variable kommen?
   - Default: 0.0 (vereinfacht)

2. **CAGR-Berechnung:**
   - Soll CAGR nur berechnet werden wenn `periods >= periods_per_year`?
   - Oder auch für kürzere Zeiträume (mit Warnung)?

3. **Turnover-Berechnung:**
   - Soll Turnover auf Basis von Notional oder auf Basis von Positions-Änderungen berechnet werden?
   - Vorschlag: Notional-basiert (einfacher, konsistent mit bestehender Logik)

4. **Backwards Compatibility:**
   - Sollen bestehende Funktionen (`compute_metrics()`) weiterhin dict zurückgeben?
   - Oder sollen sie `PerformanceMetrics` zurückgeben (Breaking Change)?

5. **Pydantic-Model:**
   - Soll `PerformanceMetrics` auch als Pydantic-Model verfügbar sein (für API)?
   - Oder nur Dataclass (einfacher)?

---

## Nächste Schritte

1. ✅ Analyse bestehender Implementierungen (abgeschlossen)
2. ⏳ Plan erstellen (dieses Dokument)
3. ⏳ Implementierung von `qa/metrics.py`
4. ⏳ Tests schreiben
5. ⏳ Integration in bestehende Module
6. ⏳ Dokumentation aktualisieren

