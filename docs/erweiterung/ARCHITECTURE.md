# ERWEITERUNG — Architektur

## Schichten

```
┌──────────────────────────────────────────────────────┐
│                    altdata                           │
│  Wikipedia · Trends · SEC · FINRA · COT · FRED-MD    │
│  Yahoo-Options · GDELT · Reddit · CoinGecko · WB     │
│  → FetchResult { df, source, as_of, rows, notes }    │
│  → Disk-Cache via Parquet                            │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│                    signals                           │
│  Cross-sectional residuals · Options-implied         │
│  Attention composite · Lead-lag networks             │
│  Statistical arbitrage · PEAD-v2 · Macro-nowcast     │
│  → DataFrame [date, symbol, <signal_value>]          │
│  → PIT-shift explicit per source                     │
└──────────────────────────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
┌─────────────────────┐  ┌─────────────────────────┐
│         ml          │  │         portfolio       │
│  Conformal CP       │  │  HRP                    │
│  Stacking ensemble  │  │  Black-Litterman        │
│  Triple-barrier     │  │  CVaR-LP                │
│  + meta-labeling    │  │  Risk-Parity (ERC)      │
│                     │  │  Kelly                  │
└─────────────────────┘  └─────────────────────────┘
              │                       │
              └───────────┬───────────┘
                          ▼
┌──────────────────────────────────────────────────────┐
│                       risk                           │
│  EVT/GPD · Dynamic-DD-Control · APC-Crisis-Score     │
│  → exposure-multiplier ∈ [0, 1]                      │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│                      meta                            │
│  Strategy-Orchestrator (Equal/InvVol/Hedge/HRP)      │
│  Regime-Router (Vol × Trend × Crisis)                │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│                    execution                         │
│  Almgren-Chriss schedule · Adaptive slippage         │
│  → ausführliche cost-aware Order-Generierung         │
└──────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────┐
│                    backtest                          │
│  Combinatorial Purged CV · Walk-Forward              │
│  Deflated Sharpe · White Reality Check · Hansen-SPA  │
│  Performance metrics (Sharpe/Sortino/Calmar/...)     │
│  → rigorose Validierung gegen Backtest-Overfitting   │
└──────────────────────────────────────────────────────┘
```

## PIT-Shift-Konventionen

Jedes altdata-Modul codiert seinen PIT-Shift im Wrapper:

| Quelle | Shift | Begründung |
|--------|-------|------------|
| Wikipedia | T+1 | Tagesaggregat ist erst am Folgetag verfügbar |
| Google Trends | T+1 | Daily SVI hat ~24h Verzögerung |
| FINRA Reg-SHO | T+1 | Veröffentlicht am Folge-Werktag |
| CFTC COT | Mo nach Reporting-Di | Veröffentlichung Fr nach Markschluss |
| SEC EDGAR Filings | filing_date | report_date enthält Look-Ahead |
| FRED-MD | T+~1 Monat | Macro-Data hat erhebliche Verzögerung |
| GDELT | T+1 | Tagesaggregat |
| Reddit | T+0 | Real-Time Streams |

## Cache-Strategie

- Disk-Cache pro Modul unter `output/erweiterung_cache/<source>/<sha1>.parquet`.
- Per `ERWEITERUNG_CACHE_DIR` env-Variable überschreibbar.
- Hash-Keys aus (source, query-params, date-range).
- TTL pro Quelle:
  - Yahoo-Options: 4h (real-time changes)
  - alle anderen: dauerhaft (oder bis File mtime > N Tage — ToDo)

## Fehlerbehandlung

- **Boundary-Pattern:** HTTP-Calls mit `@retry_with_backoff` und `@rate_limited`. Bei Endgültigem Scheitern: leeres `FetchResult` (kein Crash).
- **Optional-Imports:** `pytrends`, `yfinance`, `vaderSentiment`, `scipy.optimize.linprog`, `sklearn.covariance.LedoitWolf` — alle lazy importiert, mit klaren `RuntimeError("X required")`-Meldungen wenn nicht verfügbar.
- **Fallbacks:** CVaR-Optimizer hat SA-Fallback ohne SciPy. ADF-Test hat Variance-Ratio-Fallback ohne Statsmodels.

## Test-Strategie

- 91 Unit-Tests, alle offline (Mock + Synthetic).
- Network-Tests sind in einer optionalen Marker-Suite (`@pytest.mark.network`) — nicht enthalten in dieser Version.
- Integration-Test via `scripts/erweiterung/run_demo_backtest.py` — synthetische Daten, deterministisch via Seed.
