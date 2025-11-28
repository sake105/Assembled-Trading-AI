# Assembled Trading AI – Context Pack (for Cursor)

## Mission & Prinzipien
Robustes, transparentes, risikoorientiertes E2E-System (Daten→Features→Signale→Portfolio→orders.csv),
Human-in-the-Loop via SAFE-Bridge, realistische Ziele, strenge QA-Gates. Quellen: interne Analysen. [Siehe Unternehmensanalyse.] 
(Details: Mission/Vision, Risiko-Rahmen, Modularität, EOD-Fokus). :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

## Pipeline (heute produktiv)
pull_intraday.ps1 → resample(1m→5m) → sprint9_execute.py → sprint9_backtest.py → sprint10_portfolio.py  
Artefakte:  
- `output/aggregates/5min.parquet` (symbol,timestamp,close)  
- `output/orders_5min.csv` (timestamp,symbol,side,qty,price)  
- `output/performance_report_5min.md`, `output/portfolio_report.md`

## PowerShell Leitlinien
- PS7 only. Keine Bash-Heredocs (`python - <<'PY'` ist VERBOTEN).  
- Inline-Python so:
$code = @'
import pandas as pd; print("ok")
'@
$code | & "$Repo/.venv/Scripts/python.exe" -

bash
Code kopieren
- Utilities in `scripts/ps/ps_py_utils.ps1` nutzen.

## Resample – verbindliche Spezifikation
Input: `data/raw/1min/*.parquet` mit Spalten mind. `timestamp,symbol,close` (UTC).  
Output: `output/aggregates/5min.parquet` **mit Schema exakt** `['symbol','timestamp','close']`.  
Algorithmus: `df.set_index('timestamp').groupby('symbol')['close'].resample('5min').last().dropna().reset_index()`  
Sortierung: `sort_values(['timestamp','symbol'])`.

## Kosten/Backtest
Backtests sind **offline** und nutzen nur lokale Daten.  
`python scripts/sprint9_backtest.py --freq 5min --start-capital 10000` → Equity/Report.  
Portfolio-Simulation: `scripts/sprint10_portfolio.py` (comm_bps, spread_w, impact_w parametrisierbar).

## Do / Don’t für Cursor
**Do**
- Streng an SAFE-Bridge halten (nur CSV-Orders).
- Resample/Backtest/Portfolio ohne Netzwerk.
- PS/Heredoc-Anti-Pattern aktiv entfernen.
- Kleine, testbare Python-Funktionen mit Type Hints.

**Don’t**
- Keine Live-Broker-Calls vorschlagen oder einbauen.
- Keine externen Dienste „mal eben“ integrieren.
- Keine Strukturänderungen, die run_all_sprint10.ps1 brechen.

## Quick-Checks
- `python -c "import pandas as pd; d=pd.read_parquet('output/aggregates/5min.parquet'); print(d.columns.tolist()[:3], len(d))"`
- `Get-Content .\output\performance_report_5min.md -Raw`
