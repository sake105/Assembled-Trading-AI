```md
# TASKS — Häufige Aufgaben & Befehle

> Alle Befehle im Repo-Root (`F:\Python_Projekt\Aktiengerüst`) ausführen.  
> PowerShell 7: `pwsh -NoProfile -ExecutionPolicy Bypass -File ...`

---

## 0) Schnell-Checks
- 5-Min-Preise vorhanden?
  ```powershell
  python -c "import pandas as pd; d=pd.read_parquet('output/aggregates/5min.parquet'); \
  print('rows',len(d),'symbols',d['symbol'].nunique(),'first',d['timestamp'].min(),'last',d['timestamp'].max())"
Orders-Range:

powershell
Code kopieren
python -c "import pandas as pd; o=pd.read_csv('output/orders_5min.csv',parse_dates=['timestamp']); \
print('orders',len(o),'first',o['timestamp'].min(),'last',o['timestamp'].max())"
1) Live-Daten ziehen (Yahoo)
powershell
Code kopieren
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\live\pull_intraday.ps1 `
  -Symbols AAPL,MSFT -Days 2
Tipps bei Rate-Limit (429):

Erst nur ein Symbol ziehen, nach ~1–2 min das nächste.

Falls weiterhin leer: später erneut.

2) 1m → 5m resamplen
Variante A (Einzelschritt, robust):

powershell
Code kopieren
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_all_sprint10.ps1 -SkipPull
# Das Skript resampled zu Beginn und schreibt: output/aggregates/5min.parquet
Variante B (dev-Tooling):

powershell
Code kopieren
$code = @"
import pandas as pd, pathlib as pl
root = pl.Path('data/raw/1min')
dfs = []
for p in root.glob('*.parquet'):
    df = pd.read_parquet(p)[['timestamp','symbol','close']].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = (df.set_index('timestamp')
            .groupby('symbol', group_keys=False)['close']
            .resample('5min').last().dropna().reset_index())
    dfs.append(df)
out = pd.concat(dfs, ignore_index=True).sort_values(['timestamp','symbol'])
pl.Path('output/aggregates').mkdir(parents=True, exist_ok=True)
out.to_parquet('output/aggregates/5min.parquet', index=False)
print('[DONE] wrote output/aggregates/5min.parquet',
      'rows=',len(out),'symbols=',out['symbol'].nunique(),
      'first=',out['timestamp'].min(),'last=',out['timestamp'].max())
"@
$code | .\.venv\Scripts\python.exe -
3) Orders erzeugen (EMA 20/60)
powershell
Code kopieren
python .\scripts\sprint9_execute.py --freq 5min --ema-fast 20 --ema-slow 60 `
  --price-file output\aggregates\5min.parquet
# schreibt: output/orders_5min.csv
4) Backtest
powershell
Code kopieren
# 5min
python .\scripts\sprint9_backtest.py --freq 5min --start-capital 10000
Get-Content .\output\performance_report_5min.md -Raw

# 1d (nutzt daily.parquet)
python .\scripts\sprint9_backtest.py --freq 1d --start-capital 10000
Get-Content .\output\performance_report_1d.md -Raw
5) Portfolio (Kostenmodell)
powershell
Code kopieren
# realistischere 5min-Kosten
python .\scripts\sprint10_portfolio.py `
  --freq 5min --start-capital 10000 `
  --commission-bps 0.5 --spread-w 0.5 --impact-w 1.0
Get-Content .\output\portfolio_report.md -Raw
6) Kosten-Sensitivität (Grid)
powershell
Code kopieren
python .\scripts\sprint9_cost_grid.py --freq 5min `
  --commission-bps 0 0.5 1.0 `
  --spread-w 0.25 0.5 1.0 `
  --impact-w 0.5 1.0 2.0 `
  --ema-fast 20 --ema-slow 60 --notional 10000
Get-Content .\output\cost_grid_report.md -Raw
7) Alles aus einer Hand
powershell
Code kopieren
# Normaler Lauf: Pull → Resample → Execute → Backtest → Portfolio
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_all_sprint10.ps1

# Wenn Yahoo limitiert:
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_all_sprint10.ps1 -SkipPull