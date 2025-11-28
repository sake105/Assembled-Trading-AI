docs/cursor/RULES.md
# RULES — Assembled Trading AI (Cursor Ground Rules)

## Zweck
Klare Leitplanken für Änderungen in PowerShell/Python und saubere Datenflüsse
(1m → 5m → Orders → Backtest → Portfolio). Cursor soll **klein & sicher** patchen,
nicht „Alles neu“.

---

## Projekt-Layout (Wahrheiten)
- PowerShell-Skripte: `scripts/*.ps1`, `scripts/live/*.ps1`, `scripts/tools/*.ps1`
- Python: `scripts/*.py`, `scripts/dev/*.py`, `scripts/features/*.py`
- Daten (Input): `data/raw/1min/*.parquet`
- Aggregates (Output): `output/aggregates/{5min,daily}.parquet`
- Ergebnisse: `output/*.csv`, `output/*.md`, `logs/*.log`
- Watchlist: `watchlist.txt`

**Kerneingaben / -ausgaben**
- **5min Parquet**: Spalten `symbol, timestamp (UTC), close`
- **daily Parquet**: Spalten `symbol, timestamp (UTC), close` (gleiche Reihenfolge)
- **orders_5min.csv / orders_1d.csv**: mindestens `timestamp, symbol, price, qty`  
  (`qty` positiv = BUY, negativ = SELL; zusätzliche Spalten erlaubt)
- **equity_curve_{5min,1d}.csv**: `timestamp, equity`
- **performance_report_{5min,1d}.md**: kurze Kennzahlen
- **portfolio_report.md**: Kennzahlen zu Kosten/Trades

---

## PowerShell-Regeln
1. **PS 7** verwenden (Befehl ist `pwsh`, nicht `powershell`).
2. Skriptpfade immer relativ zu **$PSScriptRoot** auflösen, z. B.:
   ```ps1
   $repoRoot = Split-Path -LiteralPath $PSScriptRoot -Parent
   $pythonExe = Join-Path $repoRoot '.venv/Scripts/python.exe'


Kein manuelles Voranstellen eines zweiten Laufwerks-Pfads.
3. Join-Path/Resolve-Path statt Stringkonkatenation mit \.
4. Fehler hart failen:

$ErrorActionPreference = 'Stop'
function Fail([string]$m){ throw $m }


Python-Heredocs vermeiden. Wenn nötig, als Here-String bauen und in
Python via STDIN pipen:

$code = @"


import pandas as pd
print(pd.version)
"@
$code | & $pythonExe -

(Nie `python - <<'PY'` in PS – das ist Bash-Syntax.)
6. Konsolenlogs klar präfixen: `[RUNALL]`, `[LIVE]`, `[RESAMPLE]`, `[EXEC]`, `[BT9]`, `[PF10]`.

---

## Python-Regeln
1. **UTC-Zeit**: `pd.to_datetime(..., utc=True)`; `timestamp` nie lokalisieren.
2. **I/O stabil**: `read_parquet`/`to_parquet` & `read_csv`/`to_csv(index=False)`.
3. **Schemas prüfen** (zur Not `assert set(cols).issubset(...)`).
4. **GroupBy/Apply**: künftiges pandas-Verhalten (Warning) berücksichtigen:  
`df.groupby(..., group_keys=False).apply(func, include_groups=False)`
5. **Resample** je Symbol:
```py
(df.set_index("timestamp")
   .groupby("symbol", group_keys=False)["close"]
   .resample("5min").last().dropna()
   .reset_index())


Deterministisch: keine zufälligen Seeds, keine versteckten globalen States.

CLI: argparse verwenden, Standardwerte an unsere Pipeline angleichen.