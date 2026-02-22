# 60-Sekunden-Diagnose: „Orders werden erzeugt, aber nie gefüllt“

Wenn der Benchmark **0 Trades / 0 Return** liefert, liegt oft einer dieser Fälle vor:

- **0 Trades** (nur Fills gezählt), oder  
- **Trades/Orders vorhanden, aber status=rejected, fill_qty=0** → Equity flach → Return 0.

Mit der folgenden Diagnose siehst du in einem Run-Verzeichnis: Equity-Kurve, Trades-CSV, Status/Reject-Reasons, QC-Report.

---

## 1) 60-Sekunden-Diagnose (PowerShell, Copy/Paste)

Passe ggf. nur **$RunDir** an (Beispiel: `output\system_run\benchmark\trend_baseline\1y`).

```powershell
$RunDir = "output\system_run\benchmark\trend_baseline\1y"

"== RunDir =="
Resolve-Path $RunDir

"== Equity file =="
Get-ChildItem $RunDir -Filter "equity_curve_*.csv" | Select-Object FullName, Length

"== Equity rows + min/max =="
$eq = Get-ChildItem $RunDir -Filter "equity_curve_*.csv" | Select-Object -First 1
if ($eq) {
  $eqdf = Import-Csv $eq.FullName
  "equity_rows: $($eqdf.Count)"
  if ($eqdf.Count -gt 0) {
    $eqVals = $eqdf | ForEach-Object {[double]$_.equity}
    "equity_min: $([double]($eqVals | Measure-Object -Minimum).Minimum) equity_max: $([double]($eqVals | Measure-Object -Maximum).Maximum)"
    "equity_head:"
    $eqdf | Select-Object -First 5 | Format-Table
  }
} else {
  "No equity_curve_*.csv found"
}

"== Trades file =="
Get-ChildItem $RunDir -Filter "trades_*.csv" | Select-Object FullName, Length

"== Trades status distribution + fills =="
$tr = Get-ChildItem $RunDir -Filter "trades_*.csv" | Select-Object -First 1
if ($tr) {
  $tdf = Import-Csv $tr.FullName
  "trade_rows: $($tdf.Count)"
  if ($tdf.Count -gt 0) {
    "status_counts:"
    $tdf | Group-Object status | Sort-Object Count -Descending | Format-Table Name, Count

    if ($tdf[0].PSObject.Properties.Name -contains "fill_qty") {
      $fills = ($tdf | ForEach-Object {[double]$_.fill_qty} | Where-Object { $_ -gt 0 }).Count
      "fills_count(fill_qty>0): $fills"
    }

    if ($tdf[0].PSObject.Properties.Name -contains "reject_reason") {
      "reject_reason_counts:"
      $tdf | Group-Object reject_reason | Sort-Object Count -Descending | Select-Object -First 12 | Format-Table Name, Count
    } else {
      "reject_reason column NOT present (older run or schema mismatch)."
    }

    "trades_head:"
    $tdf | Select-Object -First 10 timestamp,symbol,side,qty,price,fill_qty,fill_price,status,reject_reason | Format-Table
  }
} else {
  "No trades_*.csv found"
}

"== QC report (if present) =="
$q = Join-Path $RunDir "qc_report.json"
if (Test-Path $q) { Get-Content $q -TotalCount 120 } else { "No qc_report.json found" }
```

---

## 2) Was die Ausgabe bedeutet

### A) status_counts ist leer oder es gibt gar keine trades_*.csv

→ Vermutlich **keine Orders erzeugt** (Signal/Universe/Filter blockt vorher) oder der Run ist vorher abgebrochen.  
→ **anomalies.json** / stderr prüfen.

### B) status=rejected dominiert und fills_count ist 0

→ Es **gibt Orders**, aber die Fill-/Gate-Pipeline lässt nichts durch.

Entscheidend ist dann **reject_reason_counts**:

| reject_reason | Bedeutung |
|---------------|-----------|
| **OUTSIDE_SESSION**, **NOT_AT_SESSION_CLOSE**, **NOT_TRADING_DAY** | Session-Gate blockiert (bei 1D-Bars häufig, wenn timestamps 00:00 UTC sind). |
| **INSUFFICIENT_CASH** | Cash-Gate blockiert (Notional/Qty/Preis oder Budget/Leverage). |
| **QC_FAIL_MIN_FILL_QTY** / **LIMIT_NOT_REACHED** | Fill-Regeln / Limit-Logik blockieren. |

**Lösung für 1D-EOD mit 00:00 UTC:** Benchmark bzw. Backtest mit **--no-strict-session-gate** laufen lassen (siehe unten).

### C) Fills > 0, aber Equity bleibt konstant (min=max)

→ Sehr wahrscheinlich **Equity-Timeline / cash_delta Aggregation** (oder Equity-CSV hat nur 1 Punkt).  
→ Als Nächstes prüfen: Anzahl unique timestamps im Slice, ob `simulate_with_costs` wirklich Preise bekommt.

---

## 3) Häufigster Grund bei 1D-Parquets: Session-Gate

Viele EOD-Parquets haben **timestamp auf 00:00:00+00:00**. Wenn das Session-Gate „nur am Session-Close“ füllt, wird alles rejected.

**Schnelltest (ohne Code ändern):**

- Einmal einen Run **ohne striktes Session-Gate** fahren.

**CLI-Flag (empfohlen für 1d EOD):**

```powershell
# Einzelner Backtest
py -3 scripts/run_backtest_strategy.py --freq 1d --price-file "path/to/slice.parquet" --out output/run --no-strict-session-gate

# Benchmark (alle Runs mit relaxed session gate)
py -3 scripts/dev/run_strategy_benchmark.py --output-root output/system_run --dataset $P --max-variants 12 --oos-sweep --no-strict-session-gate
```

---

## 4) Warum bei dir „0 Trades / 0 Return“ überall steht

- Benchmark läuft technisch durch (OOS-Report existiert).
- Die **tatsächlichen Fills sind 0** (oder die Equity-Timeline hat nur 1 Punkt).
- Daraus: total_return 0, sharpe None/leer, max_dd 0, Anomalien `constant_equity`.

Das ist **kein** „Strategie ist schlecht“-Signal, sondern **Execution/Fill/QC blockiert die Simulation**. Die 60-Sekunden-Diagnose plus **--no-strict-session-gate** für 1d-EOD beheben den häufigsten Fall.
