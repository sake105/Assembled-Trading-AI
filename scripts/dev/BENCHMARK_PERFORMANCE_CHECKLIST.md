# Benchmark: Realistische Performance messen & was posten

Copy/Paste-Ablauf für PowerShell – 1:1 nutzbar.

---

## Schritt 1: Parquet prüfen (Dataset-Minicheck)

```powershell
$P = "<DEIN_PARQUET_PFAD>"
py -3 scripts/dev/check_parquet_dataset.py $P
```

Wenn **unique_days** deutlich > 252 ist und **schema_ok=True**, ist das Dataset benchmark-tauglich.

---

## Schritt 2: Benchmark laufen lassen (empfohlen: ohne --quick)

```powershell
$P = "<DEIN_PARQUET_PFAD>"
py -3 scripts/dev/run_strategy_benchmark.py `
  --output-root output/system_run `
  --dataset $P `
  --max-variants 12 `
  --oos-sweep `
  --no-strict-session-gate
```

**Hinweis:** `--no-strict-session-gate` ist für 1d-EOD-Daten mit Timestamps 00:00 UTC empfohlen (vermeidet „Orders rejected“ durch Session-Gate). Siehe `scripts/dev/BENCHMARK_DIAGNOSTIC.md`.

**Schnell-Variante (weniger Varianten, nur 1y):**

```powershell
$P = "<DEIN_PARQUET_PFAD>"
py -3 scripts/dev/run_strategy_benchmark.py `
  --output-root output/system_run `
  --dataset $P `
  --quick `
  --max-variants 8 `
  --oos-sweep
```

---

## Schritt 3: Die 4 Outputs ausgeben (zum hier posten)

### A) Scoreboard Top 20 (nach total_return)

```powershell
Import-Csv output/system_run/benchmark/scoreboard.csv |
  Sort-Object {[double]$_.total_return} -Descending |
  Select-Object -First 20 variant_id,horizon,total_return,sharpe_ratio,max_drawdown_pct,total_trades,turnover,total_cost_pct,robustness_score,stability_score_v2 |
  Format-Table -AutoSize
```

Optional nach **stability_score_v2**:

```powershell
Import-Csv output/system_run/benchmark/scoreboard.csv |
  Sort-Object {[double]$_.stability_score_v2} -Descending |
  Select-Object -First 20 variant_id,horizon,total_return,sharpe_ratio,max_drawdown_pct,total_trades,turnover,robustness_score,stability_score_v2 |
  Format-Table -AutoSize
```

### B) OOS Sweep Report (komplett)

```powershell
Get-Content output/system_run/benchmark/oos_sweep_report.md
```

### C) Anomalies (erste 40 Zeilen)

```powershell
Get-Content output/system_run/benchmark/anomalies.json -TotalCount 40
```

### D) Filter sweep Top 20 (falls vorhanden)

```powershell
if (Test-Path output/system_run/benchmark/filter_sweep_results.csv) {
  Import-Csv output/system_run/benchmark/filter_sweep_results.csv |
    Sort-Object {[double]$_.total_return} -Descending |
    Select-Object -First 20 variant_id,param_name,param_value,horizon,total_return,sharpe_ratio,max_drawdown_pct,turnover,total_trades |
    Format-Table -AutoSize
} else {
  "filter_sweep_results.csv not found (run with --sweep-filters if you want it)."
}
```

**Filter-Sweep nachholen** (wenn noch nicht gelaufen):

```powershell
$P = "<DEIN_PARQUET_PFAD>"
py -3 scripts/dev/run_strategy_benchmark.py `
  --output-root output/system_run `
  --dataset $P `
  --quick `
  --sweep-filters
```

---

## Interpretation (mit den neuen Fixes)

**Wenn wieder viele Trades rejected:**
- `reject_reason == INSUFFICIENT_CASH` -> Sizing/Notional/Allokation
- `QC_FAIL_MIN_FILL_QTY` -> Fill-Config / min-fill
- Session-Gate -> Timing / Kalender

**Wenn Trades filled, aber Return ~0:**
- Kosten vs Edge: `turnover`, `total_cost_pct`, `cost_share_of_return`, `gross_total_return_est`
- Signal zu selten: laengere MAs, groesseres Universe
- Turnover zu hoch: seltener rebalancen / Top-N-Churn reduzieren

**Schnelle Hebel (ohne Konzeptwechsel):**
1. Universe erweitern -> mehr Trades
2. Kosten/Turnover senken (Filter + weniger Rebalancing)
3. Entry-Filter (RSI/Vol/Regime) kalibrieren: schlechte Trades raus, gute drin
