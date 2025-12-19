## Workflows – Batch-Backtests & Parallelisierung (P4)

**Last Updated:** 2025-12-15  
**Status:** Active Workflow for P4 – Batch Runner & Parallelisierung

---

## Overview

**Ziel:** Viele Backtests systematisch und reproduzierbar ausführen, z.B. für:

- **Parameter-Sweeps:** Variation von `max_gross_exposure`, `rebalance_freq`, Kostenparametern.
- **Strategie-Vergleiche:** Core vs. Core+ML vs. ML-only Bundles.
- **Universe-Vergleiche:** Verschiedene Universes (ETFs, AI/Tech, Sektoren).
- **Regime-On/Off:** Runs mit und ohne Regime-Overlay / Risk-Overlay.

Der Batch-Runner baut auf der optimierten Backtest-Engine (P3) auf und nutzt nur **lokale Daten** (`data_source="local"`). Er ändert **keine** Finanzlogik, sondern orchestriert bestehende Backtest-Skripte.

**Entry Points:**

- Python-Script: `scripts/batch_backtest.py`
- Zentrale CLI: `python scripts/cli.py batch_backtest ...`

---

## Batch-Config – Format & Beispiel

Der Batch-Runner liest eine YAML- oder JSON-Config. Empfohlen ist YAML (besser lesbar).

### Minimaler YAML-Beispiel-Config

```yaml
batch_name: ai_tech_core_vs_ml_2015_2020
description: "Compare AI/Tech core bundle vs. ML bundles on 2015–2020"
output_root: "output/batch_backtests"

defaults:
  freq: "1d"
  data_source: "local"
  strategy: "multifactor_long_short"
  rebalance_freq: "M"
  max_gross_exposure: 1.0
  start_capital: 100000.0
  generate_report: true
  generate_risk_report: true
  generate_tca_report: false
  symbols_file: "config/universe_ai_tech_tickers.txt"

runs:
  - id: "core_2015_2020"
    bundle_path: "config/factor_bundles/ai_tech_core_bundle.yaml"
    start_date: "2015-01-01"
    end_date: "2020-12-31"

  - id: "core_ml_2015_2020"
    bundle_path: "config/factor_bundles/ai_tech_core_ml_bundle.yaml"
    start_date: "2015-01-01"
    end_date: "2020-12-31"

  - id: "ml_alpha_only_2015_2020"
    bundle_path: "config/factor_bundles/ai_tech_ml_alpha_bundle.yaml"
    start_date: "2015-01-01"
    end_date: "2020-12-31"
    generate_tca_report: true
```

### Wichtige Felder

- **batch_name**: Kurzname des Batches (wird für Output-Verzeichnis verwendet).
- **description**: Freitext-Beschreibung (landet im Markdown-Report).
- **output_root**: Basis-Output-Verzeichnis (Standard: `output/batch_backtests`).

**defaults:**

- `freq`: `"1d"` oder `"5min"` (aktuell primär 1d).
- `data_source`: **muss** `"local"` sein (keine Live-APIs).
- `strategy`: z.B. `"multifactor_long_short"`.
- `rebalance_freq`: `"D"`, `"W"` oder `"M"`.
- `max_gross_exposure`: z.B. `1.0` (100 %).
- `start_capital`: Startkapital, z.B. `100000.0`.
- `generate_report`: `true` → klassischer Performance-/QA-Report.
- `generate_risk_report`: `true` → zusätzlicher Risk-Report pro Run.
- `generate_tca_report`: `true` → zusätzlicher TCA-Report pro Run.
- `symbols_file` oder `universe`: Universe-Definition wie in anderen Workflows.

**runs:**

- `id`: eindeutiger Run-Name (wird Teil des Output-Pfads).
- `bundle_path`: YAML-Konfiguration des Factor-Bundles.
- `start_date`, `end_date`: Zeitraum.
- Optional: Overrides der Defaults (z.B. andere `rebalance_freq` oder `max_gross_exposure`).

---

## Standard-Workflows

### 1. Parameter-Sweep (z.B. max_gross_exposure, Rebalance-Frequenz)

**Ziel:** Sensitivität der Strategie auf zentrale Parameter testen.

**Beispiel-Setup:**

- Fixes Bundle: `ai_tech_core_bundle.yaml`.
- Variation:
  - `max_gross_exposure`: 0.5, 1.0, 1.5.
  - `rebalance_freq`: D, W, M.

**Konfig-Ausschnitt:**

```yaml
defaults:
  freq: "1d"
  data_source: "local"
  strategy: "multifactor_long_short"
  symbols_file: "config/universe_ai_tech_tickers.txt"
  start_capital: 100000.0

runs:
  - id: "core_M_1_0"
    bundle_path: "config/factor_bundles/ai_tech_core_bundle.yaml"
    rebalance_freq: "M"
    max_gross_exposure: 1.0
    start_date: "2015-01-01"
    end_date: "2020-12-31"

  - id: "core_W_1_0"
    bundle_path: "config/factor_bundles/ai_tech_core_bundle.yaml"
    rebalance_freq: "W"
    max_gross_exposure: 1.0
    start_date: "2015-01-01"
    end_date: "2020-12-31"

  - id: "core_M_1_5"
    bundle_path: "config/factor_bundles/ai_tech_core_bundle.yaml"
    rebalance_freq: "M"
    max_gross_exposure: 1.5
    start_date: "2015-01-01"
    end_date: "2020-12-31"
```

**CLI-Aufruf:**

```powershell
python scripts/cli.py batch_backtest `
  --config-file configs/batch_backtests/ai_tech_core_param_sweep.yaml
```

**Auswertung:**

- `batch_summary.csv` → sortieren nach `sharpe`, `final_pf`, `max_drawdown` (über Risk-Reports).
- Pro Run: Performance-/Risk-Reports anschauen und interpretieren.

---

### 2. Strategie-Vergleich (Core vs. Core+ML vs. ML-only)

**Ziel:** Wirkung von ML-Alpha-Faktoren in Strategien vergleichen.

**Beispiel-Setup:**

- Universe: `config/universe_ai_tech_tickers.txt`.
- Bundles:
  - `ai_tech_core_bundle.yaml`
  - `ai_tech_core_ml_bundle.yaml`
  - `ai_tech_ml_alpha_bundle.yaml`

**Config-Runs:**

```yaml
runs:
  - id: "core_only"
    bundle_path: "config/factor_bundles/ai_tech_core_bundle.yaml"
    start_date: "2015-01-01"
    end_date: "2025-12-03"

  - id: "core_plus_ml"
    bundle_path: "config/factor_bundles/ai_tech_core_ml_bundle.yaml"
    start_date: "2015-01-01"
    end_date: "2025-12-03"

  - id: "ml_alpha_only"
    bundle_path: "config/factor_bundles/ai_tech_ml_alpha_bundle.yaml"
    start_date: "2015-01-01"
    end_date: "2025-12-03"
    generate_tca_report: true
```

**Interpretation:**

- **Sharpe/CAGR**: Liefert ML-Alpha komplexerweise Mehrwert über Core?
- **Risk-Profile** (Risk-Reports): Wie verändern sich Drawdowns, Regime-Profile?
- **Kosten-Impact** (TCA-Reports bei ML-only): Ist die höhere Turnover durch ML noch tragbar?

---

### 3. Regime-On/Off & Universe-Vergleiche

**Ziel:** Strategien mit/ohne Regime-Overlay und über verschiedene Universes vergleichen.

**Ideen:**

- Runs mit `--use-regime-overlay` on/off (in `extra_args` oder explizit in der Config).
- Universes:
  - `macro_world_etfs_tickers.txt`
  - `universe_ai_tech_tickers.txt`
  - weitere Sektor-Universes.

**Beispiel (stark gekürzt):**

```yaml
defaults:
  freq: "1d"
  data_source: "local"
  strategy: "multifactor_long_short"
  start_capital: 100000.0

runs:
  - id: "etfs_core_no_regime"
    bundle_path: "config/factor_bundles/macro_world_etfs_core_bundle.yaml"
    symbols_file: "config/macro_world_etfs_tickers.txt"
    start_date: "2010-01-01"
    end_date: "2020-12-31"

  - id: "etfs_core_regime"
    bundle_path: "config/factor_bundles/macro_world_etfs_core_bundle.yaml"
    symbols_file: "config/macro_world_etfs_tickers.txt"
    start_date: "2010-01-01"
    end_date: "2020-12-31"
    extra_args:
      use_regime_overlay: true
      regime_config_file: "config/regime/macro_world_etfs_regime.yaml"
```

---

## Output-Struktur & Interpretation

### Verzeichnisstruktur

Standardmäßig:

```text
output/batch_backtests/
  {batch_name}/
    batch_summary.csv
    batch_summary.md
    runs/
      {run_id}/
        backtest/
          equity_curve.parquet
          positions.parquet
          performance_report_1d.md
          ...
        risk_report/      # optional
        tca/              # optional
```

### batch_summary.csv

Spalten (aktueller Stand):

- `run_id`
- `strategy`
- `bundle_path`
- `start_date`
- `end_date`
- `status` (`success`, `failed`, `skipped`)
- `exit_code`
- `runtime_sec`
- `backtest_dir`
- `equity_curve_path`
- `performance_report_path`
- `error`

**Typische Nutzung:**

- In pandas laden und nach Kennzahlen sortieren:

```python
import pandas as pd
from pathlib import Path

summary = pd.read_csv(Path("output/batch_backtests/ai_tech_core_vs_ml_2015_2020/batch_summary.csv"))
print(summary[["run_id", "status", "runtime_sec"]])
```

### batch_summary.md

Markdown-Übersicht mit:

- Batch-Name + Beschreibung.
- Tabelle mit `run_id`, `strategy`, `bundle`, `start_date`, `end_date`, `status`, `runtime_sec`.

Ideal für schnelle visuelle Vergleiche (z.B. direkt im Repo oder in einem Report verlinkt).

---

## Best Practices

- **Naming:**
  - `batch_name` kurz, aber aussagekräftig (`ai_tech_core_vs_ml_2015_2020`).
  - `run_id` beschreibt Variation (`core_M_1_0`, `core_W_1_0`, `core_M_1_5`, `ml_alpha_only`).

- **Output-Verzeichnisse:**
  - Pro Batch eigenes Verzeichnis unter `output/batch_backtests/{batch_name}`.
  - Für Experimente pro Thema eigene Configs unter `configs/batch_backtests/`.

- **Reproduzierbarkeit:**
  - Config-Dateien versionieren (Git).
  - Keine zufällige Seedauswahl in Strategielogik ohne festen Seed.
  - Nur lokale Daten (`data_source="local"`).

- **Integration mit Profiling (P1):**
  - Über `scripts/profile_jobs.py` mit Job `BATCH_BACKTEST` kann ein Batch mit `--dry-run` profiliert werden.
  - Für echte Performance-Tests: `--dry-run` entfernen und auf repräsentativem Subset testen.

- **Fehlerhandling:**
  - Ein Run kann fehlschlagen, der Batch läuft weiter (sofern `--fail-fast` nicht gesetzt ist).
  - Den Status im CSV/Markdown prüfen, bevor man Ergebnisse interpretiert.

---

## Verweise

- Design: `docs/BATCH_BACKTEST_P4_DESIGN.md`
- Backtest Engine: `src/assembled_core/qa/backtest_engine.py`
- Batch Runner: `scripts/batch_backtest.py`
- CLI Entry: `scripts/cli.py batch_backtest`
- Profiling: `scripts/profile_jobs.py`


