# Runbook: Paper Engine Run + Artifacts

This runbook covers the Paper Engine Upgrade artifacts introduced in Phases 0–10 of the `wir-m-ssen-unsere-paper-floating-pebble` plan: how to start a paper run, where every artifact lands, and how to compare two runs.

## 1. Start a paper run

Typical one-day invocation:

```bash
python -m src.assembled_core.execution.unified_paper_engine \
  --date 2025-01-15 \
  --run-id daily_paper_$(date +%Y%m%d)
```

Programmatic (tests, notebooks):

```python
from src.assembled_core.execution.unified_paper_engine import (
    UnifiedPaperConfig, UnifiedPaperEngine,
)

cfg = UnifiedPaperConfig(
    run_id="daily_paper_20260417",
    random_seed=42,                 # Phase 1.5 — determinism
    enable_tca=True,                # Phase 7
    enable_attribution=True,        # Phase 9
    enable_manifest=True,           # Phase 8
)
eng = UnifiedPaperEngine(cfg)
result = eng.run_paper_day("2026-04-17")
```

Recommended flags:

| Flag | Default | Notes |
|---|---|---|
| `random_seed` | `None` (stochastic) | Set for reproducible runs. |
| `enable_tca` | `True` | Phase 7 per-order CSV + aggregate JSON. |
| `enable_attribution` | `True` | Phase 9 cost / regime / factor breakdown. |
| `enable_manifest` | `True` | Phase 8 per-run manifest + cross-run index. |
| `enable_circuit_breaker` | `False` | Phase 3 — opt-in per run; enable for crisis-day simulations. |
| `enable_sor` | `False` | Phase 4 — opt-in; activates SmartOrderRouter cost component. |
| `shadow_mode` | `False` | Phase 6 — parallel real-broker submit for observability-only compare. |

## 2. Artifact map

Each successful run emits the following files (paths relative to the configured output roots):

```
output/
├── paper_ledger/ledger_{date}.parquet                         (Phase 0)
├── paper_lifecycle/lifecycle_{run_id}_{date}.jsonl            (Phase 1)
├── paper_tca/
│   ├── tca_{run_id}_{date}.csv                                (Phase 7)
│   └── tca_{run_id}_{date}.json
├── paper_attribution/
│   ├── attribution_{run_id}_{date}.csv                        (Phase 9)
│   └── attribution_{run_id}_{date}.json
├── reconciliation_alerts/
│   └── reconcile_alert_{run_id}_{date}.json                   (Phase 6, only on warn/fail)
├── shadow_compare/
│   └── shadow_compare_{run_id}_{date}.csv                     (Phase 6b, only if shadow_mode=True)
├── paper_intents/
│   └── intents_{run_id}_{date}.jsonl                          (Phase 6)
└── manifests/
    ├── {run_id}/manifest_{date}.json                          (Phase 8)
    ├── {run_id}/manifest.latest.json                          (pointer, file copy)
    └── index.csv                                              (cross-run aggregate index)
```

## 3. Reading a manifest

`manifests/{run_id}/manifest_{date}.json` is the canonical entry point for a run. Key fields:

- `run_id`, `date`, `started_at_utc`, `finished_at_utc`, `status`
- `git_sha`, `config_hash` (16-char SHA-256 prefix)
- `phase_versions.paper_engine = "phase8"` — bump on breaking schema changes
- `artifacts` — `{name: path}` map; only existing files are listed
- `metrics` — lightweight key metrics also mirrored into `index.csv`

The `manifest.latest.json` pointer is a regular file copy (not a symlink) for Windows compatibility.

## 4. Comparing two runs

Quick diff via the cross-run index:

```bash
python -c "
import pandas as pd
df = pd.read_csv('output/manifests/index.csv')
print(df.pivot_table(index='date', columns='run_id', values=['final_equity','avg_cost_bps']))
"
```

Full artifact-by-artifact compare:

```python
from pathlib import Path
import json
a = json.loads(Path('output/manifests/run_A/manifest_2025-01-15.json').read_text())
b = json.loads(Path('output/manifests/run_B/manifest_2025-01-15.json').read_text())
print('git:  ', a['git_sha'], b['git_sha'])
print('cfg:  ', a['config_hash'], b['config_hash'])
print('pnl:  ', a['metrics']['total_return'], b['metrics']['total_return'])
```

If `config_hash` matches and `git_sha` matches and `random_seed` is set, fills must be bit-identical. A mismatch is a regression signal.

## 5. Attribution drilldown

`paper_attribution/attribution_{run_id}_{date}.csv` is per symbol with notional-weighted bps per cost component. Summing `spread_cost_cash + impact_cost_cash + adversarial_cost_cash + sor_cost_cash` must equal `total_cost_cash` to within float tolerance — if not, a cost term is leaking.

The JSON companion adds `regime` and `factor` aggregates (empty lists if no `regime_history` / `dominant_factor` column is available for the run).

## 6. Reconciliation alerts

`reconcile_alert_{run_id}_{date}.json` is only written when the SLO evaluation is not `ok`. Severity is one of `warn` / `fail`. Default thresholds (from `ReconcileSLO`):

| Metric | warn | fail |
|---|---|---|
| cash_diff_bps | 5 | 25 |
| position_qty_diff | 1 | 10 |
| fill_rate_min | 0.80 | 0.50 |
| slippage_p99_bps | 30 | 100 |

A `fail` alert should page the on-call; a `warn` should enter the daily review queue.

## 7. Regression protection

Extreme-regime changes are guarded by the Phase 10 regression pack:

```bash
pytest -m regression tests/regression/
```

The pack covers COVID crash (2020-03-16), 2022 bear start (2022-01-24) and 2010 flash crash (2010-05-06). Golden KPIs live in `tests/regression/golden_metrics.json`; deviations >5% fail the pack.

## 8. Cost-model calibration

After accumulating ~20 days of TCA artifacts, run the offline calibrator:

```python
from src.assembled_core.execution.cost_model_calibrator import (
    calibrate_cost_model, write_calibration_report,
)
res = calibrate_cost_model(Path("output/paper_tca"))
write_calibration_report(res, Path("configs/fill_model_calibrated.yaml"))
```

The calibrator applies a conservative 30 % shrinkage toward the priors — a single run cannot move a cost knob more than 70 % of the way to its realised value. Deployment of the recommendation is always a **manual** config change (no auto-deploy).
