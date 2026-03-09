# Paper experiments (OPS-7 / OPS-8)

**Sim vs Production:** Sim-only: `geo_spikes` / `confirm_every_n_days` sind Harness-Parameter, nicht Production. Production nutzt echte NEWS/DISCL-Artefakte; Sim ist nur Policy-Test.

## Intel mode in paper runner (OPS-11)

`configs/app.yaml` → `paper_runner.intel.mode` steuert, ob und wie Intel in den Paper Runner einfließt:

- **`mode=none`**: Kein Intel – weder Pipelines noch Sim; TradingContext hat keine News-/Disclosures-Artefakte.
- **`mode=sim`**: BENCH-Harness (BENCH-1/BENCH-2) – deterministischer Intel-Sim für Policy-Tests (Confirm-Gate, geo_spikes). Für A/B-Experimente und Sim-only-Läufe.
- **`mode=real`**: Echte Pipelines – vor dem Trading Cycle werden NEWS- und/oder DISCLOSURES-Pipeline ausgeführt; der Trading Cycle lädt die erzeugten Artefakte (z. B. `triggers_latest.json`). Für artifact-getriebene Forward-/Paper-Läufe.

**Kurz:** Sim = Policy-Test mit festen Szenarien; Real = Lauf mit echten NEWS/DISCL-Artefakten. Der Status pro Pipeline steht in `run_kpis.json` unter `intel_orchestration` (news/disclosures mit `ran`, `status`).

## Run A/B experiments

Use `run_paper_experiment` with a baseline and a treatment (e.g. confirm gate), then `compare_paper_experiments` to get delta KPIs. Mit **OPS-12** kannst du zusätzlich `--app-overrides` setzen und so `paper_runner.intel.mode` (none|sim|real) pro Experiment festlegen, ohne `configs/app.yaml` zu ändern.

## Real Intel Serien (OPS-12) — echter Compare auf NEWS/DISCL-Artefakten

**Reihenfolge:** Zuerst 1-Tages-Smoke, dann die drei Serien. Nicht blind 90 Tage laufen lassen.

### 1) Zuerst: 1-Tages-Smoketest in Real

Vor den großen Serien prüfen, ob Intel-Orchestrierung und Artefakte funktionieren.

**Smoke: Real Intel, 1 Tag**

```powershell
py -3 scripts/cli.py run_paper_experiment --name real_smoke `
  --start 2025-10-31 --end 2025-10-31 --mode paper `
  --overrides "{}" `
  --app-overrides '{"paper_runner":{"intel":{"mode":"real","run_news_pipeline":true,"run_disclosures_pipeline":true}}}'
```

**Danach prüfen:**

- **`output/runs/_experiments/real_smoke/runs/2025-10-31/run_kpis.json`**
  - `intel_orchestration.news.status`
  - `intel_orchestration.disclosures.status`
- **Intel-Artefakte vorhanden?**
  - `output/intel/news/triggers_latest.json`
  - `output/intel/disclosures/triggers_latest.json`

Wenn dort **ERROR** steht, ist das ein Daten-/Pipeline-Thema, kein A/B-Thema – zuerst beheben.

### 2) Danach: die drei echten Serien

Nur wenn der Smoke sauber ist:

**A) Ohne Intel**

```powershell
py -3 scripts/cli.py run_paper_experiment --name real_none `
  --start 2025-06-26 --end 2025-10-31 --mode paper `
  --overrides "{}" `
  --app-overrides '{"paper_runner":{"intel":{"mode":"none"}}}'
```

**B) Real Intel, Gate aus**

```powershell
py -3 scripts/cli.py run_paper_experiment --name real_gate_off `
  --start 2025-06-26 --end 2025-10-31 --mode paper `
  --overrides '{"risk_state_machine":{"hysteresis":{"disclosures_confirm_gate":{"enabled":false}}}}' `
  --app-overrides '{"paper_runner":{"intel":{"mode":"real","run_news_pipeline":true,"run_disclosures_pipeline":true}}}'
```

**C) Real Intel, Conditional Gate an**

```powershell
py -3 scripts/cli.py run_paper_experiment --name real_gate_cond `
  --start 2025-06-26 --end 2025-10-31 --mode paper `
  --overrides '{"risk_state_machine":{"hysteresis":{"disclosures_confirm_gate":{"enabled":true,"min_geo_score":3,"on_states":["WATCH","COOLDOWN"]},"disclosures_min_severity":1}}}' `
  --app-overrides '{"paper_runner":{"intel":{"mode":"real","run_news_pipeline":true,"run_disclosures_pipeline":true}}}'
```

**D) Vergleiche**

```powershell
py -3 scripts/cli.py compare_paper_experiments --a real_none --b real_gate_off
py -3 scripts/cli.py compare_paper_experiments --a real_gate_off --b real_gate_cond
```

### 3) Real-World-Hinweis

Für **real_gate_off** und **real_gate_cond** sind die Ergebnisse nur sinnvoll, wenn die Real-Pipelines tatsächlich Material liefern. Typische Bremsen:

- **NEWS** läuft, liefert aber kaum/keine aktiven Trigger.
- **DISCL** läuft, aber House PTR/EDGAR liefern im Zeitraum nichts oder `status=DEGRADED`/`ERROR`.

Deshalb in **`run_kpis.json`** und ggf. **`reasons_latest.json`** besonders prüfen:

- `intel_orchestration`
- `triggers_summary` / `top_triggers`
- `risk_state`
- `qc_flags`

### 4) Reporting nach den Compares

Sobald die beiden Compare-Dateien vorliegen, posten:

- **Aus Compare:** `a`, `b`, `delta`
- **Aus den Summaries:** `active_pct`, `disclosures_confirm_blocks`, `risk_state_distribution`, `alerts_count_by_level`

**Hinweis für später:** In `run_kpis.json` zusätzlich `news_triggers_summary` und `disclosures_triggers_summary` zu schreiben (nicht nur ein gemeinsames `triggers_summary`) erleichtert die Diagnose, welcher Intel-Layer gerade leer ist. Kein Blocker für den Smoke.

## EOD date range (OPS-8)

**Vor Experimenten immer `inspect_eod_range` ausführen, um gültige Start/Ende zu wählen.**

```bash
py -3 scripts/cli.py inspect_eod_range --freq 1d --write-json true
```

Nimm `min_utc`/`max_utc` bzw. die empfohlenen Fenster `last_30_trading_days` / `last_90_trading_days` als `--start` und `--end` für `run_paper_experiment`. Mindestens 30 Trading Days, besser 60–90.

## Sim Treatment 2 (OPS-10): Confirm-Gate sichtbar + Turnover entschärft

Cooldown 6h, per_run State, Turnover-Cap 0.80, Intel-Sim Confirm alle 7 Tage (`configs/app.yaml`: `paper_runner.intel_sim.disclosures_confirm_every_n_days: 7`).

- **Gate OFF:** `run_paper_experiment --name sim_t2_gate_off --start 2025-06-26 --end 2025-10-31 --mode paper --overrides '{"risk_state_machine":{"persistence":{"mode":"per_run","per_run_dir":"output/state/runs"},"cooldown":{"hours":6}},"turnover_budget":{"enabled":true,"cap":0.80,"behavior":"scale"}}'`
- **Gate ON:** wie oben, `--name sim_t2_gate_on` und in `overrides` zusätzlich `"hysteresis":{"require_disclosures_confirm":true,"disclosures_min_severity":1}` unter `risk_state_machine`.
- **Compare:** `compare_paper_experiments --a sim_t2_gate_off --b sim_t2_gate_on`
- **Debug (Reason-Counts):** `Get-ChildItem output\runs\_experiments\sim_t2_gate_on\runs -Recurse -Filter run_kpis.json | ForEach-Object { (Get-Content $_.FullName -Raw | ConvertFrom-Json).risk_state.reason } | Group-Object | Sort-Object Count -Descending`

Erwartung: `disclosures_confirm_blocks` > 0 bei Gate ON, `active_pct` niedriger bei Gate ON.

## Conditional Gate Zielprofil (Treatment-3, DISCL-6)

Das gewünschte Profil (Blocks 5–15, ACTIVE_pct 0,20–0,35) setzt voraus, dass der Intel-Sim an einigen Tagen **geo_score=3** liefert. Dafür muss **geo_spikes** aktiviert sein (`configs/app.yaml` → `paper_runner.intel_sim.geo_spikes.enabled: true`, `every_n_days: 7`, `geo_score: 3`). Ohne geo_spikes liefert der Sim nur 1/2 → Conditional Gate mit `min_geo_score=3` blockt nie; mit `min_geo_score=2` verhält er sich wie das Hard Gate.

## Treatment-4: Conditional Gate Zielprofil (spikes=7, confirm=9)

Spike- und Confirm-Zyklen unsynchron halten: `disclosures_confirm_every_n_days: 9`, `geo_spikes.every_n_days: 7` (in `configs/app.yaml`).

- **Gate OFF:** `run_paper_experiment --name sim_t4_gate_off --start 2025-06-26 --end 2025-10-31 --mode paper --overrides '{"risk_state_machine":{"persistence":{"mode":"per_run","per_run_dir":"output/state/runs"},"cooldown":{"hours":6},"hysteresis":{"disclosures_confirm_gate":{"enabled":false}}},"turnover_budget":{"enabled":true,"cap":0.80,"behavior":"scale"}}'`
- **Conditional Gate (min_geo_score=3):** wie oben, `--name sim_t4_gate_cond` und in `risk_state_machine.hysteresis`: `"disclosures_confirm_gate":{"enabled":true,"min_geo_score":3,"on_states":["WATCH","COOLDOWN"]},"disclosures_min_severity":1`.
- **Compare:** `compare_paper_experiments --a sim_t4_gate_off --b sim_t4_gate_cond`

Zielprofil: disclosures_confirm_blocks ~6–10, ACTIVE_pct zwischen OFF und Hard Gate (z. B. 0,20–0,50), alerts_warn bei Gate cond niedriger als OFF.

## Treatment-5 (optional): ACTIVE_pct mit activate_score=3

Feinjustierung Richtung ACTIVE_pct 0,20–0,35: `activate_score: 3` (weniger Aktivierungen, gleicher Cooldown).

- **Run:** `run_paper_experiment --name sim_t5_gate_cond_act3 --start 2025-06-26 --end 2025-10-31 --mode paper --overrides '{"risk_state_machine":{"persistence":{"mode":"per_run","per_run_dir":"output/state/runs"},"cooldown":{"hours":6},"hysteresis":{"activate_score":3,"deactivate_score":1,"disclosures_confirm_gate":{"enabled":true,"min_geo_score":3,"on_states":["WATCH","COOLDOWN"]}}},"turnover_budget":{"enabled":true,"cap":0.80,"behavior":"scale"}}'`
- **Compare:** `compare_paper_experiments --a sim_t4_gate_cond --b sim_t5_gate_cond_act3` (erwartet: active_pct in sim_t5 niedriger).
