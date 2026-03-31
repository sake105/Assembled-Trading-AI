# ASSEMBLED Trading AI — ROADMAP_LOG.md

Purpose: chronological execution log for milestone completions, blockers, state changes, and significant decisions.
Rule: append-only. Do not edit past entries. Add new entries at the bottom.

---

## Log format

```
### [DATE] [MILESTONE-ID] — [SHORT TITLE]

**Status change:** old → new
**What was done:**
**What was verified:**
**What remains open:**
**Next step:**
```

---

### 2026-03-29 M0 — Repo Governance & Policy Baseline (close)

**Status change:** specified → partially implemented (deliverables complete; CI not fully confirmed)

**What was done:**

- `docs/roadmap/MASTER_ROADMAP.md` committed (milestone chain M0–M7, acceptance criteria, stop conditions).
- `docs/roadmap/ROADMAP_STATE.md` committed and updated to reflect real repo state.
- `docs/roadmap/ROADMAP_LOG.md` created (this file — first entry).
- Governance layer aligned: `CLAUDE.md`, `.claude/rules/` (10 rule modules), `AGENTS.md`, `.cursor/rules/`.
- Claude Code subagents defined with routing guidance in `CLAUDE.md §15.5`.
- `docs/STRATEGY_POLICY.md` implemented (risk targets, scope v1, state machine principles).
- `configs/policy.yaml` implemented (risk_state_machine, georisk_overlay, profit_lock, market_stress, turnover_budget — all substantive, some TBDs remain for numeric thresholds).
- `docs/learning/` implemented: templates for incidents, patterns, anti-patterns; checklists for PR, release, runbook.
- `docs/SECURITY_SECRETS.md` + `.claude/rules/20-security-and-secrets.md` implemented.
- Intel loaders implemented: `src/assembled_core/intel/news_triggers_loader.py`, `disclosures_triggers_loader.py`.
- News/disclosures configs implemented: `configs/news/news.yaml`, `configs/news/sources.yaml`, `configs/disclosures/`.
- Data layer stubs completed (10 modules): `factor_store.py`, `panel_store.py`, `universe.py`, `security_master.py`, `news/contract.py`, `news/store.py`, `shipping/contract.py`, `data_source.py`, `altdata/finnhub_events.py`, `altdata/finnhub_news_macro.py`.
- Test bug fixed: `tests/test_factor_store_roundtrip.py` base_date off-by-11-months.

**What was verified:**

- 94 tests pass locally (Windows, Python 3.11.9), branch `cursor/development-environment-setup-8e96`.
- Zero ImportError collection failures remaining.
- Governance files read and confirmed internally consistent.
- `configs/policy.yaml` validated by reading — no format errors, substantive content confirmed.

**What remains open:**

- CI confirmation for data stub changes not yet done (local only).
- `.env` secret exposure: key must be treated as compromised until rotated at provider level.
- `docs/cursor/` audit for stale context not yet completed.
- Startup hook error status unknown — should be investigated before relying on automated hook runs.
- `configs/policy.yaml` numeric TBDs (soft DD %, hard DD %, target vol, position weight limits).
- Claude-Mem integration not validated as fully operational.

**Next step:** M1-T01 — define or verify NEWS output schema contract (`news.triggers.v1`) and create spec document.

---

### 2026-03-29 M1 — NEWS v1 MVP (trigger scoring + entity linking + worker)

**Status change:** partially implemented → locally tested

**What was done:**

- Diagnosed test failures: 12 failures across news test suite before this session.
- **M1-T09 (trigger scoring wired):** `score_triggers()` was fully implemented in
  `src/assembled_core/events/news/trigger_scoring.py` but never called from `pipeline.py`.
  `triggers_latest.json` was always emitted with `count: 0, items: []`.
  Fix: imported `score_triggers`, added `trigger_scoring_cfg` extraction, inserted call after
  health computation, wired `trigger_items` into `triggers_wrapper`, added
  `health.metrics["triggers"]` with `trigger_count` and `max_severity`.
- **entity_linking.py:** Stub (`link_news_to_symbols(news, symbols=None)`) replaced with
  full implementation supporting `mapping_df`, `security_master_df`, `missing` parameter
  ("keep"/"drop"/"keep_unknown"/"raise"), whitespace stripping, priority (mapping_df > security_master).
- **M1-T13 (worker entry point):** Created `scripts/run_news_worker.py` with:
  - `--cadence hourly|daily`, `--sources`, `--news`, `--output-dir`, `--no-lock` args
  - File-based lock (`_WorkerLock` via `os.O_CREAT|O_EXCL`) — concurrent runs skip cleanly
  - Structured log output: `[START]`, `[OK]`, `[WARN]`, `[ERROR]` prefixes
  - Exit code 1 on pipeline exception, 0 otherwise (including lock-skip)

**What was verified:**

- `tests/test_news_trigger_scoring.py`: 10/10 pass (was 9/10 before)
- `tests/test_news_entity_linking.py`: 13/13 pass (was 2/13 before)
- Full news suite (112 tests): 112/112 pass — no regression
- `scripts/run_news_worker.py --help`: imports cleanly, argparse works
- Platform: win32, Python 3.11.9, pytest 9.0.1. CI not confirmed.

**M1 acceptance criteria status:**

- ✅ Health gate works (severity capping in DEGRADED/ERROR state)
- ✅ Triggers have correct schema and are scored (8 topic rules, severity 0–3)
- ✅ Duplicates do not leak (dedupe_store + in-memory dedupe)
- ✅ Output schemas deterministic and versioned
- ✅ Worker entry point exists with locking
- ✅ 112 tests covering all components
- ⚠ TTL/decay: config key present (`ttl_decay.enabled: false`) — not yet implemented in code;
  documented as known deferred item in `docs/news/NEWS_SPEC.md`

**What remains open:**

- CI confirmation pending.
- TTL/decay not implemented (disabled by config; deferred).
- `entity_linking.py` only handles `ticker` and `entity` columns — no NLP/entity-extraction
  from free-form headlines (out of scope for M1 per NEWS_SPEC.md).

**Next step:** M2-T01 or continue with M1 stop-condition review before advancing.

---

### 2026-03-29 M2 — DISCLOSURES v1 (test fix + worker entry point)

**Status change:** partially implemented → locally tested

**What was done:**

- Audited full disclosures pipeline: all 13 modules exist in
  `src/assembled_core/events/disclosures/`, configs, intel loader, risk confirm,
  docs, and output artifacts already present. Pipeline was substantially complete.
- **Test bug fixed** (`test_pipeline_fetch_report_includes_house_ptr_stats`):
  - Root cause: `house_ptr` is `active: false` in `configs/disclosures/sources.yaml`
    (intentionally, because `index_url` is a placeholder). Test was patching
    `fetch_house_ptr_filings` but the pipeline never reached that branch for inactive sources.
  - Fix: test now writes a test-local `sources_test.yaml` with `house_ptr: active: true`
    and passes it to `run_disclosures_pipeline`. Production config remains unchanged.
- **Worker entry point created** (`scripts/run_disclosures_worker.py`):
  - Same structure as `run_news_worker.py`: argparse, `_WorkerLock`, structured log output.
  - Args: `--cadence`, `--sources`, `--disclosures`, `--output-dir`, `--no-lock`.
  - Exit code 1 on pipeline exception, 0 otherwise (including lock-skip).

**What was verified:**

- `tests/test_disclosures.py`: 21/21 pass (was 20/21 before).
- Combined news + disclosures suite (133 tests): 133/133 pass — no regression.
- `scripts/run_disclosures_worker.py --help`: imports cleanly, argparse works.
- Platform: win32, Python 3.11.9, pytest 9.0.1. CI not confirmed.

**M2 acceptance criteria status:**

- ✅ Idempotent ingest (dedupe by fingerprint)
- ✅ EDGAR Form 4 fetch with cache + stale-on-error
- ✅ House PTR fetch (active: false in prod config until real index_url configured)
- ✅ Normalization to DisclosureEvent schema
- ✅ Trigger scoring (severity, confidence, TTL, decay, QC caps)
- ✅ Health gates (DEGRADED/ERROR severity capping)
- ✅ TradingContext integration (disclosures_triggers_loader.py)
- ✅ News geo confidence boost (disclosures_confirm.py)
- ✅ Worker entry point with locking
- ✅ 21 tests covering all components
- ⚠ House PTR `index_url` is a placeholder — source disabled in prod until real URL configured
- ⚠ 13D/13G fetch (`fetch_edgar`) is a stub — returns empty results (documented in DISCLOSURES_SPEC.md)
- ⚠ PDF parsing not implemented (download_pdfs: false in config; deferred)

**What remains open:**

- CI confirmation pending.
- 13D/13G full fetch implementation (stub, deferred to later sprint).
- House PTR real index_url configuration (operational, not code).
- PDF content parsing (config-disabled, deferred).

**Next step:** M3-T01 — audit risk state machine current state before starting.

---

### 2026-03-30 M3 — Risk / State Machine v1 (audit + bug fix)

**Status change:** partially implemented → locally tested

**What was done:**

- Full audit of `src/assembled_core/risk/` (14 modules) and
  `src/assembled_core/pipeline/trading_cycle.py` (1200 lines).
- **Bug fixed** (`risk_metrics.py` line 410): `compute_risk_by_regime()` passed
  `equity=equity_from_returns` to `compute_basic_risk_metrics()` which has no `equity` parameter.
  Removed stale kwarg and unused `equity_from_returns` variable.

**M3 task status:**

- M3-T01 ✅ Spec: `state_machine.py` — WATCH/ACTIVE/COOLDOWN/PAUSE fully specified and implemented.
- M3-T02 ✅ Persistent state format: `RiskStateRecord` dataclass, JSON, atomic write with retry.
- M3-T03 ✅ Deterministic transition function: `compute_next_state(ctx, policy, now_utc, prev)`.
- M3-T04 ✅ Hysteresis/debounce: `activate_score=2` vs `deactivate_score=1`, cooldown 24h, confidence_floor.
- M3-T05 ✅ Drawdown computation: `market_stress.py` (vol_z + dd_lookback); `risk_metrics.py` fixed.
- M3-T06 ⚠️ Drawdown guards: pre_trade_checks.py has `drawdown_threshold`+`de_risk_scale` mechanism;
  but `policy.yaml` `risk_limits.max_drawdown.soft/hard/kill` remain TBD (numeric values not filled).
- M3-T07 ✅ Exposure caps per state: `georisk_overlay.py` maps state → multiplier
  (WATCH=1.0, ACTIVE=0.70, COOLDOWN=0.85, PAUSE=0.0), applied in `run_trading_cycle`.
- M3-T08 ✅ Trading cycle integration: `run_trading_cycle` calls `compute_next_state`, saves state,
  applies geo overlay to target positions, applies turnover budget and profit lock.
- M3-T09 ✅ Tests: 17 state machine tests, 10 risk_metrics tests, 7 drawdown_derisk tests — all pass.
- M3-T10 ⚠️ Docs: reason field logged in RiskStateRecord; no dedicated M3 spec document written.

**What was verified:**

- 17/17 `test_risk_state_machine.py` — all transitions, persistence, atomic write retry,
  disclosures confirm gate, ephemeral/per_run modes.
- 116/116 full risk test suite (`tests/test_risk_*.py`) — all pass.
- 150/150 combined NEWS + DISCLOSURES + risk state machine suite — all pass.
- Platform: win32, Python 3.11.9, pytest 9.0.1. CI not confirmed.

**M3 acceptance criteria status:**

- ✅ Drawdown transitions deterministic and tested
- ✅ No flip-flop (hysteresis + cooldown timer)
- ✅ Integrates with trading cycle
- ✅ State survives restart (atomic JSON persistence)
- ✅ Logs explain transition reasons (reason field)

**Known limitations (not blocking for v1):**

- Exposure cap (PAUSE=0.0) only fires when `ctx.news_geo` is set. Without news pipeline running,
  `compute_exposure_multiplier` returns 1.0 regardless of state.
- `policy.yaml` numeric drawdown thresholds remain TBD.

**What remains open:**

- CI confirmation pending.
- M3-T06: fill in numeric thresholds in `policy.yaml` (operational config decision).
- M3-T10: write brief M3 spec or architecture doc.
- Exposure cap robustness: consider enforcing PAUSE=0.0 even when `news_geo` is absent.

**Next step:** M4-T01 — spec stop_worker, reconcile_worker, minimal kill_switch_worker.

---

### 2026-03-30 M4 — Execution Workers (Ops v1) (core implementation)

**Status change:** discussed → locally tested (core deliverables implemented and verified)

**What was done:**

- **M4-T01/T02/T03 (worker contracts + intent store + idempotency keys):**
  Created `src/assembled_core/execution/intent_store.py` — append-only JSONL store
  for hard operational actions. Key functions:
  - `make_daily_key(action, date_str)` — stable idempotency key per action per UTC day
  - `make_run_key(action, run_id)` — stable key scoped to a specific run_id
  - `record_intent(action, key, metadata, store_path)` — append record (caller checks first)
  - `has_intent(key, store_path)` — check for existing record before acting
  - `load_intents(store_path)` — load all records; tolerant of malformed lines
  - `filter_intents_by_action(action, store_path)` — filter by action type

- **M4-T04 (stop_worker):**
  Created `scripts/run_stop_worker.py`. Records STOP intent + writes `.stop_active`
  sentinel file. Idempotent: skips on second run (same-day key), `--force` to override.
  Args: `--reason`, `--output-dir`, `--force`.

- **M4-T05 (reconcile_worker):**
  Created `scripts/run_reconcile_worker.py`. Loads ledger parquet via
  `build_positions_from_ledger`, loads broker snapshot CSV, calls
  `reconcile_ledger_vs_broker`, writes timestamped JSON manifest, records RECONCILE intent.
  Args: `--ledger-path`, `--broker-path`, `--ledger-cash`, `--broker-cash`, `--output-dir`,
  `--cash-tol`, `--qty-tol`.

- **M4-T06 (kill_switch_worker):**
  Created `scripts/run_kill_switch_worker.py`. Records KILL intent + writes
  `.kill_switch_active` sentinel. Optional `--positions-path` generates a SAFE-Bridge
  flatten orders CSV (human review required before execution — not auto-submitted).
  Args: `--reason`, `--output-dir`, `--positions-path`, `--force`.

- **M4-T07 (audit logging):**
  All three workers produce structured `[START]`/`[OK]`/`[WARN]`/`[SKIP]`/`[ERROR]` logs.
  RECONCILE worker writes timestamped JSON manifests. Intent store is the cross-worker
  audit trail.

- **M4-T08 (idempotency tests):**
  Created `tests/test_execution_intent_store.py` with 27 tests covering:
  key generation (make_daily_key, make_run_key), load/record/filter, has_intent,
  malformed line tolerance, blank line skip, force-override pattern, daily vs run key
  distinction, caller-level idempotency simulation.

**What was verified:**

- 27/27 `test_execution_intent_store.py` — all pass.
- 97/97 combined M1+M2+M3+M4 suite (intent_store + kill_switch + selected news/risk tests).
- All three worker scripts: `--help` imports cleanly, smoke-runs produce expected output.
- Stop worker idempotency: second run logs `[SKIP]` and exits 0 without re-writing.
- Reconcile worker: empty ledger/broker inputs → `ok=True` manifest written.
- Kill switch worker: sentinel written, intent recorded, exit 0.
- Platform: win32, Python 3.13.7, pytest 9.0.2.

**M4 task status:**

- M4-T01 ✅ Worker contracts and command surfaces defined (argparse, structured logs)
- M4-T02 ✅ Intent store designed and implemented (JSONL, append-only, tolerant)
- M4-T03 ✅ Idempotency key rules defined (daily key, run key, caller-checks pattern)
- M4-T04 ✅ stop_worker implemented
- M4-T05 ✅ reconcile_worker implemented
- M4-T06 ✅ kill_switch_worker implemented (sentinel + optional SAFE-Bridge flatten)
- M4-T07 ✅ Audit logging: structured log output + intent store + reconcile manifests
- M4-T08 ✅ Tests: 27 intent_store tests; smoke-runs for all three workers
- M4-T09 ⚠️ Docs: ROADMAP docs being updated; no dedicated M4 ops spec doc yet

**M4 acceptance criteria status:**

- ✅ Stop execution not dependent on main cycle (standalone worker, file sentinel)
- ✅ Reconcile establishes source-of-truth consistency (reads ledger + broker; writes manifest)
- ✅ Kill switch can pause and flatten safely in paper (sentinel + optional SAFE-Bridge CSV)
- ✅ Repeated runs do not duplicate hard actions (intent_store idempotency keys)

**Pre-existing failures (not caused by M4):**

- 5 failures in `test_execution_safe_orders.py`, `test_execution_pre_trade_integration.py`,
  `test_execution_order_generation_vectorized.py` — confirmed as pre-existing (those files
  are unchanged, not in git diff).

**What remains open:**

- CI confirmation pending.
- M4-T09: write a brief M4 ops spec or operational runbook.
- `kill_switch_worker` does not auto-submit flatten orders — manual step required.
- `stop_worker` sentinel file is informational only; trading cycle does not auto-check it
  (adding that check is a future wiring task in `run_trading_cycle`).

**Next step:** M5-T01 — spec / input contract for Crisis-Alpha v1. Or address M3/M4 open
items first (numeric drawdown thresholds in policy.yaml, M4 ops spec doc).

---

### 2026-03-30 M5 — Crisis-Alpha v1 (core implementation)

**Status change:** discussed → locally tested (core subsystem implemented and verified)

**What was done:**

- **M5-T01 (spec + input contract):** Designed `CrisisAlphaContext` dataclass as the
  input contract for the crisis pipeline. Fields: `timestamp_utc`, `geo_score`,
  `geo_sources`, `social_only`, `market_stress_ok`, `health_ok`, `daily_pnl`,
  `daily_loss_limit`, `news_trigger_items`, `open_positions`.

- **M5-T02 (health gate + input contract):** `gates.py` implements 6 gate checks.
  All gates are pure functions returning `(bool, reason)` for testability.
  `run_all_activation_gates()` applies them in priority order (fail-fast).

- **M5-T03 (persistent state machine):** `state_machine.py` implements
  WATCH/ACTIVE/COOLDOWN/PAUSE with:
  - Hysteresis (activate_geo_score=2.0, deactivate_geo_score=1.0)
  - Cooldown timer (24h minimum in COOLDOWN before WATCH return)
  - Daily loss guard (any state → PAUSE when daily_loss_breached)
  - Health gate (ACTIVE → COOLDOWN when health_ok=False)
  - Social-only guard (social_only=True blocks WATCH→ACTIVE)
  - Atomic JSON persistence (tempfile + os.replace, same pattern as M3)
  - Tolerant load (missing/corrupt file → WATCH default)

- **M5-T04 (geo score aggregation):** Worker script extracts geo_score from
  `triggers_latest.json` by filtering geo-relevant topic categories, finding
  max severity, counting distinct sources, detecting social-only.

- **M5-T05 (evidence rules):** `check_evidence_gate()` requires min qualifying
  triggers (severity >= 1). `check_source_gate()` requires min distinct sources.

- **M5-T06 (market stress confirmation):** `check_market_stress_gate()` requires
  `market_stress_ok=True` for WATCH→ACTIVE.

- **M5-T07 (baskets):** `baskets.py` defines 5 default ETF instruments:
  GLD, TLT, SHY (DEFENSIVE), SH (INVERSE_EQUITY), VIXY (VOLATILITY).
  Policy-overridable via `crisis_alpha.basket_overrides` or full `baskets` replacement.

- **M5-T08 (entries):** `entry.py` implements `equal_weight` and `geo_weighted` methods.
  Only runs when state=ACTIVE. Risk budget applied to all generated weights.

- **M5-T09 (risk budgets):** `risk_budget.py` enforces per-instrument max_weight caps
  and gross exposure cap (0.30). `apply_risk_budget()` applies both in sequence.

- **M5-T10 (exits and deactivation):** `exit_rules.py` implements time_stop (8h),
  break_even (0.5%), no_overnight checks per position. `check_deactivation_triggers()`
  returns flatten_all signal when state≠ACTIVE, daily loss breached, or health ERROR.

- **M5-T11 (runner and worker):** `scripts/run_crisis_alpha_worker.py` standalone
  script with argparse, structured logs, JSON manifest output. Supports `--dry-run`,
  `--reset-pause`, and CLI geo_score overrides for testing.

- **M5-T12 (scenario tests):** `test_crisis_alpha_pipeline.py` includes 4 scenario tests:
  geo shock activation and recovery (WATCH→ACTIVE→COOLDOWN→WATCH),
  false activation blocked (social-only + no market stress),
  health error forces COOLDOWN,
  daily loss → PAUSE + manual reset → WATCH.

**What was verified:**

- 30/30 `test_crisis_alpha_state_machine.py` — all transitions, persistence, roundtrip.
- 22/22 `test_crisis_alpha_gates.py` — all 6 gates + combined gate runner.
- 18/18 `test_crisis_alpha_pipeline.py` — entry, deactivation, exits, dry_run, scenarios.
- 70/70 M5 total.
- 114/114 combined M4+M5 + risk state machine — no regression.
- Worker smoke-runs: WATCH (geo_score=0.5) and ACTIVE (geo_score=2.5) paths verified.
- ACTIVE path generates 5 positions, total gross = 0.30 (cap enforced correctly).
- Platform: win32, Python 3.13.7, pytest 9.0.2.

**M5 acceptance criteria status:**

- ✅ Social-only cannot activate
- ✅ Degraded health cannot activate (health_ok gate blocks WATCH→ACTIVE)
- ✅ Deactivation and cooldown work (hysteresis + cooldown timer)
- ✅ Max daily loss pauses
- ✅ All transitions and protective actions deterministic and logged

**Known limitations (not blocking for v1):**

- Order submission is not automated — pipeline returns target_weights for manual review.
- Geo score is derived from topic-filtered triggers only; future: configurable topic list.
- No intraday re-evaluation (one-shot worker design); future: scheduled cadence via cron.

**What remains open:**

- CI confirmation pending.
- M5-T13: write a dedicated Crisis-Alpha spec document.
- No-overnight enforcement requires operator to act on the flagged positions.
- Worker does not integrate with the main trading cycle hook yet (standalone only).

**Next step:** M6-T01 — define target vol and realized vol calc for Risk v1.1 Upgrades.
Or address open items first (M3 numeric thresholds, M5 spec doc).

---

### 2026-03-31 M6 — Risk v1.1 Upgrades (partial — core modules)

**Status change:** specified → locally tested (core modules M6-T01/T02/T05/T07)

**What was done:**

- `src/assembled_core/risk/vol_targeting.py` — new module (M6-T01/T02):
  - `compute_realized_vol(returns, lookback_days, annualize_factor, min_observations)` →
    annualized realized vol from returns series; returns nan if insufficient data.
  - `compute_vol_scale_factor(realized, target, min_scale, max_scale)` →
    target/realized clamped to [min_scale, max_scale]; 1.0 on invalid inputs.
  - `apply_vol_targeting_to_weights(target_weights, scale_factor)` →
    scales symbol→weight dict, returns new dict (no mutation).
  - `compute_vol_targeting_result(equity_curve, policy, now_idx)` →
    policy-driven main entry; returns (scale, realized_vol, target_vol).

- `src/assembled_core/risk/zombie_killer.py` — new module (M6-T05):
  - `check_zombie_position(position, now_utc, max_hold_days, min_gain_pct)` →
    (is_zombie, reason); handles missing prices (conservative flag), unparseable ts (safe skip),
    long and short sides.
  - `get_zombie_positions(positions, now_utc, policy)` →
    policy-driven scan; reads zombie_killer.{enabled, max_hold_days, min_gain_pct}.

- `src/assembled_core/risk/correlation_guard.py` — new module (M6-T07):
  - `compute_correlation_matrix(prices, symbols, lookback_days)` →
    pivot timestamp×symbol returns, compute pairwise Pearson; empty if < 2 symbols or < 3 bars.
  - `detect_correlated_clusters(corr_matrix, threshold)` →
    union-find over positive correlations (≥ threshold); returns sorted list-of-lists.
    Negative (hedging) correlations excluded — only positive clustering creates concentration risk.
  - `apply_correlation_guard(target_weights, prices, policy)` →
    (adjusted_weights, reasons); proportional scale-down of over-concentrated clusters.

- 5 test files (96 tests, all passing):
  - `tests/test_risk_vol_targeting.py` — 28 tests (4 classes: realized vol, scale factor,
    apply to weights, compute_vol_targeting_result; covers disabled, nan, clamping, high/low vol)
  - `tests/test_risk_zombie_killer.py` — 19 tests (check_zombie_position: 12 tests;
    get_zombie_positions: 7 tests; covers edge cases, policy config, tuple output)
  - `tests/test_risk_correlation_guard.py` — 23 tests (correlation matrix: 7; cluster detection: 6;
    apply_correlation_guard: 10; covers disabled, no prices, single symbol, transitive clustering)
  - `tests/test_risk_profit_lock.py` — 12 tests covering existing profit_lock.py
    (disabled, insufficient data, trigger, multiplier clamping, cooldown preservation/expiry)
  - `tests/test_risk_turnover_budget.py` — 14 tests covering existing turnover_budget.py
    (estimate_turnover: 7 tests; apply_turnover_gate: 7 tests; scale and block behaviors)

**What was verified:**

- 96/96 M6 tests pass locally.
- 210/210 targeted tests pass (M6 + M5 + M3 state machine + M4 intent store).
- Bug found and corrected: test_cooldown_expires_resets had wrong now_idx (16 < lookback_days=20
  → profit_lock returns early before processing cooldown expiry); fixed to now_idx=25.
- Pre-existing failures in broader marker-based run: `exchange_calendars` not installed
  (pre-existing environment gap; not a M6 regression).

**Known limitations (not blocking for v1.1):**

- vol_targeting.py not yet wired into trading_cycle.py (M6-T03 pending).
- Attribution report (M6-T08) not yet implemented.
- Parameter stability checks (M6-T09) not yet implemented.
- All three modules are pure functions / overlays — no integration tests with trading_cycle yet.

**What remains open:**

- CI confirmation pending.
- M6-T03: integrate vol_targeting as optional overlay hook in trading_cycle.py.
- M6-T08/T09: attribution + parameter stability checks.
- M3 numeric drawdown thresholds in policy.yaml still TBD.
- M5-T13: dedicated Crisis-Alpha spec doc.

**Next step:** M6-T03 — wire vol_targeting into trading_cycle.py as optional multiplier,
or M6-T08 — implement attribution report.

---

## 2026-03-31 — Session (2) — M6-T03: vol_targeting wired into trading_cycle.py

**Scope:** Wire `vol_targeting.py` into `trading_cycle.py` as an optional multiplicative exposure multiplier.

**What was done:**

- Delegated pre-implementation safety review to `risk-execution-reviewer` subagent.
  Reviewer identified 3 blocking issues before coding:
  1. Guard bug: original `if final_multiplier < 1.0` would silently skip vol scale factors > 1.0
  2. Leverage conflict: `max_scale=1.5` default inconsistent with `leverage_allowed: false` in policy
  3. PIT safety: must pass `ctx.equity_curve_index` as `now_idx`, not `-1`, to avoid look-ahead in backtest

- `src/assembled_core/pipeline/trading_cycle.py` patched (3 changes):
  1. Added import: `from src.assembled_core.risk.vol_targeting import compute_vol_targeting_result`
  2. Added vol_targeting overlay block after profit_lock, before final_multiplier composition:
     - guard: `vt_cfg.get("enabled", False)` + equity_curve + equity_curve_index presence check
     - PIT-safe call: `compute_vol_targeting_result(ctx.equity_curve, vt_cfg, now_idx=ctx.equity_curve_index)`
     - always writes `result.meta["vol_targeting"]` with scale_factor, realized_vol, target_vol
  3. Extended formula: `final_multiplier = geo_multiplier * profit_lock_mult * vol_scale_factor`
  4. Fixed guard: `if abs(final_multiplier - 1.0) > 1e-9` — handles > 1.0 and < 1.0
  5. Updated log: now shows `geo`, `profit_lock`, `vol`, `final` for full observability

- `configs/policy.yaml` patched:
  - Added `vol_targeting:` section: `enabled: false`, `max_scale: 1.00` (no leverage), 
    `min_scale: 0.50`, `lookback_days: 20`, `target_vol_annual: 0.15`

**What was verified:**

- 106/106 phase12 tests pass (up from 96 in session 1 — additional tests collected, no regressions)
- `from src.assembled_core.pipeline.trading_cycle import run_trading_cycle` import clean
- `git stash` baseline test confirmed pre-trade integration failure is pre-existing circular import,
  not a regression from this patch
- Format string failures in drift-check script: pre-existing, unrelated to M6 changes

**Known limitations:**

- Vol targeting is wired but disabled by default (`enabled: false`). To enable, set `enabled: true` in
  `configs/policy.yaml` and calibrate `target_vol_annual` against realized historical vol.
- No dedicated integration test for the trading_cycle wire-up path exists yet.
  Verification relied on import smoke test + phase12 suite + baseline check.
- CI not confirmed — only locally tested.

**What remains open:**

- M6-T08: attribution report (per-symbol contribution to portfolio vol/return) — new standalone module
- M6-T09: parameter stability checks
- M3 open: numeric drawdown thresholds in policy.yaml still `TBD`
- M5 open: dedicated Crisis-Alpha spec document not yet written
- Pre-existing failures noted: circular import in pre-trade integration test; format string bug in
  drift-check script. Both pre-date this session.

**Next step:** M6-T08 — attribution report (new standalone module, lower blast radius than
trading_cycle changes).

---
