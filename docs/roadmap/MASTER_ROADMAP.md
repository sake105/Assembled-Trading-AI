# ASSEMBLED Trading AI — MASTER_ROADMAP.md

Version: v1.0 (operational long form)  
Status: working planning document  
Audience: Claude Code, future-you, human reviewer  
Primary goal: turn a large, long-horizon strategy and engineering vision into a stepwise, testable, low-drift execution plan.

---

## 1. Purpose of this document

This file is the **long-form master roadmap** for the Assembled Trading AI project. It is intentionally detailed so that work can be resumed after breaks, context limits, session resets, or model switches without losing the overall direction.

This document is **not** the daily cockpit. That role belongs to `ROADMAP_STATE.md`.

This document answers:
- what the system is supposed to become,
- what the main milestones are,
- what each milestone includes and explicitly does **not** include,
- which dependencies must be respected,
- what concrete deliverables must exist,
- what tests and acceptance criteria define completion,
- when work must stop before adding more complexity.

It must be read together with:
- `CLAUDE.md`
- `.claude/rules/`
- `AGENTS.md`
- project architecture and policy docs
- `ROADMAP_STATE.md` for the current execution position.

---

## 2. Operating principles for roadmap execution

### 2.1 Truth before tempo

Every milestone must be treated with the status taxonomy already used in the project governance:
- discussed
- specified
- skeleton present
- partially implemented
- implemented
- locally tested
- CI-validated

No task is considered done merely because it was planned, partially coded, or described in a doc.

### 2.2 Smallest safe step

Claude Code or any other agent must not attempt a milestone as a single giant transformation. Each milestone must be broken into:
- spec / contract,
- config / interfaces,
- minimal implementation,
- tests,
- integration,
- documentation,
- audit / state update.

### 2.3 No feature growth while foundations are unstable

If a lower-level dependency is broken, inconsistent, or not yet validated, the next milestone is blocked.

Examples:
- if repo governance is weak, new sensor systems should not be added;
- if health/freshness is unreliable, crisis activation work must not progress;
- if reconcile is unstable, execution complexity must not grow.

### 2.4 Explicit in-scope and out-of-scope boundaries

Each milestone below states both what must be built and what must intentionally wait.

### 2.5 Paper-first, safety-first

The project is still treated as a **paper-safe system** first. The roadmap supports stronger operations and realism over time, but leverage, blind automation, and fragile live behavior remain out of scope until lower milestones are proven.

---

## 3. System target picture

The long-term target is a modular trading backend that combines:
- a robust baseline EOD/weekly core,
- governance and policy enforcement,
- NEWS as an early-warning sensor,
- DISCLOSURES as confirm/context,
- a stateful risk engine,
- separate idempotent execution workers,
- a crisis-alpha subsystem,
- realism upgrades,
- auditability, incident learning, and reproducibility.

At a high level the intended architecture is:

```text
[Data Sources]
  |-- Market data (OHLCV, proxies)
  |-- NEWS v1 (RSS Tier A + GDELT Tier B)
  |-- DISCLOSURES v1 (House PTR + SEC EDGAR + strategic docs)
  |-- Optional social WATCH layer (later, not core)

[Ingestion Workers]
  |-- news_worker
  |-- disclosures_worker
  |-- optional social_worker
  --> emit artifacts: events / clusters / triggers / signals / health

[Trading Core]
  |-- TradingContext loader
  |-- Regime / state machine
  |-- Portfolio engine
  |-- Execution intents
  |-- Ledger / paper / backtest

[Execution Workers]
  |-- rebalance_worker
  |-- stop_worker
  |-- reconcile_worker
  |-- kill_switch_worker
  |-- crisis_alpha_worker

[QA / Monitoring / Learning]
  |-- QC gates
  |-- health / freshness monitoring
  |-- audit logs
  |-- reports
  |-- incident / pattern learning
```

This matches the staged architecture described in the uploaded roadmap set, especially the staged release view (v1.0 / v1.1 / v2), the milestone chain M0–M7, and the cross-cutting requirements around idempotency, health/freshness, audit logs, and tests. fileciteturn9file0

---

## 4. Release model

### 4.1 v1.0 — Robust Base + Sensors + Paper Safe

Primary intent:
- a stable paper-safe base system,
- governance and policy in place,
- NEWS and DISCLOSURES integrated as machine-readable signal/context layers,
- minimal but real state/risk handling,
- first worker split for safer operations.

Included in v1.0:
- policy page and placeholder config,
- learning folder and templates,
- NEWS v1,
- DISCLOSURES v1,
- minimal risk/state machine,
- existing ledger/reconcile plus basic worker split,
- crisis alpha either WATCH-only or EOD-only conservative mode.

Explicitly not included in v1.0:
- fully solved survivorship issues,
- corporate actions feed beyond current adjusted-price assumptions,
- social as a core trading source,
- leverage.

This definition is directly aligned with the staged release description in the uploaded master roadmap. fileciteturn9file0

### 4.2 v1.1 — Institutional guards + stronger workers + Crisis Alpha ACTIVE

Primary intent:
- more institutional risk discipline,
- stronger operational worker separation,
- controlled crisis activation,
- stronger guardrails against overtrading and clustering risk.

Expected additions:
- vol targeting,
- time stop / zombie killer,
- break-even and profit-lock behavior,
- turnover budget,
- correlation / clump guard,
- idempotent hard-action workers,
- Crisis Alpha ACTIVE on ETFs,
- health gates everywhere.

### 4.3 v2 — Realism + optional advanced extensions

Primary intent:
- realism upgrades,
- optional overlays and optional premium data paths,
- stronger historical fidelity.

Expected additions:
- exchange calendars,
- corporate actions feed / events,
- universe snapshots / security master,
- optional FX overlay,
- optional tail-risk airbag,
- optional paid news providers.

---

## 5. Master milestone map

The practical milestone chain is:
- M0 — Repo Governance & Policy Baseline
- M1 — NEWS v1 MVP
- M2 — DISCLOSURES v1 MVP
- M3 — Risk / State Machine v1
- M4 — Execution Workers (Ops v1)
- M5 — Crisis-Alpha v1
- M6 — Risk v1.1 Upgrades
- M7 — Realism Upgrades v2
- M8 — Evidence Engine
- M9 — Policy Calibration
- M10 — ETF Universe
- M11 — Post-Trade Learning Loop
- M12 — Broker Adapter
- M13 — Autonomous Operations

M0–M7 were defined in the original roadmap set. M8–M13 were added during the 2026-03 execution cycle to address evidence quality, policy operationalization, universe management, learning feedback, broker connectivity, and operational autonomy.


This sequencing is explicitly present in the uploaded roadmap, including the sprint ordering and milestone dependencies. fileciteturn9file0turn9file1

---

## 6. Milestone M0 — Repo Governance & Policy Baseline

### 6.1 Objective

Create the minimum governance baseline so that all later work is constrained by:
- explicit policy,
- project working agreements,
- learning/incident discipline,
- architecture visibility,
- repo hygiene.

### 6.2 Why M0 exists

Without M0, later milestones may be implemented in conflicting ways, with fragile docs, unclear status reporting, missing incidents, and poor constraints for AI-assisted coding.

### 6.3 In scope

Must include:
- `docs/STRATEGY_POLICY.md`
- `configs/policy.yaml`
- `docs/learning/` with templates/checklists
- governance docs / checklists
- clarified repo working agreements
- project instruction layers (`CLAUDE.md`, `.claude/rules/`, `AGENTS.md`, cursor rules) aligned with real repo state
- secret-handling discipline
- PR / change / incident expectations

### 6.4 Out of scope

Do not treat M0 as a product-feature milestone. It is not about alpha generation or new trading features.

### 6.5 Key deliverables

Required deliverables:
- strategy and policy documentation,
- policy config placeholders,
- learning and incident templates,
- governance folder and navigation,
- initial checklist-driven working model.

This is directly aligned with the roadmap’s M0 deliverables and acceptance criteria. fileciteturn9file0

### 6.6 Detailed task decomposition

Recommended tasks:
- M0-T01: document working agreements
- M0-T02: define truth-status vocabulary and completion taxonomy
- M0-T03: create policy/config placeholders
- M0-T04: define learning folder templates
- M0-T05: align agent-governance docs
- M0-T06: document secret-handling and committed-secret incident response
- M0-T07: define PR checklist and incident expectations
- M0-T08: add roadmap governance docs and navigation

### 6.7 Acceptance criteria

Minimum acceptance criteria:
- repo has explicit working agreements,
- PR checklist exists,
- policy placeholders exist,
- governance layer is internally consistent,
- incidents and patterns have a defined home,
- secret-handling is documented as an operational rule.

### 6.8 Validation

Validation must include:
- file existence checks,
- documentation consistency audit,
- manual inspection of conflicting instruction layers,
- no unresolved contradiction between Claude and Cursor governance layers.

### 6.9 Blockers / stop conditions

Do not progress beyond M0 if:
- governance docs contradict the real codebase,
- committed-secret handling remains vague,
- current instruction layers still point to obsolete entry points.

---

## 7. Milestone M1 — NEWS v1 MVP (Sensor)

### 7.1 Objective

Build a robust, low-cost, machine-readable NEWS subsystem that ingests RSS + GDELT, normalizes and deduplicates events, clusters them, detects novelty/burst behavior, emits trigger signals with TTL/decay, and writes health/freshness outputs for downstream gating.

This follows the uploaded NEWS v1 roadmap in structure and scope. fileciteturn9file2

### 7.2 Why M1 matters

The NEWS layer is the first automated external sensor. It should improve awareness and gating, not become a noisy pseudo-HFT layer.

### 7.3 In scope

Required scope:
- hourly or daily ingest cadence,
- Tier-A RSS + Tier-B GDELT,
- source registry,
- canonicalization,
- normalization to a shared event schema,
- dedupe store,
- clustering,
- baseline store and burst detection,
- taxonomy mapping,
- trigger scoring 0–3,
- confidence and evidence rules,
- TTL/decay,
- health/freshness metrics,
- atomic emit of `triggers_latest.json` and `health_latest.json`,
- TradingContext loader integration,
- worker entry point(s),
- unit/integration/golden tests,
- docs and learning patterns.

### 7.4 Out of scope

Not for v1:
- social as a core signal source,
- premium wire providers,
- high-latency-cost NLP-heavy pipeline,
- opaque ML-first trigger generation,
- aggressive intraday trading behavior.

### 7.5 Expected repo placement

Recommended package placement:
- `src/assembled_core/intel/news/...`
- data artifacts under `data/news/...`
- config under `configs/news/...`
- docs under `docs/news/...`
- scripts for hourly/daily worker entry points.

This placement and artifact model is explicitly described in the uploaded NEWS roadmap. fileciteturn9file2

### 7.6 Deliverables

Minimum deliverables:
- news worker with RSS + GDELT ingest,
- minimal normalize/dedupe,
- `triggers_latest.json`,
- `health_latest.json`,
- TradingContext loader reads these files,
- tests with fixtures.

### 7.7 Suggested task decomposition

Recommended breakdown:
- M1-T01: create NEWS spec and output schema versioning
- M1-T02: create source registry and config
- M1-T03: build fetch layer
- M1-T04: implement canonicalization + normalization
- M1-T05: implement dedupe + near-duplicate logic
- M1-T06: implement clustering
- M1-T07: implement baseline store + burst detection
- M1-T08: implement taxonomy mapping and evidence rules
- M1-T09: implement trigger scoring and TTL/decay
- M1-T10: implement health/freshness metrics
- M1-T11: implement writer + atomic outputs
- M1-T12: integrate with TradingContext
- M1-T13: add workers and locking
- M1-T14: add unit/integration/golden tests
- M1-T15: add docs and learning patterns

### 7.8 Acceptance criteria

Acceptance criteria include:
- hourly run stable with no crashes,
- health gate works,
- degraded health forces WATCH-only behavior,
- triggers have TTL/decay,
- duplicates do not leak into outputs,
- output schemas are deterministic and versioned.

These criteria are directly stated across the uploaded milestone summary and the full NEWS roadmap. fileciteturn9file0turn9file2

### 7.9 Validation

Required validation:
- fixture feeds,
- duplicate detection tests,
- outage simulation,
- stale-on-error behavior,
- golden hash comparisons for trigger outputs.

### 7.10 Stop conditions

Do not proceed to aggressive trigger-dependent systems if:
- health frequently degrades without clear recovery,
- dedupe is unstable,
- event schemas are not deterministic,
- output writing is non-atomic.

---

## 8. Milestone M2 — DISCLOSURES v1 MVP (Confirm / Context)

### 8.1 Objective

Build a disclosure subsystem that automatically ingests House PTR, SEC Form 4 / 13D / 13G, and selected strategic official documents, parses and normalizes them into deterministic event schemas, maps instruments cautiously, scores them with freshness and confidence penalties, and emits machine-readable signals plus health outputs.

This milestone follows the uploaded DISCLOSURES v1 roadmap. fileciteturn9file2

### 8.2 Why M2 matters

DISCLOSURES are not ideal ultra-fast triggers; they are delayed, document-heavy, and mapping-sensitive. Their first job is to act as:
- confirm layer,
- context layer,
- slow-intel enhancer,
- signal quality amplifier.

### 8.3 In scope

Must include:
- source registry and config,
- scheduled worker design,
- House PTR discovery and download,
- SEC incremental ingestion,
- strategic docs ingestion,
- unified `DisclosureEvent` / `StrategicDocEvent` schemas,
- document-level and event-level dedupe,
- parsing with QC gates,
- instrument mapping with explicit confidence thresholds,
- scoring (impact × confidence × freshness),
- TTL/decay,
- aggregated signal generation,
- health monitoring,
- atomic emit,
- TradingContext integration,
- parser fixtures and idempotency tests,
- documentation and learning patterns.

### 8.4 Out of scope

Not for v1:
- Senate full automation,
- full 13F coverage,
- paid mapping providers,
- social or unofficial scrapers as a core input.

### 8.5 Deliverables

Required deliverables:
- House PTR downloader and minimal PDF parsing,
- SEC poller with Form 4 / 13D / 13G parsing,
- `signals_latest.json`,
- `health_latest.json`,
- tests with fixtures.

### 8.6 Suggested task decomposition

Recommended breakdown:
- M2-T01: create disclosure spec and configs
- M2-T02: source registry and cadence setup
- M2-T03: implement House index discoverer
- M2-T04: implement House PDF downloader
- M2-T05: implement SEC new-filings detector
- M2-T06: implement filing downloader
- M2-T07: implement strategic docs fetcher
- M2-T08: implement normalization schemas
- M2-T09: implement dedupe keys and stores
- M2-T10: implement parsing with parse QC
- M2-T11: implement mapping ladder + map confidence
- M2-T12: implement scoring and freshness decay
- M2-T13: aggregate signals
- M2-T14: health metrics and degraded behavior
- M2-T15: emit artifacts atomically
- M2-T16: integrate with TradingContext
- M2-T17: add tests and fixtures
- M2-T18: add docs and learning outputs

### 8.7 Acceptance criteria

Acceptance criteria include:
- idempotent ingest,
- parse QC works,
- mapping confidence affects signals,
- degraded health produces WATCH-only behavior,
- delayed low-confidence data does not masquerade as strong trade evidence.

These criteria are explicitly reflected in the uploaded roadmap. fileciteturn9file0turn9file2

### 8.8 Validation

Required validation:
- repeated runs do not duplicate events,
- parser fixtures for multiple document variants,
- golden hashes on signal outputs,
- outage simulation still writes health outputs.

### 8.9 Stop conditions

Do not rely on DISCLOSURES for stronger trading influence if:
- parser QC is poor,
- mapping confidence is unstable,
- event dedupe is unreliable,
- health reporting does not accurately reflect degraded ingest.

---

## 9. Milestone M3 — Risk / State Machine v1

### 9.1 Objective

Formalize a persistent, deterministic risk and state machine for the baseline system. The system must know whether it is in NORMAL, DE_RISK, CRISIS, or PAUSE and adjust exposure and risk behavior accordingly.

This milestone is present both in the milestone summary and in the dedicated risk/state-machine roadmap. fileciteturn9file0turn9file2

### 9.2 Why M3 matters

The trading core is only trustworthy if risk state is explicit, persistent, deterministic, and testable.

### 9.3 In scope

Must include:
- persistent risk state,
- deterministic transitions,
- drawdown guards,
- exposure caps per state,
- logs/reporting,
- integration with the trading cycle,
- volatility-targeting-ready architecture,
- no flip-flop behavior.

### 9.4 Out of scope

Not all v1.1 guards need to be present yet. M3 should deliver the formal state model and basic caps/guards, not the full advanced risk stack.

### 9.5 Deliverables

Required deliverables:
- persistent state storage,
- drawdown guards basic,
- exposure caps per state,
- logs and reporting.

### 9.6 Suggested task decomposition

Recommended breakdown:
- M3-T01: create risk state machine spec
- M3-T02: define persistent state format
- M3-T03: implement deterministic transition function
- M3-T04: add hysteresis / debounce
- M3-T05: implement drawdown computation
- M3-T06: implement soft and hard drawdown guards
- M3-T07: integrate exposure caps
- M3-T08: integrate with trading cycle hook
- M3-T09: add tests for transitions and persistence
- M3-T10: add logging and docs

### 9.7 Acceptance criteria

Acceptance criteria:
- drawdown transitions deterministic and tested,
- no flip-flop,
- integrates with trading cycle,
- state survives restart,
- logs explain transition reasons.

### 9.8 Validation

Required validation:
- unit tests for transition logic,
- synthetic equity scenarios,
- persistence tests,
- compatibility with current trading cycle path.

### 9.9 Stop conditions

Do not proceed to operational worker splitting or crisis activation if:
- state transitions are nondeterministic,
- persistence is fragile,
- drawdown thresholds cause oscillation,
- integration with the trading cycle is unclear.

---

## 10. Milestone M4 — Execution Workers (Ops v1)

### 10.1 Objective

Separate critical operational actions from the main trading cycle into idempotent workers so that stop handling, reconcile, and kill behavior do not depend on a single monolithic run loop.

This milestone is directly defined in the uploaded roadmap and is especially tied to worker splitting and idempotency. fileciteturn9file0

### 10.2 Why M4 matters

A single central run loop is too fragile for high-stakes stop/reconcile/kill behavior.

### 10.3 In scope

Must include:
- `stop_worker`,
- `reconcile_worker`,
- minimal `kill_switch_worker`,
- intent store,
- idempotency keys for hard actions,
- audit logging,
- source-of-truth consistency focus.

### 10.4 Out of scope

Do not try to build a full broker-grade OMS in M4. This is operational hardening, not enterprise OMS completion.

### 10.5 Deliverables

Required deliverables:
- stop worker + reconcile worker minimum,
- intent store + idempotency keys,
- minimal kill switch worker,
- audit logging.

### 10.6 Suggested task decomposition

Recommended breakdown:
- M4-T01: define worker contracts and command surfaces
- M4-T02: create intent store design
- M4-T03: define idempotency key rules
- M4-T04: implement stop worker
- M4-T05: implement reconcile worker
- M4-T06: implement minimal kill worker
- M4-T07: add audit logging and run manifests
- M4-T08: add tests for re-run safety and duplicate prevention
- M4-T09: add docs and operational notes

### 10.7 Acceptance criteria

Acceptance criteria:
- stop execution not dependent on the main cycle,
- reconcile establishes source-of-truth consistency,
- kill switch can pause and flatten safely in paper,
- repeated runs do not duplicate hard actions.

### 10.8 Validation

Required validation:
- idempotent replay tests,
- failure / retry tests,
- reconcile mismatch simulation,
- logging and manifest validation.

### 10.9 Stop conditions

Do not proceed to active crisis-trading behavior if:
- reconcile drift is frequent,
- stop logic is not independently reliable,
- worker idempotency is unproven.

This stop logic is also consistent with the roadmap’s explicit “fix execution before adding features” philosophy. fileciteturn9file0

---

## 11. Milestone M5 — Crisis-Alpha v1

### 11.1 Objective

Build a separate crisis subsystem that can move through WATCH, ACTIVE, COOLDOWN, and PAUSE using:
- News-derived geo triggers,
- evidence requirements,
- market stress confirmation,
- strict risk budgets,
- deterministic activation and deactivation rules,
- high-liquidity ETF baskets only.

The detailed structure follows the uploaded Crisis-Alpha roadmap. fileciteturn9file2

### 11.2 Why M5 matters

Crisis behavior must be separated from the stable baseline system so that it can be activated conservatively, audited clearly, and paused safely.

### 11.3 In scope

Must include:
- crisis alpha spec,
- CrisisAlphaContext,
- state machine with persistence,
- evidence rules,
- market stress confirmation,
- basket definitions,
- simple robust entry setups,
- strict risk budgets,
- scale-invariant exits,
- deactivation logic,
- crisis runner / worker,
- detailed logging and scenario tests.

### 11.4 Out of scope

Not for v1:
- leverage,
- options,
- single-name intraday daytrading,
- complex ML-driven crisis logic,
- social-only activation.

### 11.5 Deliverables

Required deliverables:
- crisis alpha state machine,
- activation gates using news triggers + evidence + market confirmation + health,
- ETF basket config,
- exits (break-even, trail, time stop, no overnight),
- tests against false activation.

### 11.6 Suggested task decomposition

Recommended breakdown:
- M5-T01: create crisis spec and config
- M5-T02: define input contract and health gate
- M5-T03: implement persistent crisis state machine
- M5-T04: implement geo score aggregation
- M5-T05: implement evidence rules
- M5-T06: implement market stress confirmation
- M5-T07: define crisis baskets and filters
- M5-T08: implement simple entries
- M5-T09: implement risk budgets
- M5-T10: implement exits and deactivation
- M5-T11: implement runner and workers
- M5-T12: add scenario tests and integration tests
- M5-T13: add docs and learning patterns

### 11.7 Acceptance criteria

Acceptance criteria:
- social-only cannot activate,
- degraded health cannot activate,
- deactivation and cooldown work,
- max daily loss pauses,
- all transitions and protective actions are deterministic and logged.

These match the milestone summary and full crisis roadmap. fileciteturn9file0turn9file2

### 11.8 Validation

Required validation:
- scenario tests: shock begins, normalizes, health degrades, daily loss exceeded,
- integration tests with NEWS outputs,
- no false ACTIVE on social-only evidence,
- no orphaned positions after deactivation.

### 11.9 Stop conditions

Do not enable ACTIVE crisis behavior if:
- health gating is weak,
- stop/kill workers are not reliable,
- evidence rules are bypassable,
- no-overnight and loss-pause behavior are not proven.

---

## 12. Milestone M6 — Risk v1.1 Upgrades

### 12.1 Objective

Enhance the baseline risk engine with more institutional discipline:
- volatility targeting,
- turnover budget,
- time stop / zombie killer,
- profit lock,
- correlation / clump guard,
- attribution and parameter stability checks.

This is explicitly named in the milestone summary and supported by the deeper risk-engine roadmap. fileciteturn9file0turn9file2

### 12.2 Why M6 matters

Once baseline states and workers exist, the next risk problem is not survival alone but quality of sizing, turnover, clustering risk, and avoided givebacks.

### 12.3 In scope

Must include:
- portfolio-level vol targeting,
- turnover budget,
- time stop / zombie killer,
- profit lock,
- correlation guard,
- attribution,
- parameter stability checks.

### 12.4 Out of scope

M6 is not a total re-architecture. It should extend the baseline portfolio/risk/control hooks that already exist.

### 12.5 Deliverables

Required deliverables:
- vol targeting,
- turnover budget,
- time stop / zombie killer,
- profit lock,
- correlation guard,
- attribution + parameter stability checks.

### 12.6 Suggested task decomposition

Recommended breakdown:
- M6-T01: define target vol and realized vol calc
- M6-T02: implement exposure scaling
- M6-T03: connect vol targeting to existing risk hooks
- M6-T04: implement turnover accounting and budget enforcement
- M6-T05: implement time stop / zombie killer rules
- M6-T06: implement break-even/profit lock rules
- M6-T07: implement correlation / cluster guard
- M6-T08: implement attribution reports
- M6-T09: implement parameter stability checks
- M6-T10: add tests and reporting

### 12.7 Acceptance criteria

Acceptance criteria:
- out-of-sample stability improves,
- turnover is controlled,
- fewer givebacks occur,
- reports are reproducible.

### 12.8 Validation

Required validation:
- before/after report comparisons,
- turnover metrics,
- stability checks over windows/regimes,
- deterministic report generation.

### 12.9 Stop conditions

Do not proceed to realism upgrades if:
- turnover remains uncontrolled,
- vol scaling is unstable,
- parameter sensitivity remains opaque.

---

## 13. Milestone M7 — Realism Upgrades v2

### 13.1 Objective

Improve historical realism and interpretability of the backtest / paper environment:
- exchange calendars,
- better cost model,
- corporate actions feed / events,
- universe snapshots.

This milestone is directly named in the uploaded roadmap. fileciteturn9file0turn9file1

### 13.2 Why M7 matters

Earlier milestones may work directionally, but realism gaps can still create misleading confidence.

### 13.3 In scope

Must include:
- exchange calendar handling,
- better cost model,
- corporate actions feed or handling layer,
- universe snapshots or security master-like support,
- explicit labeling of realism level in outputs.

### 13.4 Out of scope

Not every advanced historical fidelity problem must be solved fully here, but the project must become transparent about realism assumptions.

### 13.5 Deliverables

Required deliverables:
- exchange calendar,
- better cost model,
- corporate actions feed,
- universe snapshots.

### 13.6 Acceptance criteria

Acceptance criteria:
- backtests labeled by realism level,
- delta between approximate and more realistic runs is measurable and understood,
- realism assumptions are documented.

### 13.7 Validation

Required validation:
- compare pre/post realism backtests,
- ensure calendars and corporate actions are used consistently,
- run manifest or report includes realism metadata.

### 13.8 Stop conditions

Do not claim institutional-grade realism if:
- calendars are still ignored,
- corporate actions remain untracked,
- universe history is still implicit and unstable.

---

## 14. Milestone M8 — Evidence Engine

### 14.1 Objective

Build an evidence quality layer that grades multi-source confirmation, detects misinfo risk, and gates crisis/trade activation on evidence strength.

### 14.2 In scope

- `EvidenceGrade` enum (A/B/C/D) with activation permissions,
- evidence grading function based on source count, confirmation, and trigger quality,
- misinfo risk scoring (single-source penalty, social-only penalty, contradiction detection),
- action gating: prevent ACTIVE state entry on low-grade evidence,
- integration with Crisis Alpha gates.

### 14.3 Deliverables

- `src/assembled_core/events/evidence_engine/grades.py`
- `src/assembled_core/events/evidence_engine/grader.py`
- `src/assembled_core/events/evidence_engine/misinfo_risk.py`
- `src/assembled_core/events/evidence_engine/action_gate.py`

### 14.4 Acceptance criteria

- Grade-D evidence cannot activate crisis alpha,
- single-source triggers receive misinfo penalty,
- grades are deterministic and testable.

### 14.5 Implementation status

Status: **locally tested**. All deliverables exist and are integrated with Crisis Alpha. 65 targeted tests pass.

---

## 14b. Milestone M9 — Policy Calibration

### 14b.1 Objective

Replace all placeholder/TBD values in `configs/policy.yaml` with concrete, researched risk parameters. Guard with regression tests.

### 14b.2 Deliverables

- `configs/policy.yaml` — fully calibrated (target_vol, drawdown thresholds, position weights, turnover caps, state machine thresholds, health gates),
- `tests/test_policy_calibration.py` — 9 tests ensuring no TBD values and sane ranges.

### 14b.3 Implementation status

Status: **locally tested**. Policy fully calibrated. 9 tests pass.

---

## 14c. Milestone M10 — ETF Universe

### 14c.1 Objective

Define and operationalize a curated ETF universe for crisis alpha baskets, macro overlays, and diversified paper trading.

### 14c.2 Deliverables

- `configs/universe_etf_v1.yaml` — 30+ liquid ETFs by asset class,
- `src/assembled_core/data/universe_etf.py` — loader with filtering by asset class, group, defensive purpose,
- 22 tests.

### 14c.3 Implementation status

Status: **locally tested**. 22 tests pass.

---

## 14d. Milestone M11 — Post-Trade Learning Loop

### 14d.1 Objective

Build a feedback loop that analyzes past trades, computes signal hit rates, and stores learning records for strategy improvement.

### 14d.2 Deliverables

- `src/assembled_core/qa/post_trade_analyzer.py` — forward returns, signal hit rate, learning record builder,
- `src/assembled_core/qa/learning_store.py` — atomic JSONL append/load/summarize,
- `scripts/run_post_trade_analysis.py`,
- 29 tests.

### 14d.3 Known gaps

The `_post_trade_worker` in `daily_scheduler.py` is a stub returning `status="skip"`. Full scheduler integration deferred until operational data is available.

### 14d.4 Implementation status

Status: **locally tested** (core modules). Scheduler integration is a stub.

---

## 14e. Milestone M12 — Broker Adapter

### 14e.1 Objective

Build a broker connectivity layer with paper-first safety design.

### 14e.2 Deliverables

- `src/assembled_core/execution/broker_adapter.py` — `BrokerAdapter` ABC, `AlpacaAdapter` (paper-only default), dual-gate safety, `BrokerOrder` / `BrokerPosition` dataclasses, `create_adapter_from_env` factory,
- 19 tests.

### 14e.3 Acceptance criteria

- Paper mode is the enforced default,
- live mode requires explicit double opt-in (`force_paper=False` AND `ALPACA_ALLOW_LIVE=true`),
- adapter factory works from environment variables.

### 14e.4 Implementation status

Status: **locally tested**. Paper-only by design.

---

## 14f. Milestone M13 — Autonomous Operations

### 14f.1 Objective

Build an orchestration framework for autonomous daily trading cycles.

### 14f.2 Deliverables

- `src/assembled_core/ops/daily_scheduler.py` — `DailyScheduler`, `WorkerResult`, 4 workers (ingest, post-trade, reconcile, health), `build_cycle_summary`, `run_daily_cycle`, `schedule_loop`,
- 10 tests.

### 14f.3 Known gaps

3 of 4 workers (`_ingest_worker`, `_post_trade_worker`, `_reconcile_worker`) are stubs returning `status="skip"`. Only `_health_check_worker` performs real work. The framework is operational but substantive worker implementations await configuration.

### 14f.4 Implementation status

Status: **skeleton present / locally tested** (framework). Worker implementations are stubs.

---

## 15. Practical sprint ordering

The uploaded roadmap suggests this practical sequence:  
- Sprint 1: M0 + M1, parallel start of M3  
- Sprint 2: M2 + M4, finish M3  
- Sprint 3: M5  
- Sprint 4: M6 core  
- Sprint 5: M7 realism
- Sprint 6: M8 (Evidence Engine) + M9 (Policy Calibration) + M10 (ETF Universe)
- Sprint 7: M11 (Post-Trade Learning) + M12 (Broker Adapter)
- Sprint 8: M13 (Autonomous Operations) fileciteturn9file0

For execution discipline, use this ordering as a default but always obey local blockers and stop conditions.

---

## 16. Cross-cutting requirements for every milestone

These apply everywhere and are repeatedly emphasized in the roadmap set:
- idempotency for all workers,
- atomic outputs (temp + rename),
- health/freshness in every subsystem,
- audit logs for every important decision,
- tests at unit + integration + deterministic/golden level where appropriate,
- learning loop: every significant bug should generate an incident/pattern entry.

These requirements are explicitly stated in the uploaded master roadmap. fileciteturn9file0

Additionally, the backend roadmap contributes cross-cutting engineering themes that must be respected throughout execution:
- no secrets in repo,
- clear packaging under `src/assembled_core`,
- testability and CI discipline,
- structured logging and observability,
- promotion path from research to paper to safer operations. fileciteturn9file1

---

## 17. Global stop conditions

Work must pause and stabilize before adding more complexity when any of the following are true:
- NEWS health frequently degrades and gating is unreliable,
- reconcile drift is frequent,
- drawdown guards are not being respected,
- state machines flip-flop,
- outputs are not atomic,
- source-of-truth files are inconsistent,
- tests are not deterministic enough to trust changes.

This aligns with the uploaded roadmap’s explicit stop-condition section. fileciteturn9file0

---

## 18. Definition of done policy across milestones

A milestone is only considered complete when all of the following are true:
- required files/modules/configs exist,
- output contracts are documented,
- tests required by the milestone pass locally,
- no unresolved contradiction exists with governance docs,
- `ROADMAP_STATE.md` is updated,
- `ROADMAP_LOG.md` or equivalent execution log records the outcome,
- milestone status can be expressed honestly using the project status taxonomy.

Recommended milestone-close questions:
1. What is implemented?
2. What is only specified?
3. What is only partially integrated?
4. What was actually tested?
5. What remains risky?
6. What is the next smallest safe step?

---

## 19. How Claude Code should use this file

Claude Code should not treat this file as a command to implement everything at once.

Correct usage pattern:
1. Read `ROADMAP_STATE.md` first.
2. Identify current milestone and current task.
3. Read only the relevant milestone section(s) from this file.
4. Execute the smallest safe step.
5. Update state and log.

Incorrect usage pattern:
- “Implement the whole roadmap.”
- “Continue everything.”
- “Do all milestones in sequence without checkpoints.”

---

## 20. Maintenance rules for this document

Update this document only when:
- milestone structure changes,
- dependencies change,
- scope changes,
- acceptance criteria change,
- a major risk or stop condition is discovered.

Do **not** update this file for every tiny implementation detail. Day-to-day position belongs in `ROADMAP_STATE.md`, and chronological history belongs in `ROADMAP_LOG.md`.

---

## 21. Default next action

If no current action is set elsewhere, the default first action is:
- check `ROADMAP_STATE.md` for the current execution position,
- identify the highest-priority open gap or next milestone task,
- execute the smallest safe step and update state.

As of 2026-04-02, M0–M13 are locally tested. Key remaining work:
- wire `zombie_killer` and `correlation_guard` into `trading_cycle.py` (M6 gap),
- complete M2 instrument mapping and 13D/13G fetch (M2 gaps),
- replace M13 scheduler worker stubs with real implementations,
- achieve full CI green on all 6 GitHub Actions workflows,
- begin optimization work (TA expansion, Monte Carlo, advanced position sizing).

This default is justified by the milestone dependency chain described in the roadmap set: M1 depends on M0, while M3 can start in parallel only if governance is already stable enough. fileciteturn9file0
