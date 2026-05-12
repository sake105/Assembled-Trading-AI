# Hexagonal Architecture — Migration Plan

> Audit C-001 → C-007. The Month-1 skeleton is **shipped** (Wave 17,
> commit chain following d672468). This document is the file-by-file
> map for the remaining five months of the migration.
>
> **Status today (Wave 17):** skeleton + 6 ports + 4 adapter classes +
> bootstrap container + 1 use-case + layering-invariant test. No
> existing call sites have been moved yet — the new path coexists with
> the legacy path.

## Why incremental, not big-bang

The repo has ~50 modules with deep cross-references. CLAUDE.md §10
explicitly forbids big-bang refactors without scope; the audit itself
estimates 32-80h spread over 6 months. A weekend "rewrite everything"
attempt would break ~thousands of tests in ways that take days to
unwind. We migrate **one bounded context per sprint**, with the legacy
path staying green at every step.

## What is in place (Month 1 — Wave 17)

| Layer | File(s) | Status |
|---|---|---|
| `domain/` | trading, risk, accounting, research, operations — all empty packages | scaffold |
| `ports/` | clock, alert_channel, audit_logger, event_bus, order_router, prices_repository | **6 Protocols shipped** |
| `application/use_cases/` | record_kill_switch_trip | **1 use-case shipped** |
| `adapters/outbound/` | alerting_adapter, clock_adapter, audit_logger_adapter | **4 adapter classes shipped** |
| `bootstrap/` | container.Container + build_production_container + build_test_container | **shipped** |
| `tests/test_hexagonal_layering.py` | invariant: domain MUST NOT import from adapters | **shipped** |

## Month 2 — Application use-cases

Migrate the existing pipeline-orchestrator scripts into use-cases.
Each use-case is a class with a single ``execute`` method and gets a
``Container`` in its constructor.

| Existing script | New use-case | Effort |
|---|---|---|
| `scripts/run_daily.py` | `application/use_cases/run_eod_pipeline.RunEodPipeline` | 4h |
| `scripts/run_backtest_strategy.py` | `application/use_cases/run_backtest.RunBacktest` | 6h |
| `scripts/run_api.py` | inbound adapter: `adapters/inbound/http/main.py` (no use-case needed; just routes) | 2h |
| Paper-trading routes | `application/use_cases/submit_paper_order.SubmitPaperOrder` | 6h |

**Acceptance criterion**: each use-case has a unit test that uses
`build_test_container()` to swap real I/O for in-memory fakes.

## Month 3 — Event-Sourcing for Order Lifecycle

Audit C-005. Move `execution/order_lifecycle.OrderLifecycleTracker`
into the domain layer as an Aggregate root, with append-only
``OrderEvent`` records persisted via `AuditLogger`. The existing
`OrderState` enum stays exactly as it is — only the storage shape
changes.

| Existing file | New file | Notes |
|---|---|---|
| `execution/order_lifecycle.py` | `domain/trading/order.py` + `domain/trading/order_events.py` | enum + transitions stay |
| (none) | `adapters/outbound/event_store_sqlite.py` | sqlite append-only event_store table |
| (none) | `application/use_cases/replay_order_history.py` | rebuild state from events |

**Acceptance**: the existing `tests/test_property_fsm_pit.py` Order-FSM
property tests pass unmodified against the new aggregate.

## Month 4 — Plugin architecture for strategies

Audit C-004. Move `src/assembled_core/strategies/` into a plugin
registry that uses `pyproject.toml` entry points.

```toml
# pyproject.toml
[project.entry-points."assembled_trading.strategies"]
trend_baseline = "assembled_core.strategies.trend_baseline:TrendBaseline"
multifactor_v2 = "assembled_core.strategies.multifactor_v2:MultifactorV2"
```

Then `application/strategy_registry.load_strategies()` discovers
every entry-point at startup. Third parties — and our own ERWEITERUNG
branch — can ship strategies as separate pip-installable packages.

## Month 5 — Per-bounded-context tests

Move the existing tests under `tests/domain/{trading,risk,...}/` and
have each BC's test-suite forbid imports outside its own BC + ports.
Use the same `test_hexagonal_layering.py` pattern that already exists.

## Month 6 — Property + Mutation testing

Already partly shipped (Wave 1 property tests + Wave 8 snapshots + audit
plan E-007 mutmut). Round out by adding mutmut against
`domain/risk/` once that BC has real code.

## Concrete file-mapping registry

When in doubt about where a module belongs, this table is authoritative:

| Current path | Hexagonal home | Why |
|---|---|---|
| `execution/kill_switch.py` | `domain/risk/kill_switch.py` + adapter in `adapters/outbound/kill_switch_state_jsonl.py` | state-machine is domain; file I/O is adapter |
| `execution/order_lifecycle.py` | `domain/trading/order.py` | pure FSM logic |
| `execution/pre_trade_checks.py` | `domain/risk/pre_trade_gate.py` | pure rules |
| `execution/paper_trading_engine.py` | split — engine state to `domain/trading/paper_engine.py`, broker API to `adapters/outbound/paper_broker.py` | classic split |
| `execution/broker_adapter.py` | `adapters/outbound/broker/{alpaca,ibkr,paper}.py` | already adapter shape |
| `qa/metrics.py` | `domain/research/metrics.py` | pure-math, no I/O |
| `qa/qa_gates.py` | `domain/research/quant_gates.py` | pure rules |
| `accounting/reconciliation.py` | `domain/accounting/reconciliation.py` + adapter for AlertManager | rules + adapter |
| `ops/alerting.py` | `adapters/outbound/alerting_adapter.py` (already done) | pure I/O |
| `data/factor_store.py` | `adapters/outbound/factor_store_parquet.py` | pure I/O |
| `data/sources/*.py` | `adapters/outbound/data_sources/*.py` | pure I/O |
| `api/app.py` | `adapters/inbound/http/main.py` | inbound adapter |
| `api/routers/*.py` | `adapters/inbound/http/routers/*.py` | inbound adapter |
| `api/auth.py` | `adapters/inbound/http/auth.py` | inbound adapter |
| `api/middleware.py` | `adapters/inbound/http/middleware.py` | inbound adapter |
| `signals/meta_model.py` | `domain/research/meta_model.py` (pure inference) + `adapters/outbound/model_store_joblib.py` (persistence) | classic split |
| `features/*.py` | `domain/research/features/*.py` | pure transformations |
| `utils/clock_drift.py` | adapter behind `ports.clock` | clock-related infra |
| `utils/retry.py` | stays under `utils/` (cross-cutting helper) | no layer ownership |
| `utils/async_fetch.py` | `adapters/outbound/http_fetch.py` | adapter |
| `utils/bulkhead.py` | `adapters/outbound/bulkhead.py` (or pull into utils) | infra |

Anything not in this table stays where it is until a use-case forces
the move.

## Rollback plan

Every migration sprint ships behind a feature flag (env var
`ASSEMBLED_USE_HEXAGONAL=1`). The legacy path stays the default
until the new path has run paper-track green for two weeks.

## Acceptance criteria for "done"

- All five bounded contexts have at least one module under
  `domain/<bc>/`.
- All inbound entry points (CLI, REST, scheduler) live under
  `adapters/inbound/`.
- All third-party-library imports are confined to `adapters/`.
- `tests/test_hexagonal_layering.py` passes (it already does for the
  skeleton).
- The existing test suite passes unchanged.
