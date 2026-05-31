# INTEL_SPEC.md — Intel, News & Research Arms

**Version:** 1.0  
**Date:** 2026-04-19  
**Status:** Active (post goofy-questing-crystal remediation)

---

## 1. Overview

The Intel arm consists of three pipelines:

| Pipeline | Entry Point | Output |
|----------|-------------|--------|
| News | `run_news_worker.py` | `output/intel/news/` |
| Disclosures | `run_disclosures_worker.py` | `output/intel/disclosures/` |
| Intel Cycle (GDELT) | `run_intel_cycle.py` | `data/intel/` |

All three respect `policy.intel.kill_switch.enabled` (T4.7).

---

## 2. Module Map

### Active (production)

| Module | Path | Purpose |
|--------|------|---------|
| News pipeline | `events/news/pipeline.py` | Fetch→normalize→dedupe→cluster→score→emit |
| News dedupe (SQLite) | `events/news/dedupe_store.py` | WAL-mode SQLite, 14-day window |
| News dedupe (in-memory) | `intel/news_dedupe.py` | LRU OrderedDict, 7-day TTL |
| News normalizer | `events/news/normalize.py` | Timestamp clamping, schema validation |
| News cluster | `intel/news_cluster.py` | 1-hour buckets by `published_at`, TTL=360min |
| GeoTrigger scoring | `intel/geo_trigger.py` | Classify → score → `GeoTrigger` objects |
| Shock propagation | `intel/shock_propagation.py` | Geo → symbol basket shocks |
| Crisis state machine | `events/crisis_alpha/` | Pure functions, pydantic, 63 tests |
| Crisis worker (v0) | `intel/crisis_alpha_worker.py` | **DEPRECATED** — see T3.6 |
| Disclosures pipeline | `events/disclosures/pipeline.py` | EDGAR + House PTR |
| EDGAR fetcher | `events/disclosures/fetch_edgar.py` | Form 4 + stale-cache guard |
| House PTR fetcher | `events/disclosures/fetch_house_ptr.py` | Stale-cache guard |
| Market confirmation | `intel/market_confirmation.py` | Used by run_intel_cycle.py |
| Bayesian confidence | `intel/bayesian_confidence.py` | Shadow-mode (T2.8), not yet flipped |
| Health monitor | `intel/health_monitor.py` | Component freshness |
| News triggers loader | `intel/news_triggers_loader.py` | Read-only snapshot loader |
| Disclosures loader | `intel/disclosures_triggers_loader.py` | Read-only snapshot loader |
| Trigger snapshot store | `intel/trigger_snapshot_store.py` | Per-run PIT archival (T6.1/X1-lite) |
| Entity-Ticker Linker | **NOT YET IMPLEMENTED** | Tracked as X2 |
| PIT Snapshot Store | **X1-lite in trigger_snapshot_store.py** | Full X1 TBD |

### Archived (2026-04-19, 0 call-sites)

Moved to `archive/intel_research_2026q2/`:

- `shock_correlation.py`, `multichannel_propagation.py`, `feedback_loops.py` (V1)
- `sensitivity_analysis.py` (V3)
- `escalation_model.py`, `escalation_tracker.py`, `scenario_trees.py`, `wargaming.py` (V2)
- `hegemonic_dynamics.py`, `structural_cycles.py` (V6)
- `features/earnings_call_nlp.py`, `features/analyst_features.py` (V7)
- `events/news/fetch_acled.py` (N5 stub)

See `archive/intel_research_2026q2/README.md` for per-module decision notes.

---

## 3. Data Flow

```
RSS / GDELT / EDGAR / House PTR
         ↓
    [Fetch + Normalize]       ← published_at clamped to fetched_utc (T2.4)
         ↓
    [Dedupe]                  ← WAL SQLite (T1.1) + LRU in-memory (T1.5)
         ↓
    [Cluster]                 ← 1-hour buckets by published_at (T2.2)
         ↓
    [Score → GeoTrigger]      ← Bayesian confidence shadow (T2.8)
         ↓
    [Shock Propagation]       ← Geographic → basket impacts
         ↓
    [Crisis State Machine]    ← Evidence-grade gate (T2.5 default-deny)
         ↓
    [TradingContext]          ← GeoRisk overlay + signal_layer (T4.5 pending X2)
```

---

## 4. Policy Gates

All gates in `configs/policy.yaml` under `intel:`:

| Key | Default | Purpose |
|-----|---------|---------|
| `intel.kill_switch.enabled` | `false` | Halt all three workers |
| `intel.crisis_alpha.enabled` | `false` | Wire crisis_alpha v1 into trading loop |
| `intel.crisis_alpha.shadow_only` | `true` | Log result, no order impact |
| `intel.signal_layer.enabled` | `false` | Intel → signal scoring (pending X2) |
| `intel.disclosures_triggers.enabled` | `false` | Load disclosures into TradingContext |

See `configs/policy.yaml` for full feature flag list (`features:` section).

---

## 5. Shadow-Mode Tasks (not yet flipped)

| Task | What | Flip condition |
|------|------|---------------|
| T2.1 | news_features.py as_of gate | User review after 250-day replay diff |
| T2.3 | altdata Form-4 T+2 latency | User review after shadow diff |
| T2.8 | Bayesian cluster confidence | User review after 7-day logging |
| T4.1 | Crisis-Alpha in trading loop | Enabled by default after Step 3 |

---

## 6. MNPI Policy

This system operates on **public data only**:
- SEC EDGAR filings (public, time-delayed)
- House PTR disclosures (public, time-delayed)
- GDELT (public news aggregation)
- Reuters/AP via public APIs

**No MNPI-derived signals.** See CLAUDE.md, Abschnitt „MNPI".

---

## 7. PIT Correctness

Key PIT rules enforced:
- `news_features.py`: `as_of` parameter gates event data (shadow — T2.1)
- `altdata_earnings_insider_factors.py`: Form-4 T+2 latency via `ASSEMBLED_STRICT_PIT_CHECKS` (shadow — T2.3)
- `news_cluster.py`: bucket by `published_at`, not `now` (active — T2.2)
- `normalize.py`: `published_at = min(published_at, fetched_at)` (active — T2.4)

---

## 8. Open Work

| Task | Priority | Blocking |
|------|----------|---------|
| X2 Entity→Ticker Linker | High | T4.4, T4.5 flip |
| T2.1 flip (as_of gate) | High | User review |
| T2.8 flip (Bayesian) | Medium | 7-day shadow log |
| T6.3 Source-Uptime-SLA | Medium | — |
| T6.4 News-PnL-Attribution | Medium | — |
| X3 IC-Loop | Medium | — |
| T5.2 GDELT hardening | Low | — |
| T5.3 House-PTR PDF parser | Low | — |
