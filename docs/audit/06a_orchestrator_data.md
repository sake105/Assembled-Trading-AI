# 06a — Cycle Orchestration + Data/Feature Ingress (DEEP-AUDIT Round 2, Agent A)

- **Date:** 2026-05-30
- **Cluster:** CYCLE ORCHESTRATION + DATA/FEATURE INGRESS (the spine of the trading machine)
- **Mode:** read-only analysis. Nothing was changed. No `src/`, `scripts/`, `.github/`, `.claude/` file was edited, deleted or moved. The only file written is this report.
- **Modules inspected (read fully unless noted):**
  - `src/assembled_core/pipeline/trading_cycle_v2.py` (790 ln, full) — the one live/paper cycle
  - `src/assembled_core/pipeline/trading_cycle.py` (16 ln) — thin re-export shim
  - `src/assembled_core/pipeline/trading_cycle_shared.py` (1557 ln; ctx/result classes + `_filter_prices_for_as_of` + `_generate_orders_default` + `_apply_risk_controls_default` read)
  - `src/assembled_core/pipeline/orchestrator.py` (header + structure) — the second, documented EOD-batch pipeline
  - `src/assembled_core/pipeline/_shared_eod.py` (111 ln, full)
  - `src/assembled_core/pipeline/_tc_features.py` (691 ln, full)
  - `src/assembled_core/pipeline/_tc_signals.py` (1104 ln, full)
  - `src/assembled_core/pipeline/_tc_sizing.py` (2474 ln; `size_positions` + dispatch read)
  - `src/assembled_core/pipeline/_tc_risk.py` (374 ln, full)
  - `src/assembled_core/pipeline/_tc_execution.py` (698 ln, full)
  - `src/assembled_core/pipeline/dispatcher.py` (Strangler-Fig `SignalDispatcher`)
  - `src/assembled_core/data/prices_ingest.py` (`load_eod_prices`)

Round-1 findings are NOT repeated except where a deeper related consequence was traced:
- R1 M-6 = `trading_cycle_v2.py:603-608` (`pl_update` overwrites `prices_filtered`+`prices_latest`) — see **A2-06** for the propagation chain.
- R1 = `prices_ingest.py:134-146` (strips non-OHLCV columns).

---

## Findings table

| ID | Modul:Zeile | Fund | Beleg-Snippet | Schwere | betrifft |
|----|-------------|------|---------------|---------|----------|
| A2-01 | `_tc_risk.py:213-222, 225-243, 246-254, 297-314` | **Risk gates FAIL-OPEN on exception.** Every post-pre-trade gate (VaR 220, auto-DD kill-switch 241, circuit-breaker 252, fat-finger 313) is wrapped in `try/except Exception → log.warning("… no-op") `. A gate that *raises* (bad policy, NaN, missing column) does **not** block orders — it silently passes them through. The gate that is supposed to be the last line of defense becomes a no-op exactly when its inputs are malformed. | `except Exception as e:` / `log.warning("[RISK] var_gate evaluation raised — gate no-op: %s", e)` (220-221); `log.warning("[RISK] fat_finger_guard raised — hard cap not applied: %s", e)` (314) | HOCH | Live/Paper |
| A2-02 | `trading_cycle_shared.py:879-881` | **Order-fill price taken via `groupby.last()` WITHOUT timestamp sort.** `_generate_orders_default` slices `<= as_of` (875) then `p.groupby("symbol")["close"].last()` — `.last()` returns the *last row in the current frame order*, not the latest timestamp. PIT-correctness depends on the caller pre-sorting `ctx.prices` ascending by `(symbol, timestamp)`. The `TradingContext.prices` docstring *requires* this (line 55) but it is unenforced; an unsorted/append-out-of-order frame yields a stale or future-within-window fill price. | `prices_for_orders = (p.groupby("symbol", group_keys=False)["close"].last().reset_index())` | MITTEL | Live/Paper + OOS |
| A2-03 | `trading_cycle_shared.py:428-432` | **Same unsorted `groupby.last()` in `_filter_prices_for_as_of` eod/paper/live branch.** The backtest branch sorts (`sort_values(["timestamp","symbol"])` at 424) BEFORE the per-symbol reduction; the eod/paper/live branch does `groupby("symbol").last()` (429) and only sorts the *result* by symbol (432) — never by timestamp before the reduce. The "last row" therefore again depends on input order. Divergent treatment between modes for the very same PIT reduction. | `filtered = filtered.groupby("symbol", group_keys=False, dropna=False).last()` (429) — no prior timestamp sort | MITTEL | Live/Paper |
| A2-04 | `_tc_sizing.py:2062-2065` ∧ `_tc_signals.py:620-623` | **Two cycle stages bypass the documented policy cache** (`ctx._policy_cache`) and re-read policy from disk per bar. `ingest_data` explicitly caches: *"Cache on ctx so pipeline stages skip redundant disk reads"* (`trading_cycle_v2.py:144-148`). `_tc_features` (93), `_tc_signals` (50), `_tc_risk` (52), `_tc_execution` (49) all honour the cache. But `size_positions` (`load_policy()` at 2063) and `_ensemble_signals_if_enabled` (`load_policy()` at 623) re-read fresh — redundant I/O AND a mid-cycle inconsistency window if policy.yaml is edited during a run. | `try: policy = load_policy()` (2063) vs `policy = getattr(ctx, "_policy_cache", None)` (other 4 stages) | MITTEL | Live/Paper |
| A2-05 | `_tc_signals.py` (≈13 layers) + `_tc_features.py` macro/news merges | **Pervasive `except Exception → log.debug("… skipped")`: a load/compute failure is indistinguishable from "no signal".** Every enrichment layer (intel, sector, earnings, bayesian, mean-reversion, multifactor, GNN, ensemble, pairs) wraps its body and degrades to "no contribution" at DEBUG level. At default log level a broken altdata feed, a corrupt parquet, or a raised exception produces the *exact same* observable output as a legitimately empty signal — error masking at ingress (matches anti-pattern E-025 "fail-open masks corruption"). | layer pattern `except Exception: … log.debug("[layer] … skipped")` | MITTEL | Live/Paper + OOS |
| A2-06 | `trading_cycle_v2.py:606-608` → `_tc_sizing.py:2067` → `_tc_risk.py:81` | **R1 M-6 consequence traced: latest-only frame degrades vol/cov/EVT.** `pl_update` (latest row per symbol in backtest-snapshot mode) overwrites BOTH `result.prices_latest` *and* `result.prices_filtered` (608). That same `prices_filtered` is then handed to `size_positions(prices_filtered=…)` (620) → `prices_for_sizing` (2067) and to `check_risk` → `_prices_for_risk = prices_filtered` (81), which feeds the EVT/Copula return pivot. With a 1-row-per-symbol frame, any lookback statistic (realized vol, covariance, EVT tail) collapses to a degenerate/empty estimate — the risk layer silently sizes/checks on a single price point. | `if pl_update is not None: result.prices_latest = pl_update; result.prices_filtered = pl_update` (606-608); `_prices_for_risk = prices_filtered if prices_filtered is not None else ctx.prices` (81) | HOCH | OOS (backtest-snapshot); Live/Paper if snapshot path ever enabled |
| A2-07 | `_tc_risk.py:63-67` | **`enable_risk_controls=False` is a total pre-trade bypass.** Fast-path returns `orders_filtered = orders.copy()` with *zero* gates run — QA gate, VaR, auto-DD, circuit-breaker, fat-finger all skipped. This is by design, but it is a single boolean on the context that disables every safety check at once; no per-gate granularity, no audit-log of the bypass. | `if not getattr(ctx, "enable_risk_controls", True): result.orders_filtered = orders.copy(); return result` | MITTEL | Live/Paper |
| A2-08 | `_shared_eod.py:9-24` ∧ `orchestrator.py:4-13` | **Two genuine EOD-producing code paths (documented, not stealth).** `orchestrator.py` (stateless EOD batch, `assembled-run-daily`) and `trading_cycle_v2.py` (live/paper) share *only* signal generation via `_shared_eod.compute_signals_by_mode`; sizing/risk/order paths diverge. The divergence is openly documented as deferred (*"This divergence is intentional for now. Full convergence (Option A) … is deferred."*). Not a stealth second-truth, but a real maintenance/behaviour-drift surface: a fix in one path is not automatically in the other. | `# This divergence is intentional for now. Full convergence (Option A)` (_shared_eod.py:~17) | NIEDRIG (MITTEL drift-risk) | Live/Paper + OOS |
| A2-09 | `trading_cycle.py` (16 ln) | **`pipeline/trading_cycle.py` is a thin re-export shim, NOT a second cycle.** It re-exports `run_trading_cycle`, `TradingCycleResult` from `trading_cycle_v2`. The old monolith lives under `archive/pipeline_legacy_2026q2/trading_cycle.py` and is not on the import path. Verdict on the "two-truths" question: there is **one** live trading-cycle. (Recorded so the next auditor does not re-flag the shim as a duplicate.) | re-export only; no cycle body | NIEDRIG (informational) | neither |

---

## Answers to the 6 deep questions

**(1) State threading `_tc_features → _tc_signals → _tc_sizing → _tc_risk → _tc_execution` — silently-dropped frames?**
The cycle threads state through the mutable `TradingCycleResult` and explicit kwargs (`trading_cycle_v2.py:597-650`). Two real losses:
- **A2-06 (HOCH):** `prices_filtered` is overwritten by the latest-only `pl_update` (608) and that degraded frame is what `size_positions` and `check_risk` consume — the full-history frame computed by `ingest_data` is discarded for those two stages in snapshot mode.
- The pre-trade meta computed inside risk steps is folded into `result.meta` (good), but the `_rej_counts` reasons are local and only partially surfaced.

**(2) Two-truths / legacy-cycle verdict.**
**One live trading-cycle** (`trading_cycle_v2`); `trading_cycle.py` is a re-export shim (**A2-09**); the legacy monolith is archived off-path. There **is** a second, separate, *documented-as-deferred* EOD-batch pipeline (`orchestrator.py`) sharing only signal gen (**A2-08**). Verdict: not a stealth two-truths, but a real, acknowledged divergence surface.

**(3) Determinism / ordering.**
The principal ordering hazards are the two unsorted `groupby.last()` reductions (**A2-02**, **A2-03**): correctness is *delegated to the caller's sort contract* (`TradingContext.prices` docstring line 55) and `load_eod_prices` does sort (`prices_ingest.py:170`), so on the canonical path it is "usually safe" — but it is **unenforced** in the shared functions, so any caller that builds `ctx.prices` differently (paper_runner passes prices in as a param; tests; future ingest variants) can silently get a wrong "last" row. The latest-bar signal reduce in `_tc_signals` and `prices_latest` are sorted explicitly (good).

**(4) PIT at ingress — tail-read leaks?**
**PIT-safe on the feature/intel/pairs paths**: `_tc_features` slices `<= as_of` (162-178) before feature compute; market-stress is PIT-sliced (`trading_cycle_v2.py:297-317`, a prior fix); `_add_pairs_signals_if_enabled` slices `<= as_of` with an as_of-None guard. The **residual** PIT risk is the *order-dependence* of the `<= as_of` + unsorted `.last()` pattern (**A2-02/03**), not a tail-read past `as_of`. No new look-ahead-past-`as_of` leak found in this cluster.

**(5) Error masking.**
Two distinct masking surfaces: **A2-01** (risk gates fail-open — a raising gate stops blocking) and **A2-05** (enrichment layers degrade to DEBUG-logged no-ops — broken feed ≈ empty signal). Both make a *failure* observationally identical to a *benign zero*.

**(6) dtype / tz / index consistency.**
`_generate_orders_default` defensively re-coerces tz (`pd.to_datetime(..., utc=True)` at 873-874) before the `<= as_of` compare — good, this avoids tz-naive/aware comparison errors. `TradingContext.prices` is contract-documented (schema + sort, lines 54-55) but not validated at construction. No new dtype/tz overflow found in this cluster beyond the unsorted-index dependence already captured in A2-02/03.

---

## Cycle state-flow map

```
run_trading_cycle (trading_cycle_v2.py:532)
  │
  ├─[ctx setup] load_policy() ONCE → object.__setattr__(ctx,"_policy_cache",policy)   (140-148)
  │                                   └─ contract: stages reuse cache (HONOURED by
  │                                      _tc_features/_signals/_risk/_execution;
  │                                      VIOLATED by size_positions:2063 & ensemble:623 → A2-04)
  │
  ├─ ingest_data(ctx)                              (597)  prices, prices_latest
  │     └─ _filter_prices_for_as_of (shared:348)
  │          ├─ backtest: sort(ts,sym) THEN reduce  (424)  ✓
  │          └─ eod/paper/live: groupby.last() UNSORTED (429) → A2-03
  │     result.prices_filtered = prices ; result.prices_latest = prices_latest
  │
  ├─ build_features(prices,ctx) → (features, pl_update)   (603)
  │     PIT slice <= as_of (162-178) ✓
  │     if pl_update is not None:                          (606-608)
  │        result.prices_latest  = pl_update   (latest-only)
  │        result.prices_filtered = pl_update  (latest-only)  ◄── R1 M-6 / A2-06 origin
  │
  ├─ generate_signals(features,ctx)                       (612)
  │     ctx.signal_fn(features) → latest-bar reduce (sorted ✓)
  │     ~13 enrichment layers, each try/except→log.debug("skipped") → A2-05
  │     _ensemble_signals_if_enabled: load_policy() fresh (623) → A2-04
  │
  ├─ size_positions(signals,ctx,                          (617)
  │        prices_filtered = result.prices_filtered  ◄── latest-only in snapshot mode
  │     )  load_policy() fresh (2063) → A2-04
  │        prices_for_sizing = prices_filtered (2067) ◄── degraded vol input → A2-06
  │
  ├─ route_orders(...)                                    (629)
  │     _generate_orders_default (shared:810):
  │        slice <= as_of (875) THEN groupby.last() UNSORTED (879-881) → A2-02
  │        → fill price for notional→shares
  │
  ├─ check_risk(...)                                      (~635)
  │     fast-path if enable_risk_controls=False → orders pass through, ZERO gates (64-67) → A2-07
  │     _prices_for_risk = prices_filtered (81) ◄── latest-only → degenerate EVT/Copula → A2-06
  │     gates VaR/auto-DD/CB/fat-finger each try/except→ "gate no-op" (220/241/252/313) → A2-01
  │
  └─ book_fills(...)  → fills, ledger
```

Separate, documented EOD-batch pipeline (NOT this cycle):
```
orchestrator.py  (assembled-run-daily, stateless batch)
   └─ shares ONLY: _shared_eod.compute_signals_by_mode
   └─ sizing/risk/order paths diverge — convergence deferred (A2-08)
```

---

## Severity roll-up (NEW findings this round)

- **HOCH (2):** A2-01 (risk gates fail-open), A2-06 (latest-only frame degrades vol/cov/EVT in snapshot mode — R1 M-6 traced to risk impact).
- **MITTEL (5):** A2-02, A2-03 (unsorted `groupby.last()` ×2), A2-04 (policy-cache bypass), A2-05 (enrichment error masking), A2-07 (`enable_risk_controls` total bypass).
- **NIEDRIG / informational (2):** A2-08 (documented dual EOD pipeline drift-surface), A2-09 (shim is not a second cycle).

---

## Statement

This report is the result of read-only inspection. **Nothing was changed** in any source, script, workflow, config, or rule file. The only filesystem write performed by this audit is this document (`docs/audit/06a_orchestrator_data.md`). All findings are evidenced by `file:line` references and verbatim snippets quoted from the inspected files as they stood on 2026-05-30. Findings that would require execution to fully confirm (e.g., whether the backtest-snapshot path is reachable in the current live/paper config) are noted in-line as conditional ("if … enabled").
